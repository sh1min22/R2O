import json
from neo4j import GraphDatabase
from datetime import datetime
import pandas as pd
import concurrent
from concurrent.futures import ThreadPoolExecutor
from neo4j.exceptions import TransientError, ServiceUnavailable
from queue import Empty as QueueEmpty
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError, wait, FIRST_COMPLETED
import os
import math
from tqdm import tqdm
from multiprocessing import Process, Manager
import time
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Union, List, Set, Tuple
import json
from py2neo import Graph
from collections import defaultdict

from utils.logger import logger



TIMEOUT = 600
BATCH_SIZE = 2000
MAX_JOIN = 1000000000

EDGE_COUNTS = {
    "hasCreator": 87095182,
    "hasTag": 117466392,
    "isLocatedIn": 87268567,
    "replyOf": 67126524,
    "containerOf": 19968658,
    "hasMember": 53977424,
    "hasModerator": 1503655,
    "hasInterest": 3853216,
    "knows": 6017657,
    "likes": 96127147,
    "studyAt": 132707,
    "workAt": 360671,
    "isPartOf": 1454,
    "isSubclassOf": 70
}


def merge_to_list(a, b):
    set_a = set(a) if isinstance(a, list) else {a}
    set_b = set(b) if isinstance(b, list) else {b}

    return list(set_a | set_b)  # 取并集并转换回 list


def parse_query_results(results):
    tmp_dict = defaultdict(set)

    for record in tqdm(results, desc="Processing records", unit="record"):
        for key, value in record.items():
            tmp_dict[key].add(value)

    return dict(tmp_dict)


def incremental_join_results(result1, result2, join_keys):
    if not result1 or not result2:
        return []

    if isinstance(join_keys, str):
        join_keys = [join_keys]
    if len(result1) <= len(result2):
        smaller, larger = result1, result2
    else:
        smaller, larger = result2, result1

    index = {}
    for row in smaller:
        key_val = tuple(row.get(k) for k in join_keys)
        index.setdefault(key_val, []).append(row)

    merged = []
    for row_l in larger:
        key_val = tuple(row_l.get(k) for k in join_keys)
        if key_val in index:
            for row_s in index[key_val]:
                merged_row = {**row_s, **row_l}
                # 🔽 强制补回 Join Key 字段（避免结果中丢失）
                for k, v in zip(join_keys, key_val):
                    merged_row[k] = v
                merged.append(merged_row)
    return merged


def join_dicts(mid_res_1, mid_res_2):
    joined_dict = {}

    if not mid_res_1:
        return mid_res_2
    if not mid_res_2:
        return mid_res_1
    all_keys = set(mid_res_1.keys()).union(mid_res_2.keys())

    for key in all_keys:
        if key in mid_res_1 and key in mid_res_2:
            joined_dict[key] = mid_res_1[key].intersection(mid_res_2[key])
        elif key in mid_res_1:
            joined_dict[key] = mid_res_1[key]
        else:
            joined_dict[key] = mid_res_2[key]

    return joined_dict


def merge_dicts(mid_res_1, mid_res_2):
    merged_dict = {}

    if not mid_res_1:
        return mid_res_2
    if not mid_res_2:
        return mid_res_1
    all_keys = set(mid_res_1.keys()).union(mid_res_2.keys())

    for key in all_keys:
        if key in mid_res_1 and key in mid_res_2:
            merged_dict[key] = mid_res_1[key].union(mid_res_2[key])
        elif key in mid_res_1:
            merged_dict[key] = mid_res_1[key]
        else:
            merged_dict[key] = mid_res_2[key]

    return merged_dict


def rewrite_query(q1, q2):
    inter_edges = q1.edge_set & q2.edge_set
    q1_conditions = set()
    q2_conditions = set()
    q12_conditions = set()

    for u, v in inter_edges:
        if u in q1.target_node:
            q1_conditions.add(f"size(keys({u})) > 2")
            q1_conditions.add(f"size(keys({v})) < 2")
            q12_conditions.add(f"size(keys({u})) > 2")
        elif v in q1.target_node:
            q1_conditions.add(f"size(keys({v})) > 2")
            q1_conditions.add(f"size(keys({u})) < 2")
            q12_conditions.add(f"size(keys({v})) > 2")
        if u in q2.target_node:
            q2_conditions.add(f"size(keys({u})) > 2")
            q2_conditions.add(f"size(keys({v})) < 2")
            q12_conditions.add(f"size(keys({u})) > 2")
        elif v in q2.target_node:
            q2_conditions.add(f"size(keys({v})) > 2")
            q2_conditions.add(f"size(keys({u})) < 2")
            q12_conditions.add(f"size(keys({v})) > 2")

    q1_where_sentence = f" WHERE {' AND '.join(sorted(q1_conditions))}" if q1_conditions else ""
    q2_where_sentence = f" WHERE {' AND '.join(sorted(q2_conditions))}" if q2_conditions else ""
    q1_query = q1.query + q1_where_sentence
    q2_query = q2.query + q2_where_sentence
    q_1 = SubQuery(q1_query, q1.return_str, q1.target_node, q1.key_set, q1.edge_set)
    q_2 = SubQuery(q2_query, q2.return_str, q2.target_node, q2.key_set, q2.edge_set)

    q12_target_node = merge_to_list(q1.target_node, q2.target_node)
    q12_where_sentence = f" WHERE {' AND '.join(q12_conditions)}" if q12_conditions else ""
    q_12 = SubQuery(merge_clauses(q1.query, q2.query) + q12_where_sentence,
                    merge_return_str(q1.return_str, q2.return_str), q12_target_node, q1.key_set | q2.key_set,
                    q1.edge_set | q2.edge_set)
    return q_1, q_2, q_12



def merge_return_str(return_str1, return_str2):
    pattern = r"RETURN\s+(.+)"

    match1 = re.search(pattern, return_str1)
    match2 = re.search(pattern, return_str2)

    if not match1 or not match2:
        return "Invalid RETURN statements"

    vars1 = set(match1.group(1).split(", "))
    vars2 = set(match2.group(1).split(", "))

    merged_vars = ", ".join(sorted(vars1 | vars2))  # 使用 sorted 保持稳定顺序（可选）

    return f" RETURN {merged_vars}"


def parse_clause(clause):
    clause_str = clause.strip()
    if clause_str.upper().startswith("MATCH"):
        clause_str = clause_str[5:].strip()
    pattern = r"\(\s*(?P<var1>\w+)(?::(?P<label1>\w+))?(?:\s*(?P<props1>\{[^}]+\}))?\s*\)\s*-\[\s*(?:\w*\s*:)?\s*(?P<rel>\w+)\s*\]\s*->\s*\(\s*(?P<var2>\w+)(?::(?P<label2>\w+))?(?:\s*(?P<props2>\{[^}]+\}))?\s*\)"
    matches = re.finditer(pattern, clause_str)
    edges = []
    for m in matches:
        edge = {
            'var1': m.group('var1'),
            'label1': m.group('label1'),
            'props1': m.group('props1'),
            'rel': m.group('rel'),
            'var2': m.group('var2'),
            'label2': m.group('label2'),
            'props2': m.group('props2')
        }
        edges.append(edge)
    return edges


def merge_edges(edges1, edges2):
    merged = {}
    for edge in edges1 + edges2:
        key = (edge['var1'], edge['rel'], edge['var2'])
        if key not in merged:
            merged[key] = edge
        else:
            existing = merged[key]
            if not existing['props1'] and edge['props1']:
                existing['props1'] = edge['props1']
            if not existing['props2'] and edge['props2']:
                existing['props2'] = edge['props2']
    return list(merged.values())


def format_node(var, label, props):
    node_str = var
    if label:
        node_str += ":" + label
    if props:
        node_str += " " + props
    return f"({node_str})"


def format_edge(edge):
    node1 = format_node(edge['var1'], edge['label1'], edge['props1'])
    node2 = format_node(edge['var2'], edge['label2'], edge['props2'])
    return f"{node1}-[:{edge['rel']}]->{node2}"


def merge_clauses(clause1, clause2):
    edges1 = parse_clause(clause1)
    edges2 = parse_clause(clause2)
    merged_edges = merge_edges(edges1, edges2)
    merged_clause = "MATCH " + ", ".join(format_edge(edge) for edge in merged_edges)
    return merged_clause


def extract_nodes(query_string):
    pattern = r'\bn\d+\b'
    nodes = re.findall(pattern, query_string)
    return set(nodes)


def get_query_list(query_file, graph_info):
    query_list = list()
    query_num = graph_info[2]
    subquery_edges = graph_info[3]
    with open(query_file, 'r') as f:
        for i in range(query_num):
            parts = f.readline().split('RETURN', 1)
            try:
                query = parts[0].strip()
                return_str = ' RETURN ' + parts[1].strip()
                key_set = extract_nodes(return_str)
                target_node = f"n{i}"
                q = SubQuery(query, return_str, target_node, key_set, subquery_edges[i])
                query_list.append(q)
            except Exception as e:
                logger.info("Incorrectly formatted query statement")
    return query_list


def get_servers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '../config/servers.json')
    with open(config_path, 'r') as file:
        servers_data = json.load(file)
    return servers_data['sites']


def execute_query(sentence, return_time=False):
    servers = get_servers()
    total_time = 0
    res = []

    try:
        for site in servers:
            path = f"{site['ip']}:{site['port']}"
            user = site['dbuser']
            pwd = site['dbpwd']
            graph = Graph(path, auth=(user, pwd))
            start_time = datetime.now()
            res.extend(graph.run(sentence).data())
            end_time = datetime.now()
            query_time = end_time - start_time
            total_time = max(query_time.total_seconds(), total_time)

        if return_time:
            return total_time

        return res

    except Exception as e:
        raise Exception(f"Database query failed: {e}")


def rewrite_return_clause(return_clause):
    parts = return_clause.replace("RETURN", "").strip().split(", ")

    rewritten_parts = []
    for part in parts:
        node_id = part.split('.')[0]
        rewritten_parts.append(f"{part} as {node_id}")

    rewritten_clause = " RETURN " + ", ".join(rewritten_parts)
    return rewritten_clause


@dataclass
class SubQuery:
    query: str
    return_str: str
    target_node: Union[str, List[str]]
    key_set: Set[str] = field(default_factory=set)
    edge_set: Set[Tuple[str, str]] = field(default_factory=set)

    def __str__(self):
        target_node_str = ', '.join(self.target_node) if isinstance(self.target_node, list) else self.target_node
        key_set_str = ', '.join(self.key_set) if self.key_set else "None"
        edge_set_str = ', '.join(f"({u} → {v})" for u, v in self.edge_set) if self.edge_set else "None"

        return (f"📌 SubQuery：\n"
                f"🔹 Query: {self.query}\n"
                f"🔹 Return: {self.return_str}\n"
                f"🔹 Target Node(s): {target_node_str}\n"
                f"🔹 Key Set: {key_set_str}\n"
                f"🔹 Edge Set: {edge_set_str}")

    def has_filter_condition(self) -> bool:
        return "{" in self.query

    def compute_degree(self) -> int:
        degree = 0
        edges = re.findall(r":(\w+)\]", self.query)
        for edge in edges:
            if edge in EDGE_COUNTS:
                degree += EDGE_COUNTS[edge]
        return degree


class GraphDatabaseManager:
    def __init__(self):
        self.graphs = {}
        self.servers = get_servers()
        self.connect_to_servers()

    def connect_to_servers(self):
        for site in self.servers:
            try:
                path = f"{site['ip']}:{site['port']}"
                user = site['dbuser']
                pwd = site['dbpwd']
                graph = Graph(path, auth=(user, pwd))
                self.graphs[site['ip']] = graph
                logger.info(f"Successfully connected to {path}, user: {user}")
            except Exception as e:
                logger.error(f"Failed to connect to {path}, error message: {e}")


class QueryHandler:
    def __init__(self, graph_info, query_file, db_manager):
        self.query_num = len(graph_info[1][0])
        self.graph_info = graph_info
        self.query_list = get_query_list(query_file, graph_info)
        self.graphs = db_manager.graphs
        self.servers = db_manager.servers
        self.manager = Manager()
        self.res_dict = {}

    def connect(self):
        for site in self.servers:
            path = f"{site['ip']}:{site['port']}"
            user = site['dbuser']
            pwd = site['dbpwd']
            self.graphs[site['ip']] = Graph(path, auth=(user, pwd))
            print(f"Successfully connected to {site['ip']}")

    def close_connections(self):
        for ip, graph in self.graphs.items():
            try:
                graph.close()
            except Exception as e:
                print(f"Error occurred while closing the connection:{ip} - {e}")

    def clear_res(self):
        self.res_dict.clear()

    def get_graph(self, site):
        return self.graphs.get(site['ip'])

    def run_query(self, result_dict, q, new_dict=None):

        results_all = []
        total_time = 0
        result_dict['data'] = results_all
        result_dict['elapsed'] = total_time
        server_time_dict = defaultdict(float)

        query = q.query
        node_set = q.key_set
        return_str = rewrite_return_clause(q.return_str)

        if new_dict is None:
            new_dict = self.res_dict
        common_nodes = node_set & new_dict.keys()
        if common_nodes:
            sorted_common_nodes = sorted(common_nodes, reverse=True)
            min_node = min(sorted_common_nodes, key=lambda node: len(new_dict[node]))
            node_ids = list(new_dict[min_node])
            if len(node_ids) > 80000:
                result_dict['data'] = {}
                result_dict['elapsed'] = TIMEOUT
                return
            total_batches = (len(node_ids) // BATCH_SIZE) + 1
            for batch_idx in range(total_batches):
                batch_ids = node_ids[batch_idx * BATCH_SIZE: (batch_idx + 1) * BATCH_SIZE]
                if not batch_ids:
                    continue
                if 'WHERE' in query:
                    modified_query = f"{query} AND {min_node}.id IN $ids " + return_str
                else:
                    modified_query = f"{query} WHERE {min_node}.id IN $ids " + return_str
                try:
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [
                            executor.submit(self._query_site_once, site, modified_query, batch_ids)
                            for site in self.servers
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            site, result, query_time_seconds = future.result()
                            results_all.extend(result)
                            server_time_dict[site['ip']] += query_time_seconds
                except Exception as e:
                    logger.info(f"Database query failed: {e}")
                    raise Exception(f"Database query failed: {e}")
                total_time = max(server_time_dict.values())

        else:
            query_sentence = query + return_str
            try:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self._query_site_once, site, query_sentence)
                        for site in self.servers
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        site, result, query_time_seconds = future.result()
                        results_all.extend(result)
                        server_time_dict[site['ip']] += query_time_seconds
            except Exception as e:
                logger.info(f"Database query failed: {e}")
                raise Exception(f"Database query failed: {e}")

        df = pd.DataFrame(results_all).drop_duplicates()
        results_all = df.to_dict(orient='records')
        result_dict['data'] = results_all
        result_dict['elapsed'] = total_time

    def _query_site_once(self, site, query_str, batch_ids=None):
        graph = self.get_graph(site)
        start_time = datetime.now()
        if batch_ids is not None:
            data = graph.run(query_str, ids=batch_ids).data()
        else:
            data = graph.run(query_str).data()
        elapsed = (datetime.now() - start_time).total_seconds()
        return site, data, elapsed


    def update_inter_res(self, tmp_dict):
        for key, values in tmp_dict.items():
            if key in self.res_dict:
                self.res_dict[key] &= values
            else:
                self.res_dict[key] = values

    def safe_run_query(self, q, new_list=None):
        result_dict = self.manager.dict()

        p = Process(target=self.run_query, args=(result_dict, q, new_list))
        p.start()
        p.join(timeout=TIMEOUT)
        if p.is_alive():
            logger.info(f"Query timed out after {TIMEOUT} seconds. Killing process...")
            p.terminate()
            p.join()
            logger.info("No result due to timeout.")
            return TIMEOUT, {}
        if 'data' in result_dict:
            return result_dict['elapsed'], result_dict['data']
        else:
            return TIMEOUT, {}

    def generate_where_clause(self, node_set):
        conditions = []
        for item in node_set:
            if item in self.res_dict:
                ids = self.res_dict[item]
                ids = sorted(list(ids))
                condition = f"{item}.id IN {ids}"
                conditions.append(condition)
        if not conditions:
            return ""
        else:
            where_clause = " WHERE " + " AND ".join(conditions)
            return where_clause

    def compute_query_rewriting(self, T1, res_1, q1, q2):

        T2 = T3 = T4 = T5 = 0
        q_1, q_2, q_12 = rewrite_query(q1, q2)
        res_dict_1 = parse_query_results(res_1)
        inter_key = q2.key_set.intersection(q1.key_set)
        try:
            T2, res_2 = self.safe_run_query(q2, res_dict_1)
            if T2 == TIMEOUT:
                return 0, TIMEOUT, {}, q1
            join_num_before = len(res_1) * len(res_2)

            start = datetime.now()
            join_time_before = (datetime.now() - start).total_seconds()
            T3, res_3 = self.safe_run_query(q_12)
            if T3 == TIMEOUT:
                return 0, TIMEOUT, {}, q1

            T4, res_4 = self.safe_run_query(q_1)
            if T4 == TIMEOUT:
                return 0, TIMEOUT, {}, q1
            if res_4:
                res_dict_4 = parse_query_results(res_4)
                T5, res_5 = self.safe_run_query(q_2, res_dict_4)
                join_num_after = len(res_4) * len(res_5)
                start = datetime.now()
                joined_res = incremental_join_results(res_4, res_5, inter_key)
                join_time_after = (datetime.now() - start).total_seconds()
                joined_res.extend(res_3)
            else:
                join_num_after = 0
                join_time_after = 0
                joined_res = res_3
            query_time = T3 + T4 + T5 + join_time_after
            time_diff = T1 + T2 + join_time_before - query_time
            logger.info(f"Time before rewrite: {query_time + time_diff}, time after rewrite: {query_time}")
            logger.info(f"Join size before rewrite: {join_num_before}, join size after rewrite: {join_num_after}")
            final_q = SubQuery(merge_clauses(q1.query, q2.query), merge_return_str(q1.return_str, q2.return_str),
                               q_12.target_node, q_12.key_set, q_12.edge_set)
            logger.info("final_q:")
            logger.info(final_q.query)
            logger.info(final_q.return_str)
            logger.info(final_q.target_node)
            logger.info(final_q.key_set)
            logger.info(final_q.edge_set)
            return time_diff, query_time, joined_res, final_q
        except Exception as e:
            raise Exception(f"Database query failed: {e}")
            logger.info(f"Database query failed: {e}")

