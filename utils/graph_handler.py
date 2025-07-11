from datetime import datetime
from copy import deepcopy
import mmh3
import os
import re

import torch
from torch_geometric.data import Data

def update_query_x(subquery_graphs,query_list, graph_info, res_dict):
    query_num = graph_info[2]

    for i in range(query_num):
        q = query_list[i]
        for key in q.key_set:
            if key in res_dict:
                value = res_dict[key]

                match = re.search(r'\d+', key)
                if match:
                    node = int(match.group()) 
                    subquery_graphs[i].x[node, 21] = len(value)


def map_to_number(data):
    if isinstance(data, int):
        key = f"i:{data}"
    elif isinstance(data, str):
        try:
            dt = datetime.fromisoformat(data.replace('Z', '+00:00'))
            timestamp = dt.timestamp()
            key = f"d:{timestamp}"
        except ValueError:
            key = f"s:{data}"
    else:
        raise TypeError("Unsupported data type")

    hash_64 = mmh3.hash64(key.encode())[0]
    value = hash_64 & 0xFFFFFFFFFFFFFFFF
    return value / float(0xFFFFFFFFFFFFFFFF)

def preprocess_query_graph(g_file, node_dim, edge_dim):
    e_u = list()
    e_v = list()
    subquery_edges = list()
    first_query = -1
    with open(g_file) as f:
        line = f.readline().rstrip()
        numbers = line.split()
        nodes_num = int(numbers[0])
        query_num = int(numbers[1])
        edges_num = int(numbers[2])
        nid = list(range(nodes_num))
        id_flag = f.readline().rstrip().split()
        for i in range(edges_num):
            edge = f.readline().rstrip().split()
            u = int(edge[0])
            v = int(edge[1])
            if (id_flag[u] == '1' or id_flag[v] == '1') and first_query == -1:
                first_query = i
            e_u.append(u)
            e_v.append(v)

        if first_query == -1: first_query = 0
        g_nid = deepcopy(nid)
        g_edges = [deepcopy(e_u), deepcopy(e_v)]
        subquery_graphs = []
        for i in range(query_num):
            query_edges = set()
            e_u = list()
            e_v = list()
            line = f.readline().rstrip()
            numbers = line.split()
            edge_num = int(numbers[1])
            for _ in range(edge_num):
                edge = f.readline().rstrip().split()
                u = int(edge[0])
                v = int(edge[1])
                e_u.append(u)
                e_v.append(v)
                source_node = f"n{u}"
                target_node = f"n{v}"
                query_edges.add((source_node, target_node))
            node_features_list = []
            edge_features_list = []
            for j in range(node_dim):
                values = f.readline().rstrip().split()
                values = [float(map_to_number(num)) for num in values]
                node_features_list.append(values)
            for _ in range(edge_dim):
                values = f.readline().rstrip().split()
                values = [float(map_to_number(num)) for num in values]
                edge_features_list.append(values)
            subquery = Data(
                x=torch.tensor(node_features_list, dtype=torch.float).t(),
                edge_index=torch.tensor([e_u, e_v], dtype=torch.long),
                edge_attr=torch.tensor(edge_features_list, dtype=torch.float).t()
            )
            subquery_graphs.append(subquery)
            subquery_edges.append(query_edges)
        graph_info = [
            g_nid,
            g_edges,
            query_num,
            subquery_edges
        ]

    return subquery_graphs, graph_info, first_query


def edge_selection(query_mask,origin_p, chosen_query):
        if chosen_query is None:
            chosen_querys = [i for i, mask in enumerate(query_mask) if mask]
            if not chosen_querys:
                adjacent_edges = set(i for i in range(len(query_mask)))
                return sorted(adjacent_edges)
            edge_list = origin_p[1]
            e_u = edge_list[0]
            e_v = edge_list[1]
            neighbors = set()
            for q_id in chosen_querys:
                target_node = q_id
                for u, v in zip(e_u, e_v):
                    if u == target_node:
                        neighbors.add(v)
                    elif v == target_node:
                        neighbors.add(u)
            for q_id in chosen_querys:
                if q_id in neighbors: neighbors.remove(q_id)
        else:
            neighbors = check_neighbor(chosen_query,origin_p)
        neighbors = [i for i in neighbors if not query_mask[i]]
        return sorted(neighbors)

def check_neighbor(chosen_id, origin_p):
        edge_list = origin_p[1]
        e_u = edge_list[0]
        e_v = edge_list[1]
        neighbors = set()
        target_node = chosen_id
        for u, v in zip(e_u, e_v):
            if u == target_node:
                neighbors.add(v)
            elif v == target_node:
                neighbors.add(u)
        return sorted(neighbors)