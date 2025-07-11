from utils.query_handler import *
from natsort import natsorted
from itertools import islice

query_dir = "/data1/ljy/partition_30/SF3/node_query_list_no_r"
query_dir = "/data1/ljy/partition_30/SF10/node_query_list_no_r"
query_dir = "/data1/ljy/partition_30/c++/node_query_list_no_r"

feature_dir = "/data1/ljy/partition_30/SF3/new_FeatureVector"
feature_dir = "/data1/ljy/partition_30/SF10/new_FeatureVector/"
feature_dir = "/data1/ljy/partition_30/c++/new_FeatureVector"
db_manager = GraphDatabaseManager()

def sort_queries(query_list: List[SubQuery]) -> List[SubQuery]:
    """基于改进启发式规则对子查询排序"""
    sorted_list = []
    
    first_query = min((q for q in query_list if q.has_filter_condition()), key=lambda q: q.compute_degree(), default=None)
    
    if not first_query:
        raise ValueError("The query list should contain at least one subquery with attribute filter conditions")
    
    sorted_list.append(first_query)
    remaining_queries = list(query_list)
    remaining_queries.remove(first_query)

    while remaining_queries:
        candidate_queries = [q for q in remaining_queries if
                             any(node == q.target_node for selected in sorted_list for node in selected.key_set)]
        if not candidate_queries:
            candidate_queries = list(remaining_queries)  # 如果没有连接的，就随便选
        
        next_query = min(candidate_queries, key=lambda q: q.compute_degree())
        
        sorted_list.append(next_query)
        remaining_queries.remove(next_query)
    
    return sorted_list

def get_rank(query_list: List[SubQuery]) -> List[int]:
    sorted_list: List[SubQuery] = []

    first_query = min(
        (q for q in query_list if q.has_filter_condition()),
        key=lambda q: q.compute_degree(),
        default=None
    )
    if first_query is None:
        raise ValueError("The query list should contain at least one subquery with attribute filter conditions")

    sorted_list.append(first_query)
    remaining_queries = list(query_list)
    remaining_queries.remove(first_query)

    while remaining_queries:
        candidate_queries = [
            q for q in remaining_queries
            if any(node == q.target_node
                   for selected in sorted_list
                   for node in selected.key_set)
        ]
        if not candidate_queries:
            candidate_queries = remaining_queries

        next_query = min(candidate_queries, key=lambda q: q.compute_degree())
        sorted_list.append(next_query)
        remaining_queries.remove(next_query)

    rank = [query_list.index(q) for q in sorted_list]

    return rank

def test():
    query_files = natsorted(os.listdir(query_dir))  # 获取查询文件并按名称排序
    feature_files = natsorted(os.listdir(feature_dir))  # 获取特征文件并按名称排序
    for query_file, feature_file in tqdm(islice(zip(query_files, feature_files),0,101), total=len(query_files), desc="Processing queries"):
        run_query_file(query_file, feature_file)

def run_query_file(query_file, feature_file):
    query_file_path = os.path.join(query_dir, query_file)
    feature_file_path = os.path.join(feature_dir, feature_file)
    subquery_graphs, graph_info, init_first_subquery = preprocess_query_graph(feature_file_path,
                                                                        22, 5)
    query_handler = QueryHandler(graph_info, query_file_path,db_manager)
    query_num = len(query_handler.query_list)
    sorted_queries = sort_queries(query_handler.query_list)
    for q in sorted_queries:
        logger.info(f"Query: {q.query}, Has Filter: {q.has_filter_condition()}, Degree: {q.compute_degree()}")
    mid_res = []
    mid_res_list = []
    pre_key = set()
    total_time = 0
    query_handler.clear_res()
    for i in range(query_num):
        q = sorted_queries[i]
        T,res_q = query_handler.safe_run_query(q)
        if not res_q:
            return
        if mid_res:
            if not pre_key: 
                inter_key = q.key_set
            else:
                inter_key = pre_key.intersection(q.key_set)
            key_list = list(inter_key)

            joined_res = incremental_join_results(mid_res,res_q,key_list)
        else:
            joined_res = res_q
        if not joined_res:
            return
        total_time += T
        pre_key = pre_key.union(q.key_set)
        mid_res = joined_res
        mid_res_list.append(len(mid_res))
        mid_res_dict = parse_query_results(mid_res)
        query_handler.update_inter_res(mid_res_dict)
        logger.info(f"The query time for the {i}th statement is:{T}")
        
    logger.info(f"The query time for {query_file} is:{total_time}")
    logger.info(f"intermediate result set:{mid_res_list}")


if __name__ == '__main__':
    test()

        
