import copy
import argparse
from itertools import islice
from natsort import natsorted
from utils.logger import logger
from models.gnn_new import GCN
from utils.graph_handler import *
from utils.query_handler import *
from models.algo import A2cAlgorithm as A2cAlgorithm_global
from models.algo_local import A2cAlgorithm as A2cAlgorithm_local
from models.actor_critic_new import actor_critic as actor_critic_global
from models.actor_critic_local import actor_critic as actor_critic_local
from typing import Any, Tuple, List, Optional

query_dir = "/data1/partition_30/c++/node_query_list_no_r/"
feature_dir = "/data1/partition_30/c++/new_FeatureVector/"
db_manager = GraphDatabaseManager()

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


def evaluate_single_query(query_file: str,
                          feature_file: str,
                          args,
                          global_model: Optional[A2cAlgorithm_global],
                          local_model:  Optional[A2cAlgorithm_local],
                          use_global: bool = True,
                          use_local: bool = True,
                          num_runs: int = 3):
    subquery_graphs, graph_info, init_first_subquery = preprocess_query_graph(
        feature_file, args.node_in_dim, args.edge_in_dim
    )
    query_handler = QueryHandler(graph_info, query_file, db_manager)
    if not use_global:
        rank = get_rank(query_handler.query_list)
    gcn = GCN(args)
    gcn.eval()

    runs_info = {}
    total_times = []

    for run_idx in range(num_runs):
        timeout_skip = False
        gnera_time = 0
        gnera_start = datetime.now()
        query_handler.clear_res()
        query_plan = []
        mid_res = []
        query_num = graph_info[2]
        query_mask = [False] * query_num
        first_iteration = True
        first_subquery = init_first_subquery
        states, actions, rewards, dones = list(), list(), list(), list()
        total_query_time = 0
        while not all(query_mask):
            update_query_x(subquery_graphs, query_handler.query_list, graph_info, query_handler.res_dict)
            query_x_all = []
            for graph in subquery_graphs:
                query_x = gcn(graph.x, graph.edge_index, graph.edge_attr)
                query_x_all.append(query_x)
            subquery_features = torch.stack(query_x_all, dim=0)
            curr_state = [subquery_features, query_mask]
            query_x_all, query_mask = curr_state
            curr_state_copy = [query_x_all.clone(), copy.deepcopy(query_mask)]
            states.append(curr_state_copy)
            if first_iteration:
                first_iteration = False
                q1 = query_handler.query_list[first_subquery]
            else:
                if use_global and global_model is not None:
                    first_subquery, origin_action = \
                        global_model.actor_critic.act(curr_state, graph_info, deterministic=True)
                else:
                    first_subquery = next(idx for idx in rank if not query_mask[idx])
                q1 = query_handler.query_list[first_subquery]
            gnera_time += (datetime.now() - gnera_start).total_seconds()
            T1, res_1 = query_handler.safe_run_query(q1)
            gnera_start = datetime.now()
            if not res_1:
                timeout_skip = True
            query_mask[first_subquery] = True
            if timeout_skip:
                break

            if use_local and local_model is not None:
                while True:
                    next_subquery, origin_action = local_model.actor_critic.act(curr_state, graph_info, first_subquery,deterministic=True)
                    if next_subquery == -1:
                        query_time = T1
                        result = res_1
                        break

                    q2 = query_handler.query_list[next_subquery]
                    gnera_time += (datetime.now() - gnera_start).total_seconds()
                    time_diff, query_time, result, new_q = query_handler.compute_query_rewriting(T1, res_1, q1, q2)
                    gnera_start = datetime.now()
                    if not result or query_time < 0:
                        logger.info("This group of queries is invalid")
                        return {},600
                    q1 = new_q
                    T1 = query_time
                    res_1 = result
                    query_mask[next_subquery] = True
                    curr_state[1] = query_mask
                query_plan.append(q1.query)
                mid_res.append(len(result))
                result_dict = parse_query_results(result)
                query_handler.update_inter_res(result_dict)
                total_query_time += query_time
            else:
                result_dict = parse_query_results(res_1)
                query_handler.update_inter_res(result_dict)
                query_plan.append(q1.query)
                query_mask[first_subquery] = True
                total_query_time += T1
                continue
        if timeout_skip:
            total_query_time = 600
        logger.info("All subqueries have been processed to generate the complete query_plan：")
        logger.info(query_plan)
        logger.info(f"Query time for query plan：{total_query_time}")
        gnera_time += (datetime.now() - gnera_start).total_seconds()
        logger.info(f"Generation time of query plan：{gnera_time}")
        total_times.append(total_query_time)
        logger.info(f"Intermediate result set for query plan:{mid_res}")
        runs_info[f"run_{run_idx}"] = {
            "plan": query_plan.copy(),
            "time": total_query_time
        }
    avg_time = sum(total_times) / len(total_times)
    return runs_info, avg_time



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Batch evaluation of trained global query optimization models")
    parser.add_argument('--global_model', type=str, default="global_model.pth",
                        help="global_model_offline.pth")
    parser.add_argument('--local_model', type=str, default="local_model.pth",
                        help="")
    parser.add_argument('--node_in_dim', type=int, default=22,
                        help="node feature dimension")
    parser.add_argument('--edge_in_dim', type=int, default=5,
                        help="edge feature dimension")
    parser.add_argument('--num_runs', type=int, default=3,
                        help="number of repeated evaluations per query")
    parser.add_argument('--out_dim', type=int, default=64,
                        help='dimension of output representation')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimension of hidden feature.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training')
    parser.add_argument('--loss_decay', type=float, default=1.1,
                        help='loss decay with the step of model.')
    parser.add_argument('--entropy_coef', type=float, default=1,
                        help='entropy loss coefficient.')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient.')
    args = parser.parse_args()

    model1 = actor_critic_global(args)
    global_model = A2cAlgorithm_global(model1, args)
    model2 = actor_critic_local(args)
    local_model = A2cAlgorithm_local(model2, args)

    ckpt1 = torch.load(args.global_model)
    global_model.actor_critic.load_state_dict(ckpt1)
    ckpt2 = torch.load(args.local_model)
    local_model.actor_critic.load_state_dict(ckpt2)

    global_model.actor_critic.eval()
    local_model.actor_critic.eval()

    query_files = natsorted(os.listdir(query_dir))
    feature_files = natsorted(os.listdir(feature_dir))
    assert len(query_files) == len(feature_files), \
        "Mismatch between query file and number of feature files"
    for index, (qf, ff) in enumerate(islice(zip(query_files, feature_files), 0,101)):
        qpath = os.path.join(query_dir, qf)
        fpath = os.path.join(feature_dir, ff)

        runs_info, avg_time = evaluate_single_query(
            qpath, fpath, args,
            global_model=global_model,  # 真实全局模型
            local_model=local_model,  # 真实局部模型
            use_global=True,  # 可省略，默认就是 True
            use_local=True,  # idem
            num_runs=args.num_runs
        )

        # ==== 2. (–L) ====
        # runs_info, avg_time = evaluate_single_query(
        #     qpath, fpath, args,
        #     global_model=None,  # 或 DummyGlobal()
        #     local_model=local_model,
        #     use_global=False,  # 关键：显式关掉
        #     use_local=True,
        #     num_runs=args.num_runs
        # )

        # ==== 3. (–G) ====
        # runs_info, avg_time = evaluate_single_query(
        #     qpath, fpath, args,
        #     global_model=global_model,
        #     local_model=None,  # 或 DummyLocal()
        #     use_global=True,
        #     use_local=False,  # 关键：显式关掉
        #     num_runs=args.num_runs
        # )
        logger.info(f"=== Query File: {qf} ===")
        logger.info(runs_info)
        logger.info(f"{qf} Average Execution Time over {args.num_runs} runs: {avg_time:.3f} s\n")