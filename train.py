import argparse
from datetime import datetime
import copy
import os
from itertools import islice
from natsort import natsorted
import matplotlib.pyplot as plt
import concurrent.futures
import random
import numpy as np
import torch

from utils.graph_handler import *

import pickle
from models.actor_critic_new import actor_critic as actor_critic_global
from models.actor_critic_local import actor_critic as actor_critic_local
from models.gnn_new import GCN
from models.algo import A2cAlgorithm as A2cAlgorithm_global
from models.algo_local import A2cAlgorithm as A2cAlgorithm_local
from utils.memory import Memory
from utils.local_memory import LocalMemory
from utils.query_handler import *
from utils.logger import logger
import matplotlib.pyplot as plt

epsilon=1e-3
baseline_times = {}
with open("./data/baseline_times.json", "r") as f:
    baseline_times = json.load(f)

trajectory_dir = "trajectory"
if not os.path.exists(trajectory_dir):
    os.makedirs(trajectory_dir)

def save_local_trajectory(trajectory, filename):
    full_path = os.path.join(trajectory_dir, filename)

    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            try:
                trajectory_list = pickle.load(f)
                if not isinstance(trajectory_list, list):
                    trajectory_list = [trajectory_list]
            except Exception as e:
                trajectory_list = []
    else:
        trajectory_list = []

    if not trajectory.states or not trajectory.actions or not trajectory.t_before or not trajectory.t_after:
        logger.warning("Current track data is empty or incomplete, not appended and saved")
    else:
        trajectory_list.append(trajectory)
        with open(full_path, 'wb') as f:
            pickle.dump(trajectory_list, f)
        logger.info(f"Local Trajectory saved. Number of current trajectories: {len(trajectory_list)}")

def save_trajectory(trajectory, filename):
    full_path = os.path.join(trajectory_dir, filename)
    
    if os.path.exists(full_path):
        with open(full_path, 'rb') as f:
            try:
                trajectory_list = pickle.load(f)
                if not isinstance(trajectory_list, list):
                    trajectory_list = [trajectory_list]
            except Exception as e:
                logger.warning(f"Failed to read existing tracks, error message: {e}. The list of tracks will be reconstructed.")
                trajectory_list = []
    else:
        trajectory_list = []

    if not trajectory.states or not trajectory.actions or not trajectory.true_values:
        logger.warning("Current track data is empty or incomplete, not appended and saved")
    else:
        trajectory_list.append(trajectory)
        with open(full_path, 'wb') as f:
            pickle.dump(trajectory_list, f)
        logger.info(f"Trajectory saved. Number of current trajectories: {len(trajectory_list)}")

def plot_reward_curve(reward_list, model_name, curve_type):
    plt.plot(reward_list)
    plt.title(f"{model_name} {curve_type} Reward Curve")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig(f"{model_name}_{curve_type}_reward_curve.png")
    plt.close()

def save_global_reward_curve(global_reward_list, model_name="GlobalModel"):
    plot_reward_curve(global_reward_list, model_name, "Global")

def save_local_reward_curve(local_reward_list, model_name="LocalModel"):
    plot_reward_curve(local_reward_list, model_name, "Local")


query_dir = "/data1/partition_30/c++/node_query_list_no_r/"  # 存放查询语句文件的目录
feature_dir = "/data1/partition_30/c++/new_FeatureVector"  # 存放查询特征文件的目录
timeout = 600
db_manager = GraphDatabaseManager()


def train_model_test01(args):

    query_files = natsorted(os.listdir(query_dir))
    feature_files = natsorted(os.listdir(feature_dir))

    assert len(query_files) == len(feature_files), "Mismatch between the number of query files and feature files"

    model1 = actor_critic_global(args)
    global_model = A2cAlgorithm_global(model1, args)
    model2 = actor_critic_local(args)
    local_model = A2cAlgorithm_local(model2, args)
    gcn = GCN(args)
    for query_file, feature_file in tqdm(islice(zip(query_files, feature_files),0, 31), total=30, desc="Processing queries"):
        train_single_query(query_file, feature_file, args, global_model, local_model,gcn)

def train_single_query(query_file, feature_file, args, global_model, local_model,gcn):
    query_file_path = os.path.join(query_dir, query_file)
    feature_file_path = os.path.join(feature_dir, feature_file)


    subquery_graphs, graph_info, init_first_subquery = preprocess_query_graph(feature_file_path,
                                                                        args.node_in_dim, args.edge_in_dim)
    query_handler = QueryHandler(graph_info, query_file_path,db_manager)

    global_reward_list = []
    local_reward_list = []
    num_episodes = args.num_epoch
    for epoch_num in tqdm(range(num_episodes), desc=f"Training {query_file} for {num_episodes} episodes"):
        timeout_skip = False
        query_handler.clear_res()
        query_plan = []
        query_num = graph_info[2]
        query_mask = [False] * query_num
        trajectory_first = Memory()
        first_iteration = True
        first_subquery = init_first_subquery
        states, actions, rewards, dones = list(), list(), list(), list()
        total_query_time = 0
        while not all(query_mask):
            update_query_x(subquery_graphs,query_handler.query_list,graph_info,query_handler.res_dict)
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
                first_subquery, origin_action = global_model.actor_critic.act(curr_state,graph_info)
                q1 = query_handler.query_list[first_subquery]
            
            T1,res_1 =  query_handler.safe_run_query(q1)
            if not res_1:
                timeout_skip = True
            query_mask[first_subquery] = True
            actions.append(first_subquery)
            dones.append(False)
            rewards.append(0)

            if timeout_skip:
                break

            current_group = [first_subquery]
            trajectory_second = LocalMemory()
            local_reward = 0
            while True:
                next_subquery, origin_action = local_model.actor_critic.act(curr_state, graph_info, first_subquery)
                if next_subquery == -1:
                    query_time = T1
                    T_before = T_after = query_time
                    result = res_1
                    local_reward_list.append(local_reward)
                    trajectory_second.push(curr_state, next_subquery, T_before , T_after,first_subquery)
                    break
                
                q2 = query_handler.query_list[next_subquery]
                time_diff, query_time, result, new_q = query_handler.compute_query_rewriting(T1,res_1,q1,q2)
                q1 = new_q
                T1 = query_time
                res_1 = result
                if not result or query_time < 0:
                    T_before = 1
                    T_after = 2
                else:
                    T_before = max((time_diff+query_time), epsilon)
                    T_after = max(query_time, epsilon)
                local_reward = max(0.0, (T_before - T_after) / T_before)
                if T_after/T_before < 0.01:
                    local_reward -= 0.1
                trajectory_second.push(curr_state, next_subquery, T_before,T_after,first_subquery)
                local_reward_list.append(local_reward)
                query_mask[next_subquery] = True
                curr_state[1] = query_mask
            save_local_trajectory(trajectory_second, f"new_local_trajectory_{query_file}.pkl")
            local_model.update(trajectory_second, graph_info)
            query_plan.append(q1.query)
            result_dict = parse_query_results(result)
            query_handler.update_inter_res(result_dict)
            total_query_time += query_time
        if timeout_skip:
            final_state_reward = 0.0
        else:
            final_state_reward = max(0.0, (600 - total_query_time) / 600)
            logger.info("All subqueries have been processed to generate the complete query_plan：")
            logger.info(query_plan)
            logger.info(f"Query time for query plan：{total_query_time}")

        global_reward_list.append(final_state_reward)
        rewards[-1] += final_state_reward
        dones[-1] = True

        true_values = global_model.compute_returns(rewards, dones)
        for state, action, true_val in zip(states, actions, true_values):
            trajectory_first.push(state, action, true_val)

        save_trajectory(trajectory_first, f"new_global_trajectory_{query_file}.pkl")
        global_model.update(trajectory_first, graph_info)
    save_global_reward_curve(global_reward_list)
    save_local_reward_curve(local_reward_list)



    # checkpoint_dir = "checkpoints"
    # os.makedirs(checkpoint_dir, exist_ok=True)

    # gcn_path = os.path.join(checkpoint_dir, f"gcn_{query_file}.pth")
    # torch.save(gcn.state_dict(), gcn_path)
    #
    # global_ac_path = os.path.join(checkpoint_dir, f"global_actor_critic_{query_file}.pth")
    # torch.save(global_model.actor_critic.state_dict(), global_ac_path)
    #
    # local_ac_path = os.path.join(checkpoint_dir, f"local_actor_critic_{query_file}.pth")
    # torch.save(local_model.actor_critic.state_dict(), local_ac_path)
    #
    # logger.info(f"Model saved successfully：\n  Global GCN -> {gcn_path}\n  Global AC -> {global_ac_path}\n  Local AC  -> {local_ac_path}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser("gnn")
    parser.add_argument('--node_in_dim', type=int, default=22,
                        help='input feature dim')
    parser.add_argument('--edge_in_dim', type=int, default=5,
                        help='input feature dim')
    parser.add_argument('--out_dim', type=int, default=64,
                        help='dimension of output representation')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='dimension of hidden feature.')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate for training')
    parser.add_argument('--loss_decay', type=float, default=0.9,
                        help='loss decay with the step of model.')
    parser.add_argument('--entropy_coef', type=float, default=1.5,
                        help='entropy loss coefficient.')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient.')
    parser.add_argument('--num_epoch', type=int, default=20,
                        help='number of training epoch')
    args = parser.parse_args()

    try:
        train_model_test01(args)
    except Exception as e:
        logger.exception(f"An exception occurs while the program is running:{e}")
