# R2O: A Dual-Layer Framework for Joint Rewriting and Ordering in Distributed Property Graph Query Optimization

# Overview
R2O (Rewriting to Ordering)

# File Overview

## Top-level Scripts

- `default_order_test.py`: Baseline method that executes subqueries in their default order without optimization.
- `evaluate.py`: Evaluation script for testing trained models on unseen query trajectories.
- `train.py`: Main training script that runs reinforcement learning over multiple query trajectories.

## models/

- `actor_critic_new.py`: Actor-Critic model for global subquery order optimization.
- `actor_critic_local.py`: Actor-Critic model for local subquery rewriting within partial subgraphs.
- `algo.py`: Reinforcement learning algorithm implementation for global optimization.
- `algo_local.py`: RL training algorithm specialized for local subquery rewriting.
- `gnn_new.py`: Graph neural network encoder for representing subquery structures.

## utils/

- `graph_handler.py`: Utilities for loading graph structure and generating node-edge representations.
- `local_memory.py`: Experience replay memory for local training tasks.
- `logger.py`: Logging utility for recording training and evaluation information.
- `memory.py`: Generic memory module for reinforcement learning, used in global training.
- `query_handler.py`: Functions for loading, parsing, and managing query trajectories and subqueries.


# Benchmark Queries
The benchmark queries used in our experimental evaluation exists in #queries# folder.
