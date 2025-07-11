import torch
import copy
from utils.logger import logger

class Memory(object):
    def __init__(self):
        self.states, self.actions, self.true_values = list(), list(), list()

    def push(self, state, action, true_value):
        subquery_graphs, query_mask = state
        state_copy = [subquery_graphs.clone(), copy.deepcopy(query_mask)]
        self.states.append(state_copy)
        self.actions.append(action)
        self.true_values.append(true_value)

    def print_all(self):
        logger.info(f"🔍 Memory contains {len(self.states)} transitions")
        for i in range(len(self.states)):
            subquery_features, query_mask = self.states[i]
            action = self.actions[i]
            true_value = self.true_values[i]
            logger.info(f"\n--- Transition {i} ---")
            logger.info(f"Action: {action}")
            logger.info(f"True Value (Reward): {true_value}")
            logger.info(f"Query Mask: {query_mask}")

    def pop_all(self):
        actions = torch.LongTensor(self.actions)
        true_values = torch.FloatTensor(self.true_values).unsqueeze(1)
        states = [
            [
                subquery_graphs.clone().detach(),
                copy.deepcopy(query_mask)
            ]
            for subquery_graphs, query_mask in self.states
        ]
        self.states, self.actions, self.true_values = list(), list(), list()

        return states, actions, true_values

    def get_all(self):
        actions = torch.LongTensor(self.actions)
        true_values = torch.FloatTensor(self.true_values).unsqueeze(1)
        states = [
            [
                subquery_graphs.clone().detach(),
                copy.deepcopy(query_mask)
            ]
            for subquery_graphs, query_mask in self.states
        ]

        return states, actions, true_values

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.true_values.clear()

    def get_transitions(self):
        transitions = []
        for state, action, true_value in zip(self.states, self.actions, self.true_values):
            transitions.append((state, action, true_value))
        return transitions
        
    def __iter__(self):
        return iter(self.get_transitions())