import torch
import copy
from typing import Callable, List, Tuple
from utils.logger import logger


def default_local_reward(t_before: float,
                         t_after: float,
                         eps: float = 1e-6) -> float:
    tb = t_before if t_before > eps else eps
    ta = t_after  if t_after  > eps else eps
    return (tb - ta) / tb



class LocalMemory(object):
    def __init__(self, reward_fn: Callable[[float, float], float] = None):
        self.states: List[Tuple[torch.Tensor, List[int], int]] = []
        self.actions: List[int] = []
        self.t_before: List[float] = []
        self.t_after:  List[float] = []
        self.rewards:  List[float] = []
        self.returns:  List[float] = []
        self.reward_fn = reward_fn or default_local_reward

        self.log_probs: List[torch.Tensor] = []

    def push(
        self,
        state: Tuple[torch.Tensor, List[int], int],
        action: int,
        t_before: float,
        t_after: float,
        first_query: int,
        log_prob: torch.Tensor = None
    ):
        subquery_graphs, query_mask = state[0], state[1]
        state_copy = [subquery_graphs.clone(), copy.deepcopy(query_mask)]
        if len(state) > 2:
            state_copy.append(state[2])
        state_copy.append(first_query)

        self.states.append(state_copy)
        self.actions.append(action)
        self.t_before.append(t_before)
        self.t_after.append(t_after)

        r = self.reward_fn(t_before, t_after)
        self.rewards.append(r)

        if log_prob is not None:
            self.log_probs.append(log_prob.detach().cpu().clone())
        else:
            self.log_probs.append(torch.tensor(0.0))

    def compute_returns(self, gamma: float = 0.99):
        self.returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + gamma * G
            self.returns.insert(0, G)


    def pop_all(self):
        states = [
            (sg.clone().detach(), copy.deepcopy(mask), init_id)
            for sg, mask, init_id in self.states
        ]
        actions = torch.LongTensor(self.actions)
        t_before = torch.FloatTensor(self.t_before).unsqueeze(1)
        t_after = torch.FloatTensor(self.t_after).unsqueeze(1)

        self.clear()
        return states, actions, t_before, t_after

    def get_all(self):
        states = [
            (sg.clone().detach(), copy.deepcopy(mask), init_id)
            for sg, mask, init_id in self.states
        ]
        actions = torch.LongTensor(self.actions)
        t_before = torch.FloatTensor(self.t_before).unsqueeze(1)
        t_after = torch.FloatTensor(self.t_after).unsqueeze(1)

        return states, actions, t_before, t_after

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.t_before.clear()
        self.t_after.clear()
        self.rewards.clear()
        self.returns.clear()

        self.log_probs.clear()

    def print_all(self):
        logger.info(f"🔍 LocalMemory contains {len(self.states)} transitions")
        for i, (state, act, tb, ta, r) in enumerate(zip(
                self.states, self.actions,
                self.t_before, self.t_after, self.rewards)):
            _, query_mask = state
            logger.info(f"\n--- Transition {i} ---")
            logger.info(f"Action              : {act}")
            logger.info(f"T_before | T_after  : {tb:.4f}  |  {ta:.4f}")
            logger.info(f"Instant Reward      : {r:+.6f}")
            logger.info(f"Query Mask          : {query_mask}")

    def __iter__(self):
        return iter(zip(self.states, self.actions,
                        self.t_before, self.t_after, self.rewards))
    
    def get_all_ppo(self
    ) -> Tuple[List[List], torch.LongTensor, List[float], List[float], torch.Tensor]:
        states, actions, t_before, t_after = self.get_all()

        if not hasattr(self, 'log_probs') or len(self.log_probs) != len(self.actions):
            self.log_probs = [torch.tensor(0.0) for _ in self.actions]

        old_log_probs = torch.stack(self.log_probs, dim=0).view(-1)
        return states, actions, t_before, t_after, old_log_probs