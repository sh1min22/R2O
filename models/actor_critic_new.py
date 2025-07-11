import argparse

import torch.nn as nn
from utils.graph_handler import *
from utils.logger import logger
from torch.distributions import Categorical

class actor_critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.state_norm = nn.LayerNorm(args.out_dim)

        self.actor = nn.Sequential(
            nn.Linear(args.out_dim, args.out_dim // 2),
            nn.ReLU(),
            nn.Linear(args.out_dim // 2, 1)
        )
        crit_in = args.out_dim + 2
        self.critic = nn.Sequential(
            nn.Linear(crit_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.zeros_(self.critic[-1].bias)

    def _build_state_repr(self, subquery_features, query_mask):
        selected = [i for i, m in enumerate(query_mask) if m]
        if selected:
            s = subquery_features[selected].mean(dim=0)
        else:
            s = torch.zeros_like(subquery_features[0])
        return self.state_norm(s)

    def forward(self, subquery_features, query_mask, origin_p, chosen_query=None):
        possible = edge_selection(query_mask, origin_p, chosen_query)
        if chosen_query is not None:
            possible.append(-1)

        actor_states = []
        for eid in possible:
            if eid == -1:
                valid = [e for e in possible if e != -1]
                feat = subquery_features[valid].mean(0) if valid \
                    else torch.full_like(subquery_features[0], 1e-6)
            else:
                feat = subquery_features[eid]
            actor_states.append(self.state_norm(feat).unsqueeze(0))

        actor_states = torch.cat(actor_states, dim=0)
        logits = self.actor(actor_states).T
        origin_action = logits.softmax(dim=1)

        selected = [i for i, m in enumerate(query_mask) if m]
        selected_mean = subquery_features[selected].mean(0) if selected \
            else torch.zeros_like(subquery_features[0])
        selected_mean = self.state_norm(selected_mean)

        total_edges = subquery_features.size(0)
        chosen_ratio = torch.tensor(len(selected) / total_edges,
                                    device=subquery_features.device).unsqueeze(0)
        remain_ratio = torch.tensor(len(possible) / total_edges,
                                    device=subquery_features.device).unsqueeze(0)

        state_repr = torch.cat([selected_mean, chosen_ratio, remain_ratio], dim=0)
        value = self.critic(state_repr)

        return value, origin_action, possible
    
    def act(self, current_state, origin_p, chosen_query=None,deterministic=False):
        subquery_features, query_mask = current_state
        value, origin_action, possible_edges = self.forward(subquery_features, query_mask, origin_p, chosen_query)

        if not possible_edges:
            return -1, origin_action

        probs = origin_action.squeeze(0)

        if deterministic:
            idx = torch.argmax(probs).item()
        else:
            dist = Categorical(probs)
            idx = dist.sample().item()

        next_query = possible_edges[idx]
        return next_query, origin_action
    
    def evaluate_action(self, states, actions, origin_p, chosen_query=None):
        for i in range(len(states)):
            subquery_features, query_mask = states[i]
            action = actions[i]

            value, all_action, possible_edges = self.forward(
                subquery_features, query_mask, origin_p, chosen_query
            )

            true_idx = torch.LongTensor([possible_edges.index(action.item())]).squeeze()
            dist = torch.distributions.Categorical(all_action)
            log_prob = dist.log_prob(true_idx).view(-1, 1)
            entropy = dist.entropy().mean().view(-1, 1)

            if i == 0:
                values, log_probs, entropies = value, log_prob, entropy
            else:
                values = torch.cat((values, value), 0)
                log_probs = torch.cat((log_probs, log_prob), -1)
                entropies = torch.cat((entropies, entropy), -1)

        return values, log_probs, entropies.mean()
