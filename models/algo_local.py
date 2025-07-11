import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import logger
from utils.memory import Memory
import math
import torch.nn.functional as F
from typing import List,Optional,Union

from utils.graph_handler import edge_selection



class A2cAlgorithm:
    def __init__(self, actor_critic, args):
        self.actor_critic = actor_critic
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.lr = args.learning_rate
        self.gamma = args.loss_decay

        actor_params = [p for n, p in actor_critic.named_parameters() if "critic" not in n]
        critic_params = [p for n, p in actor_critic.named_parameters() if "critic" in n]

        actor_lr = args.learning_rate * 1.2
        critic_lr = args.learning_rate * 3.0
        self.optimizer = optim.AdamW([
             {"params": actor_params, "lr": actor_lr},
             {"params": critic_params, "lr": critic_lr}
        ], weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size=1000,
                                                   gamma=0.9)

    def compute_returns(self, rewards, dones):
        returns = []
        G = rewards[-1]
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = reward
            else:
                G = reward + self.gamma * G
            returns.insert(0, G)

        return torch.FloatTensor(returns)

    def update(self, memory, origin_p):
        states, actions, t_before, t_after = memory.pop_all()
        true_values = self.compute_local_returns(t_before, t_after)

        values, log_probs, entropy = self.actor_critic.evaluate_action(states, actions, origin_p)
        advantages = true_values - values

        # critic_loss = advantages.pow(2).mean()
        critic_loss = F.smooth_l1_loss(values, true_values, reduction='mean')
        actor_loss = -(log_probs * advantages.detach()).mean()

        total_loss = self.value_loss_coef * critic_loss + actor_loss - (self.entropy_coef * entropy)

        logger.info(
            f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        logger.info(f"Entropy: {entropy.item():.4f}")
        logger.info(f"Mean Value: {values.mean().item():.4f}")
        logger.info(f"Mean Advantage: {advantages.mean().item():.4f}, Std Advantage: {advantages.std().item():.4f}")

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.1)
        self.optimizer.step()
        return values.mean().item()


    def offline_update(self, memory, origin_p, optimizer=None, warmup=False):
        states, actions, t_before, t_after = memory.get_all()

        rewards = []
        for tb, ta in zip(t_before, t_after):
            tb_, ta_ = max(tb, 1e-6), max(ta, 1e-6)
            rewards.append((tb_ - ta_) / tb_)
        device = states[0][0].device
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
        dones   = torch.ones_like(rewards)

        values, log_probs, entropy = self.actor_critic.evaluate_action(states, actions, origin_p)
        values = values.view(-1, 1)

        last_feat, last_mask = states[-1][:2]
        try:
            last_value, _, _ = self.actor_critic.forward(
                last_feat, last_mask, origin_p, chosen_query=None
            )
        except RuntimeError:
            selected = [i for i, m in enumerate(last_mask) if m]
            mean_feat = last_feat[selected].mean(0, keepdim=True) if selected \
                        else torch.zeros_like(last_feat[0]).unsqueeze(0)
            total = last_feat.size(0)
            chosen_ratio = torch.tensor(len(selected)/total, device=device).unsqueeze(0)
            remain_ratio = torch.tensor(len(selected)/total, device=device).unsqueeze(0)
            state_repr = torch.cat([mean_feat.squeeze(0), chosen_ratio, remain_ratio], dim=0).unsqueeze(0)
            last_value = self.actor_critic.critic(state_repr)
        values = torch.cat([values, last_value.view(-1,1)], dim=0)

        advantages, returns = self.compute_gae(
            rewards, values, dones,
            gamma=self.gamma, lam=0.98
        )
        returns    = returns.view(-1)
        values_trim= values[:-1].view(-1)

        adv_mean = advantages.mean()
        adv_std  = (advantages.std(unbiased=False)
                   if advantages.numel()>1 else torch.tensor(1.0, device=device))
        advantages = ((advantages - adv_mean) / (adv_std + 1e-6)).view(-1)
        advantages = torch.clamp(advantages, -3.0, 3.0)

        critic_loss = F.smooth_l1_loss(values_trim, returns, reduction='mean')
        logp = log_probs.view(-1)
        actor_loss  = -(logp * advantages.detach()).mean()
        entropy_term= 0.0 if warmup else (self.entropy_coef * entropy)

        total_loss = self.value_loss_coef * critic_loss + actor_loss - entropy_term

        opt = optimizer or self.optimizer
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        opt.step()

        return {
            "actor_loss":    actor_loss.item(),
            "critic_loss":   critic_loss.item(),
            "total_loss":    total_loss.item(),
            "entropy":       entropy.item(),
            "mean_value":    values_trim.mean().item(),
            "mean_advantage": adv_mean.item(),
            "std_advantage":  adv_std.item()
        }

    def compute_local_returns(
            self,
            tb_list: Union[List[float], torch.Tensor],
            ta_list: Union[List[float], torch.Tensor],
            method: str = "log",
            eps: float = 1e-6,
            clip_pos: Optional[float] = None,
            clip_neg: Optional[float] = None
    ) -> torch.Tensor:
        if isinstance(tb_list, torch.Tensor):
            tb_vals = tb_list.detach().cpu().flatten().tolist()
        else:
            tb_vals = tb_list

        if isinstance(ta_list, torch.Tensor):
            ta_vals = ta_list.detach().cpu().flatten().tolist()
        else:
            ta_vals = ta_list

        assert len(tb_vals) == len(ta_vals), "tb_list 和 ta_list 长度必须相同"

        rewards: List[float] = []
        for tb, ta in zip(tb_vals, ta_vals):
            tb = max(tb, eps)
            ta = max(ta, eps)

            if method == "ratio":
                r = (tb - ta) / tb
                r = r * 2.5
            elif method == "log":
                r = -math.log10(ta / tb)
            else:
                raise ValueError(f"Unknown method: {method}")

            if clip_pos is not None:
                r = min(r, clip_pos)
            if clip_neg is not None:
                r = max(r, -clip_neg)

            rewards.append(r)

        returns: List[float] = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        ret_tensor = torch.tensor(returns, dtype=torch.float32)
        return ret_tensor

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values:  torch.Tensor,
        dones:   torch.Tensor,
        gamma: float,
        lam:   float
    ):
        T = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t+1] * mask - values[t]
            gae = delta + gamma * lam * mask * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        return advantages, returns

    def ppo_offline_update(
            self,
            memory,
            origin_p,
            optimizer=None,
            eps_clip: float = 0.1,
            clip_adv: float = 1.0,
            kl_coef: float = 5e-4,
            warmup: bool = False
    ):
        states, actions, t_before, t_after, old_log_probs = memory.get_all_ppo()
        device = states[0][0].device

        rewards = torch.tensor(
            [(max(tb, 1e-6) - max(ta, 1e-6)) / max(tb, 1e-6)
             for tb, ta in zip(t_before, t_after)],
            dtype=torch.float32, device=device
        ).unsqueeze(1)
        dones = torch.ones_like(rewards)

        values, _, _ = self.actor_critic.evaluate_action(states, actions, origin_p)
        values = values.view(-1, 1)

        last_feat, last_mask = states[-1][:2]
        selected_idxs = [i for i, m in enumerate(last_mask) if m]
        if selected_idxs:
            sel_mean = last_feat[selected_idxs].mean(dim=0)
        else:
            sel_mean = torch.zeros_like(last_feat[0])
        sel_mean = self.actor_critic.state_norm(sel_mean)
        total = last_feat.size(0)
        chosen_ratio = torch.tensor(len(selected_idxs) / total,
                                    device=device).unsqueeze(0)
        possible = edge_selection(last_mask, origin_p, chosen_query=None)
        remain_ratio = torch.tensor(len(possible) / total,
                                    device=device).unsqueeze(0)
        state_repr = torch.cat([sel_mean, chosen_ratio, remain_ratio], dim=0).unsqueeze(0)
        last_val = self.actor_critic.critic(state_repr)

        if values.size(0) == rewards.size(0):
            values = torch.cat([values, last_val], dim=0)
        else:
            pad = torch.zeros(1, 1, device=device)
            values = torch.cat([values, pad], dim=0)

        advantages, returns = self.compute_gae(
            rewards, values, dones,
            gamma=self.gamma, lam=0.90
        )
        returns = returns.view(-1)
        values_pred = values[:-1].view(-1)

        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False) + 1e-6
        raw_adv_mean = adv_mean.item()
        raw_adv_std = adv_std.item()
        advantages = (advantages - adv_mean) / adv_std
        advantages = torch.clamp(advantages.view(-1), -clip_adv, clip_adv)

        _, new_log_probs, entropy = self.actor_critic.evaluate_action(states, actions, origin_p)
        new_log_probs = new_log_probs.view(-1)

        ratio = torch.exp(new_log_probs - old_log_probs.view(-1))
        clipped = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip)
        actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()

        kl_div = (old_log_probs.view(-1) - new_log_probs).mean()
        actor_loss = actor_loss + kl_coef * kl_div

        critic_loss = F.smooth_l1_loss(values_pred, returns, reduction='mean')
        entropy_term = 0.0 if warmup else (self.entropy_coef * entropy)

        total_loss = self.value_loss_coef * critic_loss + actor_loss - entropy_term

        opt = optimizer or self.optimizer
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        opt.step()
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_loss": total_loss.item(),
            "entropy": entropy.item(),
            "mean_value": values_pred.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "raw_adv_mean": raw_adv_mean,
            "raw_adv_std": raw_adv_std,
        }



