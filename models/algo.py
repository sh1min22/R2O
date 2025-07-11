import torch
import torch.nn as nn
import torch.optim as optim
from utils.logger import logger
import math
import torch.nn.functional as F


def default_local_reward(t_before: float,
                         t_after: float,
                         eps: float = 1e-6) -> float:
    tb = t_before if t_before > eps else eps
    ta = t_after  if t_after  > eps else eps
    return (tb - ta) / tb

class A2cAlgorithm:
    def __init__(self, actor_critic, args):
        """
        初始化A2C算法类

        Args:
            actor_critic (nn.Module): Actor-Critic 模型
            value_loss_coef (float): 价值损失系数
            entropy_coef (float): 熵正则化系数
            lr (float): 训练学习率
            gamma (float): 折扣因子
            optimizer (torch.optim.Optimizer): 优化器
        """

        self.actor_critic = actor_critic
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.lr = args.learning_rate
        self.gamma = args.loss_decay

        actor_params = [p for n, p in actor_critic.named_parameters() if "critic" not in n]
        critic_params = [p for n, p in actor_critic.named_parameters() if "critic" in n]

        self.optimizer = optim.RMSprop([
            {"params": actor_params, "lr": args.learning_rate * 1.2},
            {"params": critic_params, "lr": args.learning_rate * 2.2}
        ])

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

    def update(self, memory, origin_p,chosen_query=None):
        states, actions, true_values = memory.pop_all()

        values, log_probs, entropy = self.actor_critic.evaluate_action(states, actions, origin_p, chosen_query)
        advantages = true_values - values

        critic_loss = advantages.pow(2).mean()
        actor_loss = -(log_probs * advantages.detach()).mean()

        total_loss = self.value_loss_coef * critic_loss + actor_loss - (self.entropy_coef * entropy)

        logger.info(f"Actor Loss: {actor_loss.item():.4f}, Critic Loss: {critic_loss.item():.4f}, Total Loss: {total_loss.item():.4f}")
        logger.info(f"Entropy: {entropy.item():.4f}")
        logger.info(f"Mean Value: {values.mean().item():.4f}")
        logger.info(f"Mean Advantage: {advantages.mean().item():.4f}, Std Advantage: {advantages.std().item():.4f}")
    

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        self.optimizer.step()
        return values.mean().item()


    def offline_update(self, memory, origin_p,
                       global_mean, global_std,
                       optimizer=None, warmup=False, chosen_query=None):

        states, actions, old_true_values = memory.get_all()
        # true_values = self.recompute_true_values_log(old_true_values)
        true_values = self.recompute_true_values_zscore(old_true_values, global_mean, global_std)

        values, log_probs, entropy = self.actor_critic.evaluate_action(
            states, actions, origin_p, chosen_query
        )

        raw_advantages = true_values - values.detach()
        advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-6)

        critic_loss = F.smooth_l1_loss(values, true_values)
        actor_loss = -(log_probs * advantages).mean()
        entropy_term = 0.0 if warmup else (self.entropy_coef * entropy)

        total_loss = 1.5 * critic_loss + actor_loss - entropy_term

        (optimizer or self.optimizer).zero_grad()
        total_loss.backward()
        if not warmup:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
        (optimizer or self.optimizer).step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "total_loss": total_loss.item(),
            "entropy": entropy.item(),
            "mean_value": values.mean().item(),
            "mean_advantage": raw_advantages.mean().item(),
            "std_advantage": raw_advantages.std().item()
        }

    
    def recompute_true_values_log(self,old_true_values, Tmax=600):
        if isinstance(old_true_values, list):
            old_true_values = torch.FloatTensor(old_true_values)

        final_reward = old_true_values[-1].item()
        query_time = Tmax * (1 - final_reward)

        new_final_reward = -math.log(query_time + 1e-6)

        new_rewards = [0.0] * (len(old_true_values) - 1) + [new_final_reward]
        dones = [False] * (len(old_true_values) - 1) + [True]

        returns = []
        G = new_rewards[-1]
        for reward, done in zip(reversed(new_rewards), reversed(dones)):
            if done:
                G = reward
            else:
                G = reward + self.gamma * G
            returns.insert(0, G)

        return torch.FloatTensor(returns)

    def recompute_true_values_zscore(self,old_true_values, global_mean, global_std,Tmax=600):
        if isinstance(old_true_values, list):
            old_true_values = torch.FloatTensor(old_true_values)

        final_reward = old_true_values[-1].item()
        query_time = Tmax * (1 - final_reward)

        new_final_reward = -(query_time - global_mean) / global_std

        new_rewards = [0.0] * (len(old_true_values) - 1) + [new_final_reward]
        dones = [False] * (len(old_true_values) - 1) + [True]

        returns = []
        G = new_rewards[-1]
        for reward, done in zip(reversed(new_rewards), reversed(dones)):
            if done:
                G = reward
            else:
                G = reward + self.gamma * G
            returns.insert(0, G)

        return torch.FloatTensor(returns)