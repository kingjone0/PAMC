from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Dirichlet


class MetricWeightPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device if device is not None else torch.device("cpu")

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.to(self.device)

    def forward(self, state: torch.Tensor):
        if isinstance(state, np.ndarray):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        concentration_logits = self.actor(state_tensor)
        concentration = F.softplus(concentration_logits) + 1e-3
        value = self.critic(state_tensor).squeeze(-1)
        return concentration, value


class PPOAgent:
    def __init__(
        self,
        policy_net,
        state_dim,
        action_dim,
        device,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        update_epochs=4,
        value_coef=0.5,
        entropy_coef=0.01,
        dirichlet_scale=10.0,
    ):
        self.policy_net = policy_net
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = max(1, int(update_epochs))
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.dirichlet_scale = float(dirichlet_scale)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state, explore: bool = True) -> Tuple[np.ndarray, float, float, np.ndarray]:
        if isinstance(state, np.ndarray):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        else:
            state_tensor = state.to(self.device)

        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        with torch.no_grad():
            concentration, value = self.policy_net.forward(state_tensor)
            dist = Dirichlet(concentration * self.dirichlet_scale)
            if explore:
                action_tensor = dist.sample()
            else:
                action_tensor = concentration / concentration.sum(dim=-1, keepdim=True)
            log_prob = dist.log_prob(action_tensor)
            mean_weights = concentration / concentration.sum(dim=-1, keepdim=True)

        action = action_tensor.squeeze(0).cpu().numpy()
        log_prob_scalar = float(log_prob.squeeze(0).item())
        value_scalar = float(value.squeeze(0).item())
        mean_np = mean_weights.squeeze(0).cpu().numpy()
        return action, log_prob_scalar, value_scalar, mean_np

    def update(self, states, actions, log_probs, rewards, values):
        device = self.device
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=device)
        log_probs = torch.as_tensor(np.asarray(log_probs), dtype=torch.float32, device=device)
        rewards = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device)
        values = torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device)

        returns = self._compute_returns(rewards)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        final_metrics = {}
        for _ in range(self.update_epochs):
            concentration, values_pred = self.policy_net.forward(states)
            dist = Dirichlet(concentration * self.dirichlet_scale)
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - log_probs)

            obj = ratio * advantages
            obj_clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(obj, obj_clipped).mean()
            value_loss = F.mse_loss(values_pred, returns)
            entropy = dist.entropy().mean()

            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
            self.optimizer.step()

            clip_frac = ((ratio < 1 - self.clip_epsilon) | (ratio > 1 + self.clip_epsilon)).float().mean()
            final_metrics = {
                "ppo/policy_loss": float(policy_loss.item()),
                "ppo/value_loss": float(value_loss.item()),
                "ppo/entropy": float(entropy.item()),
                "ppo/ratio_mean": float(ratio.mean().item()),
                "ppo/ratio_std": float(ratio.std(unbiased=False).item()),
                "ppo/adv_mean": float(advantages.mean().item()),
                "ppo/adv_std": float(advantages.std(unbiased=False).item()),
                "ppo/ret_mean": float(returns.mean().item()),
                "ppo/ret_std": float(returns.std(unbiased=False).item()),
                "ppo/v_mean": float(values_pred.mean().item()),
                "ppo/v_std": float(values_pred.std(unbiased=False).item()),
                "ppo/clip_frac": float(clip_frac.item()),
            }

        return final_metrics

    def _compute_returns(self, rewards: torch.Tensor):
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(rewards.shape[0])):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        return returns
