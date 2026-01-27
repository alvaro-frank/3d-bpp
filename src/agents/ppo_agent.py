# ==============================================================================
# FILE: agents/ppo_agent.py
# DESCRIPTION: Implementation of a Proximal Policy Optimization (PPO) agent.
#              Features a CNN-based Actor-Critic architecture for spatial tasks.
# ==============================================================================

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

def _to_tensor(x, device):
    """
    Convert common Python/NumPy types to a float32 torch.Tensor.

    Args:
        x: Value to convert (NumPy array, list/tuple, Tensor, or scalar).
        device: Target device ('cpu', 'cuda', or a torch.device).

    Returns:
        torch.Tensor: Float32 tensor on the specified device.
    """
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Apply a binary mask (1=valid, 0=invalid) to action logits.

    Args:
        logits: Unnormalized action logits of shape (B, A) or (A,).
        mask: Optional binary mask broadcastable to logits.

    Returns:
        torch.Tensor: Masked logits with invalid entries set to a large negative value.
    """
    if mask is None:
        return logits

    mask = mask.to(logits.device).float()
    invalid = (mask < 0.5)

    if invalid.all():
        return logits
    masked = logits.masked_fill(invalid, -1e9)
    return masked


def categorical_sample_from_masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample an action from a categorical distribution defined by masked logits.

    Args:
        logits: Unnormalized action logits.
        mask: Optional binary mask for valid actions.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (sampled action index, log-probability).
    """
    masked = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=masked)
    action = dist.sample()
    logp = dist.log_prob(action)
    return action, logp

# ------------------------------------------------------------------------------
# NEURAL NETWORK ARCHITECTURE
# ------------------------------------------------------------------------------

class CNNActorCritic(nn.Module):
    """
    CNN actor-critic with shared body and separate policy/value heads.

    Args:
        obs_dim (int): Dimension of observation vectors.
        act_dim (int): Number of discrete actions.
        map_size (tuple): Dimensions (W, D) of the spatial input.
        hidden_sizes (tuple): Sizes of hidden layers for the shared MLP body.
        layer_norm (bool): Whether to apply LayerNorm.
    """
    def __init__(self, obs_dim: int, act_dim: int, map_size=(10, 10), hidden_sizes=(256, 256), layer_norm=False):
        super().__init__()
        self.map_w, self.map_d = map_size
        self.map_area = self.map_w * self.map_d
        self.extra_features_len = obs_dim - self.map_area

        # CNN Feature Extractor (Shared)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 32 * self.map_w * self.map_d
        combined_dim = cnn_out_dim + self.extra_features_len

        # Shared MLP Body
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU()
        )

        # Output Heads
        self.pi_head = nn.Linear(hidden_sizes[1], act_dim)
        self.v_head = nn.Linear(hidden_sizes[1], 1)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        nn.init.orthogonal_(self.pi_head.weight, gain=0.01)
        nn.init.orthogonal_(self.v_head.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Split and Reshape spatial vs flat features
        heightmap_flat = obs[:, :self.map_area]
        extra_features = obs[:, self.map_area:]
        heightmap_img = heightmap_flat.view(-1, 1, self.map_w, self.map_d)

        # Forward Pass
        conv_out = self.conv(heightmap_img)
        combined = torch.cat([conv_out, extra_features], dim=1)
        x = self.shared_fc(combined)

        logits = self.pi_head(x)
        value = self.v_head(x).squeeze(-1)
        
        return logits, value

# ------------------------------------------------------------------------------
# AGENT CONFIGURATION & MAIN CLASS
# ------------------------------------------------------------------------------

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch_size: int = 1024
    hidden_sizes: Tuple[int, int] = (256, 256)
    layer_norm: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    advantage_norm_eps: float = 1e-8

class PPOAgent:
    """
    PPO Agent handler for training and inference.
    """
    def __init__(self, obs_dim: int, act_dim: int, map_size=(10, 10), config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)

        self.model = CNNActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            map_size=map_size,
            hidden_sizes=self.config.hidden_sizes
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.reset_buffer()

    @torch.no_grad()
    def get_action(self, state: np.ndarray, action_space: Any, mask: Optional[np.ndarray] = None) -> int:
        """Samples an action from the current policy."""
        self.model.eval()
        s = _to_tensor(state, self.device).float().unsqueeze(0)
        logits, value = self.model(s)

        mask_t = None
        if mask is not None:
            mask_t = _to_tensor(mask, self.device).unsqueeze(0)

        action_t, logp_t = categorical_sample_from_masked_logits(logits, mask_t)
        
        self._last_action_aux = {"logp": float(logp_t.item()), "value": float(value.item())}
        return int(action_t.item())

    def store_transition(self, state, action, reward, next_state, done, mask=None, logp=None, value=None):
        """Append a single environment transition to the rollout buffer."""
        if (logp is None) or (value is None):
            with torch.no_grad():
                s = _to_tensor(state, self.device).unsqueeze(0)
                logits, v = self.model(s)
                mask_t = _to_tensor(mask, self.device).unsqueeze(0) if mask is not None else None
                masked = masked_logits(logits, mask_t)
                dist = torch.distributions.Categorical(logits=masked)
                a = torch.tensor([action], device=self.device, dtype=torch.long)
                logp = float(dist.log_prob(a).item())
                value = float(v.item())

        self.storage["states"].append(np.asarray(state, dtype=np.float32))
        self.storage["actions"].append(int(action))
        self.storage["rewards"].append(float(reward))
        self.storage["dones"].append(bool(done))
        self.storage["masks"].append(None if mask is None else np.asarray(mask, dtype=np.float32))
        self.storage["logps"].append(float(logp))
        self.storage["values"].append(float(value))

    def train(self, last_value: float = 0.0) -> Dict[str, float]:
        """Optimize the policy/value networks using the accumulated rollout."""
        if len(self.storage["states"]) == 0:
            return {}

        device = self.device
        cfg = self.config

        # Prepare Tensors
        states = _to_tensor(np.array(self.storage["states"]), device)
        actions = torch.tensor(self.storage["actions"], dtype=torch.long, device=device)
        rewards = _to_tensor(np.array(self.storage["rewards"], dtype=np.float32), device)
        dones = torch.tensor(self.storage["dones"], dtype=torch.float32, device=device)
        old_logps = _to_tensor(np.array(self.storage["logps"], dtype=np.float32), device)
        values = _to_tensor(np.array(self.storage["values"], dtype=np.float32), device)

        # GAE(λ) Calculation
        advantages = []
        gae = 0.0
        last_val = float(last_value)
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - float(dones[t].item())
            next_value = last_val if t == len(rewards) - 1 else values[t + 1].item()
            delta = rewards[t].item() + cfg.gamma * next_value * next_non_terminal - values[t].item()
            gae = delta + cfg.gamma * cfg.gae_lambda * next_non_terminal * gae
            advantages.append(gae)
        
        advantages.reverse()
        advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(cfg.advantage_norm_eps)

        # Batch Optimization
        n = states.size(0)
        batch_size = min(cfg.minibatch_size, n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        masks_np = self.storage["masks"]
        masks_tensor = torch.stack([
            _to_tensor(np.ones(self.act_dim, dtype=np.float32), device) if m is None
            else _to_tensor(m, device) for m in masks_np
        ], dim=0)

        for _ in range(cfg.epochs):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                mb_idx = idx[start:start + batch_size]
                
                logits, v = self.model(states[mb_idx])
                dist = torch.distributions.Categorical(logits=masked_logits(logits, masks_tensor[mb_idx]))

                new_logp = dist.log_prob(actions[mb_idx])
                ratio = torch.exp(new_logp - old_logps[mb_idx])
                
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages[mb_idx]
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(v, returns[mb_idx])
                entropy = dist.entropy().mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()

        # Finalize Metrics
        updates = cfg.epochs * math.ceil(n / batch_size)
        for k in metrics: metrics[k] /= max(1, updates)
        
        self.reset_buffer()
        return metrics

    def reset_buffer(self):
        """Reset the rollout storage."""
        self.storage = {k: [] for k in ["states", "actions", "rewards", "dones", "masks", "logps", "values"]}
        self._last_action_aux = {}

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.__dict__,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
        }, path)

    def load(self, path: str, map_location: Optional[str] = None):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.model.to(self.device)