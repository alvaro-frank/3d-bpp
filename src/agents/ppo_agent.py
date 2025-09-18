# src/agents/ppo_agent.py
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------- Utils -------- #

def _to_tensor(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Aplica máscara binária (1=válida, 0=inválida) aos logits.
    Ações inválidas recebem um grande negativo para ficarem com prob. ~0.
    """
    if mask is None:
        return logits
    # garante tipos/dispositivo
    mask = mask.to(logits.device).float()
    invalid = (mask < 0.5)
    # evita o caso de todas inválidas (não deve acontecer): não mascara nada se todas forem 0
    if invalid.all():
        return logits
    # usa -1e9 (em fp32 é seguro) para "remover" a ação
    masked = logits.masked_fill(invalid, -1e9)
    return masked


def categorical_sample_from_masked_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    masked = masked_logits(logits, mask)
    dist = torch.distributions.Categorical(logits=masked)
    action = dist.sample()
    logp = dist.log_prob(action)
    return action, logp


# -------- Modelo -------- #

class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(256, 256), layer_norm=False):
        super().__init__()
        layers: List[nn.Module] = []
        last = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last, h)]
            if layer_norm:
                layers += [nn.LayerNorm(h)]
            layers += [nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)
        self.pi_head = nn.Linear(last, act_dim)  # logits
        self.v_head = nn.Linear(last, 1)

        # ortho init (ligeiro boost de estabilidade)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu') if m is not self.pi_head and m is not self.v_head else 1.0)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.body(obs)
        logits = self.pi_head(x)
        value = self.v_head(x).squeeze(-1)
        return logits, value


# -------- Hiperparâmetros -------- #

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
    minibatch_size: int = 1024  # funciona com rollouts multi-episódio; para por-episódio, diminuí
    hidden_sizes: Tuple[int, int] = (256, 256)
    layer_norm: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # normalização de vantagens
    advantage_norm_eps: float = 1e-8


# -------- Agente PPO -------- #

class PPOAgent:
    """
    PPO com máscara de ações inválidas.
    Interface compatível com DQN:
      - get_action(state, action_space, mask=None) -> int
      - store_transition(state, action, reward, next_state, done, mask=None, logp=None, value=None)
      - train(last_value=0.0) -> dict (métricas)
      - save(path), load(path)
    """
    def __init__(self, obs_dim: int, act_dim: int, config: Optional[PPOConfig] = None):
        self.config = config or PPOConfig()
        self.device = torch.device(self.config.device)

        self.model = MLPActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_sizes=self.config.hidden_sizes,
            layer_norm=self.config.layer_norm
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # buffers do rollout
        self.reset_buffer()

        # metadados
        self.act_dim = act_dim
        self.obs_dim = obs_dim

    # ---------- API ---------- #

    @torch.no_grad()
    def get_action(self, state: np.ndarray, action_space: Any, mask: Optional[np.ndarray] = None) -> int:
        """
        Seleciona ação amostrando da policy (com máscara).
        `action_space` é ignorado exceto para compatibilidade (pode ser gym.spaces.Discrete).
        """
        self.model.eval()
        s = _to_tensor(state, self.device).float().unsqueeze(0)
        logits, value = self.model(s)

        mask_t = None
        if mask is not None:
            mask_t = torch.from_numpy(mask.astype(np.float32)).to(self.device).unsqueeze(0) if isinstance(mask, np.ndarray) else _to_tensor(mask, self.device).unsqueeze(0)

        action_t, logp_t = categorical_sample_from_masked_logits(logits, mask_t)
        action = int(action_t.item())
        logp = float(logp_t.item())
        v = float(value.item())

        # Guarda parcialmente para evitar recomputar na store (opcional)
        self._last_action_aux = {"logp": logp, "value": v}
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        mask: Optional[np.ndarray] = None,
        logp: Optional[float] = None,
        value: Optional[float] = None,
    ):
        # Se não nos passaram logp/value, podemos recomputar quickly
        if (logp is None) or (value is None):
            with torch.no_grad():
                s = _to_tensor(state, self.device).unsqueeze(0)
                logits, v = self.model(s)
                mask_t = None
                if mask is not None:
                    mask_t = _to_tensor(mask, self.device).unsqueeze(0)
                # log prob da ação tomada
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
        """
        Treina sobre o rollout acumulado (tipicamente um episódio, mas pode ser multi-episódio).
        `last_value`: valor de bootstrap do último estado caso a terminação não seja terminal "verdadeira".
        """
        if len(self.storage["states"]) == 0:
            return {}

        device = self.device
        cfg = self.config

        # ---- Tensors ---- #
        states = _to_tensor(np.array(self.storage["states"]), device)
        actions = torch.tensor(self.storage["actions"], dtype=torch.long, device=device)
        rewards = _to_tensor(np.array(self.storage["rewards"], dtype=np.float32), device)
        dones = torch.tensor(self.storage["dones"], dtype=torch.float32, device=device)
        old_logps = _to_tensor(np.array(self.storage["logps"], dtype=np.float32), device)
        values = _to_tensor(np.array(self.storage["values"], dtype=np.float32), device)

        # ---- GAE(λ) ---- #
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

        # normalização das vantagens
        adv_mean = advantages.mean()
        adv_std = advantages.std().clamp_min(cfg.advantage_norm_eps)
        advantages = (advantages - adv_mean) / adv_std

        # ---- Treino PPO ---- #
        n = states.size(0)
        batch_size = min(cfg.minibatch_size, n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}

        # precomputar máscaras por passo (se existirem)
        masks_np = self.storage["masks"]
        has_masks = any(m is not None for m in masks_np)
        if has_masks:
            masks_tensor = torch.stack([
                _to_tensor(np.ones(self.act_dim, dtype=np.float32), device) if m is None
                else _to_tensor(m, device)
            for m in masks_np
            ], dim=0)
        else:
            masks_tensor = None

        for _ in range(cfg.epochs):
            idx = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                mb_idx = idx[start:start + batch_size]

                s_mb = states[mb_idx]
                a_mb = actions[mb_idx]
                ret_mb = returns[mb_idx]
                adv_mb = advantages[mb_idx]
                old_logp_mb = old_logps[mb_idx]
                mask_mb = masks_tensor[mb_idx] if masks_tensor is not None else None

                logits, v = self.model(s_mb)
                masked = masked_logits(logits, mask_mb)
                dist = torch.distributions.Categorical(logits=masked)

                new_logp = dist.log_prob(a_mb)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - old_logp_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(v, ret_mb)

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (old_logp_mb - new_logp).mean().abs().item()
                # acumula métricas (média móvel simples)
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += approx_kl

        # normaliza métricas por nº de updates
        updates = cfg.epochs * math.ceil(n / batch_size)
        for k in metrics:
            metrics[k] /= max(1, updates)

        # limpar rollout
        self.reset_buffer()
        return metrics

    def reset_buffer(self):
        self.storage: Dict[str, List] = {
            "states": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "masks": [],
            "logps": [],
            "values": [],
        }
        self._last_action_aux: Dict[str, Any] = {}

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config.__dict__,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            },
            path,
        )

    def load(self, path: str, map_location: Optional[str] = None):
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # permitir continuar treino com a mesma config
        if "config" in ckpt:
            self.config = PPOConfig(**ckpt["config"])
        if "obs_dim" in ckpt:
            self.obs_dim = ckpt["obs_dim"]
        if "act_dim" in ckpt:
            self.act_dim = ckpt["act_dim"]
        self.model.to(self.device)
