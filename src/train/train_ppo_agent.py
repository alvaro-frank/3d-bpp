# src/train/train_ppo_agent.py
import os
import time
import matplotlib.pyplot as plt
import mlflow
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch

@dataclass
class TrainPPOConfig:
    """
    Training configuration for PPO.

    Attributes:
        num_episodes: Total number of training episodes.
        max_steps_per_episode: Optional hard cap on steps per episode (None = unlimited).
        log_every: Print rolling stats every N episodes.
        eval_every: Evaluate every N episodes (0 = disabled).
        eval_episodes: Number of episodes to average during evaluation.
        save_every: Save a checkpoint every N episodes.
        save_dir: Output directory for runs, plots, and checkpoints.
        seed: Optional random seed for reproducibility.
    """
    num_episodes: int = 1000
    max_steps_per_episode: Optional[int] = None
    log_every: int = 10
    eval_every: int = 0
    eval_episodes: int = 5
    save_every: int = 50
    save_dir: str = "runs/ppo"
    save_models: str = "runs/ppo/models"
    seed: Optional[int] = 42

def _ensure_dir(path: str):
    """
    Create a directory if it doesn't exist.

    Args:
        path: Directory path to create.
    """
    os.makedirs(path, exist_ok=True)

def _maybe_valid_mask(env) -> Optional[np.ndarray]:
    """
    Safely call env.valid_action_mask() if available.

    Returns:
        Optional[np.ndarray]: A float mask in [0,1] if provided by the env, else None.
    """
    if hasattr(env, "valid_action_mask") and callable(getattr(env, "valid_action_mask")):
        try:
            m = env.valid_action_mask()
            if m is not None:
                return np.asarray(m, dtype=np.float32)
        except Exception:
            pass
    return None


@torch.no_grad()
def _bootstrap_value_if_truncated(agent, last_obs, was_truncated: bool) -> float:
    """
    Estimate bootstrap value for a truncated (non-terminal) episode.

    Args:
        agent: PPO agent with `.model` and `.config.device`.
        last_obs: Final observation of the rollout.
        was_truncated: Whether the episode ended due to a time/step limit.

    Returns:
        float: Value estimate V(s_T) if truncated; otherwise 0.0.
    """
    if not was_truncated:
        return 0.0
    device = torch.device(agent.config.device if hasattr(agent, "config") else ("cuda" if torch.cuda.is_available() else "cpu"))
    obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=device).unsqueeze(0)
    if hasattr(agent, "model"):
        _, v = agent.model(obs_t)
        return float(v.item())
    return 0.0


def evaluate(env, agent, episodes: int = 5) -> Dict[str, float]:
    """
    Run a short evaluation loop and report mean return/steps.

    Args:
        env: Environment instance (must support reset/step and valid_action_mask()).
        agent: PPO agent (must implement get_action()).
        episodes: Number of evaluation episodes to average.

    Returns:
        Dict[str, float]: {'eval_return_mean', 'eval_steps_mean'}.
    """
    returns: List[float] = []
    steps: List[int] = []
    for _ in range(episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            mask = _maybe_valid_mask(env)

            if mask is None:
                raise RuntimeError("valid_action_mask() devolveu None.")
            if mask.shape[0] != agent.act_dim:
                raise RuntimeError(f"Máscara shape {mask.shape} != act_dim {agent.act_dim}.")
            if not np.isfinite(mask).all():
                raise RuntimeError("Máscara contém NaN/Inf.")
            if (mask > 1).any() or (mask < 0).any():
                raise RuntimeError("Máscara fora de [0,1].")
            if (mask <= 0).all():
                raise RuntimeError("Todas as ações inválidas neste step — precisa de um no-op válido.")

            action = agent.get_action(obs, getattr(env, "action_space", None), mask=mask)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step_out
            ep_ret += float(reward)
            ep_len += 1
        returns.append(ep_ret)
        steps.append(ep_len)
    return {"eval_return_mean": float(np.mean(returns)), "eval_steps_mean": float(np.mean(steps))}


def train_ppo_agent(env, agent, cfg: TrainPPOConfig) -> Dict[str, Any]:
    """
    Train a PPO agent for a given number of episodes, with optional periodic evaluation
    and checkpointing. Saves a learning curve plot at the end.

    Args:
        env: Environment instance (must support reset/step and optional valid_action_mask()).
        agent: PPO agent implementing get_action(), store_transition(), train(), and save().
        cfg: Training configuration (see TrainPPOConfig).

    Returns:
        Dict[str, Any]: {
            'history': list of per-episode logs,
            'best_eval_return': best evaluation mean return seen (or -inf if eval disabled)
        }
    """
    if cfg.seed is not None and hasattr(env, "seed"):
        try:
            env.seed(cfg.seed)
        except Exception:
            pass
    if cfg.seed is not None and hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        try:
            env.action_space.seed(cfg.seed)
        except Exception:
            pass
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    _ensure_dir(cfg.save_dir)
    _ensure_dir(cfg.save_models)
    history: List[Dict[str, float]] = []
    best_eval = -float("inf")

    rewards_per_episode: List[float] = []
    volume_utilizations: List[float] = []

    t0 = time.time()
    for ep in range(1, cfg.num_episodes + 1):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _ = obs

        done = False
        ep_ret, ep_len = 0.0, 0
        was_truncated = False
        last_info = {}

        while not done:
            mask = _maybe_valid_mask(env)
            action = agent.get_action(obs, getattr(env, "action_space", None), mask=mask)

            step_out = env.step(action)
            if len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
                was_truncated = bool(truncated)
            else:
                next_obs, reward, done, info = step_out
                was_truncated = bool(info.get("TimeLimit.truncated", False) or info.get("truncated", False))

            agent.store_transition(
                state=np.asarray(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_state=np.asarray(next_obs, dtype=np.float32),
                done=bool(done),
                mask=mask,
            )

            ep_ret += float(reward)
            ep_len += 1
            obs = next_obs
            last_info = info or {}

            if cfg.max_steps_per_episode and ep_len >= cfg.max_steps_per_episode:
                was_truncated = True
                break

        last_value = _bootstrap_value_if_truncated(agent, obs, was_truncated)
        ppo_metrics = agent.train(last_value=last_value)

        try:
            volume_used = env.get_placed_boxes_volume()
            bin_volume = env.bin.bin_volume()
            pct_volume_used = (volume_used / bin_volume) * 100 if bin_volume > 0 else 0.0
            boxes_placed = len(env.packed_boxes)
            skipped_boxes = len(env.skipped_boxes)
        except Exception:
            pct_volume_used = float(last_info.get("volume_utilization_pct") or 0.0)

        rewards_per_episode.append(ep_ret)
        volume_utilizations.append(float(pct_volume_used))
        
        window = 100
        start_idx = max(0, len(rewards_per_episode) - window)
        avg_ret_recent = np.mean(rewards_per_episode[start_idx:])
        
        mlflow.log_metric("avg_reward_100", avg_ret_recent, step=ep)
        mlflow.log_metric("volume_utilization", pct_volume_used, step=ep)
        mlflow.log_metric("boxes_placed", boxes_placed, step=ep)
        mlflow.log_metric("boxes_skipped", skipped_boxes, step=ep)

        log = {
            "episode": ep,
            "return": ep_ret,
            "length": ep_len,
            "time_sec": time.time() - t0,
        }
        if ppo_metrics:
            log.update({f"ppo/{k}": float(v) for k, v in ppo_metrics.items()})
        history.append(log)

        if (ep % cfg.log_every) == 0 or ep == 1:
            recent = history[-cfg.log_every:] if len(history) >= cfg.log_every else history
            avg_ret = float(np.mean([h["return"] for h in recent]))
            avg_len = float(np.mean([h["length"] for h in recent]))
            print(f"[PPO] Ep {ep:5d} | R(avg) {avg_ret:8.3f} | L(avg) {avg_len:6.1f}")

        if cfg.eval_every and (ep % cfg.eval_every == 0):
            try:
                eval_metrics = evaluate(env, agent, episodes=cfg.eval_episodes)
                print(f"[PPO][Eval] Ep {ep:5d} | Return(mean) {eval_metrics['eval_return_mean']:.3f} | Steps(mean) {eval_metrics['eval_steps_mean']:.1f}")
                if eval_metrics["eval_return_mean"] > best_eval:
                    best_eval = eval_metrics["eval_return_mean"]
                    agent.save(os.path.join(cfg.save_dir, "ppo_best.pt"))
            except Exception as e:
                print(f"[PPO][Eval] erro na avaliação: {e}")

        if (ep % cfg.save_every) == 0:
            agent.save(os.path.join(cfg.save_models, "ppo_latest.pt"))

    sanity_check_mask(env)
    
    agent.save(os.path.join(cfg.save_models, "ppo_final.pt"))

    save_path = os.path.join(cfg.save_dir, "learning_curve.png")

    rewards_smoothed = [np.mean(rewards_per_episode[i:i+window])
                        for i in range(0, len(rewards_per_episode), window)]
    utilizations_smoothed = [np.mean(volume_utilizations[i:i+window])
                             for i in range(0, len(volume_utilizations), window)]
    episodes_axis = list(range(window, len(rewards_per_episode) + 1, window))

    plt.figure(figsize=(12, 5))

    # Total Reward curve
    plt.subplot(1, 2, 1)
    plt.plot(episodes_axis, rewards_smoothed, label="Total Reward (avg/100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve (Reward)")
    plt.legend()

    # Utilization curve
    plt.subplot(1, 2, 2)
    plt.plot(episodes_axis, utilizations_smoothed, label="Volume Utilization % (avg/100)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Utilization (%)")
    plt.title("Bin Volume Utilization")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return {"history": history, "best_eval_return": best_eval}

def sanity_check_mask(env, episodes=5):
    """
    Quick validation for env.valid_action_mask() behavior.

    Checks per step:
        - shape matches env action dimension
        - no NaN/Inf, no negatives, not all zeros
        - counts valid actions

    Args:
        env: Environment instance.
        episodes: Number of episodes to probe.

    Prints:
        Problem counters and min/mean/max number of valid actions per step.
    """
    import numpy as np
    problems = {"nan":0, "neg":0, "wrong_shape":0, "all_zero":0}
    counts_valid = []
    act_dim = env.action_space.n if hasattr(env, "action_space") else len(getattr(env, "discrete_actions", []))
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            m = env.valid_action_mask()
            m = np.asarray(m, dtype=np.float32)

            if m.shape[0] != act_dim:
                problems["wrong_shape"] += 1
            if np.isnan(m).any():
                problems["nan"] += 1
            if (m < 0).any():
                problems["neg"] += 1
            if (m <= 0).all():
                problems["all_zero"] += 1
            counts_valid.append(int((m > 0.5).sum()))

            valid_idxs = np.where(m > 0.5)[0]
            a = int(np.random.choice(valid_idxs)) if len(valid_idxs) else 0
            step_out = env.step(a)
            if len(step_out) == 5:
                obs, r, term, trunc, info = step_out
                done = bool(term or trunc)
            else:
                obs, r, done, info = step_out