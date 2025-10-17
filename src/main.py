# src/main.py
import os
from pathlib import Path
import argparse
import glob
import numpy as np
import torch

from evals.evaluate_agent import evaluate_agent_on_episode
from evals.evaluate_heuristic import evaluate_heuristic_on_episode
from utils.testsets import make_test_sets, load_test_sets
from train.train_dqn_agent import train_dqn_agent as dqn_train_loop
from environment.packing_env import PackingEnv
from utils.seed import seed_all

from agents.ppo_agent import PPOAgent, PPOConfig
from train.train_ppo_agent import train_ppo_agent as ppo_train_loop, TrainPPOConfig
from agents.dqn_agent import DQNAgent


# ---------------------------- helpers ---------------------------------
def _build_env(max_boxes: int, include_noop: bool = False):
    return PackingEnv(max_boxes=max_boxes, include_noop=include_noop)

def _build_dqn(env, exploration: str = "softmax"):
    state_dim = int(np.prod(env.observation_space.shape))
    # Prefer the true action dimension from the env (handles noop consistently)
    act_dim = env.action_space.n if hasattr(env, "action_space") else len(getattr(env, "discrete_actions", []))
    return DQNAgent(state_dim, act_dim, exploration=exploration)

def _build_ppo(env):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n if hasattr(env, "action_space") else len(getattr(env, "discrete_actions", []))
    ppo_cfg = PPOConfig(
        gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
        vf_coef=0.5, ent_coef=0.01, lr=3e-4,
        epochs=4, minibatch_size=1024,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    return PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=ppo_cfg)

def _extract_state_dict(ckpt):
    """
    Return a model state_dict from a variety of checkpoint formats.
    Supported keys (in order of preference):
      - "model"
      - "model_state_dict"
      - "state_dict"
      - raw state_dict (mapping of parameter tensors)
    """
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        # Heuristic: if it looks like a state_dict (all tensor-like leaves)
        if all(isinstance(v, (dict, torch.Tensor)) for v in ckpt.values()):
            return ckpt  # assume it's already a state_dict
    # Fallback: raise a helpful error
    raise ValueError(
        "Checkpoint format not recognized. Expected keys like "
        "'model', 'model_state_dict', or 'state_dict'."
    )

def _load_weights_dqn(agent, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent

def _load_weights_ppo(agent, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent

def _auto_find_checkpoint(agent_type: str) -> str | None:
    if agent_type == "dqn":
        candidates = sorted(
            glob.glob("runs/dqn/models/*.pth") + glob.glob("runs/dqn/models/*.pt")
        )
    else:
        candidates = sorted(
            glob.glob("runs/ppo/models/*.pt") + glob.glob("runs/ppo/models/*.pth")
        )
    return candidates[-1] if candidates else None


# --------------------------- commands ---------------------------------
def cmd_train(agent_type: str, episodes: int, boxes: int, seed: int):
    seed_all(seed)
    print(f"📦 Training {agent_type.upper()} | episodes={episodes} boxes={boxes} seed={seed}")

    if agent_type == "dqn":
        agent = dqn_train_loop(num_episodes=episodes, max_boxes=boxes, generate_gif=False)
    else:
        # PPO: build env/agent to pass into PPO training loop
        env = _build_env(max_boxes=boxes, include_noop=False)
        agent = _build_ppo(env)
        train_cfg = TrainPPOConfig(
            num_episodes=episodes,
            max_steps_per_episode=None,
            log_every=10,
            eval_every=0,
            eval_episodes=5,
            save_every=50,
            save_dir="runs/ppo",
            save_models="runs/ppo/models",
            seed=seed,
        )
        os.makedirs(train_cfg.save_models, exist_ok=True)
        ppo_train_loop(env, agent, cfg=train_cfg)
        try:
            env.close()
        except Exception:
            pass

    print("✅ Training finished.")
    return 0


def cmd_evaluate(agent_type: str, boxes: int, tests: int, seed: int, model_path: str | None, make_gifs: bool):
    """
    Evaluate a trained agent against the heuristic on fixed test sets.
    If model_path is None, tries to auto-discover a checkpoint under runs/*/models/.
    """
    seed_all(seed)

    # Build env to size action/state correctly (we will reset with provided boxes per episode)
    env = _build_env(max_boxes=boxes, include_noop=False)

    # Build agent + load weights (or skip for heuristic)
    if agent_type == "dqn":
        agent = _build_dqn(env)
        model_path = model_path or _auto_find_checkpoint("dqn")
        if not model_path:
            raise FileNotFoundError("No DQN checkpoint found. Pass --model PATH or train first.")
        _load_weights_dqn(agent, model_path)
        print(f"🔁 Loaded DQN weights from: {model_path}")
    else:
        agent = _build_ppo(env)
        model_path = model_path or _auto_find_checkpoint("ppo")
        if not model_path:
            raise FileNotFoundError("No PPO checkpoint found. Pass --model PATH or train first.")
        _load_weights_ppo(agent, model_path)
        print(f"🔁 Loaded PPO weights from: {model_path}")

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_sets = make_test_sets(
        seed=seed, n_episodes=tests, n_boxes=boxes,
        box_ranges={"w_min": 1, "w_max": 5, "d_min": 1, "d_max": 5, "h_min": 1, "h_max": 5},
    )

    print("\n🤖 Evaluating Agent vs Heuristic:")
    agent_scores, heuristic_scores = [], []
    best_agent = (-1.0, None)
    best_heur  = (-1.0, None)

    for i, episode_boxes in enumerate(test_sets):
        env_seed = seed + i
        agent_score = evaluate_agent_on_episode(
            agent,
            episode_boxes,
            env_seed=env_seed,
            generate_gif=False,
            gif_name=f"runs/best_agent_ep{i}.gif",
        )
        heur_score = evaluate_heuristic_on_episode(
            episode_boxes,
            env_seed=env_seed,
            generate_gif=False,
            gif_name=f"runs/best_heuristic_ep{i}.gif",
        )

        agent_scores.append(agent_score)
        heuristic_scores.append(heur_score)
        if agent_score > best_agent[0]:
            best_agent = (agent_score, i)
        if heur_score > best_heur[0]:
            best_heur = (heur_score, i)

        print(f"Test {i+1}: Agent = {agent_score:.2f}%, Heuristic = {heur_score:.2f}%")

    print("\n📊 Agent Avg:", float(np.mean(agent_scores)))
    print("📊 Heuristic Avg:", float(np.mean(heuristic_scores)))

    # Optional GIFs of the best episodes
    if make_gifs:
        best_heur_score, best_heur_idx = best_heur
        best_agent_score, best_agent_idx = best_agent

        print(f"\n🎞️ Generating GIF for best heuristic test (Episode {best_heur_idx+1}) "
              f"with {best_heur_score:.2f}% volume used...")
        evaluate_heuristic_on_episode(
            test_sets[best_heur_idx],
            env_seed=seed + best_heur_idx,
            generate_gif=True,
            gif_name="runs/heuristic_best.gif",
        )

        print(f"\n🎞️ Generating GIF for best {agent_type.upper()} test (Episode {best_agent_idx+1}) "
              f"with {best_agent_score:.2f}% volume used...")
        gif_file = "runs/dqn/agent_best.gif" if agent_type == "dqn" else "runs/ppo/agent_best.gif"
        evaluate_agent_on_episode(
            agent,
            test_sets[best_agent_idx],
            env_seed=seed + best_agent_idx,
            generate_gif=True,
            gif_name=gif_file,
        )

    try:
        env.close()
    except Exception:
        pass

    print("✅ Evaluation finished.")
    return 0


# ----------------------------- CLI ------------------------------------
def main():
    """
    CLI with two independent commands:
      - train    : train an agent
      - evaluate : evaluate a trained agent vs heuristic
    """
    parser = argparse.ArgumentParser(description="3D-BPP: train/evaluate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train an agent")
    p_train.add_argument("--agent", choices=["dqn", "ppo"], default="dqn")
    p_train.add_argument("--episodes", type=int, default=200)
    p_train.add_argument("--boxes", type=int, default=50)
    p_train.add_argument("--seed", type=int, default=41)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate an agent vs heuristic")
    p_eval.add_argument("--agent", choices=["dqn", "ppo"], default="dqn")
    p_eval.add_argument("--model", type=str, default=None, help="Path to checkpoint; auto-detect if omitted")
    p_eval.add_argument("--tests", type=int, default=20)
    p_eval.add_argument("--boxes", type=int, default=50)
    p_eval.add_argument("--seed", type=int, default=41)
    p_eval.add_argument("--gifs", action="store_true", help="Also render GIFs of best episodes")

    args = parser.parse_args()

    if args.cmd == "train":
        return cmd_train(args.agent, args.episodes, args.boxes, args.seed)
    else:
        return cmd_evaluate(args.agent, args.boxes, args.tests, args.seed, args.model, args.gifs)


if __name__ == "__main__":
    raise SystemExit(main())
