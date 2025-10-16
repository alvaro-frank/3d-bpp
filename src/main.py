# main.py
import os
from pathlib import Path
import argparse
import numpy as np

from evals.evaluate_dqn import evaluate_agent_on_episode
from evals.evaluate_heuristic import evaluate_heuristic_on_episode
from utils.testsets import make_test_sets, save_test_sets, load_test_sets
from train.train_dqn_agent import train_dqn_agent
from environment.packing_env import PackingEnv
from heuristics.heuristic import heuristic_blb_packing
from utils.seed import seed_all
from environment.box import Box

from agents.ppo_agent import PPOAgent, PPOConfig
from train.train_ppo_agent import train_ppo_agent as ppo_train_loop, TrainPPOConfig


def train_ppo_agent(num_episodes: int, max_boxes: int, generate_gif: bool = False):
    """
    Train a PPO agent on the same environment used by DQN and return the trained agent.

    Keeps a DQN-like interface (i.e., the returned object exposes get_action(...)).

    Args:
        num_episodes (int): Number of PPO training episodes.
        max_boxes (int): Number of boxes per episode in the environment.
        generate_gif (bool): If True, environment will store frames/GIFs during training.

    Returns:
        PPOAgent: Trained PPO agent ready for evaluation.
    """
    env = PackingEnv(max_boxes=max_boxes, include_noop=False)

    obs = env.reset()
    if isinstance(obs, tuple):
        obs, _ = obs
    obs_dim = int(np.prod(np.asarray(obs).shape))
    act_dim = env.action_space.n if hasattr(env, "action_space") else len(getattr(env, "discrete_actions", []))

    ppo_cfg = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        lr=3e-4,
        epochs=4,
        minibatch_size=1024,
        device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else ("cuda" if __import__("torch").cuda.is_available() else "cpu"),
    )
    agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=ppo_cfg)

    train_cfg = TrainPPOConfig(
        num_episodes=num_episodes,
        max_steps_per_episode=None, 
        log_every=10,
        eval_every=0,
        eval_episodes=5,
        save_every=50,
        save_dir="runs/ppo",
        seed=42,
    )
    ppo_train_loop(env, agent, cfg=train_cfg)

    try:
        env.close()
    except Exception:
        pass
    return agent


def main():
    """
    End-to-end pipeline: train → generate/load fixed test sets → evaluate vs heuristic → export best GIFs.

    Steps:
        1) Train selected agent (DQN by default; PPO if specified).
        2) Generate or load fixed test sets (same episodes for all methods).
        3) Evaluate agent vs. heuristic on the same test sets.
        4) Report averages and regenerate GIFs for the best episodes.

    Command line:
        python main.py [dqn_agent|ppo_agent]

    Notes:
        - The same seed/test sets ensure reproducibility and fair comparison.
        - GIFs for the best heuristic and agent episodes are saved under `runs/`.
    """
    parser = argparse.ArgumentParser(description="Treino e avaliação 3D-BPP")
    parser.add_argument("agent", nargs="?", default="dqn_agent", choices=["dqn_agent", "ppo_agent"],
                        help="Agente a treinar: dqn_agent (default) ou ppo_agent")
    args = parser.parse_args()

    SEED = 41
    N_EPISODES = 200
    N_TESTS = 20
    N_BOXES = 50
    seed_all(SEED)

    if args.agent == "dqn_agent":
        print("📦 Training DQN Agent...")
        agent = train_dqn_agent(num_episodes=N_EPISODES, max_boxes=N_BOXES, generate_gif=False)
    else:
        print("📦 Training PPO Agent...")
        agent = train_ppo_agent(num_episodes=N_EPISODES, max_boxes=N_BOXES, generate_gif=False)

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / f"test_sets_seed{SEED}.json"

    if test_path.exists():
        test_sets = load_test_sets(str(test_path))
    else:
        test_sets = make_test_sets(
            seed=SEED,
            n_episodes=N_TESTS,
            n_boxes=N_BOXES,
            box_ranges={"w_min": 1, "w_max": 5, "d_min": 1, "d_max": 5, "h_min": 1, "h_max": 5},
        )

    print("\n🤖 Evaluating Agent vs Heuristic:")
    agent_scores = []
    heuristic_scores = []
    best_agent = (-1.0, None) 
    best_heur = (-1.0, None)

    for i, episode_boxes in enumerate(test_sets):
        env_seed = SEED + i
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

    best_heur_score, best_heur_idx = best_heur
    best_agent_score, best_agent_idx = best_agent

    print(f"\n🎞️ Generating GIF for best heuristic test (Episode {best_heur_idx+1}) "
          f"with {best_heur_score:.2f}% volume used...")
    evaluate_heuristic_on_episode(
        test_sets[best_heur_idx],
        env_seed=SEED + best_heur_idx,
        generate_gif=True,
        gif_name="runs/heuristic_best.gif",
    )

    print(f"\n🎞️ Generating GIF for best {args.agent} test (Episode {best_agent_idx+1}) "
          f"with {best_agent_score:.2f}% volume used...")
    if args.agent == 'dqn_agent':
        gif_file = "runs/dqn/agent_best.gif"
    else:
        gif_file = "runs/ppo/agent_best.gif"
    evaluate_agent_on_episode(
        agent,
        test_sets[best_agent_idx],
        env_seed=SEED + best_agent_idx,
        generate_gif=True,
        gif_name=gif_file,
    )

if __name__ == "__main__":
    main()
