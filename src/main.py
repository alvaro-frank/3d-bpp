# main.py
import os
import json
from pathlib import Path
import numpy as np

from evals.evaluate_dqn import evaluate_agent_on_episode
from evals.evaluate_heuristic import evaluate_heuristic_on_episode
from utils.testsets import make_test_sets, save_test_sets, load_test_sets
from train.train_dqn_agent import train_dqn_agent
from environment.packing_env import PackingEnv
from heuristics.heuristic import heuristic_blb_packing
from utils.seed import seed_all
from environment.box import Box

def main():
    """
    Main script:
    1. Train DQN agent.
    2. Generate or load fixed test sets for reproducibility.
    3. Evaluate agent vs heuristic on the same test sets.
    4. Report average performance and regenerate GIFs for best runs.
    """
    # 0) Global seeding FIRST (makes everything reproducible)
    SEED = 1234
    N_EPISODES = 5000
    N_TESTS = 10
    N_BOXES = 35
    seed_all(SEED)

    # 1) Train DQN
    print("📦 Training DQN Agent...")
    agent = train_dqn_agent(num_episodes=N_EPISODES, max_boxes=N_BOXES, generate_gif=False)

    # 2) Build or load IDENTICAL test sets for both methods
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
            box_ranges={"w_min": 1, "w_max": 5, "h_min": 1, "h_max": 5, "d_min": 1, "d_max": 5},
        )
        save_test_sets(str(test_path), test_sets)

    # 3) Evaluate DQN vs heuristic on same test episodes
    print("\n🤖 Evaluating DQN vs Heuristic:")
    dqn_scores = []
    heuristic_scores = []
    best_dqn = (-1.0, None) # (best score, episode idx)
    best_heur = (-1.0, None) # (best score, episode idx)

    for i, episode_boxes in enumerate(test_sets):
        env_seed = SEED + i  # per-episode reset seed
        dqn_score = evaluate_agent_on_episode(
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

        dqn_scores.append(dqn_score)
        heuristic_scores.append(heur_score)

        if dqn_score > best_dqn[0]:
            best_dqn = (dqn_score, i)
        if heur_score > best_heur[0]:
            best_heur = (heur_score, i)

        print(f"Test {i+1}: DQN = {dqn_score:.2f}%, Heuristic = {heur_score:.2f}%")

    # Report averages
    print("\n📊 DQN Avg:", float(np.mean(dqn_scores)))
    print("📊 Heuristic Avg:", float(np.mean(heuristic_scores)))

    # 4) Regenerate GIFs only for the BEST episodes
    best_heur_score, best_heur_idx = best_heur
    best_dqn_score, best_dqn_idx = best_dqn

    print(f"\n🎞️ Generating GIF for best heuristic test (Episode {best_heur_idx}) "
          f"with {best_heur_score:.2f}% volume used...")
    evaluate_heuristic_on_episode(
        test_sets[best_heur_idx],
        env_seed=SEED + best_heur_idx,
        generate_gif=True,
        gif_name=f"runs/heuristic_best_ep{best_heur_idx}.gif",
    )

    print(f"\n🎞️ Generating GIF for best DQN test (Episode {best_dqn_idx}) "
          f"with {best_dqn_score:.2f}% volume used...")
    evaluate_agent_on_episode(
        agent,
        test_sets[best_dqn_idx],
        env_seed=SEED + best_dqn_idx,
        generate_gif=True,
        gif_name=f"runs/agent_best_ep{best_dqn_idx}.gif",
    )

    # 5) Extra: side-by-side comparison on the SAME episode
    COMPARE_IDX = 0  # or pick any fixed index you want to visualize
    print(f"\n🆚 Generating comparison GIFs for episode {COMPARE_IDX} "
          f"(both agent and heuristic on identical boxes)...")

    print(test_sets[COMPARE_IDX])

    evaluate_heuristic_on_episode(
        test_sets[COMPARE_IDX],
        env_seed=SEED + COMPARE_IDX,
        generate_gif=True,
        gif_name=f"runs/heuristic_compare_ep{COMPARE_IDX}.gif",
    )

    evaluate_agent_on_episode(
        agent,
        test_sets[COMPARE_IDX],
        env_seed=SEED + COMPARE_IDX,
        generate_gif=True,
        gif_name=f"runs/agent_compare_ep{COMPARE_IDX}.gif",
    )

if __name__ == "__main__":
    main()
