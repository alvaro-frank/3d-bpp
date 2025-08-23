# main.py
import os
import json
from pathlib import Path
import numpy as np

from train.train_dqn_agent import train_dqn_agent
from environment.packing_env import PackingEnv
from utils.heuristic import heuristic_blb_packing
from utils.seed import seed_all
from environment.box import Box

# ---------- reproducible test-set utils (self-contained) ----------
def make_test_sets(seed: int, n_episodes: int, n_boxes: int, box_ranges: dict):
    """
    Move to testsets.py
    Create deterministic test sets for reproducible evaluation.

    Parameters:
    - seed (int): RNG seed for reproducibility
    - n_episodes (int): number of episodes to generate
    - n_boxes (int): number of boxes per episode
    - box_ranges (dict): min/max ranges for box dimensions

    Returns:
    - list of episodes, where each episode is a list of dicts:
      [{"w": ..., "h": ..., "d": ...}, ...]
    """
    rng = np.random.default_rng(seed)
    sets = []
    for _ in range(n_episodes):
        ep = []
        for _ in range(n_boxes):
            w = int(rng.integers(box_ranges["w_min"], box_ranges["w_max"] + 1))
            h = int(rng.integers(box_ranges["h_min"], box_ranges["h_max"] + 1))
            d = int(rng.integers(box_ranges["d_min"], box_ranges["d_max"] + 1))
            ep.append({"w": w, "h": h, "d": d})
        sets.append(ep)
    return sets


def save_test_sets(path: str, sets):
    """
    Move to testsets.py
    Save generated test sets to disk in JSON format.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sets))

def load_test_sets(path: str):
    """
    Move to testsets.py
    Load test sets from JSON file.
    """
    return json.loads(Path(path).read_text())


# ---------- evaluation helpers ----------
def evaluate_agent_on_episode(agent, episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_dqn_agent.gif"):
    """
    Move to evaluate_dqn.py
    Evaluate the trained DQN agent on ONE test episode.

    - Forces epsilon = 0 (pure exploitation) for fair evaluation.
    - Uses action masking to avoid invalid placements.
    - Optionally generates a GIF of the packing process.

    Parameters:
    - agent (DQNAgent): trained agent
    - episode_boxes (list[dict]): list of boxes with dimensions (w, h, d)
    - env_seed (int, optional): seed for reproducibility
    - generate_gif (bool): whether to record GIF
    - gif_name (str): filename for output GIF

    Returns:
    - float: percentage of bin volume used
    """
    env = PackingEnv(bin_size=(10, 10, 10), max_boxes=len(episode_boxes), generate_gif=generate_gif, gif_name=gif_name)

    # Reset env with predetermined boxes
    state = env.reset(seed=env_seed, with_boxes=episode_boxes)

    # Backup epsilon and force greedy policy
    epsilon_backup = getattr(agent, "epsilon", None)
    agent.epsilon = 0.0

    total_reward = 0.0
    done = False
    while not done:
        # Select action using mask
        mask = env.valid_action_mask()
        action = agent.get_action(state, env.action_space, mask=mask)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    # Restore epsilon after eval
    if epsilon_backup is not None:
        agent.epsilon = epsilon_backup

    # Compute utilization percentage
    volume_used = env.get_placed_boxes_volume()
    pct_volume_used = (volume_used / env.bin_volume) * 100.0
    return pct_volume_used


def evaluate_heuristic_on_episode(episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_heuristic.gif"):
    """
    Move to evaluate_heuristic.py
    Evaluate the heuristic baseline on ONE test episode.

    - Recreates the environment to ensure fairness.
    - Uses a bottom-left-back heuristic with optional rotations.
    - Optionally generates a GIF of the packing process.

    Parameters:
    - episode_boxes (list[dict]): list of boxes with dimensions (w, h, d)
    - env_seed (int, optional): seed for reproducibility
    - generate_gif (bool): whether to record GIF
    - gif_name (str): filename for output GIF

    Returns:
    - float: percentage of bin volume used
    """
    env = PackingEnv(bin_size=(10, 10, 10), max_boxes=len(episode_boxes))
    env.reset(seed=env_seed, with_boxes=episode_boxes)

    # Convert dicts -> Box objects for heuristic
    boxes_for_heur = [Box(b["w"], b["h"], b["d"], id=i) for i, b in enumerate(episode_boxes)]

    placed_boxes, _bin = heuristic_blb_packing(
        bin_size=env.bin_size,
        boxes=boxes_for_heur,
        try_rotations=True,
        generate_gif=generate_gif,
        gif_name=gif_name,
    )

    volume_used = sum(box.get_volume() for box in placed_boxes)
    pct_volume_used = (volume_used / env.bin_volume) * 100.0
    return pct_volume_used

def main():
    """
    Main experiment script:
    1. Train DQN agent.
    2. Generate or load fixed test sets for reproducibility.
    3. Evaluate agent vs heuristic on the same test sets.
    4. Report average performance and regenerate GIFs for best runs.
    """
    
    # 0) Global seeding FIRST (makes everything reproducible)
    SEED = 1234
    seed_all(SEED)

    # 1) Train DQN
    print("📦 Training DQN Agent...")
    agent = train_dqn_agent(num_episodes=100, generate_gif=False)

    # 2) Build or load IDENTICAL test sets for both methods
    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / f"test_sets_seed{SEED}.json"

    if test_path.exists():
        test_sets = load_test_sets(str(test_path))
    else:
        test_sets = make_test_sets(
            seed=SEED,
            n_episodes=10,
            n_boxes=35,
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

    # 4) Regenerate GIFs only for the BEST episodes (so you keep output tidy)
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


if __name__ == "__main__":
    main()
