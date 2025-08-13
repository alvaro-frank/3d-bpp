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
    Returns a list of episodes; each episode is a list of dicts:
    [{"w":..., "h":..., "d":...}, ...]
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(sets))


def load_test_sets(path: str):
    return json.loads(Path(path).read_text())


# ---------- evaluation helpers ----------
def evaluate_agent_on_episode(agent, episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_dqn_agent.gif"):
    """
    Evaluate the trained agent on ONE episode defined by `episode_boxes`
    (list of dicts with w/h/d). Uses action masking and epsilon=0 for fair eval.
    """
    env = PackingEnv(bin_size=(10, 10, 10), max_boxes=len(episode_boxes), generate_gif=generate_gif, gif_name=gif_name)

    # Ensure env.reset can accept with_boxes (you added this earlier);
    # if your reset doesn't have with_boxes yet, add it as discussed.
    state = env.reset(seed=env_seed, with_boxes=episode_boxes)

    # force greedy policy during evaluation
    epsilon_backup = getattr(agent, "epsilon", None)
    agent.epsilon = 0.0

    total_reward = 0.0
    done = False
    while not done:
        # use action mask during eval
        mask = env.valid_action_mask()
        action = agent.get_action(state, env.action_space, mask=mask)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    # restore epsilon
    if epsilon_backup is not None:
        agent.epsilon = epsilon_backup

    volume_used = env.get_placed_boxes_volume()
    pct_volume_used = (volume_used / env.bin_volume) * 100.0
    return pct_volume_used


def evaluate_heuristic_on_episode(episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_heuristic.gif"):
    """
    Evaluate the heuristic on ONE episode defined by `episode_boxes`.
    We construct Box objects inside the heuristic call for fairness.
    """
    # Recreate an env to get bin dimensions and exact same geometry
    env = PackingEnv(bin_size=(10, 10, 10), max_boxes=len(episode_boxes))
    env.reset(seed=env_seed, with_boxes=episode_boxes)

    # heuristic expects Box objects (your previous version used env.boxes)
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


# ---------- main ----------
def main():
    # 0) Global seeding FIRST (makes everything reproducible)
    SEED = 1234
    seed_all(SEED)

    # 1) Train DQN (your train function should also respect the RNG state)
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

    # 3) Evaluate DQN vs Heuristic on the SAME episodes (seeded)
    print("\n🤖 Evaluating DQN vs Heuristic:")
    dqn_scores = []
    heuristic_scores = []
    best_dqn = (-1.0, None)        # (score, idx)
    best_heur = (-1.0, None)       # (score, idx)

    for i, episode_boxes in enumerate(test_sets):
        env_seed = SEED + i  # per-episode reset seed (reproducible)
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
