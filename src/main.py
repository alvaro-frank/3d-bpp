
import argparse
import numpy as np
import random
from copy import deepcopy

from environment.packing_env import PackingEnv
from train.train_dqn_agent import train_dqn_agent
from utils.heuristic import heuristic_blb_packing
from utils.repro import set_global_seed, make_seed_sequence

def evaluate_dqn(agent, bin_size, max_boxes, seed, gif=False, gif_name=None):
    env = PackingEnv(bin_size=bin_size, max_boxes=max_boxes, generate_gif=gif, gif_name=(gif_name or f"dqn_{seed}.gif"))
    obs = env.reset(seed=seed)
    done = False
    while not done:
        action = agent.get_action(obs, env.action_space)
        obs, reward, done, info = env.step(action)
    score = env.get_terminal_reward() * 100.0
    return score

def evaluate_heuristic_from_boxes(bin_size, boxes, gif=False, gif_name=None):
    # deep copy boxes so we don't mutate caller
    boxes_copy = [deepcopy(b) for b in boxes]
    _, bbin = heuristic_blb_packing(bin_size, boxes_copy, try_rotations=True, generate_gif=gif, gif_name=(gif_name or "heuristic.gif"))
    used = sum(b.get_volume() for b in bbin.boxes)
    total = bbin.width * bbin.height * bbin.depth
    return 100.0 * used / total

def run_comparison(args):
    set_global_seed(args.seed)

    # Train DQN
    agent = train_dqn_agent(
        num_episodes=args.episodes,
        bin_size=tuple(args.bin_size),
        max_boxes=args.max_boxes,
        gif_name="packing_dqn_train.gif",
        generate_gif=args.train_gif,
    )

    # Create a common test set of seeds for fair comparison
    seeds = make_seed_sequence(args.seed, args.tests)

    dqn_scores, heuristic_scores = [], []
    best = {"dqn":(-1,None), "heuristic":(-1,None)}  # (score, seed)

    for s in seeds:
        # Generate identical box sequence by resetting env with seed, then copy env.boxes for heuristic
        env = PackingEnv(bin_size=tuple(args.bin_size), max_boxes=args.max_boxes)
        env.reset(seed=int(s))
        boxes_for_heur = env.boxes

        # Evaluate DQN with same seed
        dqn_score = evaluate_dqn(agent, tuple(args.bin_size), args.max_boxes, int(s), gif=False)
        heur_score = evaluate_heuristic_from_boxes(tuple(args.bin_size), boxes_for_heur, gif=False)

        dqn_scores.append(dqn_score)
        heuristic_scores.append(heur_score)

        if dqn_score > best["dqn"][0]:
            best["dqn"] = (dqn_score, int(s))
        if heur_score > best["heuristic"][0]:
            best["heuristic"] = (heur_score, int(s))

        print(f"Seed {int(s)} -> DQN: {dqn_score:.2f}% | Heuristic: {heur_score:.2f}%")

    print("\nDQN Avg:", np.mean(dqn_scores))
    print("Heuristic Avg:", np.mean(heuristic_scores))

    # GIFs for best cases
    print(f"\nGenerating GIFs...")
    # Heuristic GIF
    env = PackingEnv(bin_size=tuple(args.bin_size), max_boxes=args.max_boxes)
    env.reset(seed=best["heuristic"][1])
    evaluate_heuristic_from_boxes(tuple(args.bin_size), env.boxes, gif=True, gif_name=f"heuristic_best_{best['heuristic'][1]}.gif")
    # DQN GIF
    evaluate_dqn(agent, tuple(args.bin_size), args.max_boxes, best["dqn"][1], gif=True, gif_name=f"dqn_best_{best['dqn'][1]}.gif")

def main():
    parser = argparse.ArgumentParser(description="Train DQN, run heuristic, compare on identical test sets, and make GIFs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--tests", type=int, default=10, help="number of test seeds to compare")
    parser.add_argument("--bin_size", type=int, nargs=3, default=[10,20,10], metavar=("W","H","D"))
    parser.add_argument("--max_boxes", type=int, default=20)
    parser.add_argument("--train_gif", action="store_true")
    args = parser.parse_args()
    run_comparison(args)

if __name__ == "__main__":
    main()
