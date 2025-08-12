import argparse, os, json
from typing import List, Tuple
import numpy as np

from utils.seed import set_global_seed
from utils.box_gen import generate_dataset, BoxSpec
from baselines.heuristics import lbf_blb

from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from train.train_dqn_agent import train_dqn_agent

def eval_dqn(agent, test_sets: List[List[BoxSpec]], bin_size: Tuple[int,int,int], max_boxes: int) -> dict:
    scores = []
    for ep, specs in enumerate(test_sets):
        env = PackingEnv(bin_size=bin_size, max_boxes=max_boxes, generate_gif=False)
        try:
            _ = env.reset_with_boxes([(b.width,b.height,b.depth) for b in specs])
        except AttributeError:
            _ = env.reset(seed=ep)
        done = False
        while not done:
            action = agent.get_action(_, env.action_space)
            _, r, done, _info = env.step(action)
        util = env.get_placed_boxes_volume() / (bin_size[0]*bin_size[1]*bin_size[2])
        scores.append(util)
    return {"mean_util": float(np.mean(scores)), "per_ep": scores}

def eval_heuristic(test_sets: List[List[BoxSpec]], bin_size: Tuple[int,int,int]) -> dict:
    scores = []
    for specs in test_sets:
        boxes = [(b.width,b.height,b.depth,b.id) for b in specs]
        res = lbf_blb(bin_size[0], bin_size[1], bin_size[2], boxes, support_thresh=0.7)
        scores.append(res["utilization"])
    return {"mean_util": float(np.mean(scores)), "per_ep": scores}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bin', type=int, nargs=3, default=[10,20,10], metavar=('W','H','D'))
    p.add_argument('--boxes_per_episode', type=int, default=20)
    p.add_argument('--train_episodes', type=int, default=200)
    p.add_argument('--test_episodes', type=int, default=50)
    p.add_argument('--save_dir', type=str, default='runs/exp1')
    args = p.parse_args()

    set_global_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    train_sets = generate_dataset(args.train_episodes, args.boxes_per_episode, seed=args.seed)
    test_sets  = generate_dataset(args.test_episodes,  args.boxes_per_episode, seed=args.seed+1)

    agent = train_dqn_agent(
        num_episodes=args.train_episodes,
        bin_size=tuple(args.bin),
        max_boxes=args.boxes_per_episode,
        gif_name="packing_dqn_train.gif",
        generate_gif=False
    )

    dqn_res = eval_dqn(agent, test_sets, tuple(args.bin), args.boxes_per_episode)
    heur_res = eval_heuristic(test_sets, tuple(args.bin))

    out = {
        "seed": args.seed,
        "bin": args.bin,
        "boxes_per_episode": args.boxes_per_episode,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "dqn": dqn_res,
        "heuristic_lbf_blb": heur_res,
    }
    with open(os.path.join(args.save_dir, "compare.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== RESULTS ===")
    print(f"DQN mean utilization: {dqn_res['mean_util']:.3f}")
    print(f"Heuristic mean utilization: {heur_res['mean_util']:.3f}")
    print(f"Saved to {os.path.join(args.save_dir, 'compare.json')}")

if __name__ == "__main__":
    main()
