from train.train_dqn_agent import train_dqn_agent
from environment.packing_env import PackingEnv
import numpy as np
from utils.heuristic import heuristic_blb_packing

def evaluate_agent(agent, env_seed=42, generate_gif=True):
    env = PackingEnv(bin_size=(10, 20, 10), max_boxes=20, generate_gif=generate_gif)
    state = env.reset(seed=env_seed)
    total_reward = 0
    done = False
    while not done:
        action = agent.get_action(state, env.action_space)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    volume_used = env.get_placed_boxes_volume()
    bin_volume = env.bin_volume
    pct_volume_used = (volume_used / bin_volume) * 100

    return pct_volume_used

def evaluate_heuristic(seed=42, generate_gif=False):
    # Create environment with fixed seed to get consistent boxes
    env = PackingEnv(bin_size=(10, 20, 10), max_boxes=20)
    env.reset(seed=seed)
    
    # Run your heuristic packing algorithm
    placed_boxes, bin = heuristic_blb_packing(bin_size=env.bin_size, boxes=env.boxes, try_rotations=True, generate_gif=generate_gif, gif_name=f"packing_heuristic.gif")
    
    # Calculate reward or volume used — depending on your environment's logic
    volume_used = sum(box.get_volume() for box in placed_boxes)
    bin_volume = env.bin_volume  # total volume of the bin
    
    # Compute % volume used as a proxy for reward
    pct_volume_used = (volume_used / bin_volume) * 100
    
    # If you want, convert this to reward similar to env.reward()
    # Here, just return pct_volume_used as heuristic score
    return pct_volume_used

def main():
    print("📦 Training DQN Agent...")
    agent = train_dqn_agent(num_episodes=10, generate_gif=False)

    print("\n🤖 Evaluating DQN vs Heuristic:")
    dqn_scores = []
    heuristic_scores = []

    best_dqn_score = -1
    best_dqn_seed = None
    best_heuristic_score = -1
    best_heuristic_seed = None

    for i in range(10):
        test_seed = 1234 + i
        dqn = evaluate_agent(agent, test_seed, generate_gif=False)
        heuristic = evaluate_heuristic(test_seed, generate_gif=False)
        dqn_scores.append(dqn)
        heuristic_scores.append(heuristic)

        if dqn > best_dqn_score:
            best_dqn_score = dqn
            best_dqn_seed = test_seed
        if heuristic > best_heuristic_score:
            best_heuristic_score = heuristic
            best_heuristic_seed = test_seed

        print(f"Test {i+1}: DQN = {dqn:.2f}%, Heuristic = {heuristic:.2f}%")

    print("\n📊 DQN Avg:", np.mean(dqn_scores))
    print("📊 Heuristic Avg:", np.mean(heuristic_scores))

    print(f"\n🎞️ Generating GIF for best heuristic test (Seed {best_heuristic_seed}) with {best_heuristic_score:.2f}% volume used...")
    evaluate_heuristic(seed=best_heuristic_seed, generate_gif=True)

    print(f"\n🎞️ Generating GIF for best DQN test (Seed {best_dqn_seed}) with {best_dqn_score:.2f}% volume used...")
    evaluate_agent(agent, env_seed=best_dqn_seed, generate_gif=True)

if __name__ == "__main__":
    main()
