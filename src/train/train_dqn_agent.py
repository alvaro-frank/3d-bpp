import numpy as np
import os
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import plot_bin
from utils.visualization import create_gif
import shutil
import matplotlib.pyplot as plt

def train_dqn_agent(
    num_episodes=100,
    bin_size=(10, 10, 10),
    max_boxes=35,
    gif_name="packing_dqn.gif",
    generate_gif=False
):
    """
    Train a DQN agent on the 3D Bin Packing Problem environment.

    Parameters:
    - num_episodes (int): number of training episodes
    - bin_size (tuple): dimensions of the bin (width, height, depth)
    - max_boxes (int): number of boxes per episode
    - gif_name (str): output filename for training GIF (if enabled)
    - generate_gif (bool): whether to save training progress as GIF

    Returns:
    - DQNAgent: the trained agent
    """
    
    # Initialize environment and agent
    env = PackingEnv(bin_size=bin_size, max_boxes=max_boxes, include_noop=False)
    state = env.reset()
    state_dim = env.observation_space.shape[0] # number of features in the state
    action_dim = len(env.discrete_actions) # total number of discrete actions
    agent = DQNAgent(state_dim, action_dim, exploration="softmax") # RL agent

    rewards_per_episode = []
    volume_utilizations = [] # track utilization % per episode

    log_file = "train/train_dqn_log.txt"
    log_positions = "train/train_dqn_positions.txt"

    # Before training starts: clear old log
    with open(log_file, "w") as f:
        f.write("==== DQN Training Log ====\n\n")

    with open(log_positions, "w") as f:
        f.write("==== DQN Training Boxes Positions ====\n\n")

    # Optional: prepare directory for GIF frames
    if generate_gif:
        gif_dir = "gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0
    
    for episode in range(num_episodes):
        with open(log_file, "a") as f:
            f.write(f"==== Episode {episode+1} START ====\n")

        with open(log_positions, "a") as f:
            f.write(f"==== Episode {episode+1} START ====\n")

        state = env.reset()
        
        total_reward = 0
        done = False

        while not done:
            # Compute valid action mask from environment
            mask = env.valid_action_mask()

            # Select action using epsilon-greedy (with mask applied)
            action = agent.get_action(state, env.action_space, mask=mask)
            next_state, reward, done, info = env.step(action, log_file=log_file, pos_file=log_positions)

            if generate_gif:
                frame_path = os.path.join(gif_dir, f"frame_{frame_count:04d}.png")
                plot_bin(env.bin.boxes, env.bin_size, save_path=frame_path,
                         title=f"Episode {episode + 1} Step {frame_count}")
                frame_count += 1

            # Store transition and train agent
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            
            # Move to next state
            state = next_state
            total_reward += reward

        # Episode ended: compute utilization metrics
        rewards_per_episode.append(total_reward)
        volume_used = env.get_placed_boxes_volume()
        bin_volume = env.bin.bin_volume()
        pct_volume_used = (volume_used / bin_volume) * 100
        volume_utilizations.append(pct_volume_used)

        with open(log_file, "a") as f:
          f.write(f"==== Episode {episode+1} END ====\n")
          f.write(f"Total reward: {total_reward:.2f}\n")
          f.write(f"Boxes placed: {len(env.bin.boxes)}/{env.max_boxes}\n")
          f.write(f"Utilization: {pct_volume_used:.2f}%\n\n")

        with open(log_positions, "a") as f:
          f.write(f"==== Episode {episode+1} END ====\n")  

        #print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}, Volume Used = {pct_volume_used:.2f}%, Boxes placed = {len(env.bin.boxes)}/{env.max_boxes}")
        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(rewards_per_episode[-100:])
            mean_util   = np.mean(volume_utilizations[-100:])
            print(f"📊 Episodes {episode-98}–{episode+1}: "
                  f"Mean Reward = {mean_reward:.2f}, "
                  f"Mean Utilization = {mean_util:.2f}%, "
                  f"Epsilon = {agent.epsilon:.2f}")
                  
    if generate_gif:
        create_gif(gif_dir, gif_name)
        shutil.rmtree(gif_dir)

    save_path = os.path.join("runs", "learning_curve.png")

    window = 100
    rewards_smoothed = [np.mean(rewards_per_episode[i:i+window]) 
                        for i in range(0, len(rewards_per_episode), window)]
    utilizations_smoothed = [np.mean(volume_utilizations[i:i+window]) 
                             for i in range(0, len(volume_utilizations), window)]
    episodes_axis = list(range(window, len(rewards_per_episode)+1, window))

    # TO DO: Pass to utils/visualization.py
    # Plot learning curve AFTER all episodes
    plt.figure(figsize=(12,5))

    # Total Reward curve
    plt.subplot(1,2,1)
    plt.plot(episodes_axis, rewards_smoothed, label="Total Reward (avg/100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve (Reward)")
    plt.legend()

    # Utilization curve
    plt.subplot(1,2,2)
    plt.plot(episodes_axis, utilizations_smoothed, label="Volume Utilization % (avg/100)", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Utilization (%)")
    plt.title("Bin Volume Utilization")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return agent
