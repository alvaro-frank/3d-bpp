import numpy as np
import os
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import plot_bin
from utils.visualization import create_gif
import shutil
import matplotlib.pyplot as plt
import torch

def train_dqn_agent(
    num_episodes=100,
    bin_size=(10, 10, 10),
    max_boxes=35,
    gif_name="packing_dqn.gif",
    generate_gif=False
):
    """
    Train a DQN agent on the 3D Bin Packing Problem environment.

    Args:
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

    save_dir = "runs/dqn/models"
    os.makedirs(save_dir, exist_ok=True)

    best_avg = float("-inf")          # track best running performance
    AVG_WINDOW = 100                  # moving average window
    SAVE_EVERY = 100                  # save every N episodes

    # Optional: prepare directory for GIF frames
    if generate_gif:
        gif_dir = "gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        
        total_reward = 0
        done = False

        while not done:
            # Compute valid action mask from environment
            mask = env.valid_action_mask()

            # Select action using epsilon-greedy (with mask applied)
            action = agent.get_action(state, env.action_space, mask=mask)
            next_state, reward, done, info = env.step(action)

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
        
        start = max(0, len(rewards_per_episode) - AVG_WINDOW)
        avg_reward = sum(rewards_per_episode[start:]) / (len(rewards_per_episode) - start)
        
        print(f"Episode {episode + 1}: Total Reward = {total_reward:.2f}")

        # "best so far" checkpoint by average reward
        if avg_reward > best_avg:
            best_avg = avg_reward
            torch.save(
                agent.model.state_dict(),
                os.path.join(save_dir, f"dqn_best.pt")
            ) 

        #print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}, Volume Used = {pct_volume_used:.2f}%, Boxes placed = {len(env.bin.boxes)}/{env.max_boxes}")
        if (episode + 1) % 100 == 0:
            mean_reward = np.mean(rewards_per_episode[-100:])
            mean_util   = np.mean(volume_utilizations[-100:])
            print(f"Episodes {episode-98}–{episode+1}: "
                  f"Mean Reward = {mean_reward:.2f}, "
                  f"Mean Utilization = {mean_util:.2f}%, "
                  f"Epsilon = {agent.epsilon:.2f}")
            
    torch.save(
        agent.model.state_dict(),
        os.path.join(save_dir, "dqn_final.pt")
    )
                  
    if generate_gif:
        create_gif(gif_dir, gif_name)
        shutil.rmtree(gif_dir)

    save_path = os.path.join("runs/dqn", "learning_curve.png")

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
