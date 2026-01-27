# ==============================================================================
# FILE: train/train_dqn_agent.py
# DESCRIPTION: Training loop for the DQN agent in the 3D Bin Packing environment.
#              Handles episode generation, epsilon decay, metrics logging,
#              and model checkpointing.
# ==============================================================================
import numpy as np
import os
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import plot_bin
from utils.visualization import create_gif
import shutil
import matplotlib.pyplot as plt
import torch
import mlflow

def train_dqn_agent(
    num_episodes=100,
    bin_size=(10, 10, 10),
    max_boxes=35,
    gif_name="packing_dqn.gif",
    generate_gif=False,
    load_path=None
):
    """
    Train a DQN agent on the 3D Bin Packing Problem environment.
    
    Includes automatic epsilon decay scheduling based on total estimated steps
    and periodic logging of training metrics.

    Args:
        num_episodes (int): Number of training episodes.
        bin_size (tuple): Dimensions of the bin (width, height, depth).
        max_boxes (int): Maximum number of boxes per episode.
        gif_name (str): Output filename for training GIF (if enabled).
        generate_gif (bool): Whether to save training progress as GIF.
        load_path (str, optional): Path to a checkpoint file to resume training.

    Returns:
        DQNAgent: The trained agent instance.
    """
    
    # Initialize environment
    env = PackingEnv(bin_size=bin_size, max_boxes=max_boxes, include_noop=False)
    state = env.reset()
    state_dim = env.observation_space.shape[0] # number of features in the state
    action_dim = len(env.discrete_actions) # total number of discrete actions
    total_steps_estimate = num_episodes * max_boxes
    
    # Initialize Agent
    agent = DQNAgent(state_dim, action_dim, exploration="softmax", total_training_steps=total_steps_estimate) # RL agent
    
    # Load weights if provided
    if load_path:
        agent.load(load_path)

    rewards_per_episode = []
    volume_utilizations = [] # track utilization % per episode

    save_dir = "runs/dqn/models"
    os.makedirs(save_dir, exist_ok=True)

    best_avg = float("-inf")          # track best running performance
    AVG_WINDOW = 100                  # moving average window
    SAVE_EVERY = 100                  # save every N episodes
    
    params = {
        "agent_type": "dqn",
        "num_episodes": num_episodes,
        "bin_width": bin_size[0],
        "bin_depth": bin_size[1],
        "bin_height": bin_size[2],
        "max_boxes": max_boxes,
        "lr": 1e-3,           
        "gamma": 0.99,        
        "batch_size": 64     
    }
    mlflow.log_params(params)

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
        boxes_placed = len(env.packed_boxes)
        skipped_boxes = len(env.skipped_boxes)
        
        start = max(0, len(rewards_per_episode) - AVG_WINDOW)
        avg_reward = sum(rewards_per_episode[start:]) / (len(rewards_per_episode) - start)
        
        mlflow.log_metric("volume_utilization", pct_volume_used, step=episode)
        mlflow.log_metric("epsilon", agent.epsilon, step=episode)
        
        mlflow.log_metric("avg_reward_100", avg_reward, step=episode)
        
        mlflow.log_metric("boxes_placed", boxes_placed, step=episode)
        mlflow.log_metric("boxes_skipped", skipped_boxes, step=episode)

        # "best so far" checkpoint by average reward
        if avg_reward > best_avg:
            best_avg = avg_reward
            torch.save(
                agent.model.state_dict(),
                os.path.join(save_dir, f"dqn_best.pt")
            )
        
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(rewards_per_episode[-10:])
            avg_boxes_10 = np.mean([len(env.packed_boxes) for _ in range(10)])
            avg_util_10 = np.mean(volume_utilizations[-10:])
            
            print(f"Ep {episode + 1:5d} | R(avg10): {avg_reward_10:7.2f} | "
                  f"Boxes(avg10): {avg_boxes_10:5.1f}/{max_boxes} | "
                  f"Util(avg10): {avg_util_10:5.1f}% | Epsilon: {agent.epsilon:.2f}")

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

    # Plot Learning Curve
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
