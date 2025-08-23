import numpy as np
import os
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import plot_bin
from utils.visualization import create_gif
import shutil

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
    env = PackingEnv(bin_size=bin_size, max_boxes=max_boxes)
    state = env.reset()
    state_dim = len(state) # number of features in the state
    action_dim = len(env.discrete_actions) # total number of discrete actions
    agent = DQNAgent(state_dim, action_dim) # RL agent
    volume_utilizations = [] # track utilization % per episode

    # Optional: prepare directory for GIF frames
    if generate_gif:
        gif_dir = "gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0

    for episode in range(num_episodes):
        state = env.reset()
        
        # Legacy epsilon decay line (if step-based schedule is disabled)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
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
        volume_used = env.get_placed_boxes_volume()
        bin_volume = env.bin.bin_volume()
        pct_volume_used = (volume_used / bin_volume) * 100
        volume_utilizations.append(pct_volume_used)

        print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}, Volume Used = {pct_volume_used:.2f}%")

    if generate_gif:
        create_gif(gif_dir, gif_name)
        shutil.rmtree(gif_dir)

    return agent
