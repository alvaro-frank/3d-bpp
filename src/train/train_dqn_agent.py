import numpy as np
import os
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from utils.visualization import plot_bin
import imageio
import shutil

def train_dqn_agent(
    num_episodes=100,
    bin_size=(10, 20, 10),
    max_boxes=20,
    gif_name="packing_dqn.gif",
    generate_gif=False
):
    env = PackingEnv(bin_size=bin_size, max_boxes=max_boxes)
    state = env.reset()
    state_dim = len(state)
    action_dim = len(env.discrete_actions)
    agent = DQNAgent(state_dim, action_dim)
    volume_utilizations = []

    if generate_gif:
        gif_dir = "gif_frames_v2"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0

    for episode in range(num_episodes):
        state = env.reset()
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, env.action_space)
            next_state, reward, done, info = env.step(action)

            if generate_gif:
                frame_path = os.path.join(gif_dir, f"frame_{frame_count:04d}.png")
                plot_bin(env.bin.boxes, env.bin_size, save_path=frame_path, title=f"Episode {episode + 1} Step {frame_count}")
                frame_count += 1

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            state = next_state
            total_reward += reward

        volume_used = env.get_placed_boxes_volume()
        bin_volume = env.bin_volume
        pct_volume_used = (volume_used / bin_volume) * 100
        volume_utilizations.append(pct_volume_used)

        print(f"🎯 Episode {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}, Volume Used = {pct_volume_used:.2f}%")

    if generate_gif:
        create_gif(gif_dir, gif_name)
        shutil.rmtree(gif_dir)

    return agent

def create_gif(frame_folder, gif_name="packing_dqn_train.gif", fps=2):
    frames = []
    files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for file_name in files:
        image_path = os.path.join(frame_folder, file_name)
        frames.append(imageio.imread(image_path))
    imageio.mimsave(gif_name, frames, fps=fps)
