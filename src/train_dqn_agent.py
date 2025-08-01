import numpy as np
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from evals.evaluate_dqn import DQNAgentEvaluator

from utils.visualization import plot_bin
import os

NUM_EPISODES = 100
MAX_BOXES = 20
BIN_SIZE = (10, 10, 10)
TARGET_UPDATE = 10

env = PackingEnv(bin_size=BIN_SIZE, max_boxes=MAX_BOXES)
state = env.reset()
state_dim = len(state)
action_dim = len(env.discrete_actions)
volume_utilizations = []

agent = DQNAgent(state_dim, action_dim)

gif_dir = "gif_frames_v2"
os.makedirs(gif_dir, exist_ok=True)
frame_count = 0

for episode in range(NUM_EPISODES):
    state = env.reset()
    agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state, env.action_space)
        next_state, reward, done, info = env.step(action)

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

    print(f"🎯 Episódio {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}, Volume Used = {pct_volume_used:.2f}%")

import imageio

def create_gif(frame_folder, gif_name="packing.gif", fps=2):
    frames = []
    files = sorted([f for f in os.listdir(frame_folder) if f.endswith(".png")])
    for file_name in files:
        image_path = os.path.join(frame_folder, file_name)
        frames.append(imageio.imread(image_path))
    imageio.mimsave(gif_name, frames, fps=fps)

create_gif(gif_dir, "packing.gif", fps=2)

import shutil
shutil.rmtree(gif_dir)
