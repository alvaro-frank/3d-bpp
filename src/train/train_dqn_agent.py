import numpy as np
import os
from utils.visualization import plot_bin, create_gif
import shutil
import matplotlib.pyplot as plt
import torch
import mlflow

def train_dqn_agent(
    env,
    agent,
    num_episodes=100,
    gif_name="packing_dqn.gif",
    generate_gif=False
):
    """
    Train a DQN agent on the 3D Bin Packing Problem environment.
    """
    rewards_per_episode = []
    volume_utilizations = [] 

    save_dir = "runs/dqn/models"
    os.makedirs(save_dir, exist_ok=True)

    best_avg = float("-inf")          
    AVG_WINDOW = 100                  
    
    if generate_gif:
        gif_dir = "gif_frames"
        os.makedirs(gif_dir, exist_ok=True)
        frame_count = 0
    
    max_boxes = env.max_boxes

    for episode in range(num_episodes):
        state = env.reset()
        
        total_reward = 0
        done = False

        while not done:
            mask = env.valid_action_mask()
            action = agent.get_action(state, env.action_space, mask=mask)
            
            step_out = env.step(action)
            # Compatibilidade com Gym mais recentes
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                next_state, reward, done, info = step_out

            if generate_gif:
                frame_path = os.path.join(gif_dir, f"frame_{frame_count:04d}.png")
                plot_bin(env.bin.boxes, env.bin_size, save_path=frame_path,
                         title=f"Episode {episode + 1} Step {frame_count}")
                frame_count += 1

            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward

        # Episode ended: compute metrics
        rewards_per_episode.append(total_reward)
        volume_used = env.get_placed_boxes_volume()
        bin_volume = env.bin.bin_volume()
        pct_volume_used = (volume_used / bin_volume) * 100 if bin_volume > 0 else 0
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

        if avg_reward > best_avg:
            best_avg = avg_reward
            torch.save(agent.model.state_dict(), os.path.join(save_dir, "dqn_best.pt"))
        
        if (episode + 1) % 10 == 0:
            avg_reward_10 = np.mean(rewards_per_episode[-10:])
            avg_boxes_10 = np.mean([len(env.packed_boxes) for _ in range(10)])
            avg_util_10 = np.mean(volume_utilizations[-10:])
            print(f"Ep {episode + 1:5d} | R(avg10): {avg_reward_10:7.2f} | "
                  f"Boxes(avg10): {avg_boxes_10:5.1f}/{max_boxes} | "
                  f"Util(avg10): {avg_util_10:5.1f}% | Epsilon: {agent.epsilon:.2f}")

    torch.save(agent.model.state_dict(), os.path.join(save_dir, "dqn_final.pt"))
                  
    if generate_gif:
        create_gif(gif_dir, gif_name)
        shutil.rmtree(gif_dir)

    # Plotting à prova de erro para testes pequenos
    save_path = os.path.join("runs/dqn", "learning_curve.png")
    os.makedirs("runs/dqn", exist_ok=True)

    window = min(100, max(1, len(rewards_per_episode))) 
    rewards_smoothed = [np.mean(rewards_per_episode[i:i+window]) for i in range(0, len(rewards_per_episode), window)]
    utilizations_smoothed = [np.mean(volume_utilizations[i:i+window]) for i in range(0, len(volume_utilizations), window)]
    episodes_axis = [min(i + window, len(rewards_per_episode)) for i in range(0, len(rewards_per_episode), window)]

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(episodes_axis, rewards_smoothed, label=f"Total Reward (avg/{window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curve (Reward)")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(episodes_axis, utilizations_smoothed, label=f"Volume Utilization % (avg/{window})", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Utilization (%)")
    plt.title("Bin Volume Utilization")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    return agent