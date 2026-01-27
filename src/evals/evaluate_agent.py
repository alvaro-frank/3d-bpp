# ==============================================================================
# FILE: evals/evaluate_agent.py
# DESCRIPTION: Utility functions for evaluating agent performance.
#              Provides tools for greedy episode rollout and metrics collection.
# ==============================================================================

from environment.packing_env import PackingEnv

# ------------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ------------------------------------------------------------------------------

def evaluate_agent_on_episode(agent, episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_dqn_agent.gif"):
    """
    Evaluate a trained agent on ONE test episode.

    - Forces epsilon/temperature to 0 (pure exploitation).
    - Uses action masking to ensure valid placements.
    - Optionally generates a GIF of the packing process.

    Args:
        agent (Agent): The trained model (DQN/PPO).
        episode_boxes (list[dict]): Predetermined boxes for the test case.
        env_seed (int, optional): Seed for environment reproducibility.
        generate_gif (bool): Whether to record and save a GIF.
        gif_name (str): Filename for the output GIF.

    Returns:
        float: Percentage of bin volume utilized (0-100).
    """
    env = PackingEnv(
        bin_size=(10, 10, 10),
        max_boxes=len(episode_boxes),
        generate_gif=generate_gif,
        gif_name=gif_name,
        include_noop=False
    )

    # Reset environment with predetermined boxes
    state = env.reset(seed=env_seed, with_boxes=episode_boxes)

    # Backup exploration parameters and force greedy policy
    epsilon_backup = getattr(agent, "epsilon", None)
    agent.epsilon = 0.0

    temperature_backup = getattr(agent, "temperature", None)
    agent.temperature = 0.0

    total_reward = 0.0
    done = False
    
    # Run evaluation loop
    while not done:
        # Select action using valid action mask
        mask = env.valid_action_mask()
        action = agent.get_action(state, env.action_space, mask=mask)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    # Restore exploration parameters after evaluation
    if epsilon_backup is not None:
        agent.epsilon = epsilon_backup

    if temperature_backup is not None:
        agent.temperature = temperature_backup

    # Compute utilization metrics
    volume_used = env.get_placed_boxes_volume()
    pct_volume_used = (volume_used / env.bin.bin_volume()) * 100.0
    
    return pct_volume_used