from environment.packing_env import PackingEnv

def evaluate_agent_on_episode(agent, episode_boxes, env_seed=None, generate_gif=False, gif_name="packing_dqn_agent.gif"):
    """
    Evaluate the trained DQN agent on ONE test episode.

    - Forces epsilon = 0 (pure exploitation) for fair evaluation.
    - Uses action masking to avoid invalid placements.
    - Optionally generates a GIF of the packing process.

    Parameters:
    - agent (DQNAgent): trained agent
    - episode_boxes (list[dict]): list of boxes with dimensions (w, h, d)
    - env_seed (int, optional): seed for reproducibility
    - generate_gif (bool): whether to record GIF
    - gif_name (str): filename for output GIF

    Returns:
    - float: percentage of bin volume used
    """
    env = PackingEnv(
        bin_size=(10, 10, 10),
        max_boxes=len(episode_boxes),
        generate_gif=generate_gif,
        gif_name=gif_name,
        include_noop=True
    )

    # Reset env with predetermined boxes
    state = env.reset(seed=env_seed, with_boxes=episode_boxes)

    # Backup epsilon and force greedy policy
    epsilon_backup = getattr(agent, "epsilon", None)
    agent.epsilon = 0.0

    temperature_backup = getattr(agent, "temperature", None)
    agent.temperature = 0.0

    total_reward = 0.0
    done = False
    while not done:
        # Select action using mask
        mask = env.valid_action_mask()
        action = agent.get_action(state, env.action_space, mask=mask)
        state, reward, done, _ = env.step(action)
        total_reward += reward

    # Restore epsilon after eval
    if epsilon_backup is not None:
        agent.epsilon = epsilon_backup

    if temperature_backup is not None:
        agent.temperature = temperature_backup

    # Compute utilization percentage
    volume_used = env.get_placed_boxes_volume()
    pct_volume_used = (volume_used / env.bin.bin_volume()) * 100.0
    return pct_volume_used
