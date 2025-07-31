import numpy as np
from environment.packing_env import PackingEnv
from agents.dqn_agent import DQNAgent
from evals.evaluate_dqn import DQNAgentEvaluator

NUM_EPISODES = 500
MAX_BOXES = 20
BIN_SIZE = (10, 10, 10)
TARGET_UPDATE = 10

env = PackingEnv(bin_size=BIN_SIZE, max_boxes=MAX_BOXES)
state = env.reset()
state_dim = len(state)
action_dim = len(env.discrete_actions)
volume_utilizations = []

agent = DQNAgent(state_dim, action_dim)

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(state, env.action_space)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        agent.train()
        state = next_state
        total_reward += reward

    volume_used = env.get_placed_boxes_volume()
    bin_volume = env.bin_volume
    pct_volume_used = (volume_used / bin_volume) * 100
    volume_utilizations.append(pct_volume_used)

    #if episode % TARGET_UPDATE == 0:
        #agent.update_target()

    print(f"🎯 Episódio {episode + 1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.2f}")

evaluator = DQNAgentEvaluator(agent, env, episodes=NUM_EPISODES, render=False)
avg_reward = evaluator.evaluate()

avg_volume_utilization = sum(volume_utilizations) / len(volume_utilizations)
print(f"\n💡 Média da porcentagem de volume utilizado durante treino: {avg_volume_utilization:.2f}%")
