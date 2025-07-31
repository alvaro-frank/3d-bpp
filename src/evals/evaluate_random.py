from environment.packing_env import PackingEnv
from agents.random_agent import RandomAgent

NUM_EPISODES = 100
BOXES_PER_EPISODE = 10
BIN_SIZE = (10, 10, 10)

total_placed = 0
total_attempted = 0

env = PackingEnv(bin_size=BIN_SIZE, max_boxes=BOXES_PER_EPISODE)
agent = RandomAgent(*BIN_SIZE)

for episode in range(NUM_EPISODES):
    obs = env.reset()
    done = False

    while not done:
        box = env.current_box
        action = agent.get_action(box)
        obs, reward, done, info = env.step(action)

        if info["success"]:
            total_placed += 1
        total_attempted += 1

success_rate = total_placed / total_attempted
print(f"\n📊 Success Rate: {success_rate * 100:.2f}%")
print(f"Caixas colocadas com sucesso: {total_placed}/{total_attempted}")