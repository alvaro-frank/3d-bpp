class DQNAgentEvaluator:
    def __init__(self, agent, env, episodes=10, render=False):
        self.agent = agent
        self.env = env
        self.episodes = episodes
        self.render = render

    def evaluate(self):
        # Desativa exploração para avaliação
        self.agent.epsilon = 0.0

        total_rewards = []

        for ep in range(self.episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.agent.get_action(state, self.env.action_space)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_reward += reward

                if self.render:
                    self.env.render()

            total_rewards.append(episode_reward)
            print(f"🎯 Episódio {ep + 1}: Recompensa Total = {episode_reward:.2f}")

        avg_reward = sum(total_rewards) / self.episodes
        print(f"\n💡 Recompensa média após {self.episodes} episódios: {avg_reward:.2f}")

        return avg_reward
