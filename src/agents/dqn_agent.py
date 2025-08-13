import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, device='cpu'):
        self.device = device
        self.model = DQN(state_dim, action_dim).to(device)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.memory = deque(maxlen=10000)

        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.update_target_steps = 100
        self.step_count = 0

        self.epsilon_start = 1.0
        self.epsilon_final = 0.05
        self.epsilon_decay_steps = 500
        self.global_step = 0
        self.epsilon = self.epsilon_start

    def get_action(self, state, action_space, mask: np.ndarray = None):
        fraction = min(1.0, self.global_step / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)

        if random.random() < self.epsilon:
            if mask is not None and mask.any():
                valid_idxs = np.flatnonzero(mask)
                action = int(np.random.choice(valid_idxs))
            else:
                action = action_space.sample()
        else:
            state = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.model(state).squeeze(0).cpu().numpy()
            if mask is not None:
                invalid = ~mask
                q_values[invalid] = -1e9
            action = int(q_values.argmax())

        self.global_step += 1
        return action

    def store_transition(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_np      = np.asarray(states, dtype=np.float32)
        next_states_np = np.asarray(next_states, dtype=np.float32)
        actions_np     = np.asarray(actions, dtype=np.int64)
        rewards_np     = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        dones_np       = np.asarray(dones,   dtype=np.float32).reshape(-1, 1)  # use float for math later

        states      = torch.from_numpy(states_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        actions     = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(rewards_np).to(self.device)
        dones       = torch.from_numpy(dones_np).to(self.device)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
