import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network (DQN).

    A feedforward neural network that maps input states to Q-values,
    where each Q-value corresponds to the expected return of taking an action.
    """
    def __init__(self, input_dim, output_dim):
        """
        Initialize the network.

        Args:
            input_dim (int): Number of input features (state dimension).
            output_dim (int): Number of discrete actions.
        """
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), # first hidden layer
            nn.ReLU(),
            nn.Linear(128, 128), # second hidden layer
            nn.ReLU(),
            nn.Linear(128, output_dim) # output: Q-value for each action
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted Q-values of shape (batch_size, output_dim).
        """
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Learning Agent.

    Manages experience replay, epsilon/softmax exploration, and
    training of an online Q-network with a target network for stability.
    """
    def __init__(self, state_dim, action_dim, device='cpu', exploration="epsilon"):
        """
        Initialize the agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Number of possible discrete actions.
            device (str): Computation device ('cpu' or 'cuda').
            exploration (str): Exploration strategy ('epsilon' or 'softmax').
        """
        self.device = device
        # Online network (learned Q-function)
        self.model = DQN(state_dim, action_dim).to(device)
        
        # Target network (updated less frequently to stabilize learning)
        self.target_model = DQN(state_dim, action_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

         # Optimizer and loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = deque(maxlen=50000)

        # Training settings
        self.batch_size = 64
        self.gamma = 0.99 # discount factor
        self.update_target_steps = 100 # target network update frequency
        self.step_count = 0

        # Exploration parameters
        self.exploration = exploration
        self.epsilon_start = 1.0
        self.epsilon_final = 0.10
        self.epsilon_decay_steps = 40000 # how many steps until epsilon reaches final value
        self.global_step = 0 # total environment steps taken
        self.epsilon = self.epsilon_start # current epsilon

        self.temperature_start = 1.0
        self.temperature_final = 0.1
        self.temperature_decay_steps = 100000
        self.temperature = self.temperature_start

    def get_action(self, state, action_space, mask: np.ndarray = None):
        """
        Select an action using epsilon-greedy or softmax exploration.

        Args:
            state (np.ndarray): Current state.
            action_space (gym.spaces.Discrete): Action space object.
            mask (np.ndarray, optional): Boolean mask of valid actions.

        Returns:
            int: Selected action index.
        """
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)

        # Atualização de temperatura (softmax) e epsilon (epsilon-greedy)
        frac_T = min(1.0, self.global_step / self.temperature_decay_steps)
        self.temperature = self.temperature_start + frac_T * (self.temperature_final - self.temperature_start)

        fraction = min(1.0, self.global_step / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)

        self.global_step += 1

        # --- Normalização da máscara ---
        valid_idxs = None
        if mask is not None:
            m = np.asarray(mask)
            # converter para bool robustamente
            if m.dtype != np.bool_:
                m = m > 0.5
            if m.ndim != 1:
                raise ValueError(f"Mask deve ser 1D; recebido shape={m.shape}")
            valid_idxs = np.flatnonzero(m)
            if valid_idxs.size == 0:
                # fallback defensivo: sem válidas → ignora máscara
                m = None
                valid_idxs = None
            mask_bool = m
        else:
            mask_bool = None

        # --- Epsilon-greedy ---
        if self.exploration == "epsilon":
            if random.random() < self.epsilon:
                if valid_idxs is not None:
                    return int(np.random.choice(valid_idxs))
                else:
                    return int(action_space.sample())

            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
            if mask_bool is not None:
                q_values[~mask_bool] = -1e9
            return int(q_values.argmax())

        # --- Softmax ---
        elif self.exploration == "softmax":
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
            if mask_bool is not None:
                q_values[~mask_bool] = -1e9

            q_shifted = q_values - np.max(q_values)
            temp = max(1e-6, float(self.temperature))
            exp_q = np.exp(q_shifted / temp)

            s = exp_q.sum()
            if not np.isfinite(s) or s <= 0:
                if valid_idxs is not None:
                    return int(np.random.choice(valid_idxs))
                else:
                    return int(action_space.sample())

            probs = exp_q / s
            if mask_bool is not None:
                probs = probs * mask_bool.astype(np.float32)
                ps = probs.sum()
                if ps <= 0 or not np.isfinite(ps):
                    return int(np.random.choice(valid_idxs))
                probs = probs / ps

            return int(np.random.choice(len(q_values), p=probs))

    def store_transition(self, state, action_idx, reward, next_state, done):
        """
        Save a transition into the replay buffer.

        Args:
            state (np.ndarray): Current state.
            action_idx (int): Chosen action index.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after taking the action.
            done (bool): Whether the episode terminated.
        """
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        """
        Perform one training step on a minibatch from the replay buffer.

        Steps:
            1. Sample random transitions.
            2. Compute current Q-values with the online network.
            3. Compute target Q-values with the target network.
            4. Minimize MSE loss between current and target Q-values.
            5. Periodically update the target network.
        """
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
