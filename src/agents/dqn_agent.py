import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    
    A simple feedforward neural network that maps input states
    to Q-values for each possible action.
    """
    def __init__(self, input_dim, output_dim):
        """
        Parameters:
        - input_dim (int): number of features in the input state
        - output_dim (int): number of discrete actions available
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
        """Forward pass through the network to compute Q-values."""
        return self.net(x)


class DQNAgent:
    """
    Deep Q-Learning Agent.

    Uses an online Q-network to estimate action values and a target network
    for stable training. Supports epsilon-greedy exploration with a 
    step-based decay schedule.
    """
    def __init__(self, state_dim, action_dim, device='cpu', exploration="epsilon"):
        """
        Parameters:
        - state_dim (int): dimension of the environment's state space
        - action_dim (int): number of discrete actions
        - device (str): computation device ('cpu' or 'cuda')
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
        
        # Replay buffer to store transitions
        self.memory = deque(maxlen=50000)

        # Hyperparameters
        self.batch_size = 64
        self.gamma = 0.99 # discount factor
        self.update_target_steps = 100 # target network update frequency
        self.step_count = 0

        # Epsilon-greedy schedule
        self.exploration = exploration
        self.epsilon_start = 1.0
        self.epsilon_final = 0.10
        self.epsilon_decay_steps = 40000 # how many steps until epsilon reaches final value
        self.global_step = 0 # total environment steps taken
        self.epsilon = self.epsilon_start # current epsilon

        self.temperature_start = 1.0
        self.temperature_final = 0.1
        self.temperature_decay_steps = 20000
        self.temperature = self.temperature_start

    def get_action(self, state, action_space, mask: np.ndarray = None):
        """
        Select an action using epsilon-greedy strategy.

        With probability epsilon, choose a random valid action (exploration).
        Otherwise, choose the action with the highest predicted Q-value (exploitation).

        Parameters:
        - state (np.ndarray): current environment state
        - action_space (gym.spaces.Discrete): the action space
        - mask (np.ndarray, optional): boolean mask of valid actions

        Returns:
        - int: chosen action index
        """
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)
        
        frac_T = min(1.0, self.global_step / self.temperature_decay_steps)
        self.temperature = self.temperature_start + frac_T * (self.temperature_final - self.temperature_start)

        # Update epsilon linearly with steps
        fraction = min(1.0, self.global_step / self.epsilon_decay_steps) 
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)

        self.global_step += 1

         # --- Exploração epsilon-greedy ---
        if self.exploration == "epsilon":
            if random.random() < self.epsilon:
                if mask is not None and mask.any():
                    valid_idxs = np.flatnonzero(mask)
                    return int(np.random.choice(valid_idxs))
                else:
                    return action_space.sample()

            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
            if mask is not None:
                q_values[~mask] = -1e9
            return int(q_values.argmax())

        # --- Exploração Softmax ---
        elif self.exploration == "softmax":
            with torch.no_grad():
                q_values = self.model(state_tensor).squeeze(0).cpu().numpy()
            if mask is not None:
                q_values[~mask] = -1e9

            # Subtração para estabilidade
            q_shifted = q_values - np.max(q_values)
            exp_q = np.exp(q_shifted / self.temperature)

            if exp_q.sum() == 0 or np.isnan(exp_q.sum()):
                # fallback: escolher ação válida aleatória
                if mask is not None and mask.any():
                    return int(np.random.choice(np.flatnonzero(mask)))
                else:
                    return action_space.sample()

            probs = exp_q / exp_q.sum()
            return int(np.random.choice(len(q_values), p=probs))

    def store_transition(self, state, action_idx, reward, next_state, done):
        """
        Store a transition in the replay buffer.

        Parameters:
        - state (np.ndarray): current state
        - action_idx (int): chosen action index
        - reward (float): received reward
        - next_state (np.ndarray): state after taking the action
        - done (bool): whether the episode ended
        """
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        """
        Train the Q-network using a minibatch from the replay buffer.

        Process:
        1. Sample random transitions.
        2. Compute Q-values for current states.
        3. Compute target Q-values using the target network.
        4. Minimize loss between predicted and target Q-values.
        5. Update the target network every `update_target_steps`.
        """
        if len(self.memory) < self.batch_size:
            return # not enough samples to train yet

        # Sample a random minibatch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy
        states_np      = np.asarray(states, dtype=np.float32)
        next_states_np = np.asarray(next_states, dtype=np.float32)
        actions_np     = np.asarray(actions, dtype=np.int64)
        rewards_np     = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        dones_np       = np.asarray(dones,   dtype=np.float32).reshape(-1, 1)  # use float for math later

        # Convert to PyTorch tensors
        states      = torch.from_numpy(states_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        actions     = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(rewards_np).to(self.device)
        dones       = torch.from_numpy(dones_np).to(self.device)

        # Compute Q(s,a) using the online network
        q_values = self.model(states).gather(1, actions)
        
        # Compute target values using the target network
        next_q_values = self.target_model(next_states).max(1, keepdim=True)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q_values

        # Loss between predicted and target Q-values
        loss = self.criterion(q_values, target_q)

        # Gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network every fixed number of steps
        self.step_count += 1
        if self.step_count % self.update_target_steps == 0:
            self.target_model.load_state_dict(self.model.state_dict())
