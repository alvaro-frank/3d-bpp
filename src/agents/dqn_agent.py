# ==============================================================================
# FILE: agents/dqn_agent.py
# DESCRIPTION: Implementation of the Deep Q-Network (DQN) agent.
#              Includes the CNN-based Q-Network architecture and the Agent class
#              managing experience replay, exploration, and training.
# ==============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture with CNN layers for spatial reasoning.

    Maps the 3D bin state (heightmap) + extra features to Q-values for each action.
    """
    def __init__(self, input_dim, output_dim, map_size=(10,10)):
        """
        Initialize the neural network layers.

        Args:
            input_dim (int): Total size of the flattened input vector.
            output_dim (int): Number of possible actions (discrete).
            map_size (tuple): Dimensions of the bin base (width, depth) for the CNN.
        """
        super(DQN, self).__init__()
        
        self.map_w, self.map_d = map_size
        self.map_area = self.map_w * self.map_d
        
        self.extra_features_len = input_dim - self.map_area
        
        # Spatial feature extractor (Heightmap processing)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        cnn_out_dim = 32 * self.map_w * self.map_d
        
        # Decision head (Combining spatial and scalar features)
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + self.extra_features_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input state tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted Q-values of shape (batch_size, output_dim).
        """
        heightmap_flat = x[:, :self.map_area]
        extra_features = x[:, self.map_area:]
        
        heightmap_img = heightmap_flat.view(-1, 1, self.map_w, self.map_d)
        
        conv_out = self.conv(heightmap_img)
        
        combined = torch.cat([conv_out, extra_features], dim=1)
      
        return self.fc(combined)

class DQNAgent:
    """
    Deep Q-Learning Agent handling interaction and training.

    Manages the Online and Target networks, Experience Replay buffer,
    and Epsilon-Greedy / Softmax exploration strategies.
    """
    def __init__(self, state_dim, action_dim, map_size=(10,10), device='cpu', exploration="epsilon", total_training_steps=100000):
        """
        Initialize the DQN Agent.

        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Number of possible discrete actions.
            map_size (tuple): Bin dimensions for the CNN.
            device (str): Computation device ('cpu' or 'cuda').
            exploration (str): Strategy ('epsilon' or 'softmax').
            epsilon_decay_steps (int): Number of steps to decay epsilon from start to final.
        """
        self.device = device
        self.map_size = map_size
        
        # Online network (updates every step)
        self.model = DQN(state_dim, action_dim, map_size=map_size).to(device)
        
        # Target network (updates periodically for stability)
        self.target_model = DQN(state_dim, action_dim, map_size=map_size).to(device)
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
        self.epsilon_decay_steps = int(total_training_steps * 0.6) # how many steps until epsilon reaches final value
        self.global_step = 0 # total environment steps taken
        self.epsilon = self.epsilon_start # current epsilon

        self.temperature_start = 1.0
        self.temperature_final = 0.1
        self.temperature_decay_steps = 100000
        self.temperature = self.temperature_start

    def get_action(self, state, action_space, mask: np.ndarray = None):
        """
        Select an action based on the current policy and exploration strategy.

        Args:
            state (np.ndarray): Current observation from the environment.
            action_space (gym.spaces.Discrete): The environment's action space.
            mask (np.ndarray, optional): Binary mask of valid actions (1=valid, 0=invalid).

        Returns:
            int: The index of the selected action.
        """
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).unsqueeze(0).to(self.device)

        # Update exploration parameters (Temperature / Epsilon)
        frac_T = min(1.0, self.global_step / self.temperature_decay_steps)
        self.temperature = self.temperature_start + frac_T * (self.temperature_final - self.temperature_start)

        fraction = min(1.0, self.global_step / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + fraction * (self.epsilon_final - self.epsilon_start)

        self.global_step += 1

        # Handle Action Masking
        valid_idxs = None
        if mask is not None:
            m = np.asarray(mask)
            if m.dtype != np.bool_:
                m = m > 0.5
            if m.ndim != 1:
                raise ValueError(f"Mask should be 1D; received shape={m.shape}")
            valid_idxs = np.flatnonzero(m)
            if valid_idxs.size == 0:
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
        Store a transition tuple in the Replay Buffer.

        Args:
            state (np.ndarray): Current state.
            action_idx (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Resulting state.
            done (bool): Flag indicating if episode finished.
        """
        self.memory.append((state, action_idx, reward, next_state, done))

    def train(self):
        """
        Execute one training step using a batch from the Replay Buffer.
        
        Performs:
        1. Sampling a random batch.
        2. Computing Q(s, a) using the Online Network.
        3. Computing Target Q using the Target Network (Bellman Equation).
        4. Gradient descent on the MSE loss.
        5. Periodic Target Network update.
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
            
    def load(self, path):
        """
        Load model weights from a checkpoint file.

        Args:
            path (str): File path to the .pt or .pth checkpoint.
        """
        if not path:
            return

        print(f"[DQN] Loading weights from {path}...")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        print("[DQN] Weights loaded and Target Network synchronized.")
