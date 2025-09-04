import numpy as np
import random
import gym
import os
import imageio
import copy
import shutil
import logging

from environment.box import Box as BinBox
from gym.spaces import Box as GymBox
from environment.bin import Bin
from utils.action_space import generate_discrete_actions
from utils.visualization import plot_bin, finalize_gif
from utils.box_generator import generate_boxes

class PackingEnv(gym.Env):
    """
    Represents a packing environment where boxes are placed inside a bin.
    It manages the bin size, the boxes to be packed, and the current state of the environment
    It provides methods to reset the environment, step through actions, and render the current state.
    """
    def __init__(self, bin_size=(10, 10, 10), max_boxes=5, lookahead=10, generate_gif=False, gif_name="packing_dqn_agent.gif"):
        """
        Initializes the packing environment with a bin size, maximum number of boxes, and options for GIF generation.
        
        Parameters:
        - bin_size (tuple): dimensions of the bin (width, height, depth)
        - max_boxes (int): maximum number of boxes to pack in the bin
        - generate_gif (bool): whether to generate a GIF of the packing process
        - gif_name (str): name of the generated GIF file
        """
        self.bin_size = bin_size
        self.max_boxes = max_boxes
        self.lookahead = lookahead
        self.current_step = 0
        self.bin = None
        self.boxes = []
        self.packed_boxes = []
        self.skipped_boxes = []
        self.placed_boxes = self.packed_boxes
        self.current_box = None

        self.prev_used_volume = 0.0
        self.prev_max_height = 0.0 
        self.total_boxes_volume = 0.0
        
        self.generate_gif = generate_gif
        self.gif_name = gif_name
        self.gif_dir = "agent_gif_frames"
        self.frame_count = 0
        self.frames = []

        if self.generate_gif:
            os.makedirs(self.gif_dir, exist_ok=True)

        self.discrete_actions = generate_discrete_actions(self.bin_size[0], self.bin_size[1])
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))

        # Observation space
        self.observation_space = GymBox(
            low=0.0,
            high=1.0,
            shape=(self._obs_length(),),
            dtype=np.float32
        )    

    def _obs_length(self):
        """
        Return the length of the observation vector (for observation_space).
        """
        width, depth, height = self.bin_size

        heightmap_len = width * depth
        upcoming_len = self.lookahead * 3
        stats_len = 3

        return heightmap_len + upcoming_len + stats_len


    def reset(self, seed=None, with_boxes=None):
        """
        Resets the environment to its initial state.
        
        Parameters:
        - seed (int): random seed for reproducibility
        - with_boxes (list): optional list of boxes to place in the bin at reset
        
        Returns:
        - np.ndarray: initial observation of the environment
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.bin = Bin(*self.bin_size)
        self.current_step = 0

        if with_boxes is not None: # if boxes are provided, use them
            self.boxes = [BinBox(b["w"], b["d"], b["h"], id=i) for i, b in enumerate(with_boxes)]
        else: # otherwise generate random boxes
            self.boxes = self._generate_boxes(self.max_boxes)

        self.boxes.sort(key=lambda b: b.get_volume(), reverse=True)

        self.current_box = self.boxes[self.current_step]
        self.prev_used_volume = 0
        self.prev_max_height = 0.0
        self.placed_boxes = []
        self.skipped_boxes = []
        self.initial_boxes = list(self.boxes)
        self.total_boxes_volume = sum(box.get_volume() for box in self.boxes)

        if self.generate_gif:
            self.frame_count = 0
            if os.path.exists(self.gif_dir):
                import shutil
                shutil.rmtree(self.gif_dir)
            os.makedirs(self.gif_dir, exist_ok=True)

        return self._get_obs()

    def _get_obs(self):
        """
        Builds the state representation:
        - Heightmap: flattened 2D array of current max heights (width × depth).
        - Upcoming box sizes: dimensions of the current box (optionally with lookahead).
        - Global stats: remaining boxes, utilization, compactness, max height.

        Returns:
        - np.ndarray: state vector for the agent.
        """
        # 1) Heightmap encoding (width × depth)
        heightmap = np.zeros((self.bin.width, self.bin.depth), dtype=np.float32)
        for b in self.bin.boxes:
            x, y, z = b.position
            bw, bd, bh = b.rotate(b.rotation_type)
            for dx in range(bw):
                for dy in range(bd):
                    heightmap[x + dx, y + dy] = max(
                        heightmap[x + dx, y + dy], z + bh
                    )
        heightmap = heightmap.flatten() / self.bin.height  # normalize 0..1

        # 2) Upcoming box sizes
        upcoming = []
        if self.current_step < self.max_boxes:
          for i in range(self.current_step, min(self.current_step + self.lookahead, self.max_boxes)):
              b = self.boxes[i]
              upcoming.extend([b.width, b.depth, b.height])
        # Pad if fewer than lookahead boxes left
        while len(upcoming) < self.lookahead * 3:
            upcoming.extend([0, 0, 0])
        upcoming = np.array(upcoming, dtype=np.float32)

        # 3) Global stats
        stats = np.array([
            (self.max_boxes - self.current_step) / self.max_boxes,   # remaining ratio
            self.get_placed_boxes_volume() / self.bin.bin_volume(),  # utilization
            #self.calculate_compactness(),                           # compactness
            self.current_max_height() / self.bin.height             # normalized max height
        ], dtype=np.float32)

        obs = np.concatenate([heightmap, upcoming, stats])

        # Final state
        return obs

    def step(self, action_idx, log_file=None, pos_file=None):
        """
        Steps through the environment with a given action.
        Attempts to place the current box at the specified position and rotation.
        If successful, it updates the state and returns the new observation, reward, done flag, and info.
        If the placement fails, it applies a small penalty and skips to the next box.
        
        Parameters:
        - action_idx (int): index of the action to take, corresponding to a position and rotation
        
        Returns:
        - tuple: (observation, reward, done, info)
        - observation (np.ndarray): the new state of the environment
        - reward (float): the reward received for the action
        - done (bool): whether the episode has ended
        - info (dict): additional information about the step, such as success or failure of placement
        """

        def log(msg: str):
          if log_file:
              with open(log_file, "a") as f:
                  f.write(msg + "\n")

        log(f"Step {self.current_step}")
        x, y, rot = self.discrete_actions[action_idx] # get position and rotation from action index

        prev_compactness = self.calculate_compactness(self.bin.boxes)

        success = self.bin.place_box(self.current_box, (x, y), rot) # check if placement is valid

        # If placement failed
        if not success:
            log("    Box misplaced (ignored in simple version)")
            reward = 0.0
            self.skipped_boxes.append(self.current_box)
        # If placement succeeded
        else:
            new_compactness = self.calculate_compactness(self.bin.boxes)
            delta_compactness = new_compactness - prev_compactness

            reward = 0.0

            if self.generate_gif:
                frame_path = os.path.join(self.gif_dir, f"frame_{self.frame_count:04d}.png")
                plot_bin(self.bin.boxes, self.bin_size, save_path=frame_path,
                        title=f"Box {self.current_box.id} at {self.current_box.position} rot={rot}")
                self.frame_count += 1

            if pos_file:
                with open(pos_file, "a") as pf:
                    pf.write(
                        f"    Step {self.current_step} | "
                        f"Box {self.current_box.id} placed at {self.current_box.position} "
                        f"rot={rot}, size={self.current_box.get_rotated_size()}\n"
                    )

            self.packed_boxes.append(self.current_box)

        info = {"success": True}

        # Move to the next box
        self.current_step += 1
    
        done = self.current_step >= self.max_boxes

        # Episode ends: add terminal utilization bonus
        if done:
            reward = 10 * self.calculate_utilization_ratio()
            log(f"    Episode ended, utilization = {reward:.2f}")
            finalize_gif(self.gif_dir, self.gif_name, fps=2)
            obs = np.zeros(self._obs_length(), dtype=np.float32)
        # Otherwise load next box and continue
        else:
            self.current_box = self.boxes[self.current_step]
            obs = self._get_obs()

        return obs, reward, done, info

    def render(self):
        """
        Render the current state of the bin in text format (console output).

        Prints:
        - The number of boxes placed so far
        - The position and rotation of each placed box
        """
        print(f"Placed {len(self.bin.boxes)} boxes:")
        for b in self.bin.boxes:
            # Each box shows its ID, position (x, y, z), and chosen rotation type
            print(f"Box {b.id} at {b.position}, rotated {b.rotation_type}")

    def _generate_boxes(self, n):
        seed_used = random.randint(0, 10000)
        
        raw_boxes = generate_boxes(self.bin_size, num_items=n, seed=random.randint(0, 10000), structured=True)

        boxes = [BinBox(width=w, depth=d, height=h, id=i) for i, (w, d, h) in enumerate(raw_boxes)]

        return boxes

    def get_placed_boxes_volume(self):
        """
        Calculate the total volume of all boxes currently placed in the bin.

        Returns:
        - float: sum of volumes of every placed box
        """
        return sum(box.get_volume() for box in self.bin.boxes)

    def calculate_utilization_ratio(self):
        """
        Compute the final reward at the end of an episode.
        Reward = ratio of used volume to total bin volume.
        Range [0,1].
        """
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)
        utilization = packed_volume / self.bin.bin_volume()
        return utilization

    
    def calculate_compactness(self, boxes):
        """
        Compactness absoluta (entre 0 e 1):
        volume das caixas colocadas / volume do bounding box que as contém.
        """
        if len(boxes) <= 1:
            return 1.0

        min_x = min(b.position[0] for b in boxes)
        min_y = min(b.position[1] for b in boxes)
        min_z = min(b.position[2] for b in boxes)

        max_x = max(b.position[0] + b.width for b in boxes)
        max_y = max(b.position[1] + b.height for b in boxes)
        max_z = max(b.position[2] + b.depth for b in boxes)

        bounding_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        packed_volume = sum(b.get_volume() for b in boxes)

        if bounding_volume == 0:
            return 0.0

        return packed_volume / bounding_volume

    def calculate_stability_reward(self, placed_box, prev_compactness):
        """
        Combina suporte físico (contacto com chão ou outras caixas)
        e compactness global numa única métrica.
        """

        # ---- Parte 1: Box support (base or other boxes)
        if placed_box.position[2] == 0:
            r_support = 1.0
        else:
            support_area = 0.0
            box_area = placed_box.width * placed_box.depth

            for other in self.bin.boxes:
                if other == placed_box:
                    continue

                # sobreposição no plano X-Y
                overlap_x = max(0, min(placed_box.position[0] + placed_box.width,
                                      other.position[0] + other.width) -
                                  max(placed_box.position[0], other.position[0]))
                overlap_y = max(0, min(placed_box.position[1] + placed_box.depth,
                                      other.position[1] + other.depth) -
                                  max(placed_box.position[1], other.position[1]))
                overlap_area = overlap_x * overlap_y

                if overlap_area > 0 and placed_box.position[2] == other.position[2] + other.height:
                    support_area += overlap_area

            r_support = min(1.0, support_area / box_area)

        # ---- Parte 2: Compactness global
        new_compactness = self.calculate_compactness(self.bin.boxes)
        delta_compactness = new_compactness - prev_compactness
        r_compact = max(-1.0, min(1.0, delta_compactness * 5.0))

        # ---- Combinação (ponderada)
        alpha, beta = 0.4, 0.6
        r_stability = alpha * r_support + beta * r_compact

        return r_stability, new_compactness

    def get_terminal_reward(self):
        """
        Compute the final reward at the end of an episode.
        Defined as the ratio of used volume to total bin volume.

        Returns:
        - float: utilization score in range [0,1],
                where 1.0 means the bin is completely filled.
        """
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)
        utilization = packed_volume / self.bin.bin_volume()

        reward = 10 * utilization
        reward += 5 * (len(self.bin.boxes) / self.max_boxes)

        return reward

    def valid_action_mask(self):
        """
        Build a mask (boolean array) that indicates which actions are valid 
        for placing the current box in the current state of the bin.

        - Each index in the mask corresponds to a discrete action (x, y, rotation).
        - A True value means the box can be placed there without violating constraints.
        - A False value means the action is invalid (out of bounds or overlapping).

        This method uses deep copies of the bin and box to test placements safely,
        ensuring the real environment state is not modified.
        """
        
        # Initialize all actions as invalid
        mask = np.zeros(self.action_space.n, dtype=bool)

        if self.current_box is None:
            return mask

        for idx, (x, y, rot) in enumerate(self.discrete_actions):
            # 1) Get rotated dimensions
            w, d, h = self.current_box.rotate(rot)

            # 2) Quick bounds check (X, Y, Z)
            if (x + w > self.bin.width) or (y + d > self.bin.depth):
                continue

            # Compute placement height (z) for this (x,y)
            z = self.bin.find_lowest_z((w, d, h), x, y)

            # Depth (Z-axis) bound check
            if z + h > self.bin.depth:
                continue

            # 3) Collision check against existing boxes
            collision = False
            for b in self.bin.boxes:
                bx, by, bz = b.position
                bw, bd, bh = b.rotate(b.rotation_type)

                overlap_x = not (x + w <= bx or x >= bx + bw)
                overlap_y = not (y + h <= by or y >= by + bh)
                overlap_z = not (z + d <= bz or z >= bz + bd)

                if overlap_x and overlap_y and overlap_z:
                    collision = True
                    break

            if collision:
                continue

            # If all checks passed, mark action as valid
            mask[idx] = True

        return mask

    def current_max_height(self):
        """
        Get the maximum vertical height currently occupied in the bin.

        - If no boxes are placed, return 0.0
        - Otherwise, compute the topmost Y-coordinate across all placed boxes:
        position_y (bottom) + box_height (after applying rotation).

        Returns:
        - float: the tallest stack height reached in the bin so far
        """
        if not self.bin.boxes:
            return 0.0
        # Z-top = bottom z-position + rotated height (depends on box orientation)
        return max(b.position[2] + b.rotate(b.rotation_type)[2] for b in self.bin.boxes)
