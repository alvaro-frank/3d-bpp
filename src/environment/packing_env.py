import numpy as np
import random
import gym
import os
import imageio
import copy
import shutil

from environment.box import Box
from environment.bin import Bin
from utils.action_space import generate_discrete_actions
from utils.visualization import plot_bin, finalize_gif

class PackingEnv(gym.Env):
    """
    Represents a packing environment where boxes are placed inside a bin.
    It manages the bin size, the boxes to be packed, and the current state of the environment
    It provides methods to reset the environment, step through actions, and render the current state.
    """
    def __init__(self, bin_size=(10, 10, 10), max_boxes=5, generate_gif=False, gif_name="packing_dqn_agent.gif"):
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
        self.current_step = 0
        self.bin = None
        self.boxes = []
        self.current_box = None

        self.prev_used_volume = 0.0
        self.prev_max_height = 0.0 
        
        self.generate_gif = generate_gif
        self.gif_name = gif_name
        self.gif_dir = "agent_gif_frames"
        self.frame_count = 0
        self.frames = []

        if self.generate_gif:
            os.makedirs(self.gif_dir, exist_ok=True)

        self.discrete_actions = generate_discrete_actions(self.bin_size[0], self.bin_size[1])
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))    

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
            self.boxes = [Box(b["w"], b["h"], b["d"], id=i) for i, b in enumerate(with_boxes)]
        else: # otherwise generate random boxes
            self.boxes = self._generate_boxes(self.max_boxes)

        self.current_box = self.boxes[self.current_step]
        self.prev_used_volume = 0
        self.prev_max_height = 0.0 

        if self.generate_gif:
            self.frame_count = 0
            if os.path.exists(self.gif_dir):
                import shutil
                shutil.rmtree(self.gif_dir)
            os.makedirs(self.gif_dir, exist_ok=True)

        return self._get_obs()

    def _generate_boxes(self, n):
        """
        Generates a list of random boxes for the environment.
        
        Parameters:
        - n (int): number of boxes to generate
        
        Returns:
        - list: a list of Box objects with random dimensions
        """
        return [
            Box(
                width=random.randint(1, 5),
                height=random.randint(1, 5),
                depth=random.randint(1, 5),
                id=i
            )
            for i in range(n)
        ]

    def _get_obs(self):
        """
        Gets the current observation of the environment.
        This includes the dimensions of the current box and the number of boxes remaining to be placed.
        
        Returns:
        - np.ndarray: an array containing the current box dimensions and remaining boxes count
        """
        # Observação: dimensões da caixa atual + número de caixas restantes
        return np.array([
            self.current_box.width,
            self.current_box.height,
            self.current_box.depth,
            self.max_boxes - self.current_step
        ], dtype=np.float32)

    def step(self, action_idx):
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
        x, y, rot = self.discrete_actions[action_idx] # get position and rotation from action index

        success = self.bin.place_box(self.current_box, (x, y), rot) # check if placement is valid

        # If placement failed
        if not success:
            # Small penalty, skip this box, continue
            reward = -0.1
            info = {"success": False, "failed_placement": True}

            self.current_step += 1
            done = self.current_step >= self.max_boxes # check if all boxes are placed

            # Episode ended, add terminal reward and return empty observation
            if done:
                reward += 100.0 * self.get_terminal_reward()
                finalize_gif(self.gif_dir, self.gif_name, fps=2)
                obs = np.zeros(4, dtype=np.float32)
            # If not ended, continue with the next box
            else:
                self.current_box = self.boxes[self.current_step] # get the next box
                
                # Check if next box has any valid actions
                mask = self.valid_action_mask()
                
                # If no valid placement exists, skip with another penalty
                if not mask.any():
                    reward += -0.1
                    self.current_step += 1
                    done = self.current_step >= self.max_boxes
                    if done:
                        reward += 100.0 * self.get_terminal_reward()
                        finalize_gif(self.gif_dir, self.gif_name, fps=2)
                        obs = np.zeros(4, dtype=np.float32)
                    else:
                        self.current_box = self.boxes[self.current_step]
                        obs = self._get_obs()
                else:
                    # If valid placements exist, continue normally
                    obs = self._get_obs()

            return obs, reward, done, info

        # If placement succeeded
        if self.generate_gif:
            frame_path = os.path.join(self.gif_dir, f"frame_{self.frame_count:04d}.png")
            plot_bin(self.bin.boxes, self.bin_size, save_path=frame_path,
                     title=f"Box {self.current_box.id} at {self.current_box.position} rot={rot}")
            self.frame_count += 1

        
        placed_box = self.current_box
        total_volume = self.bin.bin_volume()

        # 1) Bottom reward: prefer placements closer to the floor
        z_pos = placed_box.position[2]
        r_bottom = (self.bin.depth - z_pos) / self.bin.depth

        # 2) Volume reward: incremental volume placed since last step
        used_volume = self.get_placed_boxes_volume()
        delta_vol = used_volume - self.prev_used_volume
        self.prev_used_volume = used_volume
        r_vol = delta_vol / total_volume
        
        # 3) Compactness reward: how tightly boxes fit together
        r_compact = self.calculate_compactness()

        # 4) Height regularizer: penalize increases in max stack height
        new_max_h = self.current_max_height()
        delta_h = new_max_h - self.prev_max_height
        self.prev_max_height = new_max_h
        r_h = -0.1 * (delta_h / self.bin.height)

        # 5) Small step cost to discourage long episodes
        r_step = -1e-3

        # Final reward (weighted combination of all components)
        reward = 0.3 * r_vol + 0.1 * r_bottom + 0.05 * r_compact + r_h + r_step
        info = {"success": True}

        # Move to the next box
        self.current_step += 1
    
        done = self.current_step >= self.max_boxes

        # Episode ends: add terminal utilization bonus
        if done:
            reward += 1.0 * self.get_terminal_reward()  # utilization in [0,1]
            finalize_gif(self.gif_dir, self.gif_name, fps=2)
            obs = np.zeros(4, dtype=np.float32)
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

    def get_placed_boxes_volume(self):
        """
        Calculate the total volume of all boxes currently placed in the bin.

        Returns:
        - float: sum of volumes of every placed box
        """
        return sum(box.get_volume() for box in self.bin.boxes)
    
    def calculate_compactness(self):
        """
        Measure how tightly the placed boxes are packed together.

        Compactness is defined as:
        (total volume of placed boxes) / (volume of the bounding box that contains them all).

        - If only one box is placed, compactness is assumed perfect (1.0).
        - If the bounding volume is zero (safety check), return 0.0.

        Returns:
        - float: compactness score in range [0,1], where higher = tighter packing.
        """
        if len(self.bin.boxes) <= 1:
            return 1.0  # Only one box placed, assume perfect compactness

        # Find the smallest coordinates occupied by any box (min corner)
        min_x = min(b.position[0] for b in self.bin.boxes)
        min_y = min(b.position[1] for b in self.bin.boxes)
        min_z = min(b.position[2] for b in self.bin.boxes)
        
        # Find the farthest extents occupied by any box (max corner)
        max_x = max(b.position[0] + b.width for b in self.bin.boxes)
        max_y = max(b.position[1] + b.height for b in self.bin.boxes)
        max_z = max(b.position[2] + b.depth for b in self.bin.boxes)

        # Volume of the minimal bounding cuboid that encloses all placed boxes
        bounding_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        
        # Total actual volume of the boxes
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)

        if bounding_volume == 0:
            return 0.0  # Avoid division by zero

        return packed_volume / bounding_volume

    def get_terminal_reward(self):
        """
        Compute the final reward at the end of an episode.
        Defined as the ratio of used volume to total bin volume.

        Returns:
        - float: utilization score in range [0,1],
                where 1.0 means the bin is completely filled.
        """
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)
        return packed_volume / self.bin.bin_volume()

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
        
        # If there's no current box, no actions are possible
        if self.current_box is None:
            return mask

        # Check each possible action
        for idx, (x, y, rot) in enumerate(self.discrete_actions):
            # 1) Quick bounds check using rotated dimensions
            w, h, d = self.current_box.rotate(rot)
            if (x + w > self.bin.width) or (y + h > self.bin.height):
                continue # skip if it doesn’t fit within bin’s X/Y bounds

            # 2) Test placement on a cloned bin/box (so the real state isn’t touched)
            bin_clone = copy.deepcopy(self.bin)
            box_clone = copy.deepcopy(self.current_box)
            
            # If placement succeeds, mark this action as valid
            if bin_clone.place_box(box_clone, (x, y), rot):
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
        # Y-top = bottom y-position + rotated height (depends on box orientation)
        return max(b.position[1] + b.rotate(b.rotation_type)[1] for b in self.bin.boxes)
