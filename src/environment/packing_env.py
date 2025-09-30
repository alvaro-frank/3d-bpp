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
    3D bin packing environment.

    Boxes are placed inside a fixed-size bin using discrete (x, y, rotation) actions.
    The environment manages the bin, a queue of boxes, observation construction,
    step transitions, and optional GIF rendering of the packing process.
    """
    def __init__(self, bin_size=(10, 10, 10), max_boxes=5, lookahead=10, generate_gif=False, gif_name="packing_dqn_agent.gif", include_noop=False):
        """
        Initialize the environment.

        Args:
            bin_size (tuple[int, int, int]): Bin dimensions (width, depth, height).
            max_boxes (int): Number of boxes to attempt to pack per episode.
            lookahead (int): How many upcoming boxes to encode in the observation (triplets of w,d,h).
            generate_gif (bool): If True, saves frames and builds a GIF of the episode.
            gif_name (str): Output GIF filename.
            include_noop (bool): If True, include a NO-OP action as the last discrete action.

        Notes:
            - Action space is Discrete over (x, y, rotation) plus optional NO-OP.
            - Observation is a flat vector: heightmap + upcoming boxes + global stats.
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

        self.include_noop = include_noop

        if self.generate_gif:
            os.makedirs(self.gif_dir, exist_ok=True)

        self.discrete_actions = generate_discrete_actions(self.bin_size[0], self.bin_size[1])
        if self.include_noop:
            self.NOOP_IDX = len(self.discrete_actions)
            self.discrete_actions.append(("noop"))
        else:
            self.NOOP_IDX = None

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
        Compute observation vector length.

        Returns:
            int: Length = (width*depth) heightmap + (lookahead*3) upcoming dims + 3 global stats.
        """
        width, depth, height = self.bin_size

        heightmap_len = width * depth
        upcoming_len = self.lookahead * 3
        stats_len = 3

        return heightmap_len + upcoming_len + stats_len


    def reset(self, seed=None, with_boxes=None):
        """
        Reset the environment to the start of a new episode.

        Args:
            seed (int | None): Optional RNG seed for reproducibility.
            with_boxes (list[dict] | None): Optional fixed boxes, each as {'w','d','h'}.

        Returns:
            np.ndarray: Initial observation vector.
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

        m = self.valid_action_mask()
        assert m.shape[0] == self.action_space.n

        return self._get_obs()

    def _get_obs(self):
        """
        Build the observation vector.

        Composition:
            1) Heightmap (width×depth), normalized to [0,1] by bin height.
            2) Upcoming box sizes: triplets (w,d,h) for current → lookahead;
               padded with zeros to length lookahead*3.
            3) Global stats:
               - remaining_ratio: (max_boxes - current_step)/max_boxes
               - utilization: placed_volume / bin_volume
               - normalized_max_height: current_max_height / bin.height

        Returns:
            np.ndarray: Flattened observation.
        """
        # 1) Heightmap encoding (width × depth)
        heightmap = np.zeros((self.bin.width, self.bin.depth), dtype=np.float32)
        for b in self.bin.boxes:
            x, y, z = b.position
            bw, bd, bh = self.dims_after_rotation(b, b.rotation_type)
            assert x + bw <= self.bin.width and y + bd <= self.bin.depth, \
                f"Box fora dos limites no _get_obs: (x={x},y={y}) + (bw={bw},bd={bd}) vs (W={self.bin.width},D={self.bin.depth})"
            for dx in range(bw):
                for dy in range(bd):
                    heightmap[x + dx, y + dy] = max(heightmap[x + dx, y + dy], z + bh)
        heightmap = heightmap.flatten() / self.bin.height  # normalize 0..1

        # 2) Upcoming box sizes
        upcoming = []
        if self.current_step < self.max_boxes:
            for i in range(self.current_step, min(self.current_step + self.lookahead, self.max_boxes)):
                b = self.boxes[i]
                upcoming.extend([b.width, b.depth, b.height])
        while len(upcoming) < self.lookahead * 3:
            upcoming.extend([0, 0, 0])
        upcoming = np.array(upcoming, dtype=np.float32)

        # 3) Global stats
        stats = np.array([
            (self.max_boxes - self.current_step) / self.max_boxes,   # remaining ratio
            self.get_placed_boxes_volume() / self.bin.bin_volume(),  # utilization
            self.current_max_height() / self.bin.height              # normalized max height
        ], dtype=np.float32)

        obs = np.concatenate([heightmap, upcoming, stats])

        return obs

    def step(self, action_idx, log_file=None, pos_file=None):
        """
        Apply a discrete action (x, y, rotation) or NO-OP and advance the environment.

        Args:
            action_idx (int): Index in the discrete action set.
            log_file (str | None): Optional file to append step logs.
            pos_file (str | None): Optional file to append placements (positions/rotations).

        Returns:
            tuple:
                - observation (np.ndarray): Next observation.
                - reward (float): Reward for this step (terminal bonus at episode end).
                - done (bool): Whether the episode is finished.
                - info (dict): Extra info (e.g., success flag, noop flag).

        Notes:
            - On successful placement, the box is added and an intermediate reward may be applied.
            - On failure or NO-OP, the box is skipped. Terminal reward adds a utilization bonus.
        """
        def log(msg: str):
          if log_file:
              with open(log_file, "a") as f:
                  f.write(msg + "\n")

        log(f"Step {self.current_step}")
        
        if self.include_noop and action_idx == self.NOOP_IDX:
            reward = 0.0 
            info = {"success": False, "noop": True}
            self.skipped_boxes.append(self.current_box)
            self.current_step += 1
            done = self.current_step >= self.max_boxes
            if done:
                reward = 10 * self.calculate_utilization_ratio()
                log(f"    Episode ended (noop), utilization = {reward:.2f}")
                finalize_gif(self.gif_dir, self.gif_name, fps=2)
                obs = np.zeros(self._obs_length(), dtype=np.float32)
            else:
                self.current_box = self.boxes[self.current_step]
                obs = self._get_obs()
            return obs, reward, done, info

        x, y, rot = self.discrete_actions[action_idx]

        prev_compactness = self.calculate_compactness(self.bin.boxes)

        success = self.bin.place_box(self.current_box, (x, y), rot)

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
        Print a simple text rendering of placed boxes (ID, position, rotation).
        """
        for b in self.bin.boxes:
            # Each box shows its ID, position (x, y, z), and chosen rotation type
            print(f"Box {b.id} at {b.position}, rotated {b.rotation_type}")

    def _generate_boxes(self, n):
        """
        Generate a list of random boxes for the episode.

        Args:
            n (int): Number of boxes to generate.

        Returns:
            list[BinBox]: Generated boxes.
        """
        raw_boxes = generate_boxes(self.bin_size, num_items=n, seed=random.randint(0, 10000), structured=True)

        boxes = [BinBox(width=w, depth=d, height=h, id=i) for i, (w, d, h) in enumerate(raw_boxes)]

        return boxes

    def get_placed_boxes_volume(self):
        """
        Compute total volume of all currently placed boxes.

        Returns:
            float: Sum of volumes.
        """
        return sum(box.get_volume() for box in self.bin.boxes)

    def calculate_utilization_ratio(self):
        """
        Compute used-volume ratio.

        Returns:
            float: Used volume / bin volume, in [0, 1].
        """
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)
        utilization = packed_volume / self.bin.bin_volume()
        return utilization

    
    def calculate_compactness(self, boxes):
        """
        Absolute compactness in [0, 1]:
        total placed volume / volume of their bounding box.

        Args:
            boxes (list[BinBox]): Boxes to evaluate.

        Returns:
            float: Compactness score (1.0 if ≤1 box).
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

    def get_terminal_reward(self):
        """
        Compute terminal reward at episode end.

        Returns:
            float: 10×utilization + 5×(placed_count / max_boxes).
        """
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)
        utilization = packed_volume / self.bin.bin_volume()

        reward = 10 * utilization
        reward += 5 * (len(self.bin.boxes) / self.max_boxes)

        return reward

    

    def dims_after_rotation(self, box, rot):
        """
        Return (w, d, h) after applying a rotation index, without mutating the box.

        Args:
            box (BinBox): Target box.
            rot (int): Rotation index in [0..5].

        Returns:
            tuple[int, int, int]: Rotated dimensions (w, d, h).
        """
        w0, d0, h0 = box.width, box.depth, box.height
        
        rot = int(rot) % 6
        if rot == 0:   # (w,d,h)
            return w0, d0, h0
        if rot == 1:   # (w,h,d)
            return w0, h0, d0
        if rot == 2:   # (d,w,h)
            return d0, w0, h0
        if rot == 3:   # (h,d,w)
            return h0, d0, d0
        if rot == 4:   # (d,h,w)
            return d0, h0, w0
        if rot == 5:   # (h,w,d)
            return h0, w0, d0
        
        return w0, d0, h0

    def valid_action_mask(self):
        """
        Build a binary mask over actions indicating which placements are valid.

        Rules:
            - Checks bounds on (x, y) and height (z + h ≤ bin.height).
            - Uses AABB overlap tests against already placed boxes.
            - If no placement is valid and NO-OP is enabled, NO-OP is set valid.

        Returns:
            np.ndarray: Float mask of shape (action_space.n,) with 1.0 for valid, 0.0 for invalid.
        """
        act_n = self.action_space.n
        mask = np.zeros(act_n, dtype=np.float32)

        if self.current_box is None:
            if self.include_noop:
                mask[self.NOOP_IDX] = 1.0
            return mask

        for idx, action in enumerate(self.discrete_actions[:-1]):
            x, y, rot = action
            w, d, h = self.dims_after_rotation(self.current_box, rot)

            if (x + w > self.bin.width) or (y + d > self.bin.depth):
                continue

            z = self.bin.find_lowest_z((w, d, h), x, y)

            if z + h > self.bin.height:
                continue

            collision = False
            for b in self.bin.boxes:
                bx, by, bz = b.position
                bw, bd, bh = self.dims_after_rotation(b, b.rotation_type)

                overlap_x = not (x + w <= bx or x >= bx + bw) 
                overlap_y = not (y + d <= by or y >= by + bd) 
                overlap_z = not (z + h <= bz or z >= bz + bh)  

                if overlap_x and overlap_y and overlap_z:
                    collision = True
                    break

            if collision:
                continue

            mask[idx] = 1.0

        if self.include_noop and mask.sum() == 0:
            mask[self.NOOP_IDX] = 1.0
        return mask

    def current_max_height(self):
        """
        Get the maximum vertical (z) height currently occupied in the bin.

        Returns:
            float: 0.0 if no boxes; otherwise max over (z_bottom + rotated_height).
        """
        if not self.bin.boxes:
            return 0.0

        return max(b.position[2] + self.dims_after_rotation(b, b.rotation_type)[2] for b in self.bin.boxes)
