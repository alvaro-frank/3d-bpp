# ==============================================================================
# FILE: environment/packing_env.py
# DESCRIPTION: 3D Bin Packing Environment using the OpenAI Gym interface.
#              Manages the bin state, box queue, and reinforcement learning rewards.
# ==============================================================================

import numpy as np
import random
import gym
import os

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

    # --------------------------------------------------------------------------
    # CORE GYM METHODS
    # --------------------------------------------------------------------------

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
        self.packed_boxes = []
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

    def step(self, action_idx):
        """
        Apply a discrete action (x, y, rotation) or NO-OP and advance the environment.

        Args:
            action_idx (int): Index in the discrete action set.

        Returns:
            tuple: (observation, reward, done, info)
        """
    
        if self.include_noop and action_idx == self.NOOP_IDX:
            reward = 0.0 
            info = {"success": False, "noop": True}
            self.skipped_boxes.append(self.current_box)
            self.current_step += 1
            done = self.current_step >= self.max_boxes
            
            if done:
                packed_count = len(self.packed_boxes)
                ratio = packed_count / self.max_boxes
                
                if ratio >= 0.8: reward += 2.0
                if ratio >= 0.9: reward += 3.0
                if ratio == 1.0: reward += 10.0
                if ratio < 0.5: reward -= 5.0
                
                if self.generate_gif:
                    finalize_gif(self.gif_dir, self.gif_name, fps=2)
                
                obs = np.zeros(self._obs_length(), dtype=np.float32)
            else:
                self.current_box = self.boxes[self.current_step]
                obs = self._get_obs()
            
            return obs, reward, done, info

        x, y, rot = self.discrete_actions[action_idx]
        prev_compactness = self.calculate_compactness(self.bin.boxes)
        prev_max_h = self.current_max_height()

        success = self.bin.place_box(self.current_box, (x, y), rot)

        if not success:
            reward = 0.0
            self.skipped_boxes.append(self.current_box)
        else:
            vol_ratio = self.current_box.get_volume() / self.total_boxes_volume
            r_vol = vol_ratio * 10.0
            
            new_compactness = self.calculate_compactness(self.bin.boxes)
            delta_compactness = new_compactness - prev_compactness
            r_comp = delta_compactness * 1.0
            
            new_max_h = self.current_max_height()
            delta_h = new_max_h - prev_max_h
            normalized_delta_h = delta_h / self.bin.height
            r_height = - (normalized_delta_h * 1.0)
            
            w, d, h = self.dims_after_rotation(self.current_box, rot)
            box_surface_area = 2 * (w*d + w*h + d*h)
            contact_area = self.calculate_contact_area(self.current_box)
            r_contact = (contact_area / box_surface_area) * 2.0

            reward = r_vol + r_comp + r_height + r_contact

            if self.generate_gif:
                frame_path = os.path.join(self.gif_dir, f"frame_{self.frame_count:04d}.png")
                plot_bin(self.bin.boxes, self.bin_size, save_path=frame_path,
                        title=f"Box {self.current_box.id} at {self.current_box.position} rot={rot}")
                self.frame_count += 1

            self.packed_boxes.append(self.current_box)

        info = {"success": True}
        self.current_step += 1
        done = self.current_step >= self.max_boxes

        if done:
            packed_count = len(self.packed_boxes)
            ratio = packed_count / self.max_boxes
            if ratio >= 0.8: reward += 2.0
            if ratio >= 0.9: reward += 3.0
            if ratio == 1.0: reward += 10.0
            if ratio < 0.5: reward -= 5.0
            if self.generate_gif:
                finalize_gif(self.gif_dir, self.gif_name, fps=2)
            obs = np.zeros(self._obs_length(), dtype=np.float32)
        else:
            self.current_box = self.boxes[self.current_step]
            obs = self._get_obs()

        return obs, reward, done, info

    def render(self):
        """Print a simple text rendering of placed boxes."""
        for b in self.bin.boxes:
            print(f"Box {b.id} at {b.position}, rotated {b.rotation_type}")

    # --------------------------------------------------------------------------
    # OBSERVATION & INTERNAL UTILS
    # --------------------------------------------------------------------------

    def _get_obs(self):
        """
        Build the observation vector: Heightmap + Upcoming boxes + Global Stats.
        """
        # 1) Heightmap encoding
        heightmap = np.zeros((self.bin.width, self.bin.depth), dtype=np.float32)
        for b in self.bin.boxes:
            x, y, z = b.position
            bw, bd, bh = self.dims_after_rotation(b, b.rotation_type)
            for dx in range(bw):
                for dy in range(bd):
                    heightmap[x + dx, y + dy] = max(heightmap[x + dx, y + dy], z + bh)
        heightmap = heightmap.flatten() / self.bin.height

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
            (self.max_boxes - self.current_step) / self.max_boxes,
            self.get_placed_boxes_volume() / self.bin.bin_volume(),
            self.current_max_height() / self.bin.height
        ], dtype=np.float32)

        return np.concatenate([heightmap, upcoming, stats])

    def valid_action_mask(self):
        """Build a binary mask indicating which placements are valid."""
        act_n = self.action_space.n
        mask = np.zeros(act_n, dtype=np.float32)

        if self.current_box is None:
            if self.include_noop: mask[self.NOOP_IDX] = 1.0
            return mask

        for idx, action in enumerate(self.discrete_actions[:-1]):
            x, y, rot = action
            w, d, h = self.dims_after_rotation(self.current_box, rot)
            if (x + w > self.bin.width) or (y + d > self.bin.depth): continue
            z = self.bin.find_lowest_z((w, d, h), x, y)
            if z + h > self.bin.height: continue

            collision = False
            for b in self.bin.boxes:
                bx, by, bz = b.position
                bw, bd, bh = self.dims_after_rotation(b, b.rotation_type)
                if not (x + w <= bx or x >= bx + bw) and \
                   not (y + d <= by or y >= by + bd) and \
                   not (z + h <= bz or z >= bz + bh):
                    collision = True
                    break
            if not collision: mask[idx] = 1.0

        if self.include_noop and mask.sum() == 0: mask[self.NOOP_IDX] = 1.0
        return mask

    # --------------------------------------------------------------------------
    # GEOMETRIC & METRIC CALCULATIONS
    # --------------------------------------------------------------------------

    def dims_after_rotation(self, box, rot):
        """Return (w, d, h) after applying a rotation index."""
        w0, d0, h0 = box.width, box.depth, box.height
        rot = int(rot) % 6
        if rot == 0: return w0, d0, h0
        if rot == 1: return w0, h0, d0
        if rot == 2: return d0, w0, h0
        if rot == 3: return h0, d0, w0
        if rot == 4: return d0, h0, w0
        if rot == 5: return h0, w0, d0
        return w0, d0, h0

    def current_max_height(self):
        """Get the maximum vertical (z) height currently occupied."""
        if not self.bin.boxes: return 0.0
        return max(b.position[2] + self.dims_after_rotation(b, b.rotation_type)[2] for b in self.bin.boxes)

    def calculate_compactness(self, boxes):
        """Absolute compactness score based on bounding box volume."""
        if len(boxes) <= 1: return 1.0
        min_x = min(b.position[0] for b in boxes)
        min_y = min(b.position[1] for b in boxes)
        min_z = min(b.position[2] for b in boxes)
        max_x = max(b.position[0] + b.width for b in boxes)
        max_y = max(b.position[1] + b.height for b in boxes)
        max_z = max(b.position[2] + b.depth for b in boxes)
        bounding_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        return sum(b.get_volume() for b in boxes) / bounding_volume if bounding_volume > 0 else 0.0

    def calculate_contact_area(self, box):
        """Calculate the total surface contact area of a box with others and bin walls."""
        bx, by, bz = box.position
        bw, bd, bh = self.dims_after_rotation(box, box.rotation_type)
        contact_area = 0.0
        if bz == 0: contact_area += bw * bd
        if bx == 0: contact_area += bd * bh
        if bx + bw == self.bin.width: contact_area += bd * bh
        if by == 0: contact_area += bw * bh
        if by + bd == self.bin.depth: contact_area += bw * bh

        for other in self.bin.boxes:
            if other.id == box.id: continue
            ox, oy, oz = other.position
            ow, od, oh = self.dims_after_rotation(other, other.rotation_type)
            def overlap(c1, l1, c2, l2): return max(0.0, min(c1 + l1, c2 + l2) - max(c1, c2))
            if (bx == ox + ow) or (bx + bw == ox): contact_area += overlap(by, bd, oy, od) * overlap(bz, bh, oz, oh)
            if (by == oy + od) or (by + bd == oy): contact_area += overlap(bx, bw, ox, ow) * overlap(bz, bh, oz, oh)
            if (bz == oz + oh) or (bz + bh == oz): contact_area += overlap(bx, bw, ox, ow) * overlap(by, bd, oy, od)
        return contact_area

    def _generate_boxes(self, n):
        """Generate a list of random boxes for the episode."""
        raw_boxes = generate_boxes(self.bin_size, num_items=n, seed=random.randint(0, 10000), structured=True)
        return [BinBox(width=w, depth=d, height=h, id=i) for i, (w, d, h) in enumerate(raw_boxes)]

    def get_placed_boxes_volume(self):
        """Compute total volume of all currently placed boxes."""
        return sum(box.get_volume() for box in self.bin.boxes)

    def calculate_utilization_ratio(self):
        """Compute used-volume ratio."""
        return sum(b.get_volume() for b in self.bin.boxes) / self.bin.bin_volume()