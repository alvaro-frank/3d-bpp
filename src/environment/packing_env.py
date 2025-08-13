import numpy as np
import random
import gym
import os
import imageio
import copy

from environment.box import Box
from environment.bin import Bin
from utils.action_space import generate_discrete_actions
from utils.visualization import plot_bin

class PackingEnv(gym.Env):
    def __init__(self, bin_size=(10, 10, 10), max_boxes=5, generate_gif=False, gif_name="packing_dqn_agent.gif"):
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

        # Ações discretas: (x, y, rot)
        self.discrete_actions = generate_discrete_actions(self.bin_size[0], self.bin_size[2])
        self.action_space = gym.spaces.Discrete(len(self.discrete_actions))      

    def reset(self, seed=None, with_boxes=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.bin = Bin(*self.bin_size)
        self.current_step = 0

        if with_boxes is not None:
            self.boxes = [Box(b["w"], b["h"], b["d"], id=i) for i, b in enumerate(with_boxes)]
        else:
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
        # Observação: dimensões da caixa atual + número de caixas restantes
        return np.array([
            self.current_box.width,
            self.current_box.height,
            self.current_box.depth,
            self.max_boxes - self.current_step
        ], dtype=np.float32)

    def step(self, action_idx):
        x, y, rot = self.discrete_actions[action_idx]

        # Try to place the box
        success = self.bin.place_box(self.current_box, (x, y), rot)

        if not success:
            # Small penalty, skip this box, continue
            reward = -0.1
            info = {"success": False, "failed_placement": True}

            self.current_step += 1
            done = self.current_step >= self.max_boxes

            if done:
                # terminal utilization bonus on episode end
                reward += 100.0 * self.get_terminal_reward()
                self._finalize_gif()
                obs = np.zeros(4, dtype=np.float32)
            else:
                self.current_box = self.boxes[self.current_step]
                # If the new box has no valid actions at all, auto‑skip with small penalty
                mask = self.valid_action_mask()
                if not mask.any():
                    # auto‑skip one time; accumulate small penalty and advance again
                    reward += -0.1
                    self.current_step += 1
                    done = self.current_step >= self.max_boxes
                    if done:
                        reward += 100.0 * self.get_terminal_reward()
                        self._finalize_gif()
                        obs = np.zeros(4, dtype=np.float32)
                    else:
                        self.current_box = self.boxes[self.current_step]
                        obs = self._get_obs()
                else:
                    obs = self._get_obs()

            return obs, reward, done, info

        # --- success path (unchanged except for comments) ---
        if self.generate_gif:
            frame_path = os.path.join(self.gif_dir, f"frame_{self.frame_count:04d}.png")
            plot_bin(self.bin.boxes, self.bin_size, save_path=frame_path,
                     title=f"Box {self.current_box.id} at {self.current_box.position} rot={rot}")
            self.frame_count += 1

        
        placed_box = self.current_box
        total_volume = self.bin_volume

        z_pos = placed_box.position[2]
        r_bottom = (self.bin.depth - z_pos) / self.bin.depth

        used_volume = self.get_placed_boxes_volume()
        delta_vol = used_volume - self.prev_used_volume
        self.prev_used_volume = used_volume
        r_vol = delta_vol / total_volume
        r_compact = self.calculate_compactness()

        new_max_h = self.current_max_height()
        delta_h = new_max_h - self.prev_max_height
        self.prev_max_height = new_max_h
        r_h = -0.1 * (delta_h / self.bin.height)

        r_step = -1e-3

        reward = 0.3 * r_vol + 0.1 * r_bottom + 0.05 * r_compact + r_h + r_step
        info = {"success": True}

        self.current_step += 1
        done = self.current_step >= self.max_boxes

        if done:
            reward += 1.0 * self.get_terminal_reward()  # utilization in [0,1]
            self._finalize_gif()
            obs = np.zeros(4, dtype=np.float32)
        else:
            self.current_box = self.boxes[self.current_step]
            obs = self._get_obs()

        return obs, reward, done, info

    def render(self):
        print(f"Placed {len(self.bin.boxes)} boxes:")
        for b in self.bin.boxes:
            print(f"Box {b.id} at {b.position}, rotated {b.rotation_type}")

    def _finalize_gif(self):
        if not self.generate_gif:
            return

        frames = []
        files = sorted([f for f in os.listdir(self.gif_dir) if f.endswith(".png")])
        for file_name in files:
            image_path = os.path.join(self.gif_dir, file_name)
            frames.append(imageio.imread(image_path))

        imageio.mimsave(self.gif_name, frames, fps=2)

        import shutil
        shutil.rmtree(self.gif_dir)
    
    @property
    def bin_volume(self):
      w, h, d = self.bin_size
      return w * h * d

    def get_placed_boxes_volume(self):
        return sum(box.get_volume() for box in self.bin.boxes)
    
    def calculate_compactness(self):
        if len(self.bin.boxes) <= 1:
            return 1.0  # Only one box placed, assume perfect compactness

        min_x = min(b.position[0] for b in self.bin.boxes)
        min_y = min(b.position[1] for b in self.bin.boxes)
        min_z = min(b.position[2] for b in self.bin.boxes)
        max_x = max(b.position[0] + b.width for b in self.bin.boxes)
        max_y = max(b.position[1] + b.height for b in self.bin.boxes)
        max_z = max(b.position[2] + b.depth for b in self.bin.boxes)

        bounding_volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)

        if bounding_volume == 0:
            return 0.0  # Avoid division by zero

        return packed_volume / bounding_volume

    def get_terminal_reward(self):
        packed_volume = sum(b.get_volume() for b in self.bin.boxes)
        return packed_volume / self.bin_volume

    def valid_action_mask(self):
        """
        Returns a boolean array (action_space.n,) where True means placing the
        current box at (x, y, rot) is feasible in the **current** state.
        Uses deepcopy of the bin and box to avoid mutating the env.
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        if self.current_box is None:
            return mask

        for idx, (x, y, rot) in enumerate(self.discrete_actions):
            # quick bounds check using non‑mutating rotated dims
            w, h, d = self.current_box.rotate(rot)
            if (x + w > self.bin.width) or (y + h > self.bin.height):
                continue

            # test the placement on a cloned bin with a cloned box (no side effects)
            bin_clone = copy.deepcopy(self.bin)
            box_clone = copy.deepcopy(self.current_box)
            if bin_clone.place_box(box_clone, (x, y), rot):
                mask[idx] = True

        return mask

    def current_max_height(self):
        if not self.bin.boxes:
            return 0.0
        # y-top is y_bottom + rotated_y_extent
        return max(b.position[1] + b.rotate(b.rotation_type)[1] for b in self.bin.boxes)
