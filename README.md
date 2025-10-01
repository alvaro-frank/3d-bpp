# 3D Bin Packing

A simple research project for the 3D Bin Packing Problem using **Reinforcement Learning**.  
It includes a custom Gym environment, DQN/PPO agents, a heuristic baseline, and 3D visualizations (with GIF export).

![Packing GIF](runs/ppo/ppo_5000steps.gif)

## Features
- Custom `gym.Env` for 3D bin packing (`src/environment/packing_env.py`).
- Two RL agents: **DQN** and **PPO** (`src/agents/`).
- Heuristic baseline (bottom-left-back) for comparison (`src/heuristics/heuristic.py`).
- Training/evaluation utilities and fixed test sets (`src/train/`, `src/evals/`, `src/utils/testsets.py`).
- 3D visualization and GIF generation (`src/utils/visualization.py`).

## Observation Space
The **state representation** encodes the environment at each step:
- **Bin state:** Occupancy grid / 3D representation of already placed boxes.
- **Next box features:** Dimensions (w, h, d) and orientation.
- **Packing statistics:** Remaining space, number of boxes placed, total volume used, etc.

This provides the agent with both spatial and sequential context to decide where to place the next box.

## Reward Shaping
The reward function is crucial for guiding the agent:
- **Positive reward** for successfully placing a box in the bin without overlap.  
- **Volume‑based reward:** Proportional to the filled volume fraction (encourages tighter packing).  
- **Penalty** for invalid placements or leaving too much unused space.  
- **Episode reward:** Cumulative sum reflects how efficiently the agent packed all boxes.

Reward shaping is designed to balance exploration (trying placements) and exploitation (maximizing packing efficiency).

## Project Structure
```
src/
  agents/              # DQN & PPO agents
  environment/         # Bin, Box, and PackingEnv (gym.Env)
  evals/               # Evaluation helpers (agents & heuristic)
  heuristics/          # Heuristic baseline(s)
  train/               # Training loops for DQN & PPO
  utils/               # Action space, box generation, seeding, viz, test sets
  runs/                # Models, logs, plots, and generated GIFs (outputs)
  main.py              # End‑to‑end script: train + evaluate + visualize
```

## Requirements
```
pip install -r requirements.txt
```

## Quick Start
Run the end‑to‑end script (train → evaluate → visuals):
```bash
python src/main.py
```
This will train an RL agent (DQN or PPO), evaluate it against fixed test sets and a heuristic baseline, and write artifacts to `src/runs/` (learning curves, models, logs and GIFs).

### Outputs
- `src/runs/*/*learning_curve.png` — learning curves
- `src/runs/*/*packing.gif` — packing animations
- `src/runs/*/*log.txt` - training logs
- `src/runs/*/*placed_boxes.txt` - placed boxes coordinates during training

## Configuration
Key knobs you may want to tweak live in the code:
- **Environment** size and number of boxes: `PackingEnv(bin_size=..., max_boxes=...)` in `src/environment/packing_env.py` or where the env is created in the training loop.
- **Training** lengths and logging cadence: see `src/train/train_dqn_agent.py` and `src/train/train_ppo_agent.py`.
- **Main pipeline** defaults (episodes, boxes, seeds): see `src/main.py`.

## Repro Tips
- Set seeds via `utils/seed.py` to make comparisons fair.
- Use the same test sets for all methods (`utils/testsets.py`).
