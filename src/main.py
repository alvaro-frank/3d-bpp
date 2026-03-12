# ==============================================================================
# FILE: main.py
# DESCRIPTION: Entry point for the 3D-BPP (3D Bin Packing Problem) project.
#              Provides a CLI to train DQN/PPO agents or evaluate them against 
#              heuristic baselines using MLflow for experiment tracking.
#              Now powered by Hydra for configuration management.
# ==============================================================================
import os
from pathlib import Path
import glob
import numpy as np
import torch
import mlflow
import hydra
from omegaconf import DictConfig, OmegaConf

from evals.evaluate_agent import evaluate_agent_on_episode
from evals.evaluate_heuristic import evaluate_heuristic_on_episode
from utils.testsets import make_test_sets, load_test_sets
from train.train_dqn_agent import train_dqn_agent as dqn_train_loop
from environment.packing_env import PackingEnv
from utils.seed import seed_all

from agents.ppo_agent import PPOAgent, PPOConfig
from train.train_ppo_agent import train_ppo_agent as ppo_train_loop, TrainPPOConfig
from agents.dqn_agent import DQNAgent, DQNConfig


# ------------------------------------------------------------------------------
# MODEL & ENVIRONMENT BUILDERS
# ------------------------------------------------------------------------------
def _build_env(max_boxes: int, include_noop: bool = False):
    return PackingEnv(max_boxes=max_boxes, include_noop=include_noop)

def _build_dqn(cfg: DictConfig, env):
    state_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n if hasattr(env, "action_space") else len(getattr(env, "discrete_actions", []))
    map_size = (cfg.environment.bin_size[0], cfg.environment.bin_size[1])

    dqn_cfg = DQNConfig(
        exploration=cfg.agent.exploration,
        lr=cfg.agent.lr,
        gamma=cfg.agent.gamma,
        batch_size=cfg.agent.batch_size,
        update_target_steps=cfg.agent.update_target_steps,
        epsilon_start=cfg.agent.epsilon_start,
        epsilon_final=cfg.agent.epsilon_final,
        temperature_start=cfg.agent.temperature_start,
        temperature_final=cfg.agent.temperature_final,
        temperature_decay_steps=cfg.agent.temperature_decay_steps
    )
    
    total_steps = cfg.episodes * cfg.environment.max_boxes

    return DQNAgent(
        state_dim=state_dim, 
        action_dim=act_dim, 
        map_size=map_size, 
        config=dqn_cfg, 
        total_training_steps=total_steps,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

def _build_ppo(cfg: DictConfig, env):
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n if hasattr(env, "action_space") else len(getattr(env, "discrete_actions", []))
    map_size = (env.bin_size[0], env.bin_size[1])
    
    ppo_cfg = PPOConfig(
        gamma=cfg.agent.gamma, 
        gae_lambda=cfg.agent.gae_lambda, 
        clip_eps=cfg.agent.clip_eps,
        vf_coef=cfg.agent.vf_coef, 
        ent_coef=cfg.agent.ent_coef, 
        lr=cfg.agent.lr,
        epochs=cfg.agent.epochs, 
        minibatch_size=cfg.agent.minibatch_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    return PPOAgent(obs_dim=obs_dim, act_dim=act_dim, map_size=map_size, config=ppo_cfg)

# ------------------------------------------------------------------------------
# CHECKPOINT & WEIGHT LOADING
# ------------------------------------------------------------------------------

def _extract_state_dict(ckpt):
    """Return a model state_dict from a variety of checkpoint formats."""
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
        if all(isinstance(v, (dict, torch.Tensor)) for v in ckpt.values()):
            return ckpt
    raise ValueError(
        "Checkpoint format not recognized. Expected keys like "
        "'model', 'model_state_dict', or 'state_dict'."
    )

def _load_weights_dqn(agent, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent

def _load_weights_ppo(agent, path: str):
    ckpt = torch.load(path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)
    agent.model.load_state_dict(state_dict)
    agent.model.eval()
    return agent

def _auto_find_checkpoint(agent_type: str) -> str | None:
    if agent_type == "dqn":
        candidates = sorted(
            glob.glob("runs/dqn/models/*.pth") + glob.glob("runs/dqn/models/*.pt")
        )
    else:
        candidates = sorted(
            glob.glob("runs/ppo/models/*.pt") + glob.glob("runs/ppo/models/*.pth")
        )
    return candidates[-1] if candidates else None


# ------------------------------------------------------------------------------
# COMMAND: TRAIN
# ------------------------------------------------------------------------------

def cmd_train(cfg: DictConfig):
    seed_all(cfg.seed)
    print(f"Training {cfg.agent.name.upper()} | episodes={cfg.episodes} boxes={cfg.environment.max_boxes} seed={cfg.seed}")
    
    experiment_name = "3D-BPP Experiment"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run() as run:
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        
        if cfg.agent.name == "dqn":
            env = _build_env(max_boxes=cfg.environment.max_boxes, include_noop=cfg.environment.get("include_noop", False))
            agent = _build_dqn(cfg, env)

            load_model = cfg.get("load_model", None)
            if load_model is not None:
                _load_weights_dqn(agent, load_model)
                print(f"Loaded DQN weights from: {load_model}")

            agent = dqn_train_loop(
                env=env,
                agent=agent,
                num_episodes=cfg.episodes, 
                generate_gif=False
            )
            
            mlflow.pytorch.log_model(agent.model, artifact_path="model")
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "3d-bpp-dqn")
            
            try:
                env.close()
            except Exception:
                pass
        else:
            env = _build_env(max_boxes=cfg.environment.max_boxes, include_noop=cfg.environment.get("include_noop", False))
            agent = _build_ppo(cfg, env)

            load_model = cfg.get("load_model", None)
            if load_model is not None:
                _load_weights_ppo(agent, load_model)
                print(f"Loaded PPO weights from: {load_model}")

            train_cfg = TrainPPOConfig(
                num_episodes=cfg.episodes,
                max_steps_per_episode=None,
                log_every=10,
                eval_every=0,
                eval_episodes=5,
                save_every=50,
                save_dir="runs/ppo",
                save_models="runs/ppo/models",
                seed=cfg.seed,
            )
            os.makedirs(train_cfg.save_models, exist_ok=True)
            ppo_train_loop(env, agent, cfg=train_cfg)
            
            mlflow.pytorch.log_model(agent.model, artifact_path="model")
            mlflow.register_model(f"runs:/{run.info.run_id}/model", "3d-bpp-ppo")
            
            try:
                env.close()
            except Exception:
                pass

    print("Training finished.")
    return 0

# ------------------------------------------------------------------------------
# COMMAND: EVALUATE
# ------------------------------------------------------------------------------

def cmd_evaluate(cfg: DictConfig):
    seed_all(cfg.seed)
    
    boxes = cfg.environment.max_boxes
    agent_type = cfg.agent.name
    tests = cfg.get("tests", 20)
    make_gifs = cfg.get("make_gifs", False)

    env = _build_env(max_boxes=boxes, include_noop=cfg.environment.get("include_noop", False))

    if agent_type == "dqn":
        agent = _build_dqn(cfg, env)
        model_path = cfg.get("load_model", None) or _auto_find_checkpoint("dqn")
        if not model_path:
            raise FileNotFoundError("No DQN checkpoint found. Pass load_model in config or train first.")
        _load_weights_dqn(agent, model_path)
        print(f"Loaded DQN weights from: {model_path}")
    else:
        agent = _build_ppo(cfg, env)
        model_path = cfg.get("load_model", None) or _auto_find_checkpoint("ppo")
        if not model_path:
            raise FileNotFoundError("No PPO checkpoint found. Pass load_model in config or train first.")
        _load_weights_ppo(agent, model_path)
        print(f"Loaded PPO weights from: {model_path}")

    out_dir = Path("runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    bin_size = (cfg.environment.bin_size[0], cfg.environment.bin_size[1], cfg.environment.bin_size[2])
    test_sets = make_test_sets(
        seed=cfg.seed, n_episodes=tests, n_boxes=boxes, bin_size=bin_size
    )

    print("\nEvaluating Agent vs Heuristic:")
    agent_scores, heuristic_scores = [], []
    best_agent = (-1.0, None)
    best_heur  = (-1.0, None)

    for i, episode_boxes in enumerate(test_sets):
        env_seed = cfg.seed + i 
        agent_score = evaluate_agent_on_episode(
            agent,
            episode_boxes,
            env_seed=env_seed,
            generate_gif=False,
            gif_name=f"runs/best_agent_ep{i}.gif",
        )
        heur_score = evaluate_heuristic_on_episode(
            episode_boxes,
            env_seed=env_seed,
            generate_gif=False,
            gif_name=f"runs/best_heuristic_ep{i}.gif",
        )

        agent_scores.append(agent_score)
        heuristic_scores.append(heur_score)
        if agent_score > best_agent[0]:
            best_agent = (agent_score, i)
        if heur_score > best_heur[0]:
            best_heur = (heur_score, i)

        print(f"Test {i+1}: Agent = {agent_score:.2f}%, Heuristic = {heur_score:.2f}%")

    print("\nAgent Avg:", float(np.mean(agent_scores)))
    print("Heuristic Avg:", float(np.mean(heuristic_scores)))

    if make_gifs:
        best_heur_score, best_heur_idx = best_heur
        best_agent_score, best_agent_idx = best_agent

        print(f"\nGenerating GIF for best heuristic test (Episode {best_heur_idx+1}) "
              f"with {best_heur_score:.2f}% volume used...")
        evaluate_heuristic_on_episode(
            test_sets[best_heur_idx],
            env_seed=cfg.seed + best_heur_idx,
            generate_gif=True,
            gif_name="runs/heuristic_best.gif",
        )

        print(f"\nGenerating GIF for best {agent_type.upper()} test (Episode {best_agent_idx+1}) "
              f"with {best_agent_score:.2f}% volume used...")
        gif_file = "runs/dqn/agent_best.gif" if agent_type == "dqn" else "runs/ppo/agent_best.gif"
        evaluate_agent_on_episode(
            agent,
            test_sets[best_agent_idx],
            env_seed=cfg.seed + best_agent_idx,
            generate_gif=True,
            gif_name=gif_file,
        )

    try:
        env.close()
    except Exception:
        pass

    print("Evaluation finished.")
    return 0


# ------------------------------------------------------------------------------
# MAIN EXECUTION (HYDRA)
# ------------------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== ACTUAL CONFIGURATION ===")
    print(OmegaConf.to_yaml(cfg))
    print("==========================")
    
    if cfg.mode == "train":
        return cmd_train(cfg)
    elif cfg.mode == "evaluate":
        return cmd_evaluate(cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}. Use 'mode=train' or 'mode=evaluate'")


if __name__ == "__main__":
    main()