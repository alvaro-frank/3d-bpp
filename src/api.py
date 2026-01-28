# ==============================================================================
# FILE: api.py
# DESCRIPTION: FastAPI implementation for the 3D-BPP project.
#              Provides endpoints to pack boxes using trained PPO/DQN agents
#              with support for recursive splitting generation.
# ==============================================================================
import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# ------------------------------------------------------------------------------
# ENVIRONMENT SETUP
# ------------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "" 
device = torch.device("cpu")

from environment.packing_env import PackingEnv
from agents.ppo_agent import PPOAgent, PPOConfig
from agents.dqn_agent import DQNAgent
from utils.box_generator import generate_boxes #
from environment.box import Box as BinBox #

app = FastAPI(title="3D-BPP Recursive Splitting API")

# ------------------------------------------------------------------------------
# DATA MODELS
# ------------------------------------------------------------------------------

class RecursivePackRequest(BaseModel):
    bin_size: List[int] = [10, 10, 10]
    num_boxes: int = 15
    seed: int = 42

def load_agent(agent_type: str, bin_size: tuple, model_path: str):
    """
    Initialize and load a trained agent onto the CPU.

    Args:
        agent_type (str): Type of agent ('ppo' or 'dqn').
        bin_size (tuple): Bin dimensions for state space initialization.
        model_path (str): Path to the saved .pt model weights.

    Returns:
        Agent: The loaded PPO or DQN agent instance.
    """
    env = PackingEnv(bin_size=bin_size, max_boxes=1)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    map_size = (bin_size[0], bin_size[1])

    if agent_type == "ppo":
        ppo_cfg = PPOConfig(device="cpu")
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, map_size=map_size, config=ppo_cfg)
    else:
        agent = DQNAgent(state_dim=obs_dim, action_dim=act_dim, map_size=map_size, device="cpu")
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
        agent.model.load_state_dict(state_dict)
        agent.model.eval()
    
    return agent

@app.post("/pack/{agent_type}")
async def pack_recursive(agent_type: str, request: RecursivePackRequest):
    if agent_type not in ["ppo", "dqn"]:
        raise HTTPException(status_code=400, detail="Agente inválido. Use 'ppo' ou 'dqn'.")

    try:
        # 1. Gerar caixas via Recursive Splitting (Structured)
        # Este método garante que a soma dos volumes das caixas é igual ao volume do bin.
        raw_boxes = generate_boxes(
            bin_size=request.bin_size, 
            num_items=request.num_boxes, 
            seed=request.seed, 
            structured=True
        )
        
        # Converter para o formato esperado pelo ambiente (lista de dicionários)
        boxes_to_pack = [{"w": b[0], "d": b[1], "h": b[2]} for b in raw_boxes]
        boxes_to_pack.sort(key=lambda b: b["w"] * b["d"] * b["h"], reverse=True)

        # 2. Configurar Ambiente e Agente
        model_path = f"runs/{agent_type}/models/{agent_type}_latest.pt"
        agent = load_agent(agent_type, tuple(request.bin_size), model_path)
        
        env = PackingEnv(bin_size=tuple(request.bin_size), max_boxes=request.num_boxes)
        state = env.reset(with_boxes=boxes_to_pack)
        
        # 3. Execução do Empacotamento
        packing_plan = []
        done = False
        
        while not done:
            mask = env.valid_action_mask()
            if hasattr(agent, "epsilon"): agent.epsilon = 0.0
            
            action = agent.get_action(state, env.action_space, mask=mask)
            state, reward, done, info = env.step(action)
            
            if env.bin.boxes:
                last_box = env.bin.boxes[-1]
                # Apenas adicionamos se for a caixa correspondente ao passo atual e não um skip
                if len(packing_plan) < len(env.bin.boxes):
                    packing_plan.append({
                        "id": last_box.id,
                        "position": last_box.position,
                        "rotation": last_box.rotation_type,
                        "original_dims": [last_box.width, last_box.depth, last_box.height],
                        "rotated_dims": last_box.get_rotated_size()
                    })

        return {
            "agent": agent_type,
            "total_generated": len(boxes_to_pack),
            "successfully_packed": len(packing_plan),
            "volume_utilization": env.get_placed_boxes_volume() / env.bin.bin_volume() * 100,
            "plan": packing_plan
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))