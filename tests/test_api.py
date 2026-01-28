# ==============================================================================
# FILE: tests/test_api.py
# DESCRIPTION: Integration tests for the FastAPI-based 3D-BPP API.
#              Verifies endpoint connectivity, request validation, and 
#              successful packing rollout execution.
# ==============================================================================

import pytest
from httpx import AsyncClient
import os

# Adiciona o diretório src ao path para garantir que as importações funcionem
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from httpx import AsyncClient, ASGITransport
from api import app

# ------------------------------------------------------------------------------
# TEST CONFIGURATION
# ------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pack_recursive_ppo_success():
    """
    Test the PPO recursive packing endpoint with a valid request.
    Verifies that the agent can successfully process a perfect-fit scenario.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "bin_size": [10, 10, 10],
            "num_boxes": 5,
            "seed": 42
        }
        # Realiza o pedido para o endpoint PPO
        response = await ac.post("/pack/ppo", json=payload)
        
    assert response.status_code == 200
    data = response.json()
    
    # Validação da estrutura de resposta
    assert data["agent"] == "ppo"
    assert "successfully_packed" in data
    assert "volume_utilization" in data
    assert isinstance(data["plan"], list)
    assert len(data["plan"]) <= data["total_generated"]

@pytest.mark.asyncio
async def test_pack_recursive_dqn_success():
    """
    Test the DQN recursive packing endpoint with a valid request.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "bin_size": [10, 10, 10],
            "num_boxes": 3,
            "seed": 123
        }
        response = await ac.post("/pack/dqn", json=payload)
        
    assert response.status_code == 200
    data = response.json()
    assert data["agent"] == "dqn"
    assert data["total_generated"] == 3

@pytest.mark.asyncio
async def test_invalid_agent_type():
    """
    Test that the API correctly rejects an invalid agent type in the URL.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {"num_boxes": 5}
        # 'random_agent' não existe no código
        response = await ac.post("/pack/random_agent", json=payload)
        
    assert response.status_code == 400
    assert "Agente inválido" in response.json()["detail"]

@pytest.mark.asyncio
async def test_request_validation_error():
    """
    Test that the API rejects requests with invalid data types.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "bin_size": "invalid_size",  # Deveria ser uma lista
            "num_boxes": "five"          # Deveria ser um inteiro
        }
        response = await ac.post("/pack/ppo", json=payload)
        
    assert response.status_code == 422  # Unprocessable Entity (FastAPI standard)