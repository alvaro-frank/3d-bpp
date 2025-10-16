# Defaults (override on the command line)
AGENT ?= ppo        # dqn|ppo
EPISODES ?= 200
BOXES ?= 50
SEED ?= 41
MODEL ?=

.DEFAULT_GOAL := help

PYTHON := py -3.10
VENV_BIN := .venv\Scripts
PIP := $(VENV_BIN)\pip.exe
PY  := $(VENV_BIN)\python.exe
SET_PYTHONPATH := set PYTHONPATH=src&&
MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )

REQ_MIN := numpy matplotlib imageio gym==0.21.0 torch

# ---------- Targets ----------
all: venv train evaluate ## Create venv, train and evaluate agent

venv: ## Create virtualenv and install minimal deps
	$(MKVENV)
	$(PY) -m pip install "pip<24.1" wheel
	$(PY) -m pip install numpy matplotlib imageio gym==0.21.0 torch
	@echo ✅ venv ready

train: ## Train agent: make train AGENT=dqn|ppo EPISODES=200 BOXES=50 SEED=41
	$(SET_PYTHONPATH) $(PY) src/main.py train --agent $(AGENT) --episodes $(EPISODES) --boxes $(BOXES) --seed $(SEED)

evaluate: ## Evaluate agent: make evaluate AGENT=dqn|ppo MODEL=path/to.ckpt TESTS=20 BOXES=50 SEED=41
	$(SET_PYTHONPATH) $(PY) src/main.py evaluate --agent $(AGENT) $(if $(MODEL),--model "$(MODEL)",) --tests $(TESTS) --boxes $(BOXES) --seed $(SEED) --gifs