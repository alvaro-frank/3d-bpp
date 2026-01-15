# Defaults (override on the command line)
AGENT ?= ppo        # dqn|ppo
EPISODES ?= 200
BOXES ?= 50
SEED ?= 41
MODEL ?=
PORT ?= 5000

PYTHON := py -3.10
VENV_BIN := .venv\Scripts
PIP := $(VENV_BIN)\pip.exe
PY  := $(VENV_BIN)\python.exe
MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )

# ---------- Targets ----------
all: setup train evaluate ## Create venv, train and evaluate agent

setup: ## Create virtualenv and install minimal deps
	$(MKVENV)
	$(PY) -m pip install -r src/requirements.txt
	@echo ✅ venv ready

train: ## Train agent: make train AGENT=dqn|ppo EPISODES=200 BOXES=50 SEED=41
	$(PY) src/main.py train --agent $(AGENT) --episodes $(EPISODES) --boxes $(BOXES) --seed $(SEED)

evaluate: ## Evaluate agent: make evaluate AGENT=dqn|ppo MODEL=path/to.ckpt TESTS=20 BOXES=50 SEED=41
	$(PY) src/main.py evaluate --agent $(AGENT) $(if $(MODEL),--model "$(MODEL)",) --tests $(TESTS) --boxes $(BOXES) --seed $(SEED) --gifs

mlflow: ## Launch MLflow UI
	$(VENV_BIN)\mlflow ui --port $(PORT)