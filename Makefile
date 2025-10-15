# Makefile (Windows + Linux/Mac compatible, simple)

.DEFAULT_GOAL := help

# ---------- OS detection & per-OS settings ----------
ifeq ($(OS),Windows_NT)
  PYTHON := py -3.10
  VENV_BIN := .venv\Scripts
  PIP := $(VENV_BIN)\pip.exe
  PY  := $(VENV_BIN)\python.exe
  SET_PYTHONPATH := set PYTHONPATH=src&&
  MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )
else
  PYTHON := python3
  VENV_BIN := .venv/bin
  PIP := $(VENV_BIN)/pip
  PY  := $(VENV_BIN)/python
  SET_PYTHONPATH :=
  MKVENV := test -d .venv || $(PYTHON) -m venv .venv
endif

REQ_MIN := numpy matplotlib imageio gym==0.21.0 torch

# ---------- Targets ----------
.PHONY: help
help: ## Show available targets
	@echo:
	@echo  Targets:
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@echo:

.PHONY: venv
venv: ## Create virtualenv and install minimal deps
	$(MKVENV)
	$(PY) -m pip install "pip<24.1" wheel
	$(PY) -m pip install numpy matplotlib imageio gym==0.21.0 torch
	@echo ✅ venv ready

.PHONY: train-dqn
train-dqn: venv ## Train DQN agent (defaults inside your code)
	$(SET_PYTHONPATH) $(PY) src/main.py dqn_agent

.PHONY: train-ppo
train-ppo: venv ## Train PPO agent (defaults inside your code)
	$(SET_PYTHONPATH) $(PY) src/main.py ppo_agent

.PHONY: clean
clean: ## Remove caches
	-$(RM) .pytest_cache 2>nul || true
	-$(RM) .mypy_cache 2>nul || true
	-$(RM) .ruff_cache 2>nul || true

.PHONY: distclean
distclean: clean ## Remove venv and run artifacts
	-$(RM) .venv 2>nul || true
	-$(RM) runs 2>nul || true