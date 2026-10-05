# L2GLS — Learning to Guide Local Search for TSP

## Quick Start

```bash
# Full pipeline: train DQN, REINFORCE, PPO + benchmark
python run_all.py

# Benchmark only (load from checkpoints/)
python run_all.py --skip_training

# Custom sizes and more instances
python run_all.py --random_sizes 20 50 100 200 500 --n_instances 30

# Only PPO
python run_all.py --agents ppo --n_instances 50
```

## Files

| File | Role |
|------|------|
| `config.py` | Centralized hyperparameters |
| `tsp_problem.py` | TSP instance, distance matrix, nearest neighbor |
| `operators.py` | 2-opt, 3-opt, relocate, swap, three-permutation |
| `gls_penalties.py` | GLS penalty mechanism (augmented cost h) |
| `state_representation.py` | 11-dim state vector for RL agent |
| `agents.py` | DQN, REINFORCE, PPO, Tabular Q-learning |
| `environment.py` | Gym-like L2GLS environment |
| `training.py` | Agent-agnostic training loop |
| `testing.py` | Baselines (NN, 2-opt, 3-opt, GLS+2-opt, GLS+3-opt) |
| `run_all.py` | **Main deliverable**: train + benchmark (random + TSPLIB) |

## Benchmark Methods

1. Nearest Neighbor (NN)
2. Pure 2-opt
3. Pure 3-opt
4. GLS + 2-opt
5. GLS + 3-opt
6. L2GLS (DQN)
7. L2GLS (REINFORCE)
8. L2GLS (PPO — hierarchical action space)

## TSPLIB Setup

Download `.tsp` files into `tsplib_data/`:
```bash
mkdir tsplib_data
# Get EUC_2D instances from: https://github.com/mastqe/tsplib
```

## Output

All results saved in `results/`:
- `random_benchmark.csv` — per-instance costs and times
- `random_summary.csv` — aggregated stats per method × size
- `tsplib_benchmark.csv` — per-instance gaps to known optima
- `training_times.json` — wall-clock training duration per agent

## Training Episodes (tuned per agent)

| Agent | Episodes | Rationale |
|-------|----------|-----------|
| DQN | 1000 | Replay buffer needs time to fill |
| REINFORCE | 1500 | High variance → needs more samples |
| PPO | 800 | Most sample-efficient (GAE + multi-epoch) |
