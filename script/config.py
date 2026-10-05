"""
L2GLS Configuration — Centralized hyperparameters and hardware-aware defaults.

Supports four agent types:
  - Tabular Q-Learning (no dependencies beyond numpy)
  - DQN (Deep Q-Network, requires PyTorch)
  - REINFORCE (Policy Gradient, requires PyTorch)
  - PPO (Proximal Policy Optimization, requires PyTorch)
"""

from dataclasses import dataclass, field
from typing import List

# ── Action Space Constants ──────────────────────────────────────────────────

N_OPERATORS = 4              # 2-opt, relocate, swap, three-perm
PENALTY_LEVELS = [0, 1, 3]  # GLS edges to penalize per step
N_PENALTY_ACTIONS = len(PENALTY_LEVELS)

# Total flat action space = penalty levels × operators = 12
# Used by Tabular, DQN, and REINFORCE (flat action decoding).
# PPO uses the hierarchical action space (two heads) directly.
N_FLAT_ACTIONS = N_PENALTY_ACTIONS * N_OPERATORS


# ── Environment ─────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    """TSP environment and local search parameters."""
    n_cities: int = 50
    alpha_gls: float = 0.3
    max_rounds: int = 200
    stagnation_limit: int = 10
    three_perm_samples: int = 100


# ── Agent Configs ───────────────────────────────────────────────────────────

@dataclass
class TabularConfig:
    """Tabular Q-Learning hyperparameters."""
    lr: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    n_bins: int = 5


@dataclass
class DQNConfig:
    """Deep Q-Network hyperparameters."""
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    buffer_size: int = 10000
    batch_size: int = 64
    target_update_freq: int = 100
    device: str = "auto"


@dataclass
class REINFORCEConfig:
    """REINFORCE (policy gradient) hyperparameters."""
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    lr: float = 1e-3
    gamma: float = 0.99
    max_grad_norm: float = 1.0
    baseline_alpha: float = 0.01
    device: str = "auto"


@dataclass
class PPOConfig:
    """PPO-Clip hyperparameters tuned for GTX 1650 (4GB VRAM, 8GB RAM)."""
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    n_epochs: int = 4
    batch_size: int = 32
    device: str = "auto"


# ── Training ────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    """Top-level training configuration."""
    env: EnvConfig = field(default_factory=EnvConfig)
    agent_type: str = "ppo"  # "tabular", "dqn", "reinforce", "ppo"

    tabular: TabularConfig = field(default_factory=TabularConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    reinforce: REINFORCEConfig = field(default_factory=REINFORCEConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    n_episodes: int = 500
    mixed_batch_sizes: List[int] = field(
        default_factory=lambda: [20, 50, 100]
    )
    seed: int = 42
    log_interval: int = 10
    checkpoint_interval: int = 50
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
