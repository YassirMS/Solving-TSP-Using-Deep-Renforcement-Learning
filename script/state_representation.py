"""
State Representation — 11-dimensional feature vector for the RL agent.

Encodes search progress, stagnation, penalty landscape, instance scale,
and per-operator recency. All features normalized to ~[0, 1].

Dimensions:
  [0]     normalized_cost    current_cost / initial_cost
  [1]     normalized_best    best_cost / initial_cost
  [2]     gap                (current - best) / best, clipped [0, 1]
  [3]     stagnation         steps_no_improve / stagnation_limit, clipped [0, 1]
  [4]     avg_penalty        mean(penalties) / 10, clipped [0, 1]
  [5]     penalty_rounds     penalty_rounds / max_rounds, clipped [0, 1]
  [6]     instance_scale     n_cities / 300, clipped [0, 1]
  [7..10] op_recency         per-operator (round - last_improved) / max_rounds
"""

import numpy as np
from typing import List

from config import N_OPERATORS
from gls_penalties import GLSPenaltyManager


STATE_DIM = 7 + N_OPERATORS  # 11


def build_state(
    current_cost: float,
    best_cost: float,
    initial_cost: float,
    steps_no_improve: int,
    penalty_manager: GLSPenaltyManager,
    penalty_rounds: int,
    operator_last_improved: List[int],
    round_num: int,
    max_rounds: int,
    stagnation_limit: int,
    n_cities: int,
) -> np.ndarray:
    """Build 11-dim float32 state vector."""
    safe_init = max(initial_cost, 1e-8)
    safe_best = max(best_cost, 1e-8)
    safe_max = max(max_rounds, 1)
    safe_stag = max(stagnation_limit, 1)

    features = [
        current_cost / safe_init,
        best_cost / safe_init,
        min((current_cost - best_cost) / safe_best, 1.0),
        min(steps_no_improve / safe_stag, 1.0),
        min(penalty_manager.get_avg_penalty() / 10.0, 1.0),
        min(penalty_rounds / safe_max, 1.0),
        min(n_cities / 300.0, 1.0),
    ]
    for op in range(N_OPERATORS):
        steps_since = round_num - operator_last_improved[op]
        features.append(min(1.0, steps_since / safe_max))

    return np.array(features, dtype=np.float32)
