"""
L2GLS Environment — Gym-like interface for the two-level decision MDP.

Each step:
  1. Agent observes 11-dim state
  2. Agent decides: penalty level (0/1/3 edges) + operator (2opt/relocate/swap/3perm)
  3. Env applies GLS penalties, then executes operator on augmented landscape
  4. Env returns: next_state, reward, done, info

Reward: continuous, proportional to cost improvement / initial_cost.
  New global best:   +10 · (old_best - new_best) / init + 0.5
  Local improvement:  +1 · (old_cost - new_cost) / init + 0.05
  No improvement:    -0.01
"""

import numpy as np
import random as py_random
from typing import Tuple, Dict, Optional, List

from config import EnvConfig, N_OPERATORS, PENALTY_LEVELS
from tsp_problem import (
    TSPInstance, TSPSolution, compute_tour_cost, nearest_neighbor_tour,
)
from operators import Operator, OPERATOR_FNS, OPERATOR_NAMES
from gls_penalties import GLSPenaltyManager
from state_representation import build_state, STATE_DIM


class L2GLSEnv:
    """
    Gym-style environment for one L2GLS search episode.

    Usage:
        env = L2GLSEnv(config)
        state = env.reset(instance) or env.reset_random(n, seed)
        while not done:
            state, reward, done, info = env.step(pen_action, op_action)
    """

    def __init__(self, config: EnvConfig):
        self.config = config
        self._rng = py_random.Random()

        # Episode state (set in reset)
        self.instance: Optional[TSPInstance] = None
        self.penalty_mgr: Optional[GLSPenaltyManager] = None
        self.tour: List[int] = []
        self.current_cost: float = 0.0
        self.initial_cost: float = 0.0
        self.best_tour: List[int] = []
        self.best_cost: float = float("inf")
        self.round_num: int = 0
        self.steps_no_improve: int = 0
        self.penalty_rounds: int = 0
        self.operator_last_improved: List[int] = [0] * N_OPERATORS

        # Per-episode stats
        self.operator_counts: Dict[str, int] = {}
        self.operator_improvements: Dict[str, int] = {}

    def reset(
        self,
        instance: TSPInstance,
        start_city: int = 0,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Reset with a specific TSP instance. Returns initial state."""
        if seed is not None:
            self._rng = py_random.Random(seed)

        self.instance = instance
        self.penalty_mgr = GLSPenaltyManager(
            instance, alpha=self.config.alpha_gls
        )
        self.tour = nearest_neighbor_tour(instance, start=start_city)
        self.current_cost = compute_tour_cost(self.tour, instance.dist_matrix)
        self.initial_cost = self.current_cost
        self.best_tour = self.tour.copy()
        self.best_cost = self.current_cost

        self.round_num = 0
        self.steps_no_improve = 0
        self.penalty_rounds = 0
        self.operator_last_improved = [0] * N_OPERATORS

        self.operator_counts = {name: 0 for name in OPERATOR_NAMES.values()}
        self.operator_improvements = {name: 0 for name in OPERATOR_NAMES.values()}

        return self._get_state()

    def reset_random(self, n: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate random instance and reset."""
        instance = TSPInstance.generate_random(n, seed=seed)
        return self.reset(instance, seed=seed)

    def step(
        self, penalty_action: int, operator_action: int
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one two-level decision step.

        Args:
            penalty_action: index into PENALTY_LEVELS (0→none, 1→1 edge, 2→3 edges)
            operator_action: Operator enum (0→2opt, 1→relocate, 2→swap, 3→3perm)
        """
        assert self.instance is not None, "Call reset() first"

        n_penalties = PENALTY_LEVELS[penalty_action]
        operator = Operator(operator_action)
        op_name = OPERATOR_NAMES[operator]
        self.operator_counts[op_name] = (
            self.operator_counts.get(op_name, 0) + 1
        )

        old_cost = self.current_cost
        old_best = self.best_cost

        # ── Decision 1: Apply GLS penalties ──
        if n_penalties > 0:
            sol = TSPSolution(tour=self.tour, cost=self.current_cost)
            self.penalty_mgr.apply_penalties(sol, n_penalties=n_penalties)
            self.penalty_rounds += 1

        # ── Augmented landscape ──
        augmented = self.penalty_mgr.compute_augmented_distances(
            self.current_cost
        )

        # ── Decision 2: Execute local search operator ──
        op_fn = OPERATOR_FNS[operator]
        kwargs = {}
        if operator == Operator.THREE_PERM:
            kwargs["n_samples"] = self.config.three_perm_samples
            kwargs["rng"] = self._rng

        new_tour, new_cost, improved = op_fn(
            self.tour, self.instance.dist_matrix, augmented, **kwargs
        )

        # ── Update search state ──
        new_global_best = False
        if new_cost < self.best_cost - 1e-10:
            self.best_tour = new_tour.copy()
            self.best_cost = new_cost
            new_global_best = True
            self.steps_no_improve = 0
            self.operator_last_improved[int(operator)] = self.round_num
            self.operator_improvements[op_name] = (
                self.operator_improvements.get(op_name, 0) + 1
            )
        elif improved and new_cost < self.current_cost - 1e-10:
            self.steps_no_improve = 0
            self.operator_last_improved[int(operator)] = self.round_num
            self.operator_improvements[op_name] = (
                self.operator_improvements.get(op_name, 0) + 1
            )
        else:
            self.steps_no_improve += 1

        if improved:
            self.tour = new_tour
            self.current_cost = new_cost

        # ── Reward (continuous, magnitude-aware) ──
        safe_init = max(self.initial_cost, 1e-8)
        if new_global_best:
            reward = 10.0 * (old_best - self.best_cost) / safe_init + 0.5
        elif improved and new_cost < old_cost - 1e-10:
            reward = 1.0 * (old_cost - new_cost) / safe_init + 0.05
        else:
            reward = -0.01

        # ── Termination ──
        self.round_num += 1
        done = (
            self.round_num >= self.config.max_rounds
            or self.steps_no_improve >= self.config.stagnation_limit * 3
        )

        info = {
            "operator": op_name,
            "n_penalties": n_penalties,
            "improved": improved,
            "new_global_best": new_global_best,
            "current_cost": self.current_cost,
            "best_cost": self.best_cost,
            "gap_to_initial": (
                (self.best_cost - self.initial_cost) / self.initial_cost
            ),
        }
        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        return build_state(
            current_cost=self.current_cost,
            best_cost=self.best_cost,
            initial_cost=self.initial_cost,
            steps_no_improve=self.steps_no_improve,
            penalty_manager=self.penalty_mgr,
            penalty_rounds=self.penalty_rounds,
            operator_last_improved=self.operator_last_improved,
            round_num=self.round_num,
            max_rounds=self.config.max_rounds,
            stagnation_limit=self.config.stagnation_limit,
            n_cities=self.instance.n,
        )

    @property
    def state_dim(self) -> int:
        return STATE_DIM
