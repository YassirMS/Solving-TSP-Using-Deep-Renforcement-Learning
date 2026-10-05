"""
Guided Local Search (GLS) Penalty Mechanism (Voudouris & Tsang, 1999).

Escapes local optima by penalizing edges in the stuck solution.

Augmented objective:  h(s) = g(s) + λ · Σ p_i · I_i(s)
Utility function:     util(i,j) = I(i,j in s) · d(i,j) / (1 + p(i,j))
Lambda:               λ = α · cost(current) / n
"""

import numpy as np
from typing import List, Tuple

from tsp_problem import TSPInstance, TSPSolution


class GLSPenaltyManager:
    """
    Manages per-edge penalty counters and augmented distance computation.

    Parameters:
        instance: TSP instance (provides distance matrix and n).
        alpha:    λ = alpha · cost / n. Default 0.3 (Voudouris & Tsang).
    """

    def __init__(self, instance: TSPInstance, alpha: float = 0.3):
        self.instance = instance
        self.alpha = alpha
        self.n = instance.n
        self.penalties = np.zeros((self.n, self.n), dtype=np.float64)

    def compute_lambda(self, current_cost: float) -> float:
        """λ = α · cost / n."""
        return self.alpha * current_cost / self.n

    def compute_augmented_distances(self, current_cost: float) -> np.ndarray:
        """d'[i][j] = d[i][j] + λ · p[i][j]."""
        lam = self.compute_lambda(current_cost)
        return self.instance.dist_matrix + lam * self.penalties

    def compute_utilities(
        self, solution: TSPSolution
    ) -> List[Tuple[Tuple[int, int], float]]:
        """Edge utilities sorted descending. Highest = best penalty candidate."""
        n = len(solution.tour)
        utilities = []
        for idx in range(n):
            i = solution.tour[idx]
            j = solution.tour[(idx + 1) % n]
            edge_cost = self.instance.dist_matrix[i, j]
            pen = self.penalties[i, j]
            utilities.append(((i, j), edge_cost / (1.0 + pen)))
        utilities.sort(key=lambda x: -x[1])
        return utilities

    def apply_penalties(self, solution: TSPSolution, n_penalties: int = 1):
        """Penalize the top-utility edges in the current solution."""
        utilities = self.compute_utilities(solution)
        for k in range(min(n_penalties, len(utilities))):
            i, j = utilities[k][0]
            self.penalties[i, j] += 1
            self.penalties[j, i] += 1

    def get_avg_penalty(self) -> float:
        """Mean penalty level across all edges."""
        if self.penalties.sum() == 0:
            return 0.0
        return float(np.mean(self.penalties))

    def get_max_penalty(self) -> float:
        """Maximum penalty on any edge."""
        return float(np.max(self.penalties))

    def reset(self):
        """Reset all penalties for a new episode."""
        self.penalties = np.zeros((self.n, self.n), dtype=np.float64)
