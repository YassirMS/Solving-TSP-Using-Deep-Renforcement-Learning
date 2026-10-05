"""
TSP Problem Representation.

Defines the TSP instance (coordinates + distance matrix), solution (tour +
penalties), tour cost computation, and nearest-neighbor heuristic.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class TSPInstance:
    """
    A TSP problem instance.

    Attributes:
        coords:      (n, 2) city positions in [0, 1]^2.
        n:           Number of cities.
        dist_matrix: (n, n) precomputed Euclidean distances.
    """
    coords: np.ndarray

    def __post_init__(self):
        self.n: int = len(self.coords)
        diff = self.coords[:, np.newaxis, :] - self.coords[np.newaxis, :, :]
        self.dist_matrix: np.ndarray = np.sqrt(np.sum(diff ** 2, axis=-1))

    @staticmethod
    def generate_random(n: int, seed: Optional[int] = None) -> "TSPInstance":
        """Generate random Euclidean TSP with cities in [0, 1]^2."""
        rng = np.random.RandomState(seed)
        return TSPInstance(coords=rng.uniform(0, 1, size=(n, 2)))


@dataclass
class TSPSolution:
    """A TSP tour with optional GLS penalty counters."""
    tour: List[int]
    cost: float
    penalties: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @staticmethod
    def edge_key(i: int, j: int) -> Tuple[int, int]:
        """Canonical undirected edge: (min, max)."""
        return (min(i, j), max(i, j))

    def get_edges(self) -> List[Tuple[int, int]]:
        """All tour edges as canonical pairs."""
        n = len(self.tour)
        return [
            self.edge_key(self.tour[idx], self.tour[(idx + 1) % n])
            for idx in range(n)
        ]

    def get_penalty(self, i: int, j: int) -> int:
        return self.penalties.get(self.edge_key(i, j), 0)

    def increment_penalty(self, i: int, j: int):
        key = self.edge_key(i, j)
        self.penalties[key] = self.penalties.get(key, 0) + 1


def compute_tour_cost(tour: List[int], dist_matrix: np.ndarray) -> float:
    """Total tour length around the cycle (vectorized)."""
    indices = np.array(tour)
    next_indices = np.roll(indices, -1)
    return float(np.sum(dist_matrix[indices, next_indices]))


def nearest_neighbor_tour(instance: TSPInstance, start: int = 0) -> List[int]:
    """
    Nearest Neighbor construction heuristic.

    Greedily visits the closest unvisited city.
    Complexity: O(n^2). Typical quality: 20-25% above optimal.
    """
    n = instance.n
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True

    for _ in range(n - 1):
        current = tour[-1]
        dists = np.where(visited, np.inf, instance.dist_matrix[current])
        best_next = int(np.argmin(dists))
        tour.append(best_next)
        visited[best_next] = True

    return tour
