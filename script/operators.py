"""
Local Search Operators for TSP.

Four operators managed by the RL agent:
  1. Two-opt:    Reverse a tour segment (Croes, 1958)
  2. Relocate:   Move a city to a different position (Or, 1976)
  3. Swap:       Exchange two cities' positions
  4. Three-perm: Permute three randomly chosen cities

Each accepts optional augmented distances (for GLS) but returns true cost.
All return (new_tour, new_cost, improved: bool).
"""

import numpy as np
import random
from enum import IntEnum
from itertools import permutations
from typing import List, Tuple, Optional

from tsp_problem import compute_tour_cost


class Operator(IntEnum):
    """Local search operator identifiers."""
    TWO_OPT = 0
    RELOCATE = 1
    SWAP = 2
    THREE_PERM = 3


def two_opt_step(
    tour: List[int],
    dist_matrix: np.ndarray,
    augmented_cost: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[List[int], float, bool]:
    """
    Best-improvement 2-opt step. O(n^2).

    Tries all segment reversals, applies the single best improving move.
    """
    n = len(tour)
    d = augmented_cost if augmented_cost is not None else dist_matrix
    best_delta = -1e-10
    best_i, best_j = -1, -1

    for i in range(n - 1):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue  # Reversing entire tour is a no-op
            a, b = tour[i], tour[i + 1]
            c, d_node = tour[j], tour[(j + 1) % n]
            delta = d[a, c] + d[b, d_node] - d[a, b] - d[c, d_node]
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j

    if best_i >= 0:
        new_tour = (
            tour[: best_i + 1]
            + tour[best_i + 1 : best_j + 1][::-1]
            + tour[best_j + 1 :]
        )
        return new_tour, compute_tour_cost(new_tour, dist_matrix), True

    return tour, compute_tour_cost(tour, dist_matrix), False


def relocate_step(
    tour: List[int],
    dist_matrix: np.ndarray,
    augmented_cost: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[List[int], float, bool]:
    """
    Best-improvement relocate step. O(n^2).

    Removes each city and tries reinserting at every other position.
    FIX: insertion index is correctly adjusted after removal.
    """
    n = len(tour)
    d = augmented_cost if augmented_cost is not None else dist_matrix
    best_delta = -1e-10
    best_remove_idx = -1
    best_insert_after = -1

    for i in range(n):
        prev_i = tour[(i - 1) % n]
        city = tour[i]
        next_i = tour[(i + 1) % n]
        removal_saving = d[prev_i, next_i] - d[prev_i, city] - d[city, next_i]

        for j in range(n):
            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                continue
            a_node = tour[j]
            b_node = tour[(j + 1) % n]
            insertion_cost = (
                d[a_node, city] + d[city, b_node] - d[a_node, b_node]
            )
            delta = removal_saving + insertion_cost
            if delta < best_delta:
                best_delta = delta
                best_remove_idx = i
                best_insert_after = j

    if best_remove_idx >= 0:
        city = tour[best_remove_idx]
        new_tour = tour.copy()
        new_tour.pop(best_remove_idx)
        # Adjust: if insert-after was past the removed index, it shifted left
        insert_after = best_insert_after
        if best_insert_after > best_remove_idx:
            insert_after -= 1
        new_tour.insert(insert_after + 1, city)
        return new_tour, compute_tour_cost(new_tour, dist_matrix), True

    return tour, compute_tour_cost(tour, dist_matrix), False


def swap_step(
    tour: List[int],
    dist_matrix: np.ndarray,
    augmented_cost: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[List[int], float, bool]:
    """
    Best-improvement swap step. O(n^2).

    Handles adjacent and non-adjacent cases with correct delta formulas.
    """
    n = len(tour)
    d = augmented_cost if augmented_cost is not None else dist_matrix
    best_delta = -1e-10
    best_i, best_j = -1, -1

    for i in range(n):
        for j in range(i + 1, n):
            pi = (i - 1) % n
            ni = (i + 1) % n
            pj = (j - 1) % n
            nj = (j + 1) % n
            ci, cj = tour[i], tour[j]

            if ni == j or pj == i:  # Adjacent
                old = d[tour[pi], ci] + d[ci, cj] + d[cj, tour[nj]]
                new = d[tour[pi], cj] + d[cj, ci] + d[ci, tour[nj]]
            else:  # Non-adjacent
                old = (
                    d[tour[pi], ci] + d[ci, tour[ni]]
                    + d[tour[pj], cj] + d[cj, tour[nj]]
                )
                new = (
                    d[tour[pi], cj] + d[cj, tour[ni]]
                    + d[tour[pj], ci] + d[ci, tour[nj]]
                )

            delta = new - old
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j

    if best_i >= 0:
        new_tour = tour.copy()
        new_tour[best_i], new_tour[best_j] = new_tour[best_j], new_tour[best_i]
        return new_tour, compute_tour_cost(new_tour, dist_matrix), True

    return tour, compute_tour_cost(tour, dist_matrix), False


def three_perm_step(
    tour: List[int],
    dist_matrix: np.ndarray,
    augmented_cost: Optional[np.ndarray] = None,
    n_samples: int = 100,
    rng: Optional[random.Random] = None,
    **kwargs,
) -> Tuple[List[int], float, bool]:
    """
    Three-permutation operator.

    Samples random triples of positions, tries all 6 orderings.
    FIX: skips adjacent positions to avoid broken delta computation.
    Uses explicit RNG for reproducibility.
    """
    n = len(tour)
    d = augmented_cost if augmented_cost is not None else dist_matrix
    _rng = rng or random.Random()

    best_delta = -1e-10
    best_positions = None
    best_perm = None
    actual_samples = min(n_samples, n * (n - 1) * (n - 2) // 6)

    for _ in range(actual_samples):
        positions = sorted(_rng.sample(range(n), 3))
        i, j, k = positions

        # Skip if any two are adjacent (mod n)
        if (j - i) % n <= 1 or (k - j) % n <= 1 or (n + i - k) % n <= 1:
            continue

        cities = [tour[i], tour[j], tour[k]]
        preds = [tour[(i - 1) % n], tour[(j - 1) % n], tour[(k - 1) % n]]
        succs = [tour[(i + 1) % n], tour[(j + 1) % n], tour[(k + 1) % n]]

        current_cost = sum(
            d[preds[m], cities[m]] + d[cities[m], succs[m]] for m in range(3)
        )

        for perm in permutations(cities):
            perm_cost = sum(
                d[preds[m], perm[m]] + d[perm[m], succs[m]] for m in range(3)
            )
            delta = perm_cost - current_cost
            if delta < best_delta:
                best_delta = delta
                best_positions = positions
                best_perm = list(perm)

    if best_positions is not None:
        new_tour = tour.copy()
        for pos, city in zip(best_positions, best_perm):
            new_tour[pos] = city
        return new_tour, compute_tour_cost(new_tour, dist_matrix), True

    return tour, compute_tour_cost(tour, dist_matrix), False


def three_opt_step(
    tour: List[int],
    dist_matrix: np.ndarray,
    augmented_cost: Optional[np.ndarray] = None,
    **kwargs,
) -> Tuple[List[int], float, bool]:
    """
    First-improvement 3-opt step.

    Tries removing three edges and reconnecting the three resulting segments
    in all 7 non-trivial ways. Accepts the first improving move found.
    Complexity: O(n^3) worst case, but early-exit makes it faster in practice.
    """
    n = len(tour)
    d = augmented_cost if augmented_cost is not None else dist_matrix
    best_delta = -1e-10
    best_move = None

    for i in range(n - 2):
        for j in range(i + 2, n - 1):
            for k in range(j + 2, n + (0 if i > 0 else -1)):
                # Indices in the tour
                a, b = tour[i], tour[(i + 1) % n]
                c, dd_node = tour[j], tour[(j + 1) % n]
                e, f = tour[k % n], tour[(k + 1) % n]

                old_cost = d[a, b] + d[c, dd_node] + d[e, f]

                # Type 1: reverse segment B (equivalent to 2-opt on i,j)
                new1 = d[a, c] + d[b, dd_node] + d[e, f]
                delta = new1 - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("reverse_B", i, j, k)

                # Type 2: reverse segment C
                new2 = d[a, b] + d[c, e] + d[dd_node, f]
                delta = new2 - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("reverse_C", i, j, k)

                # Type 3: reverse both B and C
                new3 = d[a, c] + d[b, e] + d[dd_node, f]
                delta = new3 - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("reverse_BC", i, j, k)

                # Type 4: swap segments B and C (no reversal)
                new4 = d[a, dd_node] + d[e, b] + d[c, f]
                delta = new4 - old_cost
                if delta < best_delta:
                    best_delta = delta
                    best_move = ("swap_BC", i, j, k)

            # Early exit after finding first improvement for speed on large n
            if best_move is not None and n > 100:
                break
        if best_move is not None and n > 100:
            break

    if best_move is not None:
        move_type, i, j, k = best_move
        seg_A = tour[:i + 1]
        seg_B = tour[i + 1:j + 1]
        seg_C = tour[j + 1:k % n + 1] if k % n > j else tour[j + 1:]
        seg_D = tour[k % n + 1:] if k % n + 1 < n else []

        if move_type == "reverse_B":
            new_tour = seg_A + seg_B[::-1] + seg_C + seg_D
        elif move_type == "reverse_C":
            new_tour = seg_A + seg_B + seg_C[::-1] + seg_D
        elif move_type == "reverse_BC":
            new_tour = seg_A + seg_B[::-1] + seg_C[::-1] + seg_D
        elif move_type == "swap_BC":
            new_tour = seg_A + seg_C + seg_B + seg_D
        else:
            new_tour = tour

        if len(new_tour) == n:
            return new_tour, compute_tour_cost(new_tour, dist_matrix), True

    return tour, compute_tour_cost(tour, dist_matrix), False


# ── Dispatch tables ──────────────────────────────────────────────────────────

OPERATOR_FNS = {
    Operator.TWO_OPT: two_opt_step,
    Operator.RELOCATE: relocate_step,
    Operator.SWAP: swap_step,
    Operator.THREE_PERM: three_perm_step,
}

OPERATOR_NAMES = {
    Operator.TWO_OPT: "2-opt",
    Operator.RELOCATE: "relocate",
    Operator.SWAP: "swap",
    Operator.THREE_PERM: "3-perm",
}
