"""
Testing and Benchmarking for L2GLS — Works with any agent type.

Usage:
    python testing.py --agent ppo --checkpoint checkpoints/PPO_final.pt
    python testing.py --agent tabular --checkpoint checkpoints/TABULAR_final.npz
    python testing.py --agent reinforce --checkpoint checkpoints/REINFORCE_final.pt
    python testing.py --agent dqn --checkpoint checkpoints/DQN_final.pt
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional, Dict

from config import (
    EnvConfig, TrainConfig,
    TabularConfig, DQNConfig, REINFORCEConfig, PPOConfig,
)
from tsp_problem import (
    TSPInstance, TSPSolution, compute_tour_cost, nearest_neighbor_tour,
)
from operators import two_opt_step, three_opt_step
from gls_penalties import GLSPenaltyManager
from environment import L2GLSEnv
from agents import BaseAgent, create_agent


# ═══════════════════════════════════════════════════════════════════════════════
# Single Instance Solver
# ═══════════════════════════════════════════════════════════════════════════════

def solve_instance(
    agent: BaseAgent,
    instance: TSPInstance,
    config: EnvConfig,
    n_restarts: int = 5,
) -> Tuple[List[int], float, Dict]:
    """Solve one TSP with multi-restart greedy policy."""
    env = L2GLSEnv(config)
    best_tour, best_cost = None, float("inf")
    all_costs = []

    for r in range(n_restarts):
        state = env.reset(instance, start_city=r % instance.n, seed=r)
        done = False
        while not done:
            pen_act, op_act = agent.select_action_greedy(state)
            state, _, done, _ = env.step(pen_act, op_act)
        all_costs.append(env.best_cost)
        if env.best_cost < best_cost:
            best_cost = env.best_cost
            best_tour = env.best_tour.copy()

    return best_tour, best_cost, {
        "best": best_cost,
        "mean": float(np.mean(all_costs)),
        "std": float(np.std(all_costs)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════════

def solve_pure_2opt(instance: TSPInstance, max_iters: int = 200) -> float:
    tour = nearest_neighbor_tour(instance)
    cost = compute_tour_cost(tour, instance.dist_matrix)
    for _ in range(max_iters):
        tour, cost, improved = two_opt_step(tour, instance.dist_matrix)
        if not improved:
            break
    return cost


def solve_gls_2opt(instance, max_rounds=200, stag_limit=5) -> float:
    pmgr = GLSPenaltyManager(instance, alpha=0.3)
    tour = nearest_neighbor_tour(instance)
    cost = compute_tour_cost(tour, instance.dist_matrix)
    best, stag = cost, 0
    for _ in range(max_rounds):
        aug = pmgr.compute_augmented_distances(cost)
        tour, cost, improved = two_opt_step(tour, instance.dist_matrix, aug)
        if cost < best - 1e-10:
            best, stag = cost, 0
        else:
            stag += 1
        if stag >= stag_limit:
            pmgr.apply_penalties(TSPSolution(tour=tour, cost=cost), n_penalties=1)
            stag = 0
    return best


def solve_pure_3opt(instance: TSPInstance, max_iters: int = 200) -> float:
    """Pure 3-opt local search (no GLS)."""
    tour = nearest_neighbor_tour(instance)
    cost = compute_tour_cost(tour, instance.dist_matrix)
    for _ in range(max_iters):
        tour, cost, improved = three_opt_step(tour, instance.dist_matrix)
        if not improved:
            break
    return cost


def solve_gls_3opt(instance, max_rounds=200, stag_limit=5) -> float:
    """GLS with 3-opt as the local search operator."""
    pmgr = GLSPenaltyManager(instance, alpha=0.3)
    tour = nearest_neighbor_tour(instance)
    cost = compute_tour_cost(tour, instance.dist_matrix)
    best, stag = cost, 0
    for _ in range(max_rounds):
        aug = pmgr.compute_augmented_distances(cost)
        tour, cost, improved = three_opt_step(tour, instance.dist_matrix, aug)
        if cost < best - 1e-10:
            best, stag = cost, 0
        else:
            stag += 1
        if stag >= stag_limit:
            pmgr.apply_penalties(TSPSolution(tour=tour, cost=cost), n_penalties=1)
            stag = 0
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark(
    agent: BaseAgent,
    agent_name: str = "L2GLS",
    n: int = 50,
    n_instances: int = 10,
    seed: int = 1000,
    config: Optional[EnvConfig] = None,
    n_restarts: int = 5,
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Compare L2GLS agent vs baselines: NN, 2-opt, GLS+2-opt."""
    if config is None:
        config = EnvConfig(n_cities=n)

    l2gls_label = f"L2GLS ({agent_name})"
    names = ["NN", "2-opt", "GLS+2-opt", l2gls_label]
    results = {m: {"costs": [], "times": []} for m in names}

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  BENCHMARK: {n} cities, {n_instances} instances, agent={agent_name}")
        print(f"{'=' * 70}")
        print(f"  {'#':>3} | {'NN':>8} | {'2-opt':>8} | {'GLS':>8} | {agent_name:>8}")
        print(f"  {'─'*3}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*8}")

    for i in range(n_instances):
        inst = TSPInstance.generate_random(n, seed=seed + i)

        t0 = time.time()
        nn_cost = compute_tour_cost(nearest_neighbor_tour(inst), inst.dist_matrix)
        results["NN"]["costs"].append(nn_cost)
        results["NN"]["times"].append(time.time() - t0)

        t0 = time.time()
        c2 = solve_pure_2opt(inst)
        results["2-opt"]["costs"].append(c2)
        results["2-opt"]["times"].append(time.time() - t0)

        t0 = time.time()
        cg = solve_gls_2opt(inst, max_rounds=config.max_rounds)
        results["GLS+2-opt"]["costs"].append(cg)
        results["GLS+2-opt"]["times"].append(time.time() - t0)

        t0 = time.time()
        _, cl, _ = solve_instance(agent, inst, config, n_restarts)
        results[l2gls_label]["costs"].append(cl)
        results[l2gls_label]["times"].append(time.time() - t0)

        if verbose:
            print(f"  {i+1:3d} | {nn_cost:8.3f} | {c2:8.3f} | {cg:8.3f} | {cl:8.3f}")

    if verbose:
        best_m = min(names, key=lambda m: np.mean(results[m]["costs"]))
        best_v = np.mean(results[best_m]["costs"])

        print(f"\n{'─' * 70}")
        print(f"  {'Method':<18} | {'Mean':>8} | {'Std':>7} | {'Time':>7} | {'Gap':>7}")
        print(f"  {'─'*18}─┼─{'─'*8}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*7}")
        for m in names:
            c, t = results[m]["costs"], results[m]["times"]
            gap = (np.mean(c) - best_v) / best_v * 100
            star = " *" if m == best_m else ""
            print(
                f"  {m:<18} | {np.mean(c):8.4f} | {np.std(c):7.4f} | "
                f"{np.mean(t):6.3f}s | {gap:+6.2f}%{star}"
            )
        print(f"{'=' * 70}\n")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════════════════

def visualize_tour(coords, tour, title="TSP Tour", save_path=None):
    """Plot a TSP tour. Saves to file if save_path given."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available"); return

    fig, ax = plt.subplots(figsize=(8, 8))
    n = len(tour)
    for i in range(n):
        c1, c2 = coords[tour[i]], coords[tour[(i + 1) % n]]
        ax.plot([c1[0], c2[0]], [c1[1], c2[1]], "b-", alpha=0.5, lw=1)
    ax.scatter(coords[:, 0], coords[:, 1], c="red", s=40, zorder=5)
    ax.scatter(coords[tour[0], 0], coords[tour[0], 1],
               c="green", s=120, zorder=6, marker="*", label="Start")
    dist = np.sqrt(np.sum((coords[:, None] - coords[None]) ** 2, axis=-1))
    ax.set_title(f"{title}\nCost: {compute_tour_cost(tour, dist):.4f} | n={len(coords)}")
    ax.legend(); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def _get_default_config(agent_type: str):
    """Return the default config object for an agent type."""
    return {
        "tabular": TabularConfig(),
        "dqn": DQNConfig(),
        "reinforce": REINFORCEConfig(),
        "ppo": PPOConfig(),
    }[agent_type.lower()]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Test L2GLS Agent")
    p.add_argument("--agent", type=str, default="ppo",
                    choices=["tabular", "dqn", "reinforce", "ppo"])
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint. Auto-detected if not given.")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--instances", type=int, default=10)
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--max_rounds", type=int, default=200)
    p.add_argument("--save_plot", type=str, default=None)
    a = p.parse_args()

    agent_cfg = _get_default_config(a.agent)
    agent = create_agent(a.agent, agent_cfg)

    # Auto-detect checkpoint path if not given
    if a.checkpoint is None:
        ext = ".npz" if a.agent == "tabular" else ".pt"
        a.checkpoint = f"checkpoints/{a.agent.upper()}_final{ext}"

    if os.path.exists(a.checkpoint):
        agent.load(a.checkpoint)
        print(f"Loaded: {a.checkpoint}")
    else:
        print(f"WARNING: {a.checkpoint} not found. Using random policy.")

    env_cfg = EnvConfig(n_cities=a.n, max_rounds=a.max_rounds)
    benchmark(
        agent, agent_name=a.agent.upper(),
        n=a.n, n_instances=a.instances, seed=a.seed,
        config=env_cfg, n_restarts=a.restarts,
    )

    if a.save_plot:
        inst = TSPInstance.generate_random(a.n, seed=a.seed)
        tour, cost, _ = solve_instance(agent, inst, env_cfg, a.restarts)
        visualize_tour(inst.coords, tour,
                       title=f"L2GLS {a.agent.upper()} — {a.n} cities",
                       save_path=a.save_plot)
