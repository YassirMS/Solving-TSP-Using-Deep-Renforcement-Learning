#!/usr/bin/env python3
"""
L2GLS — Complete Pipeline: Train + Benchmark (Random + TSPLIB)
==============================================================

Trains DQN, REINFORCE, PPO with per-agent tuned episodes, then benchmarks
against all baselines (NN, 2-opt, 3-opt, GLS+2-opt, GLS+3-opt) on:
  - Random instances: N in {20, 50, 100, 200, 500}
  - TSPLIB instances: all EUC_2D up to 500 cities

All results stored in results/ as JSON + CSV for the report.

Usage:
    python run_all.py                           # full pipeline
    python run_all.py --skip_training           # benchmark only (load checkpoints)
    python run_all.py --agents ppo dqn          # subset
    python run_all.py --random_sizes 20 50 100  # custom sizes
"""

import os
import sys
import json
import csv
import time
import numpy as np
import random as py_random
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from config import (
    TrainConfig, EnvConfig,
    DQNConfig, REINFORCEConfig, PPOConfig,
    N_OPERATORS, PENALTY_LEVELS,
)
from tsp_problem import (
    TSPInstance, TSPSolution, compute_tour_cost, nearest_neighbor_tour,
)
from operators import two_opt_step, three_opt_step
from gls_penalties import GLSPenaltyManager
from environment import L2GLSEnv
from agents import BaseAgent, create_agent
from training import train, set_all_seeds
from testing import (
    solve_instance, solve_pure_2opt, solve_gls_2opt,
    solve_pure_3opt, solve_gls_3opt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-agent tuned configs (key: each agent needs different episode counts)
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_config(agent_type: str, seed=42, ckpt_dir="checkpoints", log_dir="logs"):
    env = EnvConfig(n_cities=50, alpha_gls=0.3, max_rounds=200,
                    stagnation_limit=10, three_perm_samples=100)

    configs = {
        "dqn": TrainConfig(
            env=env, agent_type="dqn",
            dqn=DQNConfig(hidden_dim=64, n_hidden_layers=2, lr=1e-3,
                          gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
                          epsilon_decay=0.997, buffer_size=20000,
                          batch_size=64, target_update_freq=100, device="auto"),
            n_episodes=4000,  # DQN needs replay buffer to fill
            mixed_batch_sizes=[20, 50, 100],
            seed=seed, log_interval=50, checkpoint_interval=250,
            checkpoint_dir=ckpt_dir, log_dir=log_dir,
        ),
        "reinforce": TrainConfig(
            env=env, agent_type="reinforce",
            reinforce=REINFORCEConfig(hidden_dim=64, n_hidden_layers=2,
                                      lr=5e-4, gamma=0.99, max_grad_norm=1.0,
                                      baseline_alpha=0.01, device="auto"),
            n_episodes=4000,  # REINFORCE high variance → needs more episodes
            mixed_batch_sizes=[20, 50, 100],
            seed=seed, log_interval=50, checkpoint_interval=300,
            checkpoint_dir=ckpt_dir, log_dir=log_dir,
        ),
        "ppo": TrainConfig(
            env=env, agent_type="ppo",
            ppo=PPOConfig(hidden_dim=64, n_hidden_layers=2, lr_actor=3e-4,
                          lr_critic=1e-3, gamma=0.99, gae_lambda=0.95,
                          clip_epsilon=0.2, entropy_coef=0.01,
                          value_loss_coef=0.5, max_grad_norm=0.5,
                          rollout_steps=128, n_epochs=4, batch_size=32,
                          device="auto"),
            n_episodes=4000,   # PPO most sample-efficient
            mixed_batch_sizes=[20, 50, 100],
            seed=seed, log_interval=50, checkpoint_interval=200,
            checkpoint_dir=ckpt_dir, log_dir=log_dir,
        ),
    }
    return configs[agent_type.lower()]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Train
# ═══════════════════════════════════════════════════════════════════════════════

def train_agents(agent_types, seed=42, ckpt_dir="checkpoints", log_dir="logs"):
    trained = {}
    for i, at in enumerate(agent_types):
        print(f"\n{'#'*74}")
        print(f"  TRAINING {i+1}/{len(agent_types)}: {at.upper()}")
        print(f"{'#'*74}")
        cfg = get_train_config(at, seed=seed, ckpt_dir=ckpt_dir, log_dir=log_dir)
        t0 = time.time()
        agent = train(cfg, verbose=True)
        dt = time.time() - t0
        trained[at] = (agent, dt)
        print(f"  {at.upper()} trained in {dt:.1f}s ({dt/60:.1f} min)")
    return trained


def load_agents(agent_types, ckpt_dir="checkpoints"):
    configs = {"dqn": DQNConfig(), "reinforce": REINFORCEConfig(), "ppo": PPOConfig()}
    loaded = {}
    for at in agent_types:
        agent = create_agent(at, configs[at])
        path = os.path.join(ckpt_dir, f"{at.upper()}_final.pt")
        if os.path.exists(path):
            agent.load(path)
            print(f"  Loaded {at.upper()} from {path}")
        else:
            print(f"  WARNING: {path} not found — using random policy")
        loaded[at] = agent
    return loaded


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Random Instance Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_random(agents, test_sizes, n_instances=20, n_restarts=5,
                     max_rounds=200, seed=100000, results_dir="results"):
    """Benchmark on random Euclidean instances."""
    os.makedirs(results_dir, exist_ok=True)

    baseline_names = ["NN", "2-opt", "3-opt", "GLS+2-opt", "GLS+3-opt"]
    agent_names = [f"L2GLS({t.upper()})" for t in agents.keys()]
    all_methods = baseline_names + agent_names

    env_config = EnvConfig(max_rounds=max_rounds)
    rows = []  # per-instance rows for CSV

    print(f"\n{'='*90}")
    print(f"  RANDOM BENCHMARK — sizes={test_sizes}, {n_instances} inst/size")
    print(f"{'='*90}")

    for sz in test_sizes:
        print(f"\n  ── n={sz} ──")
        hdr = f"  {'#':>3} | {'NN':>8} | {'2-opt':>8} | {'3-opt':>8} | {'GLS2':>8} | {'GLS3':>8}"
        for t in agents: hdr += f" | {t.upper()[:5]:>7}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))

        for i in range(n_instances):
            inst_seed = seed + sz * 10000 + i
            inst = TSPInstance.generate_random(sz, seed=inst_seed)
            row = {"size": sz, "instance": i, "seed": inst_seed}

            # NN
            t0 = time.time()
            nn_cost = compute_tour_cost(nearest_neighbor_tour(inst), inst.dist_matrix)
            row["NN_cost"], row["NN_time"] = nn_cost, time.time() - t0

            # 2-opt
            t0 = time.time()
            c2 = solve_pure_2opt(inst, max_iters=200)
            row["2-opt_cost"], row["2-opt_time"] = c2, time.time() - t0

            # 3-opt
            t0 = time.time()
            c3 = solve_pure_3opt(inst, max_iters=100)
            row["3-opt_cost"], row["3-opt_time"] = c3, time.time() - t0

            # GLS+2-opt
            t0 = time.time()
            cg2 = solve_gls_2opt(inst, max_rounds=max_rounds)
            row["GLS+2-opt_cost"], row["GLS+2-opt_time"] = cg2, time.time() - t0

            # GLS+3-opt
            t0 = time.time()
            cg3 = solve_gls_3opt(inst, max_rounds=min(max_rounds, 100))
            row["GLS+3-opt_cost"], row["GLS+3-opt_time"] = cg3, time.time() - t0

            # L2GLS agents
            for at, agent in agents.items():
                t0 = time.time()
                _, cl, _ = solve_instance(agent, inst, env_config, n_restarts)
                row[f"L2GLS({at.upper()})_cost"] = cl
                row[f"L2GLS({at.upper()})_time"] = time.time() - t0

            rows.append(row)

            # Print row
            vals = [nn_cost, c2, c3, cg2, cg3]
            for at in agents: vals.append(row[f"L2GLS({at.upper()})_cost"])
            print(f"  {i+1:3d} |" + " |".join(f" {v:8.3f}" for v in vals))

    # Save per-instance CSV
    csv_path = os.path.join(results_dir, "random_benchmark.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Summary
    print(f"\n{'='*90}")
    print(f"  RANDOM BENCHMARK SUMMARY")
    print(f"{'='*90}")

    summary_rows = []
    for sz in test_sizes:
        sz_rows = [r for r in rows if r["size"] == sz]
        if not sz_rows: continue

        print(f"\n  ── n={sz} ({len(sz_rows)} instances) ──")
        print(f"  {'Method':<20} {'Mean':>9} {'Std':>8} {'Time':>8} {'Gap':>8}")
        print(f"  {'─'*20}─{'─'*9}─{'─'*8}─{'─'*8}─{'─'*8}")

        method_means = {}
        for m in all_methods:
            key = f"{m}_cost"
            costs = [r[key] for r in sz_rows if key in r]
            if costs: method_means[m] = np.mean(costs)

        best_val = min(method_means.values()) if method_means else 1.0

        for m in all_methods:
            cost_key = f"{m}_cost"
            time_key = f"{m}_time"
            costs = [r[cost_key] for r in sz_rows if cost_key in r]
            times = [r[time_key] for r in sz_rows if time_key in r]
            if not costs: continue
            mc, sc, mt = np.mean(costs), np.std(costs), np.mean(times)
            gap = (mc - best_val) / best_val * 100
            star = " *" if mc <= best_val + 1e-8 else ""
            print(f"  {m:<20} {mc:9.4f} {sc:8.4f} {mt:7.3f}s {gap:+7.2f}%{star}")
            summary_rows.append({"size": sz, "method": m, "mean": mc,
                                  "std": sc, "time": mt, "gap_pct": gap})

    sum_path = os.path.join(results_dir, "random_summary.csv")
    if summary_rows:
        with open(sum_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\n  Saved: {csv_path}, {sum_path}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: TSPLIB Benchmark
# ═══════════════════════════════════════════════════════════════════════════════

# Known optima (EUC_2D, verified by Concorde)
TSPLIB_OPTIMA = {
    "burma14": 3323, "gr17": 2085, "gr21": 2707, "gr24": 1272,
    "fri26": 937, "bayg29": 1610, "bays29": 2020, "dantzig42": 699,
    "swiss42": 1273, "att48": 10628, "gr48": 5046, "hk48": 11461,
    "eil51": 426, "berlin52": 7542, "brazil58": 25395,
    "st70": 675, "eil76": 538, "pr76": 108159,
    "gr96": 55209, "rat99": 1211, "rd100": 7910,
    "kroA100": 21282, "kroB100": 22141, "kroC100": 20749,
    "kroD100": 21294, "kroE100": 22068, "eil101": 629,
    "lin105": 14379, "pr107": 44303, "gr120": 6942,
    "pr124": 59030, "bier127": 118282, "ch130": 6110,
    "pr136": 96772, "ch150": 6528, "kroA150": 26524,
    "kroB150": 26130, "pr152": 73682, "u159": 42080,
    "rat195": 2323, "d198": 15780, "kroA200": 29368,
    "kroB200": 29437, "tsp225": 3916, "a280": 2579,
    "pr299": 48191, "lin318": 42029, "rd400": 15281,
    "fl417": 11861, "pr439": 107217, "pcb442": 50778,
    "d493": 35002,
}


def parse_tsplib(filepath):
    """Parse TSPLIB .tsp file → (name, coords, edge_weight_type)."""
    name, dim, ewt = "", 0, ""
    coords, reading = [], False
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith("NAME"):
                name = line.split(":")[-1].strip() if ":" in line else line.split()[-1]
            elif line.startswith("DIMENSION"):
                dim = int(line.split(":")[-1].strip() if ":" in line else line.split()[-1])
            elif line.startswith("EDGE_WEIGHT_TYPE"):
                ewt = line.split(":")[-1].strip() if ":" in line else line.split()[-1]
            elif line.startswith("NODE_COORD_SECTION"):
                reading = True; continue
            elif line in ("EOF", "DISPLAY_DATA_SECTION"):
                reading = False; continue
            if reading:
                parts = line.split()
                if len(parts) >= 3:
                    try: coords.append((float(parts[1]), float(parts[2])))
                    except ValueError: pass
    return name, np.array(coords, dtype=np.float64), ewt


def make_tsplib_instance(coords):
    """Create TSPInstance with TSPLIB integer-rounded distances."""
    n = len(coords)
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.round(np.sqrt(np.sum(diff**2, axis=-1))).astype(np.float64)
    inst = TSPInstance.__new__(TSPInstance)
    inst.coords = coords
    inst.n = n
    inst.dist_matrix = dist
    return inst


def benchmark_tsplib(agents, tsplib_dir="tsplib_data", max_rounds=200,
                     n_restarts=5, max_n=500, results_dir="results"):
    """Benchmark on TSPLIB EUC_2D instances up to max_n cities."""
    os.makedirs(results_dir, exist_ok=True)

    # Find available .tsp files
    available = []
    if os.path.isdir(tsplib_dir):
        for fn in sorted(os.listdir(tsplib_dir)):
            if fn.endswith(".tsp"):
                name, coords, ewt = parse_tsplib(os.path.join(tsplib_dir, fn))
                if ewt == "EUC_2D" and len(coords) <= max_n and len(coords) > 0:
                    opt = TSPLIB_OPTIMA.get(name, TSPLIB_OPTIMA.get(name.lower()))
                    if opt: available.append((name, coords, opt))

    if not available:
        print(f"  No TSPLIB EUC_2D instances found in {tsplib_dir}/ (up to {max_n} cities)")
        print(f"  Download from: https://github.com/mastqe/tsplib")
        return []

    available.sort(key=lambda x: len(x[1]))

    baseline_names = ["NN", "2-opt", "3-opt", "GLS+2-opt", "GLS+3-opt"]
    agent_names = [f"L2GLS({t.upper()})" for t in agents.keys()]
    all_methods = baseline_names + agent_names
    env_config = EnvConfig(max_rounds=max_rounds)
    rows = []

    print(f"\n{'='*90}")
    print(f"  TSPLIB BENCHMARK — {len(available)} instances, max n={max_n}")
    print(f"{'='*90}")

    for name, coords, optimum in available:
        inst = make_tsplib_instance(coords)
        n = inst.n
        row = {"instance": name, "n": n, "optimum": optimum}

        # NN
        t0 = time.time()
        nn_cost = compute_tour_cost(nearest_neighbor_tour(inst), inst.dist_matrix)
        row["NN_cost"], row["NN_time"] = nn_cost, time.time() - t0
        row["NN_gap"] = (nn_cost - optimum) / optimum * 100

        # 2-opt
        t0 = time.time()
        c2 = solve_pure_2opt(inst, max_iters=300)
        row["2-opt_cost"], row["2-opt_time"] = c2, time.time() - t0
        row["2-opt_gap"] = (c2 - optimum) / optimum * 100

        # 3-opt (limit iterations for large instances)
        t0 = time.time()
        c3 = solve_pure_3opt(inst, max_iters=min(100, max(20, 500 // n)))
        row["3-opt_cost"], row["3-opt_time"] = c3, time.time() - t0
        row["3-opt_gap"] = (c3 - optimum) / optimum * 100

        # GLS+2-opt
        t0 = time.time()
        cg2 = solve_gls_2opt(inst, max_rounds=max_rounds)
        row["GLS+2-opt_cost"], row["GLS+2-opt_time"] = cg2, time.time() - t0
        row["GLS+2-opt_gap"] = (cg2 - optimum) / optimum * 100

        # GLS+3-opt
        t0 = time.time()
        cg3 = solve_gls_3opt(inst, max_rounds=min(max_rounds, 50))
        row["GLS+3-opt_cost"], row["GLS+3-opt_time"] = cg3, time.time() - t0
        row["GLS+3-opt_gap"] = (cg3 - optimum) / optimum * 100

        # L2GLS agents
        for at, agent in agents.items():
            label = f"L2GLS({at.upper()})"
            t0 = time.time()
            _, cl, _ = solve_instance(agent, inst, env_config, n_restarts)
            row[f"{label}_cost"] = cl
            row[f"{label}_time"] = time.time() - t0
            row[f"{label}_gap"] = (cl - optimum) / optimum * 100

        rows.append(row)

        # Print
        gaps = [row.get(f"{m}_gap", None) for m in all_methods]
        print(f"  {name:<12} n={n:>3} opt={optimum:>8} |" +
              " |".join(f" {g:+6.1f}%" if g is not None else "    N/A" for g in gaps))

    # Save
    csv_path = os.path.join(results_dir, "tsplib_benchmark.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Summary by size group
    print(f"\n{'='*90}")
    print(f"  TSPLIB SUMMARY — Mean Gap to Optimum (%)")
    print(f"{'='*90}")

    groups = [("n≤50", lambda r: r["n"] <= 50), ("51-100", lambda r: 50 < r["n"] <= 100),
              ("101-200", lambda r: 100 < r["n"] <= 200), ("201-500", lambda r: 200 < r["n"] <= 500),
              ("ALL", lambda r: True)]

    for gname, gfilt in groups:
        gr = [r for r in rows if gfilt(r)]
        if not gr: continue
        print(f"\n  ── {gname} ({len(gr)} instances) ──")
        print(f"  {'Method':<20} {'Mean Gap':>9} {'Std':>8} {'Time':>8}")
        print(f"  {'─'*20}─{'─'*9}─{'─'*8}─{'─'*8}")
        for m in all_methods:
            gkey = f"{m}_gap"
            tkey = f"{m}_time"
            gaps = [r[gkey] for r in gr if gkey in r and r[gkey] is not None]
            times = [r[tkey] for r in gr if tkey in r]
            if not gaps: continue
            print(f"  {m:<20} {np.mean(gaps):+8.2f}% {np.std(gaps):7.2f}% {np.mean(times):7.3f}s")

    if rows:
        print(f"\n  Saved: {csv_path}")

    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="L2GLS Complete Pipeline")
    p.add_argument("--agents", nargs="+", default=["dqn", "reinforce", "ppo"],
                   choices=["dqn", "reinforce", "ppo"])
    p.add_argument("--random_sizes", nargs="+", type=int, default=[20, 50, 100, 200])
    p.add_argument("--n_instances", type=int, default=20)
    p.add_argument("--n_restarts", type=int, default=5)
    p.add_argument("--max_rounds", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_training", action="store_true")
    p.add_argument("--tsplib_dir", type=str, default="tsplib_data")
    p.add_argument("--max_tsplib_n", type=int, default=500)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--results_dir", type=str, default="results")
    args = p.parse_args()

    set_all_seeds(args.seed)
    total_t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║     L2GLS — Complete Pipeline: Train + Benchmark            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  Agents:      {[a.upper() for a in args.agents]}")
    print(f"  Random:      sizes={args.random_sizes}, {args.n_instances} inst/size")
    print(f"  TSPLIB:      dir={args.tsplib_dir}, max_n={args.max_tsplib_n}")

    # Phase 1: Train or Load
    training_times = {}
    if args.skip_training:
        agents_dict = load_agents(args.agents, args.checkpoint_dir)
    else:
        trained = train_agents(args.agents, seed=args.seed,
                               ckpt_dir=args.checkpoint_dir, log_dir="logs")
        agents_dict = {t: a for t, (a, _) in trained.items()}
        training_times = {t: dt for t, (_, dt) in trained.items()}

    # Save training times
    os.makedirs(args.results_dir, exist_ok=True)
    if training_times:
        with open(os.path.join(args.results_dir, "training_times.json"), "w") as f:
            json.dump({k.upper(): {"seconds": v, "minutes": v/60}
                       for k, v in training_times.items()}, f, indent=2)

    # Phase 2: Random Benchmark
    random_rows = benchmark_random(
        agents_dict, test_sizes=args.random_sizes,
        n_instances=args.n_instances, n_restarts=args.n_restarts,
        max_rounds=args.max_rounds, seed=args.seed + 100000,
        results_dir=args.results_dir,
    )

    # Phase 3: TSPLIB Benchmark
    tsplib_rows = benchmark_tsplib(
        agents_dict, tsplib_dir=args.tsplib_dir,
        max_rounds=args.max_rounds, n_restarts=args.n_restarts,
        max_n=args.max_tsplib_n, results_dir=args.results_dir,
    )

    total = time.time() - total_t0
    print(f"\n{'='*70}")
    print(f"  DONE — Total: {total:.0f}s ({total/60:.1f} min)")
    if training_times:
        for t, dt in training_times.items():
            print(f"    {t.upper():<12} training: {dt:.0f}s")
    print(f"  Results in: {args.results_dir}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
