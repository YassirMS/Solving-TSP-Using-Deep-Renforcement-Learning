"""
Training Loop — Agent-agnostic, supports Tabular, DQN, REINFORCE, and PPO.

The loop adapts its update strategy to the agent type:
  - Tabular/DQN: update() called every step (off-policy, per-step learning)
  - REINFORCE:   update() is a no-op per step; end_episode() does the update
  - PPO:         update() called when rollout buffer reaches rollout_steps

Usage:
    python training.py --agent ppo
    python training.py --agent reinforce --n_episodes 1000
    python training.py --agent tabular --sizes 20 50
    python training.py --agent dqn --lr 1e-3
"""

import os
import csv
import time
import numpy as np
import random as py_random
from typing import Optional

from config import TrainConfig, EnvConfig
from environment import L2GLSEnv
from agents import create_agent, BaseAgent, PPOAgent, TabularQLearningAgent


def set_all_seeds(seed: int):
    """Reproducibility across all RNGs."""
    py_random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # No torch needed for tabular


def _get_agent_config(config: TrainConfig):
    """Return the config object matching config.agent_type."""
    return {
        "tabular": config.tabular,
        "dqn": config.dqn,
        "reinforce": config.reinforce,
        "ppo": config.ppo,
    }[config.agent_type.lower()]


def train(config: Optional[TrainConfig] = None, verbose: bool = True) -> BaseAgent:
    """
    Train any L2GLS agent.

    The loop detects the agent type and adapts:
      - When to call update() (per-step vs per-rollout vs per-episode)
      - When to call end_episode()
      - When to call decay_epsilon()

    Returns the trained agent.
    """
    if config is None:
        config = TrainConfig()

    set_all_seeds(config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    agent_config = _get_agent_config(config)
    agent = create_agent(config.agent_type, agent_config)
    env = L2GLSEnv(config.env)

    is_ppo = isinstance(agent, PPOAgent)
    is_tabular = isinstance(agent, TabularQLearningAgent)
    agent_name = config.agent_type.upper()

    # CSV logger
    log_path = os.path.join(config.log_dir, f"training_{agent_name}.csv")
    log_fields = [
        "episode", "instance_size", "best_cost", "initial_cost",
        "improvement_pct", "episode_reward", "episode_length",
        "policy_loss", "value_loss", "entropy", "approx_kl",
        "time_sec", "op_2opt", "op_relocate", "op_swap", "op_3perm",
    ]
    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    if verbose:
        print("=" * 70)
        print(f"  L2GLS Training — {agent_name} Agent")
        print("=" * 70)
        device_str = getattr(agent, "device", "cpu")
        print(f"  Agent: {agent_name} | Device: {device_str} | State dim: {env.state_dim}")
        print(f"  Episodes: {config.n_episodes} | Sizes: {config.mixed_batch_sizes}")
        print(f"  Seed: {config.seed}")
        print("-" * 70)

    last_stats: dict = {}
    recent_costs, recent_imps, recent_rews = [], [], []

    for episode in range(config.n_episodes):
        t0 = time.time()

        # Mixed curriculum: sample instance size
        n_cities = py_random.choice(config.mixed_batch_sizes)
        state = env.reset_random(n=n_cities, seed=config.seed + episode)
        ep_reward, ep_steps, done = 0.0, 0, False

        while not done:
            pen_act, op_act, info = agent.select_action(state)
            next_state, reward, done, step_info = env.step(pen_act, op_act)
            agent.store_transition(state, pen_act, op_act, info, reward, done)

            # ── Update strategy depends on agent type ──
            if is_ppo:
                # PPO: update when rollout buffer is full
                if agent.buffer_size >= config.ppo.rollout_steps:
                    last_stats = agent.update(next_state)
            elif not is_tabular and config.agent_type.lower() == "dqn":
                # DQN: update every step (with replay buffer)
                last_stats = agent.update(next_state)
            # Tabular: we update every step too
            elif is_tabular:
                last_stats = agent.update(next_state)
            # REINFORCE: no per-step update (done at end_episode)

            state = next_state
            ep_reward += reward
            ep_steps += 1

        # ── End of episode ──
        agent.end_episode()   # No-op for DQN/Tabular/PPO; REINFORCE does its update here
        agent.decay_epsilon() # No-op for REINFORCE/PPO

        imp_pct = (env.initial_cost - env.best_cost) / env.initial_cost * 100
        ep_time = time.time() - t0
        recent_costs.append(env.best_cost)
        recent_imps.append(imp_pct)
        recent_rews.append(ep_reward)

        # Log to CSV
        row = {
            "episode": episode, "instance_size": n_cities,
            "best_cost": f"{env.best_cost:.6f}",
            "initial_cost": f"{env.initial_cost:.6f}",
            "improvement_pct": f"{imp_pct:.2f}",
            "episode_reward": f"{ep_reward:.4f}",
            "episode_length": ep_steps,
            "policy_loss": f"{last_stats.get('policy_loss', last_stats.get('q_loss', 0)):.6f}",
            "value_loss": f"{last_stats.get('value_loss', 0):.6f}",
            "entropy": f"{last_stats.get('entropy', 0):.4f}",
            "approx_kl": f"{last_stats.get('approx_kl', 0):.6f}",
            "time_sec": f"{ep_time:.3f}",
            "op_2opt": env.operator_counts.get("2-opt", 0),
            "op_relocate": env.operator_counts.get("relocate", 0),
            "op_swap": env.operator_counts.get("swap", 0),
            "op_3perm": env.operator_counts.get("3-perm", 0),
        }
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow(row)

        # Print progress
        if verbose and (episode + 1) % config.log_interval == 0:
            w = min(config.log_interval, len(recent_costs))
            eps_str = ""
            if hasattr(agent, "epsilon"):
                eps_str = f" | eps={agent.epsilon:.3f}"
            kl = last_stats.get("approx_kl", 0)
            print(
                f"  Ep {episode+1:4d}/{config.n_episodes} | "
                f"n={n_cities:3d} | "
                f"cost={np.mean(recent_costs[-w:]):.4f} | "
                f"imp={np.mean(recent_imps[-w:]):+.1f}% | "
                f"rew={np.mean(recent_rews[-w:]):+.2f}"
                f"{eps_str} | "
                f"{ep_time:.2f}s"
            )

        # Checkpoint
        if (episode + 1) % config.checkpoint_interval == 0:
            ext = ".npz" if is_tabular else ".pt"
            p = os.path.join(config.checkpoint_dir, f"{agent_name}_ep{episode+1}{ext}")
            agent.save(p)
            if verbose:
                print(f"  >> Checkpoint: {p}")

    # Final save
    if is_ppo and agent.buffer_size > 0:
        agent.update(state)

    ext = ".npz" if is_tabular else ".pt"
    final_path = os.path.join(config.checkpoint_dir, f"{agent_name}_final{ext}")
    agent.save(final_path)

    if verbose:
        print("-" * 70)
        print(f"  Done. Model: {final_path} | Log: {log_path}")
        if recent_costs:
            print(
                f"  Last 50 avg: cost={np.mean(recent_costs[-50:]):.4f}, "
                f"imp={np.mean(recent_imps[-50:]):.1f}%"
            )
        print("=" * 70)

    return agent


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Train L2GLS Agent")
    p.add_argument("--agent", type=str, default="ppo",
                    choices=["tabular", "dqn", "reinforce", "ppo"])
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rounds", type=int, default=200)
    p.add_argument("--sizes", nargs="+", type=int, default=[20, 50, 100])
    p.add_argument("--lr", type=float, default=None,
                    help="Learning rate (overrides agent default)")
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--log_dir", type=str, default="logs")
    a = p.parse_args()

    cfg = TrainConfig(
        env=EnvConfig(max_rounds=a.max_rounds),
        agent_type=a.agent,
        n_episodes=a.n_episodes,
        mixed_batch_sizes=a.sizes,
        seed=a.seed,
        checkpoint_dir=a.checkpoint_dir,
        log_dir=a.log_dir,
    )

    # Override learning rate if specified
    if a.lr is not None:
        if a.agent == "ppo":
            cfg.ppo.lr_actor = a.lr
        elif a.agent == "dqn":
            cfg.dqn.lr = a.lr
        elif a.agent == "reinforce":
            cfg.reinforce.lr = a.lr
        elif a.agent == "tabular":
            cfg.tabular.lr = a.lr

    train(cfg)
