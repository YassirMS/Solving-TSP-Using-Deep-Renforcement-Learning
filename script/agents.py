"""
RL Agents for L2GLS — Four implementations sharing a common interface.

All agents operate on the two-level action space:
  - penalty_action: index into PENALTY_LEVELS  (0=none, 1=one edge, 2=three edges)
  - operator_action: Operator enum value       (0=2opt, 1=relocate, 2=swap, 3=3perm)

Flat-action agents (Tabular, DQN, REINFORCE) encode both decisions into a
single flat index: action = pen_idx * N_OPERATORS + op_idx, then decode
back. PPO uses two separate output heads (hierarchical).

Common interface (every agent implements):
    select_action(state)        → (pen_act, op_act, info_dict)
    select_action_greedy(state) → (pen_act, op_act)
    store_transition(state, pen_act, op_act, info, reward, done)
    update(last_state)          → stats_dict
    end_episode()               → None  (for on-policy methods)
    decay_epsilon()             → None
    save(path)                  → None
    load(path)                  → None

Factory function:
    create_agent(agent_type, config) → agent instance
"""

import os
import random
import numpy as np
from typing import Tuple, Dict, Optional, List
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field

from config import (
    N_OPERATORS, N_PENALTY_ACTIONS, N_FLAT_ACTIONS, PENALTY_LEVELS,
    TabularConfig, DQNConfig, REINFORCEConfig, PPOConfig,
)
from state_representation import STATE_DIM


# ═══════════════════════════════════════════════════════════════════════════════
# Flat action encoding/decoding (used by Tabular, DQN, REINFORCE)
# ═══════════════════════════════════════════════════════════════════════════════

def flat_action_encode(pen_idx: int, op_idx: int) -> int:
    """Encode (penalty_index, operator_index) → flat action."""
    return pen_idx * N_OPERATORS + op_idx


def flat_action_decode(action: int) -> Tuple[int, int]:
    """Decode flat action → (penalty_index, operator_index)."""
    pen_idx = action // N_OPERATORS
    op_idx = action % N_OPERATORS
    return pen_idx, op_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Base Agent (abstract)
# ═══════════════════════════════════════════════════════════════════════════════

class BaseAgent(ABC):
    """Abstract base class — defines the interface all agents must implement."""

    @abstractmethod
    def select_action(self, state: np.ndarray) -> Tuple[int, int, Dict]:
        """Returns (pen_action, op_action, info_dict)."""

    @abstractmethod
    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, int]:
        """Deterministic action for evaluation."""

    @abstractmethod
    def store_transition(self, state, pen_act, op_act, info, reward, done):
        """Store one transition."""

    @abstractmethod
    def update(self, last_state: np.ndarray) -> Dict[str, float]:
        """Perform a learning update. Returns stats dict."""

    def end_episode(self):
        """Called at episode end. Override for on-policy methods."""
        pass

    def decay_epsilon(self):
        """Decay exploration. Override for epsilon-greedy methods."""
        pass

    def save(self, path: str):
        """Save agent state. Override for neural agents."""
        pass

    def load(self, path: str):
        """Load agent state. Override for neural agents."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TABULAR Q-LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

class TabularQLearningAgent(BaseAgent):
    """
    Tabular Q-Learning with discretized state space.

    Discretizes the 11-dim continuous state into bins and maintains a Q-table
    over the 12-action flat space. Simple, interpretable, no torch needed.

    Q(s,a) ← Q(s,a) + α · [r + γ · max_a' Q(s',a') - Q(s,a)]
    """

    def __init__(self, config: TabularConfig):
        self.config = config
        self.lr = config.lr
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.n_bins = config.n_bins

        self.q_table: Dict[tuple, np.ndarray] = {}

        # Store last transition for per-step update
        self._last_state = None
        self._last_action = None

    def _discretize(self, state: np.ndarray) -> tuple:
        """Continuous state → tuple of bin indices."""
        return tuple(
            min(self.n_bins - 1, max(0, int(v * self.n_bins)))
            for v in state
        )

    def _get_q(self, key: tuple) -> np.ndarray:
        if key not in self.q_table:
            self.q_table[key] = np.zeros(N_FLAT_ACTIONS)
        return self.q_table[key]

    def select_action(self, state: np.ndarray) -> Tuple[int, int, Dict]:
        if random.random() < self.epsilon:
            flat_action = random.randint(0, N_FLAT_ACTIONS - 1)
        else:
            key = self._discretize(state)
            flat_action = int(np.argmax(self._get_q(key)))

        pen_idx, op_idx = flat_action_decode(flat_action)
        return pen_idx, op_idx, {"flat_action": flat_action}

    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, int]:
        key = self._discretize(state)
        flat_action = int(np.argmax(self._get_q(key)))
        return flat_action_decode(flat_action)

    def store_transition(self, state, pen_act, op_act, info, reward, done):
        flat_action = info["flat_action"]
        self._last_state = state.copy()
        self._last_action = flat_action
        self._last_reward = reward

    def update(self, last_state: np.ndarray) -> Dict[str, float]:
        """Per-step Q-learning update using the most recent transition."""
        if self._last_state is None:
            return {}

        key = self._discretize(self._last_state)
        next_key = self._discretize(last_state)

        q_vals = self._get_q(key)
        next_q = self._get_q(next_key)

        target = self._last_reward + self.gamma * np.max(next_q)
        td_error = target - q_vals[self._last_action]
        q_vals[self._last_action] += self.lr * td_error

        self._last_state = None
        return {"td_error": abs(td_error)}

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        np.savez(
            path,
            q_keys=np.array([list(k) for k in self.q_table.keys()]),
            q_values=np.array(list(self.q_table.values())),
            epsilon=self.epsilon,
        )

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        keys = [tuple(k) for k in data["q_keys"]]
        values = data["q_values"]
        self.q_table = {k: v for k, v in zip(keys, values)}
        self.epsilon = float(data["epsilon"])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DQN (Deep Q-Network)
# ═══════════════════════════════════════════════════════════════════════════════

class DQNAgent(BaseAgent):
    """
    Deep Q-Network with target network and experience replay.

    Network: state (11) → FC(hidden) → ReLU → ... → FC(12)
    Outputs Q-values for all 12 flat actions.

    Improvements over original code:
      - Target network (synced every target_update_freq steps)
      - Experience replay buffer with mini-batch sampling
    """

    def __init__(self, config: DQNConfig):
        import torch
        import torch.nn as nn

        self.config = config
        self.gamma = config.gamma
        self.epsilon = config.epsilon_start
        self.epsilon_end = config.epsilon_end
        self.epsilon_decay = config.epsilon_decay
        self.batch_size = config.batch_size
        self.target_update_freq = config.target_update_freq
        self.step_count = 0

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Build Q-network
        def _make_net():
            layers = [nn.Linear(STATE_DIM, config.hidden_dim), nn.ReLU()]
            for _ in range(config.n_hidden_layers - 1):
                layers += [nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU()]
            layers.append(nn.Linear(config.hidden_dim, N_FLAT_ACTIONS))
            return nn.Sequential(*layers)

        self.q_net = _make_net().to(self.device)
        self.target_net = _make_net().to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = deque(maxlen=config.buffer_size)

    def select_action(self, state: np.ndarray) -> Tuple[int, int, Dict]:
        import torch

        if random.random() < self.epsilon:
            flat_action = random.randint(0, N_FLAT_ACTIONS - 1)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                flat_action = int(self.q_net(state_t).argmax(dim=1).item())

        pen_idx, op_idx = flat_action_decode(flat_action)
        return pen_idx, op_idx, {"flat_action": flat_action}

    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, int]:
        import torch

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            flat_action = int(self.q_net(state_t).argmax(dim=1).item())
        return flat_action_decode(flat_action)

    def _update_batch(self) -> Dict[str, float]:
        """Internal: train on a mini-batch from replay buffer."""
        import torch

        if len(self.buffer) < self.batch_size:
            return {}

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q(s, a)
        current_q = self.q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: r + γ · max_a' Q_target(s', a') · (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Sync target network periodically
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"q_loss": loss.item()}

    def store_transition(self, state, pen_act, op_act, info, reward, done):
        # We need next_state for DQN. We'll store it when update() is called.
        # Use a temporary holding slot.
        self._pending = (state.copy(), info["flat_action"], reward, done)

    def update(self, last_state: np.ndarray) -> Dict[str, float]:
        # Complete the pending transition with next_state
        if hasattr(self, "_pending") and self._pending is not None:
            s, a, r, d = self._pending
            self.buffer.append((s, a, r, last_state.copy(), float(d)))
            self._pending = None

        return self._update_batch()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        import torch
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "step_count": self.step_count,
        }, path)

    def load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.step_count = ckpt["step_count"]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. REINFORCE (Policy Gradient)
# ═══════════════════════════════════════════════════════════════════════════════

class REINFORCEAgent(BaseAgent):
    """
    REINFORCE with running-average baseline.

    Directly parameterizes π(a|s) as a softmax over 12 flat actions.
    On-policy: collects full episode trajectory, updates at episode end.

    Update: θ ← θ + α · ∇log π(a_t|s_t) · (G_t - baseline)
    """

    def __init__(self, config: REINFORCEConfig):
        import torch
        import torch.nn as nn

        self.config = config
        self.gamma = config.gamma
        self.max_grad_norm = config.max_grad_norm
        self.baseline_alpha = config.baseline_alpha

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Policy network: state → logits over 12 flat actions
        layers = [nn.Linear(STATE_DIM, config.hidden_dim), nn.ReLU()]
        for _ in range(config.n_hidden_layers - 1):
            layers += [nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(config.hidden_dim, N_FLAT_ACTIONS))
        self.policy_net = nn.Sequential(*layers).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.lr)

        # Episode buffers (on-policy: collect full episode, then update)
        self.ep_states: List[np.ndarray] = []
        self.ep_actions: List[int] = []
        self.ep_rewards: List[float] = []

        # Running baseline for variance reduction
        self.baseline = 0.0

    def select_action(self, state: np.ndarray) -> Tuple[int, int, Dict]:
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_net(state_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            flat_action = dist.sample().item()

        pen_idx, op_idx = flat_action_decode(flat_action)
        return pen_idx, op_idx, {"flat_action": flat_action}

    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, int]:
        import torch

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits = self.policy_net(state_t)
            flat_action = int(logits.argmax(dim=-1).item())
        return flat_action_decode(flat_action)

    def store_transition(self, state, pen_act, op_act, info, reward, done):
        self.ep_states.append(state.copy())
        self.ep_actions.append(info["flat_action"])
        self.ep_rewards.append(reward)

    def update(self, last_state: np.ndarray) -> Dict[str, float]:
        # REINFORCE updates at episode end, not per-step
        return {}

    def end_episode(self):
        """Perform REINFORCE update using the collected episode trajectory."""
        import torch
        import torch.nn.functional as F

        if len(self.ep_rewards) == 0:
            return

        T = len(self.ep_rewards)

        # Step 1: Compute discounted returns G_t
        returns = [0.0] * T
        G = 0.0
        for t in reversed(range(T)):
            G = self.ep_rewards[t] + self.gamma * G
            returns[t] = G

        returns_t = torch.FloatTensor(returns).to(self.device)

        # Step 2: Advantages = G_t - baseline
        advantages = returns_t - self.baseline
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update running baseline
        mean_return = returns_t.mean().item()
        self.baseline = (
            (1 - self.baseline_alpha) * self.baseline
            + self.baseline_alpha * mean_return
        )

        # Step 3: Policy gradient loss
        states_t = torch.FloatTensor(np.array(self.ep_states)).to(self.device)
        actions_t = torch.LongTensor(self.ep_actions).to(self.device)

        logits = self.policy_net(states_t)
        log_probs = F.log_softmax(logits, dim=-1)
        selected_lp = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        policy_loss = -(selected_lp * advantages).mean()

        # Step 4: Backprop
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        # Clear episode buffers
        self.ep_states.clear()
        self.ep_actions.clear()
        self.ep_rewards.clear()

    def save(self, path: str):
        import torch
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "baseline": self.baseline,
        }, path)

    def load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.baseline = ckpt["baseline"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PPO-CLIP (Proximal Policy Optimization)
# ═══════════════════════════════════════════════════════════════════════════════

class PPOAgent(BaseAgent):
    """
    PPO-Clip with two-level hierarchical actions.

    Two separate heads (penalty + operator) branching from a shared trunk.
    Uses GAE for advantage estimation and clipped surrogate objective.

    This is the most advanced agent and the recommended choice for L2GLS.
    """

    def __init__(self, config: PPOConfig):
        import torch
        import torch.nn as nn

        self.config = config

        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # ── Actor-Critic Network ──
        h = config.hidden_dim

        # Shared trunk
        trunk_layers = [nn.Linear(STATE_DIM, h), nn.ReLU()]
        for _ in range(config.n_hidden_layers - 1):
            trunk_layers += [nn.Linear(h, h), nn.ReLU()]
        self.trunk = nn.Sequential(*trunk_layers).to(self.device)

        # Action heads
        self.penalty_head = nn.Linear(h, N_PENALTY_ACTIONS).to(self.device)
        self.operator_head = nn.Linear(h, N_OPERATORS).to(self.device)
        self.value_head = nn.Linear(h, 1).to(self.device)

        # Orthogonal init
        for m in [self.trunk, self.penalty_head, self.operator_head, self.value_head]:
            for layer in m.modules() if hasattr(m, 'modules') else [m]:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.penalty_head.weight, gain=0.01)
        nn.init.orthogonal_(self.operator_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

        all_params = (
            list(self.trunk.parameters())
            + list(self.penalty_head.parameters())
            + list(self.operator_head.parameters())
            + list(self.value_head.parameters())
        )
        self.optimizer = torch.optim.Adam(all_params, lr=config.lr_actor, eps=1e-5)

        # Rollout buffer
        self.buf_states: List[np.ndarray] = []
        self.buf_pen_acts: List[int] = []
        self.buf_op_acts: List[int] = []
        self.buf_pen_lp: List[float] = []
        self.buf_op_lp: List[float] = []
        self.buf_rewards: List[float] = []
        self.buf_values: List[float] = []
        self.buf_dones: List[bool] = []

        self.update_count = 0

    @property
    def buffer_size(self) -> int:
        return len(self.buf_states)

    def _forward(self, state_t):
        """Shared trunk → three heads."""
        h = self.trunk(state_t)
        return self.penalty_head(h), self.operator_head(h), self.value_head(h)

    def select_action(self, state: np.ndarray) -> Tuple[int, int, Dict]:
        import torch

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pen_logits, op_logits, value = self._forward(state_t)
            pen_dist = torch.distributions.Categorical(logits=pen_logits)
            op_dist = torch.distributions.Categorical(logits=op_logits)
            pen_act = pen_dist.sample()
            op_act = op_dist.sample()

        return (
            pen_act.item(),
            op_act.item(),
            {
                "pen_logprob": pen_dist.log_prob(pen_act).item(),
                "op_logprob": op_dist.log_prob(op_act).item(),
                "value": value.squeeze(-1).item(),
            },
        )

    def select_action_greedy(self, state: np.ndarray) -> Tuple[int, int]:
        import torch

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pen_logits, op_logits, _ = self._forward(state_t)
        return (
            int(pen_logits.argmax(dim=-1).item()),
            int(op_logits.argmax(dim=-1).item()),
        )

    def store_transition(self, state, pen_act, op_act, info, reward, done):
        self.buf_states.append(state.copy())
        self.buf_pen_acts.append(pen_act)
        self.buf_op_acts.append(op_act)
        self.buf_pen_lp.append(info["pen_logprob"])
        self.buf_op_lp.append(info["op_logprob"])
        self.buf_rewards.append(reward)
        self.buf_values.append(info["value"])
        self.buf_dones.append(done)

    def update(self, last_state: np.ndarray) -> Dict[str, float]:
        """PPO-Clip update with GAE."""
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        cfg = self.config
        T = len(self.buf_states)
        if T == 0:
            return {}

        # Buffer → numpy
        states = np.array(self.buf_states, dtype=np.float32)
        pen_acts = np.array(self.buf_pen_acts, dtype=np.int64)
        op_acts = np.array(self.buf_op_acts, dtype=np.int64)
        old_pen_lp = np.array(self.buf_pen_lp, dtype=np.float32)
        old_op_lp = np.array(self.buf_op_lp, dtype=np.float32)
        rewards = np.array(self.buf_rewards, dtype=np.float32)
        values = np.array(self.buf_values, dtype=np.float32)
        dones = np.array(self.buf_dones, dtype=np.float32)

        # Bootstrap last value
        last_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, last_v = self._forward(last_t)
            last_val = last_v.squeeze(-1).item()

        # GAE
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_val = last_val if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * next_val * nonterminal - values[t]
            advantages[t] = last_gae = (
                delta + cfg.gamma * cfg.gae_lambda * nonterminal * last_gae
            )
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # → tensors
        s_t = torch.FloatTensor(states).to(self.device)
        pa_t = torch.LongTensor(pen_acts).to(self.device)
        oa_t = torch.LongTensor(op_acts).to(self.device)
        old_plp_t = torch.FloatTensor(old_pen_lp).to(self.device)
        old_olp_t = torch.FloatTensor(old_op_lp).to(self.device)
        adv_t = torch.FloatTensor(advantages).to(self.device)
        ret_t = torch.FloatTensor(returns).to(self.device)

        total_pl, total_vl, total_ent, total_kl, n_up = 0, 0, 0, 0, 0
        indices = np.arange(T)

        for _ in range(cfg.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, cfg.batch_size):
                mb = indices[start:min(start + cfg.batch_size, T)]

                # Forward
                h = self.trunk(s_t[mb])
                pen_logits = self.penalty_head(h)
                op_logits = self.operator_head(h)
                new_vals = self.value_head(h).squeeze(-1)

                pen_dist = torch.distributions.Categorical(logits=pen_logits)
                op_dist = torch.distributions.Categorical(logits=op_logits)

                new_plp = pen_dist.log_prob(pa_t[mb])
                new_olp = op_dist.log_prob(oa_t[mb])
                entropy = pen_dist.entropy() + op_dist.entropy()

                log_ratio = (new_plp + new_olp) - (old_plp_t[mb] + old_olp_t[mb])
                ratio = torch.exp(log_ratio)
                approx_kl = ((ratio - 1) - log_ratio).mean().item()

                surr1 = ratio * adv_t[mb]
                surr2 = (
                    torch.clamp(ratio, 1 - cfg.clip_epsilon, 1 + cfg.clip_epsilon)
                    * adv_t[mb]
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_vals, ret_t[mb])
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + cfg.value_loss_coef * value_loss
                    - cfg.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                all_params = (
                    list(self.trunk.parameters())
                    + list(self.penalty_head.parameters())
                    + list(self.operator_head.parameters())
                    + list(self.value_head.parameters())
                )
                nn.utils.clip_grad_norm_(all_params, cfg.max_grad_norm)
                self.optimizer.step()

                total_pl += policy_loss.item()
                total_vl += value_loss.item()
                total_ent += entropy_loss.item()
                total_kl += approx_kl
                n_up += 1

        # Clear buffer
        for lst in [self.buf_states, self.buf_pen_acts, self.buf_op_acts,
                     self.buf_pen_lp, self.buf_op_lp,
                     self.buf_rewards, self.buf_values, self.buf_dones]:
            lst.clear()

        self.update_count += 1
        return {
            "policy_loss": total_pl / max(n_up, 1),
            "value_loss": total_vl / max(n_up, 1),
            "entropy": total_ent / max(n_up, 1),
            "approx_kl": total_kl / max(n_up, 1),
        }

    def save(self, path: str):
        import torch
        torch.save({
            "trunk": self.trunk.state_dict(),
            "penalty_head": self.penalty_head.state_dict(),
            "operator_head": self.operator_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update_count": self.update_count,
        }, path)

    def load(self, path: str):
        import torch
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.trunk.load_state_dict(ckpt["trunk"])
        self.penalty_head.load_state_dict(ckpt["penalty_head"])
        self.operator_head.load_state_dict(ckpt["operator_head"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.update_count = ckpt["update_count"]


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_agent(agent_type: str, config) -> BaseAgent:
    """
    Create an agent by type name.

    Args:
        agent_type: "tabular", "dqn", "reinforce", or "ppo"
        config: The corresponding config dataclass (TabularConfig, DQNConfig, etc.)

    Returns:
        An agent implementing BaseAgent.
    """
    agent_type = agent_type.lower().strip()

    if agent_type == "tabular":
        return TabularQLearningAgent(config)
    elif agent_type == "dqn":
        return DQNAgent(config)
    elif agent_type == "reinforce":
        return REINFORCEAgent(config)
    elif agent_type == "ppo":
        return PPOAgent(config)
    else:
        raise ValueError(
            f"Unknown agent type '{agent_type}'. "
            f"Choose from: tabular, dqn, reinforce, ppo"
        )
