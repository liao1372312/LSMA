"""
DQN Memory Controller  (Section III-G)
========================================
Implements the RL-based storage control mechanism that decides, for each
candidate memory item in STM, whether to commit it to LTM (action=1) or
discard it (action=0).

Architecture (Section IV-C):
    Three-layer fully connected network as Q-function approximator.

State vector (Equation 1):
    s_t^i = [eM(m_t^i);  eP(q_t, c_t);  SCORE_t;  nSTM;  nLTM]
    - eM   : embedding of the memory item      (dim = embedding_dim)
    - eP   : embedding of current request+ctx  (dim = embedding_dim)
    - SCORE: interaction-level feedback signal  (scalar)
    - nSTM : normalised STM size               (scalar)
    - nLTM : normalised LTM size               (scalar)

Reward (Equation 2):
    r_t^i = Ruser(t+1) - R̄_user   if a=1 and item retrieved at t+1
          = Reval(m_t^i)            if a=0
          = 0                       otherwise

Loss (Equation 3):
    L(θ) = E[(r + γ max_a' Q_θ-(s',a') - Q_θ(s,a))²]
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qos_lsma.rl.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Q-Network  (3-layer FC as specified in Section IV-C)
# -----------------------------------------------------------------------
class QNetwork(nn.Module):
    """Three-layer fully connected Q-function approximator.

    Parameters
    ----------
    state_dim:
        Dimensionality of the state vector.
    hidden_dims:
        Sizes of the two hidden layers (default: [256, 128]).
    n_actions:
        Number of discrete actions (2: discard=0, store=1).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = None,
        n_actions: int = 2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]
        layers: List[nn.Module] = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.ReLU()])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


# -----------------------------------------------------------------------
# DQN Memory Controller
# -----------------------------------------------------------------------
class DQNMemoryController:
    """RL-based storage controller using Deep Q-Network.

    Parameters
    ----------
    embedding_dim:
        Dimension of the embedding vectors (eM and eP).
    hidden_dims:
        Hidden layer sizes for the Q-network.
    lr:
        Learning rate.
    gamma:
        Discount factor γ.
    epsilon_start / epsilon_end / epsilon_decay:
        ε-greedy exploration parameters.
    batch_size:
        Mini-batch size for DQN updates.
    target_update_freq:
        Update target network every N DQN steps.
    replay_capacity:
        Replay buffer size.
    max_stm / max_ltm:
        Maximum memory sizes (used for state normalisation).
    device:
        Torch device (auto-detected if None).
    """

    ACTION_DISCARD = 0
    ACTION_STORE = 1

    def __init__(
        self,
        embedding_dim: int = 1536,
        hidden_dims: Optional[List[int]] = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        target_update_freq: int = 10,
        replay_capacity: int = 10_000,
        max_stm: int = 100,
        max_ltm: int = 10_000,
        device: Optional[str] = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.max_stm = max_stm
        self.max_ltm = max_ltm

        # State dim: eM + eP + SCORE + nSTM + nLTM
        self.state_dim = embedding_dim * 2 + 3

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        hidden_dims = hidden_dims or [256, 128]
        self.q_net = QNetwork(self.state_dim, hidden_dims).to(self.device)
        self.target_net = copy.deepcopy(self.q_net).to(self.device)
        self.target_net.eval()

        self.optimiser = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        self._step = 0           # total DQN update steps
        self._interaction = 0    # total interactions (for target net sync)

        # Retrieval trace for delayed credit assignment (Section III-G-g)
        # Maps item_id → {"state": s, "action": a} for stored items
        self._pending_credit: Dict[str, Dict[str, Any]] = {}

        # Running baseline R̄_user for reward variance reduction (Eq. 2)
        self._r_user_baseline: float = 0.0

        logger.info(
            "DQNMemoryController: state_dim=%d, device=%s",
            self.state_dim, self.device,
        )

    # ------------------------------------------------------------------
    # State construction  (Equation 1)
    # ------------------------------------------------------------------
    def build_state(
        self,
        item_embedding: List[float],
        query_embedding: List[float],
        score: float,
        n_stm: int,
        n_ltm: int,
    ) -> np.ndarray:
        """Construct state vector s_t^i.

        s = [eM(m_t^i); eP(q_t, c_t); SCORE_t; nSTM_norm; nLTM_norm]
        """
        em = np.array(item_embedding, dtype=np.float32)
        ep = np.array(query_embedding, dtype=np.float32)
        extra = np.array(
            [
                float(score),
                float(n_stm) / max(self.max_stm, 1),   # normalise
                float(n_ltm) / max(self.max_ltm, 1),   # normalise
            ],
            dtype=np.float32,
        )
        return np.concatenate([em, ep, extra])

    # ------------------------------------------------------------------
    # Action selection (ε-greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        """Return 0 (discard) or 1 (store) for the given state."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 2)

        self.q_net.eval()
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_vals = self.q_net(s)
            action = int(q_vals.argmax(dim=1).item())
        return action

    def decay_epsilon(self) -> None:
        """Apply ε decay after each interaction."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Reward computation  (Equation 2)
    # ------------------------------------------------------------------
    def compute_reward_discard(self, item_embedding: List[float]) -> float:
        """Compute Reval(m_t^i) for a discarded item.

        Uses a lightweight heuristic:  norm of the embedding as a proxy
        for information content (higher norm → more informative).
        Scaled to [0, reval_max_score].
        """
        vec = np.array(item_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        # Normalise: typical L2 norm of a unit embedding ~1; raw ~20–50
        reval = min(norm / 10.0, 5.0)
        return reval

    def receive_delayed_reward(
        self,
        retrieved_item_ids: List[str],
        user_score: float,
    ) -> None:
        """Assign delayed reward to items that were retrieved at t+1.

        Called at the START of each new interaction, AFTER retrieval,
        to credit items that contributed to the previous interaction.

        Parameters
        ----------
        retrieved_item_ids:
            IDs of memory items retrieved and used in the current interaction.
        user_score:
            Feedback or success proxy for the CURRENT (t+1) interaction.
        """
        r_delayed = user_score - self._r_user_baseline

        # Update running baseline (EMA)
        self._r_user_baseline = (
            0.9 * self._r_user_baseline + 0.1 * user_score
        )

        for item_id in retrieved_item_ids:
            if item_id in self._pending_credit:
                pending = self._pending_credit.pop(item_id)
                s = pending["state"]
                a = pending["action"]          # was 1 (store)
                # We use a dummy next_state (zeros) with done=True
                # because the delayed feedback closes the episode for
                # this item.
                s_next = np.zeros_like(s)
                self.replay_buffer.push(s, a, r_delayed, s_next, done=True)

    # ------------------------------------------------------------------
    # DQN training step  (Equation 3)
    # ------------------------------------------------------------------
    def update(self) -> Optional[float]:
        """Sample a mini-batch and perform one DQN gradient update.

        Returns the scalar loss value, or None if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        states_t = torch.tensor(states).to(self.device)
        actions_t = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.tensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.tensor(next_states).to(self.device)
        dones_t = torch.tensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a)
        self.q_net.train()
        q_vals = self.q_net(states_t).gather(1, actions_t)

        # max_a' Q_θ-(s', a')
        with torch.no_grad():
            q_next = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target = rewards_t + self.gamma * q_next * (1.0 - dones_t)

        # TD loss  (Equation 3)
        loss = nn.functional.mse_loss(q_vals, target)
        self.optimiser.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimiser.step()

        self._step += 1
        # Periodically copy online weights → target network
        if self._step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            logger.debug("DQN: updated target network at step %d.", self._step)

        return float(loss.item())

    # ------------------------------------------------------------------
    # Online decision loop for a batch of STM items
    # ------------------------------------------------------------------
    def decide_batch(
        self,
        stm_items,          # List[MemoryItem]
        query_embedding: List[float],
        score: float,
        n_stm: int,
        n_ltm: int,
    ) -> Tuple[List, List]:
        """Decide store/discard for all items in an STM batch.

        Returns
        -------
        (to_store, to_discard):
            Two lists of MemoryItem.
        """
        to_store = []
        to_discard = []

        for item in stm_items:
            emb = item.embedding or [0.0] * self.embedding_dim
            state = self.build_state(emb, query_embedding, score, n_stm, n_ltm)
            action = self.select_action(state)

            if action == self.ACTION_STORE:
                to_store.append(item)
                # Register for potential delayed credit
                self._pending_credit[item.item_id] = {
                    "state": state,
                    "action": action,
                }
                # Immediate reward = 0 (delayed feedback will come later)
                s_next = np.zeros_like(state)
                self.replay_buffer.push(state, action, 0.0, s_next, done=False)
            else:
                to_discard.append(item)
                # Immediate surrogate reward
                reward = self.compute_reward_discard(emb)
                s_next = np.zeros_like(state)
                self.replay_buffer.push(state, action, reward, s_next, done=True)

        return to_store, to_discard

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "epsilon": self.epsilon,
                "step": self._step,
                "r_user_baseline": self._r_user_baseline,
            },
            path,
        )
        logger.info("DQNMemoryController saved to %s", path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimiser.load_state_dict(ckpt["optimiser"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._step = ckpt.get("step", 0)
        self._r_user_baseline = ckpt.get("r_user_baseline", 0.0)
        logger.info("DQNMemoryController loaded from %s", path)
