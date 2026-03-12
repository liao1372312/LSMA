"""
Replay Buffer
=============
Circular experience replay buffer for the DQN memory controller.
Stores transitions (s, a, r, s') and samples mini-batches for training.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, List, Tuple

import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]
"""Type alias: (state, action, reward, next_state, done)."""


class ReplayBuffer:
    """Fixed-size circular buffer for DQN experience replay.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.
    seed:
        Random seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 10_000, seed: int = 42) -> None:
        self.capacity = capacity
        self._buffer: Deque[Transition] = deque(maxlen=capacity)
        random.seed(seed)

    # ------------------------------------------------------------------
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Add a transition to the buffer."""
        self._buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Sample a random mini-batch.

        Returns
        -------
        Tuple of (states, actions, rewards, next_states, dones), each as
        a numpy array of shape (batch_size, …).
        """
        batch: List[Transition] = random.sample(self._buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states).astype(np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states).astype(np.float32),
            np.array(dones, dtype=np.float32),
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}/{self.capacity})"
