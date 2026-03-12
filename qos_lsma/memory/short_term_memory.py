"""
Short-Term Memory (STM) Buffer
==============================
A FIFO queue that temporarily holds candidate memory items distilled
from recent interactions before the RL controller decides whether to
promote them to Long-Term Memory (Section III-G-a).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List

from qos_lsma.memory.memory_item import MemoryItem


class ShortTermMemory:
    """In-memory FIFO buffer for newly distilled interaction fragments.

    Parameters
    ----------
    max_size:
        Maximum number of items to retain.  Oldest items are dropped when
        the buffer is full (simulating a sliding window).
    """

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max_size
        self._buffer: Deque[MemoryItem] = deque()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def insert(self, item: MemoryItem) -> None:
        """Add an item to the STM buffer, evicting the oldest if full."""
        if len(self._buffer) >= self.max_size:
            self._buffer.popleft()
        self._buffer.append(item)

    def insert_many(self, items: Iterable[MemoryItem]) -> None:
        for item in items:
            self.insert(item)

    def pop_all(self) -> List[MemoryItem]:
        """Remove and return all items currently in the buffer."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    def peek_all(self) -> List[MemoryItem]:
        """Return all items without removing them."""
        return list(self._buffer)

    def remove(self, item_id: str) -> bool:
        """Remove a specific item by id.  Returns True if found."""
        for i, item in enumerate(self._buffer):
            if item.item_id == item_id:
                del self._buffer[i]
                return True
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def size(self) -> int:
        return len(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return f"ShortTermMemory(size={self.size}/{self.max_size})"
