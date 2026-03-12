"""
Memory Item
===========
Structured record representing a single memory fragment distilled from
an interaction (Section III-D-b in the paper).

Three categories:
  • profile_fact   – inferred user constraints / intent descriptors
  • workflow_trace – successful workflow fragments under a context
  • service_usage  – concrete API selections, argument templates,
                     failure-to-fix traces
"""

from __future__ import annotations

import uuid
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------
# Node / Edge types  (Section III-D-a)
# -----------------------------------------------------------------------
NODE_TYPES = {"Intent", "UserProfile", "Service", "Workflow", "Strategy"}
EDGE_LABELS = {"uses", "calls", "mitigates", "failed_because", "fixed_by",
               "has_step", "depends_on", "prefers"}

MEMORY_CATEGORIES = {"profile_fact", "workflow_trace", "service_usage"}


@dataclass
class MemoryItem:
    """A compact, structured memory fragment.

    Attributes
    ----------
    item_id:      Unique identifier.
    category:     One of ``MEMORY_CATEGORIES``.
    content:      Human-readable text summary of the memory.
    entities:     Extracted entity dicts, e.g.
                  ``[{"name": "BookFlight", "type": "Intent"}]``.
    relations:    Relation triples, e.g.
                  ``[{"head": "BookFlight", "label": "uses", "tail": "FlightAPI"}]``.
    metadata:     Contextual tags – region, version, time_bucket, domain.
    embedding:    Dense vector (set by EmbeddingIndex after creation).
    created_at:   Unix timestamp of creation.
    utility_score:Running utility estimate (updated by RL controller).
    retrieval_count: How many times this item has been retrieved.
    node_id:      Corresponding node id in the LTM graph (set after commit).
    """

    item_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    category: str = "workflow_trace"
    content: str = ""
    entities: List[Dict[str, str]] = field(default_factory=list)
    relations: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = field(default=None, repr=False)
    created_at: float = field(default_factory=time.time)
    utility_score: float = 0.0
    retrieval_count: int = 0
    node_id: Optional[str] = None

    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.category not in MEMORY_CATEGORIES:
            raise ValueError(
                f"Invalid category '{self.category}'. "
                f"Must be one of {MEMORY_CATEGORIES}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "category": self.category,
            "content": self.content,
            "entities": self.entities,
            "relations": self.relations,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "utility_score": self.utility_score,
            "retrieval_count": self.retrieval_count,
            "node_id": self.node_id,
        }

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"MemoryItem(id={self.item_id[:8]}, "
            f"category={self.category}, "
            f"content={self.content[:60]!r})"
        )
