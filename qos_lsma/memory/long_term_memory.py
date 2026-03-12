"""
Long-Term Memory (LTM) – Graph Store
=====================================
Implements the typed property graph  G = (V, E, L)  described in
Section III-D-a of the paper.

Each node v ∈ V stores:
    • semantic type  t(v) ∈ {Intent, UserProfile, Service, Workflow, Strategy}
    • embedding vector  ev  (for semantic matching)
    • metadata (timestamps, context tags)

Each directed edge (u, v) ∈ E carries a label  ℓ(u, v) ∈ L, representing
a relation triple (uses, calls, mitigates, failed_because, fixed_by, …).
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from qos_lsma.memory.memory_item import MemoryItem, NODE_TYPES, EDGE_LABELS


class LongTermMemory:
    """Graph-based Long-Term Memory built on top of NetworkX.

    Parameters
    ----------
    max_nodes:
        Hard cap on the number of nodes to prevent unbounded growth.
        When the cap is reached, the node with the lowest utility score
        is evicted.
    """

    def __init__(self, max_nodes: int = 10_000) -> None:
        self.max_nodes = max_nodes
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        node_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        utility_score: float = 0.0,
    ) -> str:
        """Add (or update) a node in the memory graph.

        Returns the node_id.
        """
        if node_type not in NODE_TYPES:
            raise ValueError(
                f"Unknown node type '{node_type}'. Valid: {NODE_TYPES}"
            )
        if len(self._graph.nodes) >= self.max_nodes and node_id not in self._graph:
            self._evict_one()

        self._graph.add_node(
            node_id,
            node_type=node_type,
            content=content,
            embedding=embedding or [],
            metadata=metadata or {},
            utility_score=utility_score,
            created_at=time.time(),
            retrieval_count=0,
        )
        return node_id

    def update_node_embedding(
        self, node_id: str, embedding: List[float]
    ) -> None:
        if node_id in self._graph:
            self._graph.nodes[node_id]["embedding"] = embedding

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------
    def add_edge(
        self,
        src: str,
        dst: str,
        label: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed, labelled edge  (src) --[label]--> (dst)."""
        if label not in EDGE_LABELS:
            raise ValueError(
                f"Unknown edge label '{label}'. Valid: {EDGE_LABELS}"
            )
        self._graph.add_edge(src, dst, label=label, metadata=metadata or {})

    # ------------------------------------------------------------------
    # Commit a MemoryItem → graph nodes + edges
    # ------------------------------------------------------------------
    def commit_item(self, item: MemoryItem) -> str:
        """Write a MemoryItem into the graph as typed nodes and relations.

        Creates one primary node representing the item itself, plus
        lightweight entity nodes and relation edges extracted from the item.

        Returns the primary node_id.
        """
        # Primary node
        primary_id = item.node_id or str(uuid.uuid4())
        item.node_id = primary_id

        self.add_node(
            node_id=primary_id,
            node_type=self._category_to_node_type(item.category),
            content=item.content,
            embedding=item.embedding,
            metadata={**item.metadata, "item_id": item.item_id,
                      "category": item.category,
                      "created_at": item.created_at},
            utility_score=item.utility_score,
        )

        # Entity nodes
        entity_id_map: Dict[str, str] = {}
        for ent in item.entities:
            ent_name = ent.get("name", "")
            ent_type = ent.get("type", "Intent")
            if ent_type not in NODE_TYPES:
                ent_type = "Intent"
            ent_id = f"entity::{ent_type}::{ent_name}"
            if ent_id not in self._graph:
                self.add_node(
                    node_id=ent_id,
                    node_type=ent_type,
                    content=ent_name,
                    metadata={"source_item": item.item_id},
                )
            entity_id_map[ent_name] = ent_id
            # Link primary node to entity
            self.add_edge(primary_id, ent_id, label="uses")

        # Relation edges between entities
        for rel in item.relations:
            head = rel.get("head", "")
            tail = rel.get("tail", "")
            label = rel.get("label", "uses")
            if label not in EDGE_LABELS:
                label = "uses"
            head_id = entity_id_map.get(head)
            tail_id = entity_id_map.get(tail)
            if head_id and tail_id:
                self.add_edge(head_id, tail_id, label=label)

        return primary_id

    # ------------------------------------------------------------------
    # Neighbourhood expansion  (Section III-E, Step 3)
    # ------------------------------------------------------------------
    def get_neighborhood(
        self,
        node_id: str,
        hops: int = 2,
        max_edges: int = 20,
        priority_labels: Optional[Set[str]] = None,
    ) -> nx.DiGraph:
        """Return a subgraph containing all nodes within *hops* of node_id.

        Priority edges (uses, calls, failed_because, fixed_by) are included
        first; remaining edges are added until max_edges is reached.
        """
        if node_id not in self._graph:
            return nx.DiGraph()

        visited: Set[str] = {node_id}
        frontier: Set[str] = {node_id}
        edges: List[Tuple[str, str, Dict]] = []

        for _ in range(hops):
            next_frontier: Set[str] = set()
            for n in frontier:
                for nbr in list(self._graph.predecessors(n)) + list(
                    self._graph.successors(n)
                ):
                    if nbr not in visited:
                        visited.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier

        sub_nodes = visited
        candidate_edges: List[Tuple[str, str, Dict]] = []
        for u, v, data in self._graph.edges(sub_nodes, data=True):
            if v in sub_nodes:
                candidate_edges.append((u, v, data))

        # Prioritise important relation labels
        if priority_labels:
            candidate_edges.sort(
                key=lambda e: (0 if e[2].get("label") in priority_labels else 1)
            )

        edges = candidate_edges[:max_edges]

        subgraph = nx.DiGraph()
        for n in sub_nodes:
            subgraph.add_node(n, **self._graph.nodes[n])
        for u, v, data in edges:
            subgraph.add_edge(u, v, **data)

        return subgraph

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        if node_id in self._graph:
            return dict(self._graph.nodes[node_id])
        return None

    def increment_retrieval(self, node_id: str) -> None:
        if node_id in self._graph:
            self._graph.nodes[node_id]["retrieval_count"] += 1

    def update_utility(self, node_id: str, score: float) -> None:
        if node_id in self._graph:
            self._graph.nodes[node_id]["utility_score"] = score

    # ------------------------------------------------------------------
    # Stats / helpers
    # ------------------------------------------------------------------
    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    def all_node_ids(self) -> List[str]:
        return list(self._graph.nodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _category_to_node_type(category: str) -> str:
        mapping = {
            "profile_fact": "UserProfile",
            "workflow_trace": "Workflow",
            "service_usage": "Service",
        }
        return mapping.get(category, "Intent")

    def _evict_one(self) -> None:
        """Remove the node with the lowest utility score."""
        worst = min(
            self._graph.nodes(data=True),
            key=lambda n: n[1].get("utility_score", 0.0),
        )
        self._graph.remove_node(worst[0])

    def __repr__(self) -> str:
        return (
            f"LongTermMemory(nodes={self.num_nodes}, edges={self.num_edges})"
        )
