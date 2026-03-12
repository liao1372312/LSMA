"""
Memory Retrieval Module
========================
Implements the four-step retrieval pipeline from Section III-E:

  Step 1 – Query parsing and entity extraction
  Step 2 – Top-K candidate recall from the embedding index
  Step 3 – Subgraph expansion in the memory graph
  Step 4 – Memory-to-context summarisation

The output is a structured context string  C_q  that is injected into
the Planner's prompt.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from qos_lsma.memory.embedding_index import EmbeddingIndex
from qos_lsma.memory.long_term_memory import LongTermMemory

logger = logging.getLogger(__name__)


class RetrievalModule:
    """Encapsulates the full memory retrieval pipeline.

    Parameters
    ----------
    ltm:
        The long-term memory graph.
    index:
        The embedding index over LTM nodes.
    embed_fn:
        A callable  ``embed_fn(text: str) -> List[float]``  that maps a
        string to a dense embedding vector.  Typically backed by the
        OpenAI embeddings API.
    top_k:
        Number of seed nodes to retrieve from the embedding index.
    hop_size:
        Neighbourhood expansion depth (1 or 2).
    max_edges_per_seed:
        Cap on edges per seed to limit subgraph size.
    priority_relations:
        Edge labels given priority during subgraph construction.
    """

    def __init__(
        self,
        ltm: LongTermMemory,
        index: EmbeddingIndex,
        embed_fn,
        top_k: int = 10,
        hop_size: int = 2,
        max_edges_per_seed: int = 20,
        priority_relations: Optional[List[str]] = None,
    ) -> None:
        self.ltm = ltm
        self.index = index
        self.embed_fn = embed_fn
        self.top_k = top_k
        self.hop_size = hop_size
        self.max_edges_per_seed = max_edges_per_seed
        self.priority_relations: Set[str] = set(
            priority_relations or ["uses", "calls", "failed_because", "fixed_by"]
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> str:
        """Run the full 4-step retrieval and return the context string C_q.

        Parameters
        ----------
        query:
            Raw user request  q.
        context:
            Runtime context  c  (region, time_bucket, input_scale, etc.).
        metadata_filter:
            Optional key-value pairs for pre-filtering candidates by
            metadata (e.g. service version or region bucket).

        Returns
        -------
        A formatted string ready to be appended to a planner prompt.
        """
        context = context or {}

        # ---------- Step 1: query parsing ----------
        structured_query, entities = self._parse_query(query, context)
        logger.debug("Structured query: %s | Entities: %s", structured_query, entities)

        # ---------- Step 2: top-K recall ----------
        query_embedding = self.embed_fn(structured_query)
        filter_ids = self._apply_metadata_filter(metadata_filter)
        candidates = self.index.top_k(query_embedding, k=self.top_k,
                                      filter_ids=filter_ids)
        logger.debug("Candidates: %s", [(c[0][:12], round(c[1], 4)) for c in candidates])

        if not candidates:
            return self._empty_context()

        # Mark retrieved nodes
        for node_id, _ in candidates:
            self.ltm.increment_retrieval(node_id)

        # ---------- Step 3: subgraph expansion ----------
        seed_ids = [nid for nid, _ in candidates]
        subgraph = self._expand_subgraph(seed_ids)

        # ---------- Step 4: summarise to context string ----------
        context_str = self._summarise(subgraph, entities, context)
        return context_str

    # ------------------------------------------------------------------
    # Step 1: query parsing
    # ------------------------------------------------------------------
    def _parse_query(
        self,
        query: str,
        context: Dict[str, Any],
    ) -> tuple[str, List[str]]:
        """Build a structured query representation and extract entities.

        In a full deployment this step calls an LLM to extract entities
        and constraints.  Here we use a lightweight keyword approach so
        that the module works without an LLM call on every retrieval.
        """
        # Simple tokenisation – extract capitalised or quoted tokens as entities
        import re
        entities = re.findall(r'"([^"]+)"|([A-Z][a-z]+(?:[A-Z][a-z]+)*)', query)
        flat_entities = [e[0] or e[1] for e in entities]

        # Append compact context tags
        ctx_tags = " ".join(f"{k}:{v}" for k, v in context.items() if v)
        structured = f"{query} {ctx_tags}".strip()
        return structured, flat_entities

    # ------------------------------------------------------------------
    # Step 2 helper: metadata pre-filter
    # ------------------------------------------------------------------
    def _apply_metadata_filter(
        self, metadata_filter: Optional[Dict[str, str]]
    ) -> Optional[List[str]]:
        if not metadata_filter:
            return None
        allowed = []
        for nid in self.ltm.all_node_ids():
            node = self.ltm.get_node(nid)
            if node is None:
                continue
            meta = node.get("metadata", {})
            if all(meta.get(k) == v for k, v in metadata_filter.items()):
                allowed.append(nid)
        return allowed if allowed else None

    # ------------------------------------------------------------------
    # Step 3: subgraph expansion  (Equation in Section III-E)
    # ------------------------------------------------------------------
    def _expand_subgraph(self, seed_ids: List[str]) -> nx.DiGraph:
        """Merge h-hop neighbourhoods around all seed nodes.

        G_q = ⋃_{v ∈ C} N_h(v),   h ∈ {1, 2}
        """
        merged = nx.DiGraph()
        for seed_id in seed_ids:
            sub = self.ltm.get_neighborhood(
                seed_id,
                hops=self.hop_size,
                max_edges=self.max_edges_per_seed,
                priority_labels=self.priority_relations,
            )
            merged = nx.compose(merged, sub)
        return merged

    # ------------------------------------------------------------------
    # Step 4: summarise subgraph → context string
    # ------------------------------------------------------------------
    def _summarise(
        self,
        subgraph: nx.DiGraph,
        entities: List[str],
        context: Dict[str, Any],
    ) -> str:
        """Convert the subgraph into a structured prompt-ready string.

        Follows the minimal template from Section III-E-d:
            (1) profile/constraints
            (2) similar workflow fragments
            (3) effective services/APIs and argument patterns
            (4) observed failures and mitigations
            (5) context tags
        """
        profile_facts: List[str] = []
        workflow_frags: List[str] = []
        service_usages: List[str] = []
        failure_fixes: List[str] = []

        for nid, data in subgraph.nodes(data=True):
            content = data.get("content", "")
            ntype = data.get("node_type", "")
            category = data.get("metadata", {}).get("category", "")

            if category == "profile_fact" or ntype == "UserProfile":
                profile_facts.append(content)
            elif category == "workflow_trace" or ntype == "Workflow":
                workflow_frags.append(content)
            elif category == "service_usage" or ntype == "Service":
                service_usages.append(content)

        # Collect failure→fix edges
        for u, v, edata in subgraph.edges(data=True):
            label = edata.get("label", "")
            if label in ("failed_because", "fixed_by", "mitigates"):
                u_content = subgraph.nodes[u].get("content", u[:20])
                v_content = subgraph.nodes[v].get("content", v[:20])
                failure_fixes.append(f"[{label}] {u_content} → {v_content}")

        ctx_tags = ", ".join(f"{k}={v}" for k, v in context.items() if v)

        def _fmt_list(items: List[str], max_items: int = 5) -> str:
            if not items:
                return "N/A"
            return "; ".join(items[:max_items])

        summary = (
            "Retrieved memory summary:\n"
            f"(1) profile/constraints: {_fmt_list(profile_facts)}\n"
            f"(2) similar workflow fragments: {_fmt_list(workflow_frags)}\n"
            f"(3) effective services/APIs and argument patterns: {_fmt_list(service_usages)}\n"
            f"(4) observed failures and mitigations: {_fmt_list(failure_fixes)}\n"
            f"(5) context tags (region/version/time): {ctx_tags or 'N/A'}"
        )
        return summary

    @staticmethod
    def _empty_context() -> str:
        return (
            "Retrieved memory summary:\n"
            "(1) profile/constraints: N/A\n"
            "(2) similar workflow fragments: N/A\n"
            "(3) effective services/APIs and argument patterns: N/A\n"
            "(4) observed failures and mitigations: N/A\n"
            "(5) context tags (region/version/time): N/A"
        )
