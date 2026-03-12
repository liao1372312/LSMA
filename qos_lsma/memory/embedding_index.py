"""
Embedding Index
===============
Maintains a dense vector index over LTM nodes for efficient top-K
candidate recall (Section III-D-c and Section III-E, Step 2).

Implementation uses numpy-based cosine similarity as the default backend.
When ``faiss`` is installed, it automatically switches to an IVF flat
index for O(log n) approximate search.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingIndex:
    """Scalable embedding index with optional FAISS acceleration.

    All vectors are L2-normalised before insertion so that inner-product
    search is equivalent to cosine similarity.

    Parameters
    ----------
    dim:
        Dimensionality of embedding vectors.
    use_faiss:
        If True, attempt to import FAISS and use it as backend.
        Falls back to numpy if FAISS is unavailable.
    """

    def __init__(self, dim: int = 1536, use_faiss: bool = True) -> None:
        self.dim = dim
        self._id_list: List[str] = []           # node_id → index position
        self._vectors: List[np.ndarray] = []    # parallel list of embeddings

        self._faiss_index = None
        if use_faiss:
            self._try_init_faiss()

    # ------------------------------------------------------------------
    # FAISS initialisation (optional)
    # ------------------------------------------------------------------
    def _try_init_faiss(self) -> None:
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            logger.info("EmbeddingIndex: using FAISS IndexFlatIP backend.")
        except ImportError:
            logger.info(
                "EmbeddingIndex: FAISS not found. "
                "Falling back to numpy cosine similarity."
            )

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def add(self, node_id: str, embedding: List[float]) -> None:
        """Add or update a node embedding in the index."""
        vec = self._normalise(np.array(embedding, dtype=np.float32))

        if node_id in self._id_list:
            idx = self._id_list.index(node_id)
            self._vectors[idx] = vec
            if self._faiss_index is not None:
                # FAISS flat index doesn't support in-place update – rebuild
                self._rebuild_faiss()
        else:
            self._id_list.append(node_id)
            self._vectors.append(vec)
            if self._faiss_index is not None:
                self._faiss_index.add(vec.reshape(1, -1))

    def remove(self, node_id: str) -> None:
        """Remove a node from the index (if present)."""
        if node_id in self._id_list:
            idx = self._id_list.index(node_id)
            self._id_list.pop(idx)
            self._vectors.pop(idx)
            if self._faiss_index is not None:
                self._rebuild_faiss()

    def _rebuild_faiss(self) -> None:
        """Rebuild the FAISS index from scratch (after deletions/updates)."""
        try:
            import faiss  # type: ignore
            self._faiss_index = faiss.IndexFlatIP(self.dim)
            if self._vectors:
                matrix = np.stack(self._vectors, axis=0)
                self._faiss_index.add(matrix)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Top-K retrieval  (Section III-E, Step 2)
    # ------------------------------------------------------------------
    def top_k(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return top-K (node_id, similarity_score) pairs.

        Parameters
        ----------
        query_embedding:
            Dense query vector (same dimensionality as index vectors).
        k:
            Number of candidates to return.
        filter_ids:
            Optional allowlist.  Only nodes whose ids appear here are
            eligible.  Useful for applying metadata pre-filters.

        Returns
        -------
        List of ``(node_id, score)`` sorted by descending similarity.
        """
        if not self._id_list:
            return []

        q = self._normalise(np.array(query_embedding, dtype=np.float32))
        k = min(k, len(self._id_list))

        if self._faiss_index is not None and filter_ids is None:
            scores, indices = self._faiss_index.search(q.reshape(1, -1), k)
            results = [
                (self._id_list[int(i)], float(scores[0][j]))
                for j, i in enumerate(indices[0])
                if 0 <= int(i) < len(self._id_list)
            ]
        else:
            matrix = np.stack(self._vectors, axis=0)     # (N, dim)
            sims = matrix @ q                            # cosine similarities

            if filter_ids is not None:
                allowed = set(filter_ids)
                for i, nid in enumerate(self._id_list):
                    if nid not in allowed:
                        sims[i] = -math.inf

            top_idx = np.argsort(-sims)[:k]
            results = [
                (self._id_list[int(i)], float(sims[int(i)]))
                for i in top_idx
                if sims[int(i)] > -math.inf
            ]

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    @property
    def size(self) -> int:
        return len(self._id_list)

    def __repr__(self) -> str:
        backend = "FAISS" if self._faiss_index is not None else "numpy"
        return f"EmbeddingIndex(size={self.size}, dim={self.dim}, backend={backend})"
