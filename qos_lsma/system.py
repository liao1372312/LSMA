"""
QoS-LSMA – Main System  (Algorithm 1: online loop)
====================================================
Coordinates all modules into the end-to-end retrieve → compose → store
pipeline described in the paper.

Online loop (Algorithm 1):
  1.  Parse q → structured request q̂
  2.  Compute query embedding eq
  3.  Retrieve top-K candidate memory nodes  C ← TopK(I, eq)
  4.  Expand neighbourhood subgraph  Gq
  5.  Summarise Gq → context Cq
  6.  Planner proposes workflow W conditioned on (q, c, Cq)
  7.  Service Provider grounds W to concrete services/APIs in S
  8.  Executor runs the workflow; Supervisor monitors
  9.  Summarizer distills interaction → {mi} → STM
  10. DQN controller: for each mi ∈ STM → {store, discard}
  11. Commit stored items to LTM + update embedding index
  12. DQN update (mini-batch gradient step)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from qos_lsma.config import QoSLSMAConfig
from qos_lsma.memory import (
    EmbeddingIndex,
    LongTermMemory,
    MemoryItem,
    ShortTermMemory,
)
from qos_lsma.retrieval import RetrievalModule
from qos_lsma.agents import (
    ExecutorAgent,
    PlannerAgent,
    ServiceProviderAgent,
    SummarizerAgent,
    SupervisorAgent,
)
from qos_lsma.rl import DQNMemoryController

logger = logging.getLogger(__name__)


class QoSLSMA:
    """Memory-Augmented Multi-Agent Framework for Microservice Composition.

    Parameters
    ----------
    config:
        QoSLSMAConfig instance with all hyper-parameters and credentials.
    service_catalog:
        List of service/API dicts.  Each should contain at minimum
        ``name``, ``description``, and ``signature`` keys.
    service_registry:
        Optional dict mapping service names to Python callables for
        direct invocation (bypasses LLM simulation).
    embed_fn:
        Optional custom embedding callable.  If None, the system uses
        the OpenAI embeddings API specified in ``config``.
    """

    def __init__(
        self,
        config: QoSLSMAConfig,
        service_catalog: Optional[List[Dict[str, Any]]] = None,
        service_registry: Optional[Dict[str, Callable]] = None,
        embed_fn: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.service_catalog = service_catalog or []

        # ------------------------------------------------------------------
        # Embedding function
        # ------------------------------------------------------------------
        self.embed_fn = embed_fn or self._build_embed_fn()

        # ------------------------------------------------------------------
        # Memory system
        # ------------------------------------------------------------------
        self.ltm = LongTermMemory(max_nodes=config.max_ltm_nodes)
        self.stm = ShortTermMemory(max_size=config.max_stm_size)
        self.index = EmbeddingIndex(dim=config.embedding_dim)

        # ------------------------------------------------------------------
        # Retrieval module
        # ------------------------------------------------------------------
        self.retrieval = RetrievalModule(
            ltm=self.ltm,
            index=self.index,
            embed_fn=self.embed_fn,
            top_k=config.top_k_candidates,
            hop_size=config.hop_size,
            max_edges_per_seed=config.max_edges_per_seed,
            priority_relations=config.priority_relations,
        )

        # ------------------------------------------------------------------
        # Agents (all use the same LLM config)
        # ------------------------------------------------------------------
        agent_kwargs = dict(
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            max_retries=config.max_retries,
        )
        self.planner = PlannerAgent(**agent_kwargs)
        self.service_provider = ServiceProviderAgent(**agent_kwargs)
        self.executor = ExecutorAgent(
            service_registry=service_registry, **agent_kwargs
        )
        self.supervisor = SupervisorAgent(**agent_kwargs)
        self.summarizer = SummarizerAgent(**agent_kwargs)

        # ------------------------------------------------------------------
        # DQN memory controller
        # ------------------------------------------------------------------
        self.dqn = DQNMemoryController(
            embedding_dim=config.embedding_dim,
            hidden_dims=config.dqn_hidden_dims,
            lr=config.dqn_lr,
            gamma=config.dqn_gamma,
            epsilon_start=config.dqn_epsilon_start,
            epsilon_end=config.dqn_epsilon_end,
            epsilon_decay=config.dqn_epsilon_decay,
            batch_size=config.dqn_batch_size,
            target_update_freq=config.dqn_target_update_freq,
            replay_capacity=config.replay_buffer_size,
            max_stm=config.max_stm_size,
            max_ltm=config.max_ltm_nodes,
        )

        self._interaction_count: int = 0
        # Track which node ids were retrieved in the last interaction
        self._last_retrieved_node_ids: List[str] = []

    # ------------------------------------------------------------------
    # Main entry point – Algorithm 1
    # ------------------------------------------------------------------
    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        user_score: Optional[float] = None,
        metadata_filter: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Execute one interaction of the online loop.

        Parameters
        ----------
        query:
            Free-form user request  q.
        context:
            Runtime context  c  (region, time_bucket, input_scale, etc.).
        user_score:
            User feedback / success proxy for the PREVIOUS interaction
            (used for delayed reward assignment).  Pass None on first call.
        metadata_filter:
            Optional metadata pre-filter for memory retrieval.

        Returns
        -------
        Dict with keys:
          ``workflow``          – planned subtask list
          ``groundings``        – service grounding list
          ``execution_results`` – execution outcomes
          ``memory_context``    – retrieved memory context string
          ``stored_items``      – items committed to LTM this interaction
          ``dqn_loss``          – DQN training loss (or None)
        """
        context = context or {}
        self._interaction_count += 1
        logger.info(
            "=== Interaction #%d | query: %s",
            self._interaction_count, query[:80],
        )

        # ---- Delayed reward from previous interaction -----------------
        if user_score is not None and self._last_retrieved_node_ids:
            self.dqn.receive_delayed_reward(
                retrieved_item_ids=self._last_retrieved_node_ids,
                user_score=user_score,
            )

        # ---- Step 1–5: Memory retrieval ------------------------------
        memory_context = self.retrieval.retrieve(
            query=query,
            context=context,
            metadata_filter=metadata_filter,
        )
        logger.debug("Memory context:\n%s", memory_context)

        # Track which items were retrieved for delayed credit assignment
        query_emb = self.embed_fn(query)
        candidates = self.index.top_k(query_emb, k=self.config.top_k_candidates)
        self._last_retrieved_node_ids = [nid for nid, _ in candidates]

        # ---- Step 6: Planning ----------------------------------------
        workflow = self.planner.run(
            query=query,
            context=context,
            memory_context=memory_context,
        )
        logger.info("Workflow: %d steps", len(workflow))

        # ---- Step 7: Service grounding -------------------------------
        groundings = self.service_provider.run(
            workflow=workflow,
            service_catalog=self.service_catalog,
            memory_context=memory_context,
            query=query,
        )

        # ---- Step 8: Execution + supervision -------------------------
        raw_results = self.executor.run(groundings)
        if self.config.supervisor_enabled:
            execution_results = self.supervisor.evaluate(
                raw_results, groundings, self.executor
            )
        else:
            execution_results = raw_results

        overall_success = all(r.get("success") for r in execution_results)
        score_proxy = 5.0 if overall_success else 1.0
        logger.info(
            "Execution: %d/%d steps succeeded.",
            sum(r.get("success") for r in execution_results),
            len(execution_results),
        )

        # ---- Step 9: Summarise → STM ---------------------------------
        candidate_items = self.summarizer.run(
            query=query,
            workflow=workflow,
            groundings=groundings,
            execution_results=execution_results,
            context=context,
            user_score=score_proxy,
        )
        # Embed each item before putting into STM
        for item in candidate_items:
            if not item.embedding:
                item.embedding = self.embed_fn(item.content)
        self.stm.insert_many(candidate_items)

        # ---- Step 10: DQN storage decision ---------------------------
        stm_items = self.stm.pop_all()
        to_store, to_discard = self.dqn.decide_batch(
            stm_items=stm_items,
            query_embedding=query_emb,
            score=score_proxy,
            n_stm=len(stm_items),
            n_ltm=self.ltm.num_nodes,
        )
        logger.info(
            "DQN decision: store=%d, discard=%d",
            len(to_store), len(to_discard),
        )

        # ---- Step 11: Commit to LTM + update index -------------------
        for item in to_store:
            node_id = self.ltm.commit_item(item)
            if item.embedding:
                self.index.add(node_id, item.embedding)
            # Also index entity nodes just added
            for entity in item.entities:
                ent_id = f"entity::{entity.get('type','Intent')}::{entity.get('name','')}"
                ent_node = self.ltm.get_node(ent_id)
                if ent_node and not ent_node.get("embedding"):
                    ent_emb = self.embed_fn(entity.get("name", ""))
                    self.ltm.update_node_embedding(ent_id, ent_emb)
                    self.index.add(ent_id, ent_emb)

        # ---- Step 12: DQN update -------------------------------------
        dqn_loss = None
        if self._interaction_count % self.config.dqn_update_every == 0:
            dqn_loss = self.dqn.update()
            self.dqn.decay_epsilon()
            logger.debug("DQN loss: %s, ε=%.4f", dqn_loss, self.dqn.epsilon)

        return {
            "workflow": workflow,
            "groundings": groundings,
            "execution_results": execution_results,
            "memory_context": memory_context,
            "stored_items": [item.to_dict() for item in to_store],
            "dqn_loss": dqn_loss,
            "interaction_id": self._interaction_count,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_dqn(self, path: str) -> None:
        """Save DQN weights and training state."""
        self.dqn.save(path)

    def load_dqn(self, path: str) -> None:
        """Load DQN weights and training state."""
        self.dqn.load(path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_embed_fn(self) -> Callable:
        """Build the OpenAI embedding callable from config."""
        from openai import OpenAI
        client = OpenAI(
            api_key=self.config.embedding_api_key or self.config.llm_api_key,
            base_url=self.config.embedding_base_url,
        )
        model = self.config.embedding_model
        dim = self.config.embedding_dim

        def embed(text: str) -> List[float]:
            try:
                resp = client.embeddings.create(input=text, model=model)
                return resp.data[0].embedding
            except Exception as e:  # noqa: BLE001
                logger.warning("Embedding failed (%s). Returning zero vector.", e)
                return [0.0] * dim

        return embed

    def __repr__(self) -> str:
        return (
            f"QoSLSMA("
            f"interactions={self._interaction_count}, "
            f"ltm_nodes={self.ltm.num_nodes}, "
            f"index_size={self.index.size})"
        )
