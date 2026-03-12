"""
Summarizer Agent  (Section III-C, Section III-G-a)
====================================================
After an interaction completes, the Summarizer distills the dialogue and
execution trace into candidate memory items {m_t^i} ready for STM insertion.

Each item is a compact structured record containing:
  • entity types and their names
  • relation triples between entities
  • human-readable content summary
  • metadata (domain, context tags, success/failure)

Category mapping:
  • profile_fact   – inferred user intent / constraint cues
  • workflow_trace – high-level plan fragments that led to success
  • service_usage  – concrete API selections, arguments, failure-fix traces
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from qos_lsma.agents.base_agent import BaseAgent
from qos_lsma.memory.memory_item import MemoryItem

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Summarizer Agent in the QoS-LSMA framework.

After an interaction, extract the following memory items from the interaction
trace and execution results.  Return ONLY valid JSON.

Three memory categories:
  1. profile_fact   – inferred user constraints and intent descriptors
  2. workflow_trace – workflow steps that succeeded
  3. service_usage  – concrete API calls, arguments, outcomes, failure fixes

Output schema:
{
  "memory_items": [
    {
      "category": "profile_fact" | "workflow_trace" | "service_usage",
      "content": "<concise natural-language summary>",
      "entities": [{"name": "<entity name>", "type": "<node type>"}],
      "relations": [{"head": "<entity>", "label": "<edge label>", "tail": "<entity>"}],
      "metadata": {"domain": "<domain>", "success": <bool>}
    }
  ]
}

Node types: Intent, UserProfile, Service, Workflow, Strategy
Edge labels: uses, calls, mitigates, failed_because, fixed_by, has_step, depends_on, prefers
"""


class SummarizerAgent(BaseAgent):
    """Distills an interaction trace into structured memory items."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="Summarizer", **kwargs)

    def run(
        self,
        query: str,
        workflow: List[Dict[str, Any]],
        groundings: List[Dict[str, Any]],
        execution_results: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        user_score: float = 0.0,
    ) -> List[MemoryItem]:
        """Distill interaction into candidate memory items.

        Parameters
        ----------
        query:
            Original user request.
        workflow:
            Planner output (subtask list).
        groundings:
            Service Provider output (service mappings).
        execution_results:
            Executor + Supervisor output (outcomes).
        context:
            Runtime context dict.
        user_score:
            User feedback / task success proxy (0–5 scale).

        Returns
        -------
        List of MemoryItem instances ready for STM insertion.
        """
        context = context or {}
        trace_text = self._format_trace(
            query, workflow, groundings, execution_results, context
        )

        user_prompt = (
            f"## Interaction Trace\n{trace_text}\n\n"
            f"## User Feedback Score: {user_score}/5\n\n"
            "Extract memory items as JSON."
        )

        raw = self._call_llm(SYSTEM_PROMPT, user_prompt)
        raw_items = self._parse_items(raw)

        memory_items: List[MemoryItem] = []
        for d in raw_items:
            item = MemoryItem(
                category=d.get("category", "workflow_trace"),
                content=d.get("content", ""),
                entities=d.get("entities", []),
                relations=d.get("relations", []),
                metadata={
                    **d.get("metadata", {}),
                    "context": context,
                    "user_score": user_score,
                },
            )
            try:
                item.validate()
            except ValueError as e:
                logger.warning("SummarizerAgent: invalid item – %s", e)
                continue
            memory_items.append(item)

        logger.info("[Summarizer] Extracted %d memory items.", len(memory_items))
        return memory_items

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_trace(
        query: str,
        workflow: List[Dict],
        groundings: List[Dict],
        results: List[Dict],
        context: Dict,
    ) -> str:
        lines = [f"Request: {query}"]
        lines.append(f"Context: {context}")
        lines.append("\nWorkflow:")
        for s in workflow:
            lines.append(f"  Step {s.get('step')}: {s.get('subtask')} – {s.get('description', '')}")
        lines.append("\nService Grounding:")
        for g in groundings:
            svcs = ", ".join(s.get("name", "") for s in g.get("services", []))
            lines.append(f"  Step {g.get('step')}: {svcs}")
        lines.append("\nExecution Results:")
        for r in results:
            status = "✓" if r.get("success") else "✗"
            err = f"  error: {r.get('error', '')}" if not r.get("success") else ""
            lines.append(
                f"  Step {r.get('step')} [{status}] service={r.get('service')} "
                f"latency={r.get('latency_ms', 0):.1f}ms{err}"
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_items(raw: str) -> List[Dict[str, Any]]:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data = json.loads(raw)
            return data.get("memory_items", [])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("SummarizerAgent: failed to parse JSON (%s). Raw: %s", e, raw[:300])
            return []
