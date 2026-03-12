"""
Planner Agent  (Section III-F-a)
=================================
Decomposes the user request into a high-level workflow skeleton W,
conditioned on the retrieved memory context C_q.

Output format (JSON):
    {
      "workflow": [
        {"step": 1, "subtask": "...", "description": "...",
         "dependencies": [], "expected_output": "..."},
        ...
      ]
    }
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from qos_lsma.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Planner Agent in the QoS-LSMA multi-agent \
service composition framework.

Your role:
  1. Analyse the user's request and runtime context carefully.
  2. Decompose the request into an ORDERED sequence of simpler subtasks.
  3. Each subtask should map to ONE logical service invocation or operation.
  4. Reuse workflow fragments and mitigation hints from retrieved memory \
when applicable.

IMPORTANT:
  - Keep subtasks concrete and actionable.
  - Identify data dependencies between steps (which step needs output \
from a previous step).
  - Return ONLY valid JSON – no extra text.

Output schema:
{
  "workflow": [
    {
      "step": <int>,
      "subtask": "<brief name>",
      "description": "<what this step does>",
      "dependencies": [<step numbers this step depends on>],
      "expected_output": "<type or description of output>"
    }
  ]
}
"""


class PlannerAgent(BaseAgent):
    """Proposes a workflow skeleton from (q, c, C_q)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="Planner", **kwargs)

    def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Decompose user request into a workflow.

        Parameters
        ----------
        query:
            Raw user request.
        context:
            Runtime context dict (region, time_bucket, etc.).
        memory_context:
            Structured memory summary C_q from the retrieval module.

        Returns
        -------
        List of subtask dicts following the output schema above.
        """
        context = context or {}
        ctx_str = "\n".join(f"  {k}: {v}" for k, v in context.items()) or "  N/A"

        user_prompt = (
            f"## User Request\n{query}\n\n"
            f"## Runtime Context\n{ctx_str}\n\n"
            f"## Retrieved Memory\n{memory_context}\n\n"
            "Produce the workflow plan as JSON."
        )

        raw = self._call_llm(SYSTEM_PROMPT, user_prompt)
        return self._parse_workflow(raw)

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_workflow(raw: str) -> List[Dict[str, Any]]:
        """Extract the workflow list from the LLM's JSON response."""
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data = json.loads(raw)
            workflow = data.get("workflow", [])
            if not isinstance(workflow, list):
                raise ValueError("'workflow' must be a list.")
            return workflow
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("PlannerAgent: failed to parse JSON (%s). Raw: %s", e, raw[:300])
            # Graceful fallback: treat the whole response as a single step
            return [
                {
                    "step": 1,
                    "subtask": "handle_request",
                    "description": raw[:200],
                    "dependencies": [],
                    "expected_output": "result",
                }
            ]
