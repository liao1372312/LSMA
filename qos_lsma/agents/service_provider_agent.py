"""
Service Provider Agent  (Section III-F-b)
==========================================
Maps each abstract workflow step to one or more concrete services/APIs
from the service catalog S.

Grounding uses both:
  • semantic compatibility (description / signature match)
  • retrieved usage experience (argument templates, known pitfalls,
    recommended fallbacks)

Output format (JSON):
    {
      "groundings": [
        {
          "step": <int>,
          "subtask": "<subtask name>",
          "services": [
            {
              "name": "<API name>",
              "description": "<brief desc>",
              "arguments": {"<arg_name>": "<value_or_template>"},
              "fallback": "<fallback API name or null>",
              "confidence": <float 0-1>
            }
          ]
        }
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

SYSTEM_PROMPT = """You are the Service Provider Agent in the QoS-LSMA \
multi-agent service composition framework.

Your role:
  1. For each subtask in the workflow, select the MOST suitable service(s) \
from the provided service catalog.
  2. Use the retrieved memory context for argument templates, known pitfalls, \
and recommended fallbacks.
  3. Prefer services with higher reliability and lower latency based on any \
QoS statistics available.
  4. Provide a fallback service whenever possible.
  5. Return ONLY valid JSON – no extra text.

Output schema:
{
  "groundings": [
    {
      "step": <int>,
      "subtask": "<subtask name>",
      "services": [
        {
          "name": "<API/service name>",
          "description": "<brief desc>",
          "arguments": {"<arg_name>": "<value_or_template>"},
          "fallback": "<fallback service or null>",
          "confidence": <float 0-1>
        }
      ]
    }
  ]
}
"""


class ServiceProviderAgent(BaseAgent):
    """Maps workflow steps to concrete services/APIs."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="ServiceProvider", **kwargs)

    def run(
        self,
        workflow: List[Dict[str, Any]],
        service_catalog: List[Dict[str, Any]],
        memory_context: str = "",
        query: str = "",
    ) -> List[Dict[str, Any]]:
        """Ground each workflow step to concrete services.

        Parameters
        ----------
        workflow:
            List of subtask dicts from the Planner.
        service_catalog:
            List of available service/API dicts, each with at minimum
            ``name``, ``description``, and ``signature`` fields.
        memory_context:
            Structured memory summary C_q (from retrieval).
        query:
            Original user request (for context).

        Returns
        -------
        List of grounding dicts following the output schema above.
        """
        catalog_text = self._format_catalog(service_catalog)
        workflow_text = json.dumps(workflow, ensure_ascii=False, indent=2)

        user_prompt = (
            f"## Original Request\n{query}\n\n"
            f"## Workflow to Ground\n{workflow_text}\n\n"
            f"## Available Services\n{catalog_text}\n\n"
            f"## Retrieved Memory\n{memory_context}\n\n"
            "Produce the service grounding as JSON."
        )

        raw = self._call_llm(SYSTEM_PROMPT, user_prompt)
        return self._parse_groundings(raw)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_catalog(catalog: List[Dict[str, Any]], max_services: int = 50) -> str:
        lines = []
        for svc in catalog[:max_services]:
            name = svc.get("name", "")
            desc = svc.get("description", "")
            sig = svc.get("signature", "")
            lines.append(f"- {name}: {desc} | signature: {sig}")
        if len(catalog) > max_services:
            lines.append(f"... ({len(catalog) - max_services} more services omitted)")
        return "\n".join(lines) if lines else "No services available."

    @staticmethod
    def _parse_groundings(raw: str) -> List[Dict[str, Any]]:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            data = json.loads(raw)
            return data.get("groundings", [])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "ServiceProviderAgent: failed to parse JSON (%s). Raw: %s", e, raw[:300]
            )
            return []
