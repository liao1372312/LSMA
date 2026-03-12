"""
Executor Agent  (Section III-F-c)
==================================
Invokes each grounded service call and logs runtime outcomes
(responses, latency, errors).

In production, the Executor calls real HTTP endpoints.  For testability
we support injecting a ``service_registry`` dict that maps service names
to callable Python functions.

Output per step:
    {
      "step": <int>,
      "service": "<name>",
      "arguments": {...},
      "response": <any>,
      "success": <bool>,
      "latency_ms": <float>,
      "error": "<message or null>"
    }
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

import requests

from qos_lsma.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Executor Agent in the QoS-LSMA framework.
Your role is to call services and interpret their responses.
When given a service name and arguments, simulate or invoke the service and
report the outcome.  Return a JSON object with keys:
  success (bool), response (any), error (str or null).
"""


class ExecutorAgent(BaseAgent):
    """Invokes service calls and collects runtime outcomes.

    Parameters
    ----------
    service_registry:
        Dict mapping service names to ``Callable(arguments) -> Any``.
        When a service is found here, the callable is used directly
        (avoids an LLM call).  Unknown services fall back to LLM
        simulation.
    http_timeout:
        Timeout (seconds) for real HTTP calls.
    """

    def __init__(
        self,
        service_registry: Optional[Dict[str, Callable]] = None,
        http_timeout: float = 10.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(name="Executor", **kwargs)
        self.service_registry: Dict[str, Callable] = service_registry or {}
        self.http_timeout = http_timeout

    def run(
        self,
        groundings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Execute all grounded service calls.

        Parameters
        ----------
        groundings:
            Output from ServiceProviderAgent.

        Returns
        -------
        List of execution result dicts, one per grounding step.
        """
        results: List[Dict[str, Any]] = []
        for grounding in groundings:
            step = grounding.get("step", 0)
            for svc_info in grounding.get("services", []):
                result = self._invoke_one(step, svc_info)
                results.append(result)
                if result["success"]:
                    break  # Use the first successful service; fallback later
        return results

    # ------------------------------------------------------------------
    # Single service invocation
    # ------------------------------------------------------------------
    def _invoke_one(
        self, step: int, svc_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        name = svc_info.get("name", "unknown")
        arguments = svc_info.get("arguments", {})

        t0 = time.perf_counter()
        result = {
            "step": step,
            "service": name,
            "arguments": arguments,
            "response": None,
            "success": False,
            "latency_ms": 0.0,
            "error": None,
        }

        try:
            if name in self.service_registry:
                # Direct Python callable
                response = self.service_registry[name](**arguments)
                result["response"] = response
                result["success"] = True
            elif svc_info.get("endpoint"):
                # Real HTTP call
                endpoint = svc_info["endpoint"]
                http_resp = requests.post(
                    endpoint,
                    json=arguments,
                    timeout=self.http_timeout,
                )
                http_resp.raise_for_status()
                result["response"] = http_resp.json()
                result["success"] = True
            else:
                # LLM simulation (for testing / demo)
                simulated = self._simulate_via_llm(name, arguments)
                result["response"] = simulated
                result["success"] = True
        except Exception as e:  # noqa: BLE001
            result["error"] = str(e)
            logger.warning("[Executor] Step %d, service '%s' failed: %s", step, name, e)

        result["latency_ms"] = (time.perf_counter() - t0) * 1000.0
        return result

    def _simulate_via_llm(
        self, service_name: str, arguments: Dict[str, Any]
    ) -> str:
        """Use the LLM to simulate a service response (demo / testing)."""
        prompt = (
            f"Simulate the response of service '{service_name}' "
            f"called with arguments: {arguments}. "
            "Return a realistic JSON response."
        )
        return self._call_llm(SYSTEM_PROMPT, prompt)
