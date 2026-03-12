"""
Supervisor Agent  (Section III-C, Section III-F-c)
====================================================
Monitors runtime execution outcomes and applies operational strategies:
  (i)   Retry with backoff
  (ii)  Switch to a known fallback service
  (iii) Skip / abort safely

The Supervisor triggers re-execution of any agent that returns invalid
results or deviates from expected behaviour, up to ``max_retries``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from qos_lsma.agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the Supervisor Agent in the QoS-LSMA framework.
Your role is to evaluate execution outcomes and decide the recovery strategy.

Given an execution result that FAILED, output a JSON with:
{
  "action": "retry" | "fallback" | "skip" | "abort",
  "reason": "<brief explanation>",
  "fallback_service": "<service name or null>"
}

- retry:    the same service should be called again (transient error)
- fallback: switch to the fallback service specified in the grounding
- skip:     this step is non-critical, skip and continue
- abort:    the error is unrecoverable; stop the workflow
"""


class SupervisorAgent(BaseAgent):
    """Monitors execution and triggers recovery strategies."""

    def __init__(self, max_retries: int = 3, **kwargs: Any) -> None:
        super().__init__(name="Supervisor", **kwargs)
        self.max_retries = max_retries

    def run(self, **kwargs: Any) -> Any:
        """Delegate to evaluate(); required by BaseAgent abstract interface."""
        return self.evaluate(**kwargs)

    def evaluate(
        self,
        execution_results: List[Dict[str, Any]],
        groundings: List[Dict[str, Any]],
        executor,
    ) -> List[Dict[str, Any]]:
        """Evaluate results; retry or apply fallback for failed steps.

        Parameters
        ----------
        execution_results:
            Output from ExecutorAgent.run().
        groundings:
            Original groundings from ServiceProviderAgent (for fallback info).
        executor:
            ExecutorAgent instance (for re-invocation).

        Returns
        -------
        Final list of execution results after recovery attempts.
        """
        grounding_map: Dict[int, Dict] = {}
        for g in groundings:
            grounding_map[g.get("step", 0)] = g

        final_results: List[Dict[str, Any]] = []
        for result in execution_results:
            if result["success"]:
                final_results.append(result)
                continue

            recovered = self._recover(result, grounding_map, executor)
            final_results.append(recovered)

        return final_results

    # ------------------------------------------------------------------
    # Recovery logic
    # ------------------------------------------------------------------
    def _recover(
        self,
        failed_result: Dict[str, Any],
        grounding_map: Dict[int, Dict],
        executor,
    ) -> Dict[str, Any]:
        step = failed_result["step"]
        grounding = grounding_map.get(step, {})
        services = grounding.get("services", [])

        current_service_name = failed_result["service"]
        current_idx = next(
            (i for i, s in enumerate(services) if s["name"] == current_service_name),
            0,
        )

        for attempt in range(1, self.max_retries + 1):
            action, reason, fallback_name = self._decide_action(
                failed_result, services, current_idx, attempt
            )
            logger.info(
                "[Supervisor] Step %d, attempt %d: action=%s, reason=%s",
                step, attempt, action, reason,
            )

            if action == "retry":
                backoff = 2 ** (attempt - 1)
                time.sleep(backoff)
                # Re-invoke the SAME service
                svc_info = services[current_idx] if services else {}
                result = executor._invoke_one(step, svc_info)
                if result["success"]:
                    result["recovery_action"] = "retry"
                    return result
                failed_result = result

            elif action == "fallback":
                # Try the next service in the list or the named fallback
                next_idx = current_idx + 1
                if fallback_name:
                    fb_svcs = [s for s in services if s["name"] == fallback_name]
                    svc_info = fb_svcs[0] if fb_svcs else (
                        services[next_idx] if next_idx < len(services) else {}
                    )
                elif next_idx < len(services):
                    svc_info = services[next_idx]
                    current_idx = next_idx
                else:
                    svc_info = {}
                if svc_info:
                    result = executor._invoke_one(step, svc_info)
                    if result["success"]:
                        result["recovery_action"] = "fallback"
                        return result
                    failed_result = result

            elif action == "skip":
                failed_result["success"] = True   # mark as skipped-ok
                failed_result["response"] = None
                failed_result["recovery_action"] = "skip"
                return failed_result

            else:  # abort
                failed_result["recovery_action"] = "abort"
                logger.error("[Supervisor] Step %d: aborting.", step)
                return failed_result

        failed_result["recovery_action"] = "max_retries_exceeded"
        return failed_result

    def _decide_action(
        self,
        failed_result: Dict[str, Any],
        services: List[Dict[str, Any]],
        current_idx: int,
        attempt: int,
    ):
        """Decide recovery action using LLM or heuristics."""
        import json, re

        error_msg = failed_result.get("error", "")
        # Fast heuristic: transient HTTP errors → retry
        transient_keywords = ["timeout", "connection", "502", "503", "504"]
        if any(k in error_msg.lower() for k in transient_keywords) and attempt <= 2:
            return "retry", "transient error", None

        # Fallback if there are alternative services
        if current_idx + 1 < len(services):
            return "fallback", "trying next service", None

        # Ask LLM for decision
        context_str = json.dumps(
            {"error": error_msg, "service": failed_result.get("service"), "attempt": attempt},
            ensure_ascii=False,
        )
        try:
            raw = self._call_llm(SYSTEM_PROMPT, context_str)
            raw = re.sub(r"```(?:json)?", "", raw).strip()
            data = json.loads(raw)
            return (
                data.get("action", "skip"),
                data.get("reason", ""),
                data.get("fallback_service"),
            )
        except Exception:  # noqa: BLE001
            return "skip", "llm decision failed", None
