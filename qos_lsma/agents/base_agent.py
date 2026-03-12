"""
Base Agent
==========
Abstract base class for all LLM-powered agents in QoS-LSMA.

Handles:
  • OpenAI-compatible LLM calls (works with DeepSeek, GPT-4o, etc.)
  • System / user prompt construction
  • Basic retry on transient errors
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from openai import OpenAI, APIError, RateLimitError

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract LLM agent.

    Parameters
    ----------
    name:
        Human-readable agent name (used in logs).
    model:
        LLM model name, e.g. ``"deepseek-chat"`` or ``"gpt-4o"``.
    api_key:
        API key for the LLM provider.
    base_url:
        Base URL for OpenAI-compatible APIs (e.g. DeepSeek endpoint).
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the response.
    max_retries:
        Number of retries on transient API errors.
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        max_retries: int = 3,
    ) -> None:
        self.name = name
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries

        self._client = OpenAI(api_key=api_key, base_url=base_url)

    # ------------------------------------------------------------------
    # LLM call wrapper
    # ------------------------------------------------------------------
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_messages: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Call the LLM with retry logic.

        Returns the assistant response content as a plain string.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_prompt})

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content or ""
                logger.debug("[%s] LLM response: %s", self.name, content[:200])
                return content.strip()
            except RateLimitError as e:
                wait = 2 ** attempt
                logger.warning(
                    "[%s] RateLimitError on attempt %d/%d. Sleeping %ds.",
                    self.name, attempt, self.max_retries, wait,
                )
                time.sleep(wait)
                last_error = e
            except APIError as e:
                logger.warning(
                    "[%s] APIError on attempt %d/%d: %s",
                    self.name, attempt, self.max_retries, e,
                )
                last_error = e
                time.sleep(1)

        raise RuntimeError(
            f"[{self.name}] LLM call failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the agent's primary task.  Implemented by subclasses."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.model})"
