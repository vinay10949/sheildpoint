"""
Test helpers — public to test suites but not part of the framework API.

Importing from this module is the recommended way for downstream agent
projects to write tests against the framework without depending on pytest
internals. Example::

    from shieldpoint_agents._testing import FakeLMClient

    def test_my_agent():
        client = FakeLMClient([
            '{"thought":"...","action":"FINAL_ANSWER","action_input":{...}}',
        ])
        agent = MyAgent(llm_client=client, ...)
        result = agent.run(claim)
        assert result.source == "llm"
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock


class _FakeMessage(SimpleNamespace):
    pass


class _FakeChoice(SimpleNamespace):
    pass


class _FakeResponse(SimpleNamespace):
    pass


class FakeLMClient:
    """Mock OpenAI-compatible client for tests.

    Pass a list of canned responses to ``__init__``; each call to
    ``chat.completions.create`` pops the next one. If a response is an
    Exception instance, it's raised (simulating LLM failure/timeout).

    Each response can be:
    - A ``str`` — wrapped in a fake OpenAI response with ``choices[0].message.content``.
    - An ``Exception`` instance — raised directly (use ``TimeoutError``,
      ``ConnectionRefusedError``, etc. to simulate failures).
    - Any object — returned as-is (use this if you need to fake ``usage``
      fields or other response shapes).

    Records every call's ``messages`` for assertion in tests.
    """

    def __init__(self, responses: list[Any]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create
        # Also expose .models.list() so ping() works in health checks.
        self.models = MagicMock()
        self.models.list.return_value = SimpleNamespace(
            data=[SimpleNamespace(id="qwen-test")]
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError(
                "FakeLMClient ran out of canned responses — test setup issue."
            )
        nxt = self._responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if isinstance(nxt, str):
            return _FakeResponse(
                choices=[_FakeChoice(message=_FakeMessage(content=nxt))],
                usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15
                ),
            )
        # Already a response-shaped object — return as-is.
        return nxt
