"""
Shared pytest fixtures.

We strip Langfuse env vars before each test so the tracer is in a known
disabled state by default. Tests that want tracing enabled set the vars
themselves via ``monkeypatch.setenv``.

The ``FakeLMClient`` mock is in ``shieldpoint_agents._testing`` (importable
from anywhere, including the package's own tests). The ``fake_lm_client_factory``
fixture below just re-exports it for convenience.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

# Make `shieldpoint_agents` importable when running from the package dir
# (i.e. before `pip install -e .` has run).
PKG_SRC = Path(__file__).resolve().parent.parent / "src"
if str(PKG_SRC) not in sys.path:
    sys.path.insert(0, str(PKG_SRC))

# Also add the repo root so `agent_framework.*` is importable.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _clean_langfuse_env(monkeypatch):
    """Strip Langfuse env vars before each test (default = disabled)."""
    for k in (
        "LANGFUSE_HOST",
        "LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY",
        "LANGFUSE_ENABLED",
        "LANGFUSE_FLUSH_AT",
        "LANGFUSE_FLUSH_INTERVAL_MS",
    ):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture
def disabled_config() -> Any:
    """An AgentConfig with Langfuse disabled (no keys)."""
    from shieldpoint_agents import AgentConfig

    return AgentConfig.from_env()


@pytest.fixture
def sample_claim() -> dict[str, Any]:
    """A representative homeowners-claim fixture for tests."""
    return {
        "claim_id": "CLM-2026-0001",
        "adjuster_id": "ADJ-42",
        "session_id": "sess-test-001",
        "policy_id": "HO-2024-001",
        "claimant": "Alice Homeowner",
        "amount": 1_250.00,
        "description": "Wind damage to roof shingles during storm on 2026-03-14.",
        "date_of_loss": "2026-03-14",
    }


@pytest.fixture
def high_amount_claim() -> dict[str, Any]:
    """A high-value claim that should trigger manual review."""
    return {
        "claim_id": "CLM-2026-HIGH",
        "adjuster_id": "ADJ-50",
        "policy_id": "HO-2024-012",
        "claimant": "Dan Property",
        "amount": 12_500.00,
        "description": "Flood damage to basement after heavy rain.",
        "date_of_loss": "2026-02-28",
    }


@pytest.fixture
def low_amount_claim() -> dict[str, Any]:
    """A small claim that should be auto-approved."""
    return {
        "claim_id": "CLM-2026-LOW",
        "adjuster_id": "ADJ-44",
        "policy_id": "HO-2024-088",
        "claimant": "Carol Resident",
        "amount": 250.00,
        "description": "Minor hail damage to mailbox and fence.",
        "date_of_loss": "2026-05-10",
    }


@pytest.fixture
def fraud_claim() -> dict[str, Any]:
    """A claim with fraud indicators."""
    return {
        "claim_id": "CLM-2026-FRAUD",
        "adjuster_id": "ADJ-45",
        "policy_id": "HO-2024-012",
        "claimant": "Dan Property",
        "amount": 12_500.00,
        "description": "Flood damage. Intentional misrepresentation suspected.",
        "date_of_loss": "2026-02-28",
    }


@pytest.fixture
def fake_lm_client_factory():
    """Factory that builds FakeLMClient instances (re-exported from _testing)."""
    from shieldpoint_agents._testing import FakeLMClient

    return FakeLMClient
