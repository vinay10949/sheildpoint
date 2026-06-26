"""
sys.path bootstrap
==================

Makes the legacy ``agent_framework.*`` package importable when
``shieldpoint_agents`` is installed (either via ``pip install -e .`` from the
repo root, or via wheel install into a fresh environment).

Strategy
--------
When this package is loaded, we walk up from ``__file__`` looking for a
directory that contains both ``shieldpoint_agents/`` and ``agent_framework/``
as siblings. If found, that directory is prepended to ``sys.path``. If
``agent_framework`` is already importable (e.g. installed elsewhere), we
silently no-op.

This is deliberately defensive — none of the lookups raise. If the bootstrap
fails, downstream imports of ``agent_framework.*`` will simply raise
``ImportError`` at first use, which is easy to diagnose.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _candidate_parents(start: Path, depth: int = 6) -> list[Path]:
    """Walk up ``start`` yielding ancestors up to ``depth`` levels."""
    out: list[Path] = []
    cur = start.resolve()
    for _ in range(depth):
        if cur.parent == cur:
            break
        cur = cur.parent
        out.append(cur)
    return out


def ensure_repo_root_on_path() -> None:
    """Prepend the repo root to ``sys.path`` if ``agent_framework`` is missing.

    Idempotent — safe to call multiple times.
    """
    # Fast path: already importable.
    if "agent_framework" in sys.modules or _is_importable("agent_framework"):
        return

    here = Path(__file__).resolve()
    for parent in _candidate_parents(here, depth=6):
        if (parent / "agent_framework" / "__init__.py").exists():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return

    # Not found — silent. Downstream imports will raise naturally.
    return


def _is_importable(module_name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module_name) is not None
