"""
ShieldPoint Agent Framework
===========================

Top-level package for the ShieldPoint claims-automation agent framework.

Subpackages
-----------
- ``observability`` — Langfuse trace wrapper for LLM calls, tools, and spans.
- ``config``        — Centralized configuration loader (env vars + defaults).
"""

__version__ = "0.1.0"
__all__ = ["observability", "config"]
