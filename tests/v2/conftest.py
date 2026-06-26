"""Pytest configuration: add all package src dirs to sys.path."""
import os
import sys

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in [
    "state_machine_engine/src",
    "shieldpoint_agents/src",
    "zkp_circuit",
    "agent_framework",
]:
    p = os.path.join(ROOT, sub)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
