"""ShieldPoint 5-Agent State Machine Engine package.

Public API
----------
- :class:`StateMachineEngine` — the main engine class
- :class:`State`, :class:`Transition` — enums for states and transitions
- :class:`GuardConditionFailedError`, :class:`InvalidStateTransitionError` — exceptions
- :class:`StateLogEntry` — persisted transition record
- :data:`STATE_AGENT` — mapping from state to owning agent name
- :data:`INITIAL_STATE`, :data:`TERMINAL_STATES`, :data:`ESCALATION_STATES`
"""

from .state_machine import (
    ESCALATION_STATES,
    INITIAL_STATE,
    STATE_AGENT,
    TERMINAL_STATES,
    GuardConditionFailedError,
    InvalidStateTransitionError,
    State,
    StateLogEntry,
    StateMachineEngine,
    StateRecoveryError,
    Transition,
    TransitionDef,
)

__all__ = [
    "StateMachineEngine",
    "State",
    "Transition",
    "TransitionDef",
    "StateLogEntry",
    "STATE_AGENT",
    "INITIAL_STATE",
    "TERMINAL_STATES",
    "ESCALATION_STATES",
    "GuardConditionFailedError",
    "InvalidStateTransitionError",
    "StateRecoveryError",
]
