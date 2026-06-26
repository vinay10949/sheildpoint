# ShieldPoint Agent Framework (`shieldpoint_agents`)

> **Ticket:** SHLD-13 — *Scaffold Base Agent Framework with Langfuse Integration*
> **Status:** Scaffolding complete — all 7 ACs met.

Base agent framework for ShieldPoint claims automation. Every future agent
subclasses :class:`Agent` and gets, for free:

- A **Think/Plan/Act ReAct loop** with Pydantic-validated structured output
  parsing.
- **Langfuse tracing** on every LLM call and every tool invocation (prompt,
  completion, latency, token counts).
- A **ToolRegistry** that accepts Python functions with JSON-schema
  descriptors, validates arguments, and logs each invocation.
- **Graceful fallback** to a rule-based :class:`FallbackEngine` when the LLM
  call fails or times out (>10s).
- A **Dockerfile** + **docker-compose.agent.yml** wired to LM Studio
  (`host.docker.internal:1234`) and Langfuse (`langfuse:3000`).

## Package layout

```
shieldpoint_agents/
├── pyproject.toml              # PEP 517/518 build, deps, pytest config
├── Dockerfile                  # Python 3.11-slim, installs package + [api] extras
├── docker-compose.agent.yml    # Override: adds `agent-api` service to the stack
├── README.md                   # This file
├── src/
│   └── shieldpoint_agents/
│       ├── __init__.py         # Public API exports
│       ├── _bootstrap.py       # sys.path bootstrap → import agent_framework.*
│       ├── config.py           # AgentConfig (reads env at call time)
│       ├── tracer.py           # LangfuseTracer (façade over ShieldPointTracer)
│       ├── schemas.py          # Pydantic v2 models: ReActStep, ClaimDecision, AgentRunResult
│       ├── tools.py            # Tool + ToolRegistry + JSON-schema validation
│       ├── fallback.py         # FallbackEngine — rule-based claim processing
│       ├── agent.py            # Agent base class — ReAct loop, parse retries, fallback
│       ├── _lmstudio.py        # OpenAI-compatible client factory
│       ├── api.py              # FastAPI demo: /health + /run
│       └── example.py          # CLI demo: `python -m shieldpoint_agents.example`
├── tests/
│   ├── conftest.py             # Shared fixtures: FakeLMClient, sample_claim
│   ├── test_tools.py           # ToolRegistry unit tests
│   ├── test_tracer.py          # LangfuseTracer unit tests
│   ├── test_fallback.py        # FallbackEngine rule tests
│   ├── test_agent.py           # Agent ReAct loop + fallback trigger tests
│   └── test_integration.py     # End-to-end ReAct cycle (SHLD-13 AC)
└── examples/
    └── (add per-agent demos here as Sprint 2+ lands)
```

## Quickstart

### Install (dev)

```bash
cd shieldpoint_agents/
pip install -e ".[dev]"
pytest -v
```

The editable install makes both `shieldpoint_agents` and the legacy
`agent_framework` importable (the latter via `_bootstrap.py`).

### Run the demo (no LM Studio required)

```bash
python -m shieldpoint_agents.example
```

If LM Studio is reachable, the agent will use the real Qwen model. If not,
the :class:`FallbackEngine` produces a deterministic decision.

### Run as a Docker container

```bash
# From the repo root:
docker compose -f docker-compose.yml \
               -f shieldpoint_agents/docker-compose.agent.yml \
               --profile langfuse --profile agent up -d

# Health check:
curl http://localhost:8000/health

# Run a sample claim:
curl -X POST http://localhost:8000/run \
     -H 'Content-Type: application/json' \
     -d '{"claim":{"claim_id":"CLM-1","amount":1250.00,"description":"Wind damage.","policy_id":"HO-001"}}'
```

## Subclassing the Agent

```python
from shieldpoint_agents import Agent, AgentConfig, FallbackEngine, LangfuseTracer, ToolRegistry

class ClaimClassifier(Agent):
    def __init__(self):
        registry = ToolRegistry()

        @registry.register(
            name="validate_policy",
            description="Look up a policy by ID.",
            schema={
                "type": "object",
                "properties": {"policy_id": {"type": "string"}},
                "required": ["policy_id"],
            },
        )
        def validate_policy(policy_id: str) -> dict:
            # ... real DB lookup ...
            return {"policy_id": policy_id, "limit": 25_000}

        super().__init__(
            name="claim-classifier",
            tools=registry,
            tracer=LangfuseTracer(agent_name="claim-classifier"),
            fallback=FallbackEngine(),
            config=AgentConfig.from_env(),
        )

agent = ClaimClassifier()
result = agent.run(claim={...})
print(result.decision.decision)  # "approve" | "deny" | "route_to_manual_review"
```

## Multi-Agent Orchestration — `ManagerAgent` (SHLD-15)

The `ManagerAgent` extends the base `Agent` class to orchestrate three
specialist agents (`ClaimsAgent`, `FinancialAgent`, `SentimentAgent`)
for multi-claim processing.

### Quickstart

```python
from shieldpoint_agents import (
    ManagerAgent, InMemoryEpisodicMemory, ConflictResolver,
    build_specialists, AgentConfig,
)

cfg = AgentConfig.from_env()
specialists = build_specialists(config=cfg)  # ClaimsAgent, FinancialAgent, SentimentAgent
manager = ManagerAgent(
    config=cfg,
    specialists=specialists,
    memory=InMemoryEpisodicMemory(),
    conflict_resolver=ConflictResolver(strategy="weighted"),
)

result = manager.run({
    "claim_id": "CLM-2026-0001",
    "policy_id": "HO-2024-001",
    "claimant": "Alice Homeowner",
    "amount": 1_250.00,
    "description": "Wind damage to roof shingles during storm.",
})
print(result.decision.decision)        # "approve" | "deny" | "route_to_manual_review"
print(result.plan.claim_type)          # "property"
print(result.plan.stages)              # [OrchestrationStage(stage_id='stage-1', mode='parallel', ...)]
print(len(result.invocations))         # 3 (all three specialists ran)
print(len(result.conflicts))           # 0 if they agreed, 1 if they disagreed
print(result.memory_entries_used)      # ids of prior episodes (empty on first interaction)
```

### Routing logic

The manager classifies each claim into one of six types and produces an
`:class:`OrchestrationPlan` describing which specialists to invoke, in
what sequence, and in what execution mode:

| Claim type | Stages | Mode | Specialists | Conflict strategy |
|---|---|---|---|---|
| `property` | 1 | parallel | Claims + Financial + Sentiment | weighted |
| `liability` | 2 | parallel then sequential | (Claims + Sentiment) → Financial | priority |
| `auto_collision` | 1 | parallel | Claims + Financial (skip Sentiment) | weighted |
| `theft` | 3 | sequential | Sentiment → Claims → Financial | weighted |
| `fraud_suspected` | 2 | sequential then parallel | Sentiment → (Claims + Financial) | priority |
| `unknown` | 1 | parallel | all three (conservative default) | weighted |

### Conflict resolution strategies

When specialists disagree, the `ConflictResolver` applies one of four
configurable strategies:

- **`weighted`** (default) — each specialist's vote is weighted by its
  confidence score; the label with the highest aggregate weighted
  confidence wins.
- **`priority`** — a per-agent priority map decides (FinancialAgent=100,
  ClaimsAgent=80, SentimentAgent=40). Used when one specialist is
  authoritative for a given claim dimension.
- **`vote`** — majority vote across specialists. Ties fall back to the
  priority strategy.
- **`escalation`** — always route to manual review when a conflict is
  detected.

Every conflict produces a `ConflictRecord` with full audit trail:
agent names, decisions, description, strategy used, resolution, and
rationale.

### Episodic memory

Every specialist output is appended to the `EpisodicMemoryStore`. On
follow-up interactions with the same claim, prior episodes are recalled
and surfaced to specialists as additional context.

Two implementations:
- `InMemoryEpisodicMemory` — thread-safe in-process store (default, for
  tests and small deployments).
- `PostgresEpisodicMemory` — production store backed by a PostgreSQL
  table with a `JSONB` payload column. Uses psycopg3 (or psycopg2
  fallback). Lazy schema creation on first use.

```sql
CREATE TABLE agent_episodes (
    episode_id      TEXT PRIMARY KEY,
    claim_id        TEXT NOT NULL,
    agent_name      TEXT NOT NULL,
    decision_label  TEXT NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    trace_id        TEXT,
    created_at      DOUBLE PRECISION NOT NULL,
    payload         JSONB NOT NULL
);
CREATE INDEX idx_agent_episodes_claim_id ON agent_episodes (claim_id);
CREATE INDEX idx_agent_episodes_claim_agent ON agent_episodes (claim_id, agent_name);
```

### Langfuse linked tracing

The manager opens a single top-level trace (`manager_run`) and every
specialist invocation, orchestration decision, and conflict resolution
is logged as a linked span within that trace tree. Specialists share
the manager's `LangfuseTracer` instance so nested spans auto-attach
via OpenTelemetry context vars.

If Langfuse env vars are not set or the SDK is unavailable, every
trace call silently no-ops — the manager runs untraced.

### Demo

```bash
PYTHONPATH=src python -m shieldpoint_agents.manager_example
```

Processes 3 sample claims (property, liability, fraud) through the
ManagerAgent and prints the orchestration plan, invocations, conflicts,
and final decision.

## Acceptance Criteria mapping

| SHLD-13 AC | Where addressed |
|---|---|
| Python package `shieldpoint_agents` with base `Agent`, `ToolRegistry`, `LangfuseTracer` | `src/shieldpoint_agents/{__init__,agent,tools,tracer}.py` |
| Agent base class supports Think/Plan/Act ReAct loop with structured output parsing | `Agent._run_react_loop()` in `agent.py`; `schemas.ReActStep` (Pydantic v2) |
| Every LLM call decorated with Langfuse trace capturing prompt, completion, latency, tokens | `Agent._call_llm()` wraps the call in `@tracer.llm_call(...)`; legacy `ShieldPointTracer.observe_llm` captures all four fields |
| `ToolRegistry` accepts Python functions with JSON schema descriptors | `ToolRegistry.register(func, schema=...)` in `tools.py`; `jsonschema.Draft7Validator` enforces |
| Graceful fallback: if LLM call fails or times out (>10s), rule-based fallback executes and logs reason | `FallbackEngine` in `fallback.py`; `Agent.run()` catches `_FallbackSignal` and delegates; `llm_timeout_sec=10` default |
| Dockerfile builds agent container with all dependencies, connects to LM Studio and Langfuse | `Dockerfile` + `docker-compose.agent.yml`; env vars wired in both |
| Integration test: agent processes a sample claim through full Think/Plan/Act cycle | `tests/test_integration.py::TestHappyPathReActCycle::test_sample_claim_processes_through_full_cycle` |

| SHLD-15 AC | Where addressed |
|---|---|
| ManagerAgent processes claims by invoking specialist agents in determined sequence | `ManagerAgent.run()` in `manager.py`; `tests/test_manager_agent.py::TestSingleAndTwoAgentScenarios`, `TestMultiAgentParallelScenarios`, `TestMultiAgentSequentialScenarios` |
| Orchestration logic adapts sequence based on claim type | `ManagerAgent._classify_claim_type()` + `_build_plan_for_type()` in `manager.py`; `tests/test_manager_agent.py::TestRoutingAndPlan` (11 tests covering all 6 claim types) |
| Conflict resolution: when agents disagree, ManagerAgent synthesizes with documented rationale | `ConflictResolver` in `conflict.py` (4 strategies); `tests/test_manager_agent.py::TestConflictScenarios` (5 conflict scenarios) + `TestConflictResolverStrategies` (6 unit tests) |
| Episodic memory: follow-up claim interactions reference prior agent outputs | `EpisodicMemoryStore` in `memory.py` (InMemory + Postgres JSONB); `tests/test_manager_agent.py::TestEpisodicMemory` + `TestEpisodicMemoryStore` |
| All orchestration decisions logged as Langfuse linked spans | `ManagerAgent._trace_event()` in `manager.py`; specialists share the manager's tracer; `tests/test_manager_agent.py::TestLangfuseLinkedTracing` |
| Integration test: 50 multi-agent claims with at least 5 conflict scenarios handled correctly | `tests/test_manager_agent.py::TestFiftyMultiAgentClaimsWithConflicts::test_50_claims_with_at_least_5_conflicts` |

## Sub-tasks status

| Sub-task | Status | File(s) |
|---|---|---|
| Create Python project structure with `pyproject.toml` and pytest | ✅ | `pyproject.toml` |
| Implement `Agent` base class with LM Studio OpenAI-compatible client | ✅ | `src/shieldpoint_agents/agent.py`, `_lmstudio.py` |
| Implement `LangfuseTracer` class wrapping SDK calls | ✅ | `src/shieldpoint_agents/tracer.py` |
| Implement `ToolRegistry` with schema validation and invocation logging | ✅ | `src/shieldpoint_agents/tools.py` |
| Implement `FallbackEngine` with rule-based claim processing logic | ✅ | `src/shieldpoint_agents/fallback.py` |
| Write integration test with sample claim data | ✅ | `tests/test_integration.py` |
| Create `Dockerfile` and `docker-compose.agent.yml` | ✅ | both at package root |
| **SHLD-15**: Implement ManagerAgent extending base Agent class | ✅ | `src/shieldpoint_agents/manager.py` |
| **SHLD-15**: Create agent invocation framework with sequential and parallel execution modes | ✅ | `ManagerAgent._execute_stage_sequential` / `_execute_stage_parallel` in `manager.py` |
| **SHLD-15**: Implement episodic memory store using PostgreSQL JSONB columns | ✅ | `src/shieldpoint_agents/memory.py` (`PostgresEpisodicMemory` with JSONB payload column) |
| **SHLD-15**: Build conflict resolution logic with configurable strategies | ✅ | `src/shieldpoint_agents/conflict.py` (`ConflictResolver` with priority/vote/escalation/weighted) |
| **SHLD-15**: Add Langfuse linked trace instrumentation for agent-to-agent handoffs | ✅ | `ManagerAgent._trace_event()` + shared `LangfuseTracer` in `manager.py` |
| **SHLD-15**: Write integration tests covering single-agent, multi-agent, and conflict scenarios | ✅ | `tests/test_manager_agent.py` (45 tests across 13 test classes) |
