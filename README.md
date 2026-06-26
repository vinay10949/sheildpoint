# ShieldPoint Agent Framework

AI agent framework that runs LLM inference locally via **LM Studio** on macOS, with agent services orchestrated through Docker Compose. Built for private, low-latency, no-egress AI workloads.

---

## Features

- **Local inference** — LM Studio runs natively on your Mac; agent containers connect via Docker's `host.docker.internal`
- **OpenAI-compatible API** — drop-in replacement for any OpenAI SDK client
- **Containerized agents** — microservice architecture with Docker Compose
- **Observability** — self-hosted Langfuse stack (PostgreSQL + Redis + ClickHouse) for trace capture
- **No egress** — all inference traffic stays on localhost; no data leaves your network
- **Portable** — same config works on macOS dev and Linux production (A100, etc.)

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Mac host                                                   │
│                                                             │
│  ┌──────────────────────┐    ┌──────────────────────────┐  │
│  │  LM Studio desktop    │    │  Docker Desktop          │  │
│  │  (local inference)    │    │                          │  │
│  │  Port :1234          │◀───┤  host.docker.internal    │  │
│  └──────────────────────┘    │  ┌─────────────────────┐ │  │
│         ▲                    │  │ shieldpoint-net     │ │  │
│         │                    │  │  172.28.0.0/16     │ │  │
│  localhost:1234              │  │ ┌─────────────────┐│ │  │
│                              │  │ │ smoke-probe     ││ │  │
│                              │  │ │ (health check)  ││ │  │
│                              │  │ └─────────────────┘│ │  │
│                              │  │ ┌─────────────────┐│ │  │
│                              │  │ │ agent-api       ││ │  │
│                              │  │ │ (your services) ││ │  │
│                              │  │ └─────────────────┘│ │  │
│                              │  └─────────────────────┘ │  │
│                              └──────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

| Context | Endpoint |
|---------|----------|
| Mac terminal / scripts | `http://localhost:1234/v1` |
| Inside Docker container | `http://host.docker.internal:1234/v1` |

---

## Prerequisites

- **macOS** (Apple Silicon recommended)
- [LM Studio](https://lmstudio.ai) with a model downloaded (e.g., Qwen 3.6 35B A3B GGUF)
- [Docker Desktop](https://docker.com) (≥ 4.0)
- Python 3.11+ (for agent SDK scripts)

---

## Quick Start

### 1. Start LM Studio

1. Open **LM Studio** desktop app.
2. Load your model in the **My Models** tab.
3. Go to **Local Server** tab, select the model, click **Start Server**.
4. Confirm: `http://localhost:1234/v1/models` returns `200 OK`.

### 2. Configure the bundle

```bash
cd shieldpoint-lmstudio/
cp .env.example .env
./scripts/langfuse-gen-secrets.sh >> .env
```

Edit `.env` to adjust thresholds for your hardware (see [Configuration](#configuration)).

### 3. Verify LM Studio

```bash
./scripts/verify-lm-studio-desktop.sh
```

All 4 stages should pass (process, HTTP, model loaded, inference test).

### 4. Start the agent framework

```bash
make dev          # verify LM Studio + bring up containers
make smoke        # 5-stage acceptance test
```

---

## Configuration

All configuration lives in a single `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_API_KEY` | `lm-studio` | Must match LM Studio Settings → API Key |
| `QWEN_MODEL_ID` | — | Model identifier from LM Studio Local Server tab |
| `SMOKE_THROUGHPUT_ACCEPT_TPS` | `80` | Min acceptable throughput (tok/s). Adjust to your hardware baseline. |
| `SMOKE_INFERENCE_LATENCY_MAX_SEC` | `2` | Max acceptable inference latency in seconds |

Typical Mac throughput baselines: M2/M3 Max ~20–40 tok/s, M3 Ultra ~40–60 tok/s, A100 ~80–120 tok/s.

---

## Daily Operations

| Action | Command |
|--------|---------|
| Start (verify + bring up containers) | `make dev` |
| Stop containers (LM Studio keeps running) | `make down` |
| Live status | `make status` |
| Verify LM Studio | `make verify-lm-studio` |
| Run smoke test | `make smoke` |
| Tail logs (LM Studio) | `make logs` |
| Tail logs (containers) | `make logs TARGET=containers` |
| Check VRAM | `make vram` |
| Lint configs | `make lint` |
| Full clean | `make clean` |

---

## Adding Agent Services

Add your services to `docker-compose.yml`. Each container that needs LM Studio must:

1. Attach to `shieldpoint-net` network
2. Add `extra_hosts: - "host.docker.internal:host-gateway"`
3. Use `http://host.docker.internal:1234/v1` as `LM_STUDIO_BASE_URL`

```yaml
services:
  my-agent:
    image: shieldpoint/my-agent:latest
    networks: [shieldpoint-net]
    extra_hosts:
      - "host.docker.internal:host-gateway"
    environment:
      - LM_STUDIO_BASE_URL=http://host.docker.internal:1234/v1
      - LM_STUDIO_API_KEY=${LM_STUDIO_API_KEY}
      - LM_STUDIO_MODEL=${QWEN_MODEL_ID}
```

Usage in Python:

```python
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ["LM_STUDIO_BASE_URL"],
    api_key=os.environ["LM_STUDIO_API_KEY"],
)

response = client.chat.completions.create(
    model=os.environ["LM_STUDIO_MODEL"],
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=4096,
    temperature=0.1,
)
```

---

## Observability (Langfuse)

A self-hosted [Langfuse](https://langfuse.com) stack is bundled for trace capture (prompts, completions, latency, token counts). No data is sent to the cloud.

```bash
make langfuse-up             # start Langfuse stack
make langfuse-bootstrap      # create project + API keys
make langfuse-test-trace     # verify end-to-end
```

Open `http://localhost:3000` to view traces.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `host.docker.internal` unreachable from container | Update Docker Desktop (≥ 18.03) or verify `extra_hosts` entry in compose file |
| LM Studio reachable from Mac but not from container | Set LM Studio Server → Host to `0.0.0.0` (or `127.0.0.1` on Docker Desktop for Mac) |
| 403 / Unauthorized | API key mismatch — either clear key in LM Studio settings or update `LM_STUDIO_API_KEY` in `.env` |
| Low throughput | Adjust `SMOKE_THROUGHPUT_ACCEPT_TPS` in `.env` to match your hardware baseline |

---

## Production Deployment (A100 / Linux)

To deploy on a dual-A100 server:

1. Install LM Studio Linux build on the A100 host.
2. Load the same GGUF model file.
3. Start the local server.
4. Set `LM_STUDIO_BASE_URL=http://<a100-ip>:1234/v1` in `.env`.

No code changes required.

---

## License

Proprietary — ShieldPoint. All rights reserved.
