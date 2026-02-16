# RAG Private Document Chatbot

[![CI](https://github.com/mojarrad353/rag_private_document_chatbot/actions/workflows/ci.yml/badge.svg)]
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-green?logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)](https://redis.io/)

A production-ready RAG (Retrieval-Augmented Generation) chatbot that allows users to upload PDF documents and chat with them using LLMs. Built with Flask, LangChain, ChromaDB, and OpenAI.

## Architecture

```
User → Nginx (port 80) → Gunicorn/Flask (port 5000) → Celery Worker → Redis
                                  ↓                          ↓
                              ChromaDB                   OpenAI API
                           (persistent)
```

| Service | Port | Purpose |
|---------|------|---------|
| **Nginx** | 80 | Reverse proxy, rate limiting |
| **Flask App** | 5000 | Web UI + API |
| **Celery Worker** | — | Background PDF processing |
| **Redis** | 6379 | Task queue + chat history |
| **Prometheus** | 9090 | Metrics collection |
| **Grafana** | 3000 | Dashboards + log explorer |
| **Loki** | 3100 | Centralized log storage |
| **Promtail** | — | Log shipping from containers |

## Features

- **Document Ingestion**: Upload PDFs → automatic splitting, embedding, and indexing.
- **RAG Architecture**: Persistent ChromaDB for vector storage, OpenAI for generation, Redis for chat memory.
- **Async Processing**: File uploads processed in the background via **Celery** + **Redis**. Frontend polls for completion status.
- **Shared Session State**: ChromaDB persisted to disk and chat history stored in Redis — state is shared across the Flask app and Celery worker.
- **Observability**:
  - **Structured Logging**: JSON logs via `structlog` with request-scoped tracing.
  - **Centralized Logs**: **Grafana Loki** + **Promtail** — all container logs searchable in Grafana.
  - **Metrics**: Prometheus metrics at `/metrics` (token usage, cost, latency).
  - **Tracing**: Full LangChain tracing with **LangSmith**.
- **Security**:
  - Nginx reverse proxy with rate limiting.
  - Security headers (HSTS, CSP, XSS) via `flask-talisman`.
  - Non-root Docker container.
  - Input filename sanitization.
- **Health Checks**: Docker health probes + `/health` endpoint (app + Redis connectivity).
- **Production Ready**: Gunicorn WSGI server, 120s timeout for model loading, hot-reload for development.

## Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/mojarrad353/rag_private_document_chatbot.git
cd rag_private_document_chatbot
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key

# Optional: LangSmith Observability
LANGSMITH_API_KEY=your_langsmith_key
```

### 2. Run

```bash
docker compose up -d
```

### 3. Use

| What | URL |
|------|-----|
| **Chat Interface** | [http://localhost](http://localhost) |
| **Grafana** (metrics + logs) | [http://localhost:3000](http://localhost:3000) (admin/admin) |
| **Prometheus** | [http://localhost:9090](http://localhost:9090) |

### 4. Stop

```bash
docker compose down
```

## Local Development (Without Docker)

1. **Start Redis**:
   ```bash
   docker run -d -p 6379:6379 redis:alpine
   ```

2. **Start Celery Worker**:
   ```bash
   uv run celery -A src.celery_app.celery_app worker --loglevel=info
   ```

3. **Start Flask App**:
   ```bash
   uv run flask --app src/app.py run --debug
   ```

> **Tip**: Source code is volume-mounted into Docker. After code changes, just run `docker compose restart app worker` — no rebuild needed.

## Observability

### Metrics

Prometheus metrics at `http://localhost:5000/metrics`:

| Metric | Description |
|--------|-------------|
| `rag_tokens_total{type="prompt\|completion"}` | Token usage counter |
| `rag_cost_total` | Estimated cost in USD |
| `process_cpu_seconds_total` | CPU usage |
| `process_resident_memory_bytes` | Memory usage |

### Centralized Logs (Loki)

All container logs are collected by **Promtail** and shipped to **Loki**, queryable in Grafana.

1. Open [Grafana → Explore](http://localhost:3000/explore)
2. Select **Loki** as the data source
3. Example queries:

| Query | Shows |
|-------|-------|
| `{service="app"}` | All app logs |
| `{service="worker"}` | Celery worker logs |
| `{service="nginx"}` | Nginx access/error logs |
| `{level="error"}` | All errors across services |
| `{event="invoke_chain_complete"}` | RAG query completions |

### LangSmith Tracing

Set `LANGSMITH_API_KEY` in your `.env` file. Traces are sent to the `rag-private-document-chatbot` project in [LangSmith](https://smith.langchain.com/).

### Health Checks

- **Endpoint**: `http://localhost:5000/health` (checks Flask + Redis)
- **Docker**: `docker ps` shows `healthy` / `unhealthy` status

### Grafana Dashboard

1. Open [Grafana](http://localhost:3000) → **Dashboards** → **Import**
2. Upload `grafana_dashboard.json` from the project root
3. View real-time request rate, latency, token usage, and cost

## CI Pipeline

Quality gates enforced on every push:

| Check | Threshold |
|-------|-----------|
| **Pylint** | Score ≥ 9.0 |
| **Test Coverage** | ≥ 85% |
| **Black** | Formatting check |
| **Mypy** | Type checking |
| **Bandit** | Security scan |

## Project Structure

```
├── src/
│   ├── app.py              # Flask application + routes
│   ├── rag.py              # RAG service (ChromaDB + LangChain)
│   ├── tasks.py            # Celery async tasks
│   ├── celery_app.py       # Celery configuration
│   ├── config.py           # Pydantic settings
│   ├── logging_config.py   # Structlog configuration
│   └── templates/
│       └── index.html      # Chat UI
├── tests/                  # Unit tests (pytest)
├── nginx/nginx.conf        # Nginx reverse proxy config
├── prometheus.yml          # Prometheus scrape config
├── loki-config.yml         # Loki server config
├── promtail-config.yml     # Promtail log collection config
├── grafana/provisioning/   # Auto-provisioned Grafana datasources
├── grafana_dashboard.json  # Pre-built Grafana dashboard
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Full stack orchestration (8 services)
└── pyproject.toml          # Dependencies + tool config
```
