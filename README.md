# Private Document Chatbot with Multi-Doc RAG

[![CI](https://github.com/mojarrad353/rag_private_document_chatbot/actions/workflows/ci.yml/badge.svg)]
[![Python](https://img.shields.io/badge/python-3.12%2B-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)](https://python.langchain.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-green?logo=langchain&logoColor=white)](https://smith.langchain.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800?logo=grafana&logoColor=white)](https://grafana.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Redis](https://img.shields.io/badge/Redis-DC382D?logo=redis&logoColor=white)](https://redis.io/)

A high-performance, private Retrieval-Augmented Generation (RAG) chatbot designed for secure document interaction. This application uses a sophisticated **Map-Reduce Parallel Retrieval** architecture powered by **LangGraph** to concurrently query multiple documents and provide contextually accurate answers.

## 🚀 Key Features

- **Multi-Document Support**: Upload and chat with multiple PDF files simultaneously.
- **RAG Orchestration**: Built with **LangGraph** to handle complex, parallel retrieval branches (Map-Reduce pattern).
- **Asynchronous Processing**: Background file indexing using **Celery** and **Redis** to ensure a responsive UI.
- **State-of-the-Art Retrieval**: Uses **PyMuPDF** for high-fidelity extraction and **ChromaDB** for vector storage.
- **Production-Ready Observability**: Full integration with the Prometheus/Grafana/Loki stack and LangSmith for real-time monitoring.
- **Defense-in-Depth Security**: Multi-layered hardening against prompt injection, file upload attacks, session forgery, and network-level threats (see [Security](#-security) section).

## 🛠 Technology Stack

- **AI/LLM**: LangChain, LangGraph, OpenAI (GPT-4o-mini), HuggingFace (Local Embeddings).
- **Backend**: Flask (Python 3.12), Gunicorn, Celery.
- **Data Layers**: ChromaDB (Vector Store), Redis (Task Queue & Chat History).
- **Infrastructure**: Docker Compose, Nginx (Reverse Proxy & Rate Limiting).
- **Observability**: Prometheus, Grafana, Loki, Promtail, Structlog, LangSmith.
- **Security**: Flask-Talisman (CSP/Headers), Docker Socket Proxy, server-side session management.

## 📁 Project Structure 

```text
├── .github/workflows/  # CI/CD (linting, testing, security, code review)
├── grafana/           # Grafana provisioning (dashboards & datasources)
├── nginx/             # Nginx reverse proxy & rate limiting configuration
├── scripts/           # Utility scripts (AI Code Review)
├── src/               # Main application source code
│   ├── app.py         # Flask API, routes & security middleware
│   ├── rag.py         # RAG logic, LLM orchestration & I/O sanitization
│   ├── tasks.py       # Celery background tasks
│   ├── celery_app.py  # Celery initialization
│   ├── config.py      # Environment & Settings
│   ├── static/        # External CSS & JS (CSP-compliant, no inline code)
│   │   ├── style.css  # Application styles
│   │   └── app.js     # Client-side logic (session init, upload, chat)
│   └── templates/     # HTML template (index.html)
├── tests/             # Comprehensive Pytest suite (89%+ coverage)
├── docker-compose.yml # Full stack orchestration (network-isolated)
├── promtail-config.yml # Log collection config (uses socket proxy)
└── README.md          # You are here
```

## ⚙️ Setup & Installation

### Prerequisites
- Docker & Docker Compose
- OpenAI API Key

### 1. Configuration
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
GRAFANA_ADMIN_PASSWORD=your_grafana_password
```

### 2. Launch the Application
Start the entire stack using Docker Compose:
```bash
docker compose up -d
```
The application will be available at [http://localhost](http://localhost) (port 80 via Nginx — the only publicly exposed port).

## 🔒 Security

This project implements a **defense-in-depth** strategy with multiple independent security layers:

### Prompt Injection Defense
| Layer | Mechanism | Location |
|-------|-----------|----------|
| **Hardened System Prompt** | XML-delimited `<context>` tags with absolute anti-override rules placed before untrusted content | `rag.py` |
| **Input Sanitization** | Query length capped at 2000 chars, control characters stripped | `rag.py` → `sanitize_query()` |
| **Output Filtering** | Regex patterns block LLM responses containing prompt leaks, URLs, or code blocks | `rag.py` → `sanitize_output()` |

### File Upload Hardening
| Layer | Mechanism | Location |
|-------|-----------|----------|
| **Request Size Limit** | Flask `MAX_CONTENT_LENGTH = 5 MB` — rejects oversized requests before they reach application code | `app.py` |
| **Per-File Size Check** | Each file individually validated against 5 MB limit | `app.py` |
| **File Count Limit** | Maximum 10 files per upload request | `app.py` |
| **Extension Validation** | Server-side `.pdf` extension check (client `accept` attribute is bypassable) | `app.py` |
| **Content Validation** | PDF magic byte (`%PDF-`) verification — rejects renamed non-PDF files | `app.py` |
| **Filename Sanitization** | Werkzeug `secure_filename()` strips path traversal sequences | `app.py` |

### Session & Access Control
| Layer | Mechanism | Location |
|-------|-----------|----------|
| **Server-Side Session IDs** | `secrets.token_urlsafe(32)` — 256 bits of entropy, registered in Redis with 2-hour TTL | `app.py` → `/session` |
| **Session Validation** | Triple-check on every request: regex format, path safety (realpath), and Redis existence | `app.py` → `validate_session_id()` |
| **Path Traversal Prevention** | Session IDs validated against `^[a-zA-Z0-9_-]{22,64}$` — no `.`, `/`, `\` allowed | `app.py` |

### Network & Infrastructure
| Layer | Mechanism | Location |
|-------|-----------|----------|
| **Single Entry Point** | Only Nginx port 80 is exposed; Flask, Redis, Prometheus, Loki have no external ports | `docker-compose.yml` |
| **Per-Endpoint Rate Limiting** | `/chat` 2r/s, `/upload` 1r/s, `/status` 5r/s, general 30r/s | `nginx.conf` |
| **Internal-Only Metrics** | `/metrics` and `/health` restricted to Docker bridge subnets (172.16.0.0/12, 10.0.0.0/8) | `nginx.conf` |
| **CSP Without `unsafe-inline`** | All CSS/JS in external files; Content Security Policy blocks injected inline scripts | `app.py`, `static/` |
| **Security Headers** | X-Content-Type-Options, X-Frame-Options, CSP via Flask-Talisman | `app.py` |
| **Docker Socket Proxy** | Promtail connects to `tecnativa/docker-socket-proxy` instead of raw `/var/run/docker.sock` — blocks access to container env vars (API keys) | `docker-compose.yml` |
| **Grafana Localhost-Only** | Grafana bound to `127.0.0.1:3001` — not reachable from external networks | `docker-compose.yml` |

## 📊 Observability & Metrics

The system is equipped with a full observability stack to monitor AI performance and operational health:

- **Grafana Dashboard**: Access at `http://localhost:3001` (set password via `GRAFANA_ADMIN_PASSWORD` in `.env`).
- **Prometheus Metrics**: Available internally at `/metrics` (restricted to Docker network).
- **Loki Logs**: Centralized structured logging for all services via Promtail.
- **LangSmith**: Optional LLM trace monitoring (configure `LANGSMITH_API_KEY` in `.env`).

### Tracked AI Metrics
- **`rag_llm_calls_total`**: Total number of calls made to the LLM.
- **`rag_tokens_total`**: Breakdown of tokens used (Prompt vs. Completion).
- **`rag_cost_total`**: Real-time tracking of cumulative API costs in USD.

## 🧪 Testing & Quality

This project maintains high standards of code quality, enforced by GitHub Actions:

- **Pytest**: **89%+ coverage** across the core RAG, API, and security logic (32 tests).
- **Pylint**: Adheres to strict coding standards (Score > 9.0).
- **Mypy**: Full type-checking for static safety.
- **Black**: Automatic code formatting.

Run local tests with:
```bash
uv run pytest --cov=src
```


