# Private Document Chatbot with Multi-Doc Graph RAG

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
- **Graph RAG Orchestration**: Built with **LangGraph** to handle complex, parallel retrieval branches (Map-Reduce pattern).
- **Asynchronous Processing**: Background file indexing using **Celery** and **Redis** to ensure a responsive UI.
- **State-of-the-Art Retrieval**: Uses **PyMuPDF** for high-fidelity extraction and **ChromaDB** for vector storage.
- **Production-Ready Observability**: Full integration with the Prometheus/Grafana/Loki stack and LangSmith for real-time monitoring.
- **Robust Security**: Includes filename sanitization, secure headers (Talisman), and Nginx reverse proxying.

## 🛠 Technology Stack

- **AI/LLM**: LangChain, LangGraph, OpenAI (GPT-4o-mini), HuggingFace (Local Embeddings).
- **Backend**: Flask (Python 3.12), Gunicorn, Celery.
- **Data Layers**: ChromaDB (Vector Store), Redis (Task Queue & Chat History).
- **Infrastructure**: Docker Compose, Nginx (Reverse Proxy).
- **Observability**: Prometheus, Grafana, Loki, Promtail, Structlog, LangSmith.

## 📁 Project Structure 

```text
├── .github/workflows/  # CI/CD (linting, testing, security, code review)
├── grafana/           # Grafana provisioning (dashboards & datasources)
├── nginx/             # Nginx reverse proxy configuration
├── scripts/           # Utility scripts (AI Code Review)
├── src/               # Main application source code
│   ├── app.py         # Flask API & Routes
│   ├── rag.py         # Graph RAG logic & LLM orchestration
│   ├── tasks.py       # Celery background tasks
│   ├── celery_app.py  # Celery initialization
│   ├── config.py      # Environment & Settings
│   └── templates/     # UI template (index.html)
├── tests/             # Comprehensive Pytest suite (91%+ coverage)
├── docker-compose.yml # Full stack orchestration
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
```

### 2. Launch the Application
Start the entire stack using Docker Compose:
```bash
docker compose up -d
```
The application will be available at [http://localhost](http://localhost).

## 📊 Observability & Metrics

The system is equipped with a full observability stack to monitor AI performance and operational health:

- **Grafana Dashboard**: Access at `http://localhost:3000` (Default: admin/admin).
- **Prometheus Metrics**: Available at `/metrics` from the app container.
- **Loki Logs**: Centralized structured logging for all services.

### Tracked AI Metrics
- **`rag_llm_calls_total`**: Total number of calls made to the LLM.
- **`rag_tokens_total`**: Breakdown of tokens used (Prompt vs. Completion).
- **`rag_cost_total`**: Real-time tracking of cumulative API costs in USD.

## 🧪 Testing & Quality

This project maintains high standards of code quality, enforced by GitHub Actions:

- **Pytest**: Over **91.7% coverage** across the core RAG and Task logic.
- **Pylint**: Adheres to strict coding standards (Score > 9.0).
- **Mypy**: Full type-checking for static safety.
- **Black**: Automatic code formatting.

Run local tests with:
```bash
uv run pytest --cov=src
```

---
*Built for security and scale.*
