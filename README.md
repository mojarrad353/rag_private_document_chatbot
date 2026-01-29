![Python](https://img.shields.io/badge/Python-3.x-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)

---

# RAG Private Document Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that allows you to chat with your private PDF documents using LangChain, OpenAI, and ChromaDB.

## Features
- **PDF Upload**: Upload and process PDF documents.
- **RAG Architecture**: Uses ChromaDB for vector storage and OpenAI for generation.
- **Session Management**: Each user session has its own isolated context.
- **Modern Structure**: Organized with Flask, Pydantic Settings, and Docker.

## Prerequisites
- Python 3.12+
- OpenAI API Key

## Setup

### 1. Environment Variables
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL_NAME=gpt-4o-mini
```

### 2. Local Development (using uv)
This project uses `uv` for dependency management.

```bash
# Install dependencies
uv sync

# Run the application
uv run flask --app src/app.py run
```

Access the app at `http://localhost:5000`.

### 3. Docker
To run with Docker Compose:

```bash
docker compose up --build
```

## Project Structure
- `src/`: Application source code.
    - `app.py`: Flask routes and entry point.
    - `rag.py`: Core RAG logic.
    - `config.py`: Configuration settings.
- `Dockerfile`: Container definition.
- `docker-compose.yml`: Local deployment config.

## Usage
1. Open the application in your browser.
2. Upload a PDF file.
3. Start chatting with your document!
