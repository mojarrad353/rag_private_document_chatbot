FROM python:3.12-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv sync --frozen

# Copy source code
COPY src/ src/

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH"
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose port
EXPOSE 5000

# Run commands
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
