"""
This module contains the Flask application for the RAG chatbot.
It handles file uploads and chat interactions.
"""

import os
import re
import secrets
import uuid
from flask import Flask, render_template, request, jsonify, g
import structlog
from prometheus_flask_exporter import PrometheusMetrics
from werkzeug.utils import secure_filename
from .config import settings

# Explicitly set LangSmith environment variables for LangChain SDK
# This ensures that even default values from Settings class are respected
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    if settings.LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY

from .rag import rag_service
from .tasks import process_files_batch_task
from .logging_config import configure_logging

# Configure Logging
configure_logging()
logger = structlog.get_logger()

# Initialize Flask Application
app = Flask(__name__, template_folder="templates")

# --- Upload Security Constraints ---
MAX_REQUEST_SIZE = 5 * 1024 * 1024   # 5 MB total per request
MAX_FILE_SIZE = 5 * 1024 * 1024      # 5 MB per individual file
MAX_FILES_PER_REQUEST = 10           # Max number of files per upload
PDF_MAGIC_BYTES = b"%PDF-"            # First 5 bytes of a valid PDF

app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_SIZE

# Security Headers
from flask_talisman import Talisman

csp = {
    "default-src": "'self'",
    "script-src": "'self'",
    "style-src": "'self'",
}
Talisman(
    app, content_security_policy=csp, force_https=False
)  # Let Nginx handle HTTPS redirection

# Initialize Prometheus Metrics
metrics = PrometheusMetrics(app)
metrics.info("app_info", "Application info", version="0.1.0")

# Initialize System Metrics


@app.before_request
def add_request_id():
    """Adds a unique request ID to the global context and structlog context."""
    request_id = str(uuid.uuid4())
    g.request_id = request_id
    structlog.contextvars.bind_contextvars(request_id=request_id)


@app.route("/")
def home():
    """Renders the chat interface."""
    return render_template("index.html")


# --- Session Security ---
# Only alphanumeric, hyphens, and underscores allowed (22-64 chars)
_SESSION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{22,64}$")
_SESSION_TTL = 7200  # 2 hours


def _get_redis_client():
    """Get the Redis client from the RAG service (lazy-initialized)."""
    return rag_service.redis_client


def validate_session_id(session_id: str) -> tuple | None:
    """
    Validate a session ID for format, existence, and path safety.
    Returns a (jsonify, status_code) error tuple if invalid, or None if valid.
    """
    if not session_id:
        return jsonify({"error": "Session ID missing"}), 400

    # 1. Format check — prevents path traversal characters (. / \)
    if not _SESSION_ID_PATTERN.match(session_id):
        logger.warning("session_rejected", reason="invalid_format", session_id=session_id[:50])
        return jsonify({"error": "Invalid session ID format"}), 400

    # 2. Path safety — resolved path must stay inside the expected directory
    chroma_base = os.path.realpath(settings.CHROMA_PERSIST_DIRECTORY)
    resolved = os.path.realpath(os.path.join(chroma_base, session_id))
    if not resolved.startswith(chroma_base + os.sep):
        logger.warning("session_rejected", reason="path_traversal", session_id=session_id[:50])
        return jsonify({"error": "Invalid session ID"}), 400

    # 3. Existence check — session must have been created via /session
    redis_client = _get_redis_client()
    if not redis_client.exists(f"session:{session_id}"):
        logger.warning("session_rejected", reason="unknown_session", session_id=session_id[:50])
        return jsonify({"error": "Unknown or expired session. Please refresh the page."}), 403

    return None


@app.route("/session", methods=["POST"])
def create_session():
    """Create a new session with a cryptographically secure ID."""
    session_id = secrets.token_urlsafe(32)  # 43 chars, 256 bits of entropy
    redis_client = _get_redis_client()
    redis_client.setex(f"session:{session_id}", _SESSION_TTL, "active")
    logger.info("session_created", session_id=session_id)
    return jsonify({"session_id": session_id}), 201


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle uploads that exceed MAX_CONTENT_LENGTH."""
    logger.warning("upload_rejected", reason="request_too_large")
    return jsonify({"error": f"Request too large. Maximum size is {MAX_REQUEST_SIZE // (1024 * 1024)} MB."}), 413


@app.route("/health")
def health_check():
    """
    Health check endpoint.
    Checks application responsiveness and Redis connectivity.
    """
    try:
        # Check Redis connectivity via Celery
        from .celery_app import celery_app

        celery_app.control.ping(timeout=0.1)
        return jsonify({"status": "healthy", "redis": "connected"}), 200
    except Exception as e:
        logger.error("health_check_failed", error=str(e))
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


@app.route("/upload", methods=["POST"])
def upload_file():
    """Handles multiple PDF uploads with size, count, and content validation."""
    if "file" not in request.files:
        logger.warning("upload_failed", reason="no_file_part")
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist("file")
    session_id = request.form.get("session_id")

    # --- Validate session ---
    session_error = validate_session_id(session_id)
    if session_error:
        return session_error

    if not files or all(f.filename == "" for f in files):
        logger.warning("upload_failed", reason="no_selected_files")
        return jsonify({"error": "No selected files"}), 400

    # --- Validate file count ---
    if len(files) > MAX_FILES_PER_REQUEST:
        logger.warning(
            "upload_rejected",
            reason="too_many_files",
            count=len(files),
        )
        return jsonify({"error": f"Maximum {MAX_FILES_PER_REQUEST} files per upload."}), 400

    # --- Validate each file before saving anything ---
    for file in files:
        if not file or not file.filename:
            continue

        # Check extension (server-side)
        if not file.filename.lower().endswith(".pdf"):
            logger.warning(
                "upload_rejected",
                reason="invalid_extension",
                filename=file.filename,
            )
            return jsonify({"error": f"Only PDF files are allowed: {file.filename}"}), 400

        # Check individual file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)     # Reset to beginning
        if file_size > MAX_FILE_SIZE:
            size_mb = MAX_FILE_SIZE // (1024 * 1024)
            logger.warning(
                "upload_rejected",
                reason="file_too_large",
                filename=file.filename,
                size=file_size,
            )
            return jsonify({"error": f"File '{file.filename}' exceeds {size_mb} MB limit."}), 400

        # Validate PDF magic bytes (content-level check)
        header = file.read(5)
        file.seek(0)  # Reset to beginning
        if header != PDF_MAGIC_BYTES:
            logger.warning(
                "upload_rejected",
                reason="invalid_pdf_content",
                filename=file.filename,
            )
            return jsonify({"error": f"File '{file.filename}' is not a valid PDF."}), 400

    # --- All validation passed — save files ---
    filepaths = []
    try:
        for file in files:
            if file and file.filename:
                original_filename = secure_filename(file.filename)
                unique_id = uuid.uuid4().hex[:8]
                filepath = os.path.join(
                    settings.UPLOAD_FOLDER,
                    f"{session_id}_{unique_id}_{original_filename}",
                )
                file.save(filepath)
                filepaths.append(filepath)

        if not filepaths:
            return jsonify({"error": "No valid files uploaded"}), 400

        # Trigger Single Async Batch Task
        task = process_files_batch_task.delay(session_id, filepaths)

        logger.info(
            "batch_task_started",
            task_id=task.id,
            session_id=session_id,
            file_count=len(filepaths),
        )
        return (
            jsonify(
                {
                    "message": "Processing started",
                    "task_ids": [task.id],
                    "file_count": len(filepaths),
                }
            ),
            202,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("processing_error", error=str(e), session_id=session_id)
        return jsonify({"error": f"Failed to start processing: {str(e)}"}), 500

    return jsonify({"error": "Unknown error"}), 500


@app.route("/status/<task_id>", methods=["GET"])
def task_status(task_id):
    """
    Checks the status of a background task.
    """
    task = process_files_batch_task.AsyncResult(task_id)
    if task.state == "PENDING":
        response = {"state": task.state, "status": "Processing..."}
    elif task.state != "FAILURE":
        response = {
            "state": task.state,
            "status": "Task completed!",
            "result": task.result,
        }
    else:
        # something went wrong in the background job
        response = {
            "state": task.state,
            "status": str(task.info),  # this is the exception raised
        }
    return jsonify(response)


@app.route("/chat", methods=["POST"])
def chat():
    """Handles the chat logic using the user's specific PDF data."""
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    user_query = data.get("message")
    session_id = data.get("session_id")

    if not user_query:
        logger.warning("chat_failed", reason="missing_message")
        return jsonify({"error": "Missing message"}), 400

    # --- Validate session ---
    session_error = validate_session_id(session_id)
    if session_error:
        return session_error

    # Sanitize user input to mitigate prompt injection
    user_query = rag_service.sanitize_query(user_query)
    if not user_query:
        return jsonify({"error": "Invalid message content"}), 400

    try:
        logger.info("chat_request_received", session_id=session_id)
        answer = rag_service.get_answer(session_id, user_query)
        return jsonify({"answer": answer})

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("chat_error", error=str(e), session_id=session_id)
        return jsonify({"error": "An error occurred processing your request."}), 500


if __name__ == "__main__":  # pragma: no cover
    print("Starting Flask Server...")
    # Fix Bandit B201: Do not hardcode debug=True in production
    # Use environment variable FLASK_DEBUG, default to False
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode, port=5000)
