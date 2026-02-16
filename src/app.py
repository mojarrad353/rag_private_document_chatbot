"""
This module contains the Flask application for the RAG chatbot.
It handles file uploads and chat interactions.
"""

import os
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
from .tasks import process_file_task
from .logging_config import configure_logging

# Configure Logging
configure_logging()
logger = structlog.get_logger()

# Initialize Flask App
app = Flask(__name__, template_folder="templates")

# Security Headers
from flask_talisman import Talisman

csp = {
    "default-src": "'self'",
    "script-src": "'self' 'unsafe-inline'",  # Be careful with unsafe-inline
    "style-src": "'self' 'unsafe-inline'",
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
    """Handles PDF upload and processes it for the specific session."""
    if "file" not in request.files:
        logger.warning("upload_failed", reason="no_file_part")
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    session_id = request.form.get("session_id")

    if not session_id:
        logger.warning("upload_failed", reason="missing_session_id")
        return jsonify({"error": "Session ID missing"}), 400

    if file.filename == "":
        logger.warning("upload_failed", reason="no_selected_file")
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Save File Temporarily
            original_filename = secure_filename(file.filename)
            filepath = os.path.join(
                settings.UPLOAD_FOLDER, f"{session_id}_{original_filename}"
            )
            file.save(filepath)

            # Trigger Async Task
            task = process_file_task.delay(session_id, filepath)

            logger.info("async_task_started", task_id=task.id, session_id=session_id)
            return (
                jsonify({"message": "File processing started", "task_id": task.id}),
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
    task = process_file_task.AsyncResult(task_id)
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

    if not user_query or not session_id:
        logger.warning("chat_failed", reason="missing_data")
        return jsonify({"error": "Missing message or session_id"}), 400

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
