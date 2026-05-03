import io

import pytest
from unittest.mock import MagicMock, patch
from src.app import app

# Valid PDF content for upload tests (starts with %PDF- magic bytes)
VALID_PDF_CONTENT = b"%PDF-1.4 fake pdf content for testing"

# A session ID that matches the _SESSION_ID_PATTERN regex (22-64 alphanumeric + _-)
VALID_SESSION_ID = "aB3dEf7hIjKlMnOpQrStUv"


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_session():
    """Patch validate_session_id to always pass (returns None = valid)."""
    with patch("src.app.validate_session_id", return_value=None):
        yield VALID_SESSION_ID


# --- Home ---


def test_home(client):
    """Test the home page route."""
    response = client.get("/")
    assert response.status_code == 200
    assert (
        b"Chatbot" in response.data
        or b"Recall" in response.data
        or b"<!DOCTYPE html>" in response.data
    )


# --- Session Endpoint ---


@patch("src.app._get_redis_client")
def test_create_session(mock_redis, client):
    """Test the /session endpoint creates a session and returns a secure ID."""
    mock_client = MagicMock()
    mock_redis.return_value = mock_client

    response = client.post("/session")

    assert response.status_code == 201
    data = response.get_json()
    assert "session_id" in data
    assert len(data["session_id"]) >= 22  # secrets.token_urlsafe(32) produces 43 chars
    mock_client.setex.assert_called_once()


# --- Session Validation ---


def test_upload_rejects_missing_session_id(client):
    """Test file upload without session_id."""
    data = {"file": (io.BytesIO(b"%PDF-fake"), "test.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Session ID missing" in response.data


@patch("src.app._get_redis_client")
def test_upload_rejects_invalid_session_format(mock_redis, client):
    """Test that session IDs with path traversal characters are rejected."""
    data = {
        "file": (io.BytesIO(VALID_PDF_CONTENT), "test.pdf"),
        "session_id": "../../etc/passwd",
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Invalid session ID format" in response.data


@patch("src.app._get_redis_client")
def test_upload_rejects_unknown_session(mock_redis, client):
    """Test that unregistered session IDs are rejected."""
    mock_client = MagicMock()
    mock_client.exists.return_value = False
    mock_redis.return_value = mock_client

    data = {
        "file": (io.BytesIO(VALID_PDF_CONTENT), "test.pdf"),
        "session_id": VALID_SESSION_ID,
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 403
    assert b"Unknown or expired session" in response.data


@patch("src.app._get_redis_client")
def test_chat_rejects_unknown_session(mock_redis, client):
    """Test that chat rejects unregistered session IDs."""
    mock_client = MagicMock()
    mock_client.exists.return_value = False
    mock_redis.return_value = mock_client

    response = client.post(
        "/chat", json={"message": "Hi", "session_id": VALID_SESSION_ID}
    )
    assert response.status_code == 403
    assert b"Unknown or expired session" in response.data


# --- Upload (No file part) ---


@patch("src.app.validate_session_id", return_value=None)
def test_upload_file_no_file(mock_validate, client):
    """Test file upload without file part."""
    response = client.post("/upload", data={})
    assert response.status_code == 400
    assert b"No file part" in response.data


@patch("src.app.validate_session_id", return_value=None)
def test_upload_file_no_selected_file(mock_validate, client):
    """Test file upload with empty filename."""
    data = {"file": (io.BytesIO(b""), ""), "session_id": VALID_SESSION_ID}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"No selected files" in response.data


# --- Upload Success ---


@patch("src.app.validate_session_id", return_value=None)
@patch("src.app.process_files_batch_task")
@patch("os.path.exists")
def test_upload_file_success(mock_exists, mock_task, mock_validate, client):
    """Test successful file upload triggers async task."""
    mock_exists.return_value = True
    mock_result = MagicMock()
    mock_result.id = "test-task-123"
    mock_task.delay.return_value = mock_result

    data = {
        "file": (io.BytesIO(VALID_PDF_CONTENT), "test.pdf"),
        "session_id": VALID_SESSION_ID,
    }

    with patch("werkzeug.datastructures.FileStorage.save"):
        response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 202
    assert response.json["task_ids"] == ["test-task-123"]
    mock_task.delay.assert_called_once()


@patch("src.app.validate_session_id", return_value=None)
@patch("src.app.process_files_batch_task")
def test_upload_file_exception(mock_task, mock_validate, client):
    """Test exception during file processing startup."""
    mock_task.delay.side_effect = Exception("Queue error")
    data = {
        "file": (io.BytesIO(VALID_PDF_CONTENT), "test.pdf"),
        "session_id": VALID_SESSION_ID,
    }

    with patch("werkzeug.datastructures.FileStorage.save"):
        response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 500
    assert b"Failed to start processing" in response.data


# --- Upload File Validation ---


@patch("src.app.validate_session_id", return_value=None)
def test_upload_rejects_non_pdf_extension(mock_validate, client):
    """Test that non-PDF extensions are rejected server-side."""
    data = {
        "file": (io.BytesIO(b"%PDF-fake"), "malware.exe"),
        "session_id": VALID_SESSION_ID,
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Only PDF files are allowed" in response.data


@patch("src.app.validate_session_id", return_value=None)
def test_upload_rejects_invalid_pdf_content(mock_validate, client):
    """Test that files without PDF magic bytes are rejected."""
    data = {
        "file": (io.BytesIO(b"NOT-A-PDF-FILE"), "fake.pdf"),
        "session_id": VALID_SESSION_ID,
    }
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"not a valid PDF" in response.data


@patch("src.app.validate_session_id", return_value=None)
def test_upload_rejects_too_many_files(mock_validate, client):
    """Test that more than MAX_FILES_PER_REQUEST files are rejected."""
    from werkzeug.datastructures import MultiDict

    items = [("session_id", VALID_SESSION_ID)]
    for i in range(11):  # MAX_FILES_PER_REQUEST is 10
        items.append(("file", (io.BytesIO(VALID_PDF_CONTENT), f"doc{i}.pdf")))

    response = client.post(
        "/upload",
        data=MultiDict(items),
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert b"Maximum" in response.data


# --- Chat ---


@patch("src.app.validate_session_id", return_value=None)
@patch("src.app.rag_service")
def test_chat_success(mock_rag, mock_validate, client):
    """Test successful chat interaction."""
    mock_rag.get_answer.return_value = "This is the answer."
    mock_rag.sanitize_query.side_effect = lambda q: q  # Pass through
    data = {"message": "Hello", "session_id": VALID_SESSION_ID}

    response = client.post("/chat", json=data)

    assert response.status_code == 200
    assert response.json["answer"] == "This is the answer."
    mock_rag.sanitize_query.assert_called_with("Hello")
    mock_rag.get_answer.assert_called_with(VALID_SESSION_ID, "Hello")


@patch("src.app.validate_session_id", return_value=None)
@patch("src.app.rag_service")
def test_chat_exception(mock_rag, mock_validate, client):
    """Test exception during chat."""
    mock_rag.sanitize_query.side_effect = lambda q: q  # Pass through
    mock_rag.get_answer.side_effect = Exception("Chat error")
    data = {"message": "Hello", "session_id": VALID_SESSION_ID}

    response = client.post("/chat", json=data)

    assert response.status_code == 500
    assert b"An error occurred" in response.data


@patch("src.app.rag_service")
def test_chat_missing_message(mock_rag, client):
    """Test chat with missing message."""
    response = client.post("/chat", json={})
    assert response.status_code == 400
    assert b"Invalid JSON" in response.data or b"Missing message" in response.data
