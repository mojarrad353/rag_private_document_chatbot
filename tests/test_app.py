import pytest
from unittest.mock import MagicMock, patch
from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_home(client):
    """Test the home page route."""
    response = client.get("/")
    assert response.status_code == 200
    assert (
        b"Chatbot" in response.data
        or b"Recall" in response.data
        or b"<!DOCTYPE html>" in response.data
    )


import io


@patch("src.app.rag_service")
def test_upload_file_no_file(mock_rag, client):
    """Test file upload without file part."""
    response = client.post("/upload", data={})
    assert response.status_code == 400
    assert b"No file part" in response.data


@patch("src.app.rag_service")
def test_upload_file_no_selected_file(mock_rag, client):
    """Test file upload with empty filename."""
    data = {"file": (io.BytesIO(b""), ""), "session_id": "test_session"}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"No selected file" in response.data


@patch("src.app.rag_service")
def test_upload_file_missing_session_id(mock_rag, client):
    """Test file upload without session_id."""
    data = {"file": (io.BytesIO(b"content"), "test.pdf")}
    response = client.post("/upload", data=data, content_type="multipart/form-data")
    assert response.status_code == 400
    assert b"Session ID missing" in response.data


@patch("src.app.process_file_task")
@patch("os.path.exists")
def test_upload_file_success(mock_exists, mock_task, client):
    """Test successful file upload triggers async task."""
    mock_exists.return_value = True
    mock_result = MagicMock()
    mock_result.id = "test-task-123"
    mock_task.delay.return_value = mock_result

    data = {"file": (io.BytesIO(b"content"), "test.pdf"), "session_id": "test_session"}

    with patch("werkzeug.datastructures.FileStorage.save"):
        response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 202
    assert response.json["task_id"] == "test-task-123"
    mock_task.delay.assert_called_once()


@patch("src.app.process_file_task")
def test_upload_file_exception(mock_task, client):
    """Test exception during file processing startup."""
    mock_task.delay.side_effect = Exception("Queue error")
    data = {"file": (io.BytesIO(b"content"), "test.pdf"), "session_id": "test_session"}

    with patch("werkzeug.datastructures.FileStorage.save"):
        response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 500
    assert b"Failed to start processing" in response.data


@patch("src.app.rag_service")
def test_chat_success(mock_rag, client):
    """Test successful chat interaction."""
    mock_rag.get_answer.return_value = "This is the answer."
    data = {"message": "Hello", "session_id": "test_session"}

    response = client.post("/chat", json=data)

    assert response.status_code == 200
    assert response.json["answer"] == "This is the answer."
    mock_rag.get_answer.assert_called_with("test_session", "Hello")


@patch("src.app.rag_service")
def test_chat_exception(mock_rag, client):
    """Test exception during chat."""
    mock_rag.get_answer.side_effect = Exception("Chat error")
    data = {"message": "Hello", "session_id": "test_session"}

    response = client.post("/chat", json=data)

    assert response.status_code == 500
    assert b"An error occurred" in response.data


@patch("src.app.rag_service")
def test_chat_missing_data(mock_rag, client):
    """Test chat with missing data."""
    response = client.post("/chat", json={})
    assert response.status_code == 400
    assert b"Invalid JSON" in response.data or b"Missing message" in response.data

    response = client.post("/chat", json={"message": "Hi"})
    assert response.status_code == 400
