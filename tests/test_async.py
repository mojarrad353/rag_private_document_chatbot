import pytest
from unittest.mock import patch, MagicMock
from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch("src.app.process_file_task")
def test_upload_file_async(mock_task, client):
    """Test that file upload triggers an async task."""
    # Mock file save to avoid disk write
    with patch("werkzeug.datastructures.FileStorage.save"):
        # Mock the Celery task delay method
        mock_result = MagicMock()
        mock_result.id = "test-task-id-123"
        mock_task.delay.return_value = mock_result

        import io

        data = {
            "file": (io.BytesIO(b"dummy content"), "test.pdf"),
            "session_id": "test_session",
        }
        response = client.post("/upload", data=data, content_type="multipart/form-data")

    assert response.status_code == 202
    json_data = response.get_json()
    assert json_data["task_id"] == "test-task-id-123"
    assert "File processing started" in json_data["message"]

    # Verify task was called
    mock_task.delay.assert_called_once()


@patch("src.app.process_file_task")
def test_task_status_pending(mock_task, client):
    """Test checking status of a pending task."""
    mock_result = MagicMock()
    mock_result.state = "PENDING"
    mock_task.AsyncResult.return_value = mock_result

    response = client.get("/status/test-task-id-123")

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["state"] == "PENDING"
    assert json_data["status"] == "Processing..."


@patch("src.app.process_file_task")
def test_task_status_success(mock_task, client):
    """Test checking status of a completed task."""
    mock_result = MagicMock()
    mock_result.state = "SUCCESS"
    mock_result.result = {"status": "success", "session_id": "123"}
    mock_task.AsyncResult.return_value = mock_result

    response = client.get("/status/test-task-id-123")

    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data["state"] == "SUCCESS"
    assert json_data["result"]["status"] == "success"
