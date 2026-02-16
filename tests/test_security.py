import pytest
from src.app import app
from unittest.mock import patch, MagicMock


@pytest.fixture
def client():
    app.config["TESTING"] = True
    # Talisman forces HTTPS by default unless configured otherwise.
    # In app.py we set force_https=False for dev/docker, but let's check headers.
    with app.test_client() as client:
        yield client


def test_security_headers(client):
    """Test that security headers are present."""
    response = client.get("/")
    print(response.headers)
    assert "Content-Security-Policy" in response.headers
    assert "X-Content-Type-Options" in response.headers
    assert "X-Frame-Options" in response.headers


@patch("src.app.process_file_task")
def test_filename_sanitization(mock_task, client):
    """Test that filenames are sanitized and saved correctly."""
    # Mock task return
    mock_result = MagicMock()
    mock_result.id = "test-security-task-id"
    mock_task.delay.return_value = mock_result

    with patch("werkzeug.datastructures.FileStorage.save") as mock_save:
        import io

        data = {
            "file": (io.BytesIO(b"content"), "../../../etc/passwd"),
            "session_id": "test_session",
        }
        response = client.post("/upload", data=data, content_type="multipart/form-data")

        # Check successful response
        assert response.status_code == 202

        # Verify the filename was sanitized in the path passed to save()
        # secure_filename("../../../etc/passwd") -> "etc_passwd"
        # path should be "uploads/test_session_etc_passwd"
        args, _ = mock_save.call_args
        saved_path = args[0]
        assert "etc_passwd" in saved_path
        assert ".." not in saved_path

        # Verify the async task received the sanitized path
        task_args, _ = mock_task.delay.call_args
        processed_path = task_args[1]
        assert processed_path == saved_path
