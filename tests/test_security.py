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


@patch("src.app.validate_session_id", return_value=None)
@patch("src.app.process_files_batch_task")
def test_path_traversal_non_pdf_rejected(mock_task, mock_validate, client):
    """Test that path traversal with non-PDF extension is rejected by extension check."""
    with patch("werkzeug.datastructures.FileStorage.save"):
        import io

        data = {
            "file": (io.BytesIO(b"content"), "../../../etc/passwd"),
            "session_id": "aB3dEf7hIjKlMnOpQrStUv",
        }
        response = client.post("/upload", data=data, content_type="multipart/form-data")

        # Now rejected at the extension validation layer
        assert response.status_code == 400
        assert b"Only PDF files are allowed" in response.data


@patch("src.app.validate_session_id", return_value=None)
@patch("src.app.process_files_batch_task")
def test_filename_sanitization_with_pdf_extension(mock_task, mock_validate, client):
    """Test that path traversal in .pdf filenames is sanitized by secure_filename."""
    mock_result = MagicMock()
    mock_result.id = "test-security-task-id"
    mock_task.delay.return_value = mock_result

    with patch("werkzeug.datastructures.FileStorage.save") as mock_save:
        import io

        # Path traversal attempt with .pdf extension — passes extension check
        # but secure_filename strips the traversal
        data = {
            "file": (io.BytesIO(b"%PDF-1.4 fake pdf"), "../../../etc/evil.pdf"),
            "session_id": "aB3dEf7hIjKlMnOpQrStUv",
        }
        response = client.post("/upload", data=data, content_type="multipart/form-data")

        assert response.status_code == 202

        # Verify the filename was sanitized: secure_filename("../../../etc/evil.pdf") -> "etc_evil.pdf"
        args, _ = mock_save.call_args
        saved_path = args[0]
        assert "evil.pdf" in saved_path
        assert ".." not in saved_path

