import pytest
from unittest.mock import patch, MagicMock
from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@patch(
    "src.app.process_file_task"
)  # We can use this to get to celery_app via mock if needed, but easier to mock celery_app.control.ping
def test_health_check_success(mock_task, client):
    """Test health check returns 200 and healthy status."""
    with patch("src.celery_app.celery_app.control.ping") as mock_ping:
        mock_ping.return_value = ["pong"]
        response = client.get("/health")
        assert response.status_code == 200
        json_data = response.get_json()
        assert json_data["status"] == "healthy"
        assert json_data["redis"] == "connected"


def test_health_check_failure(client):
    """Test health check returns 500 when Redis/Celery is down."""
    with patch("src.celery_app.celery_app.control.ping") as mock_ping:
        mock_ping.side_effect = Exception("Redis connection failed")
        response = client.get("/health")
        assert response.status_code == 500
        json_data = response.get_json()
        assert json_data["status"] == "unhealthy"
        assert "Redis connection failed" in json_data["error"]
