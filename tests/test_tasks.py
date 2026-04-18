import pytest
from unittest.mock import patch, MagicMock
from src.tasks import process_files_batch_logic


def test_process_files_batch_logic_success():
    """Test successful batch processing of multiple files."""
    session_id = "test_session_123"
    filepaths = ["upload/file1.pdf", "upload/file2.pdf"]

    with (
        patch("src.tasks.rag_service.process_file") as mock_process,
        patch("os.path.exists", return_value=True) as mock_exists,
        patch("os.remove") as mock_remove,
    ):
        result = process_files_batch_logic(session_id, filepaths, task_id="task-123")

    assert result["status"] == "batch_complete"
    assert len(result["results"]) == 2
    assert result["results"][0]["status"] == "success"
    assert mock_process.call_count == 2
    assert mock_remove.call_count == 2


def test_process_files_batch_logic_partial_failure():
    """Test batch processing when one file fails."""
    session_id = "test_session_123"
    filepaths = ["upload/file_success.pdf", "upload/file_fail.pdf"]

    def process_side_effect(sid, path):
        if "fail" in path:
            raise Exception("Processing failed")
        return None

    with (
        patch(
            "src.tasks.rag_service.process_file", side_effect=process_side_effect
        ) as mock_process,
        patch("os.path.exists", return_value=True),
        patch("os.remove") as mock_remove,
    ):
        result = process_files_batch_logic(session_id, filepaths, task_id="task-123")

    assert result["status"] == "batch_complete"
    assert result["results"][0]["status"] == "success"
    assert result["results"][1]["status"] == "error"
    assert result["results"][1]["message"] == "Processing failed"
    assert mock_process.call_count == 2
    assert mock_remove.call_count == 2
