"""
This module contains Celery tasks for background processing.
"""

import os
from typing import List

import structlog

from .celery_app import celery_app
from .rag import rag_service

logger = structlog.get_logger()


def process_files_batch_logic(
    session_id: str, filepaths: List[str], task_id: str | None = None
):
    """
    Business logic for processing multiple uploaded files.
    Separated from the Celery task for easier unit testing.
    """
    logger.info(
        "async_batch_process_start",
        task_id=task_id,
        session_id=session_id,
        count=len(filepaths),
    )
    results = []
    for filepath in filepaths:
        try:
            rag_service.process_file(session_id, filepath)
            # Cleanup file after processing
            if os.path.exists(filepath):
                os.remove(filepath)
            results.append({"file": filepath, "status": "success"})
        except Exception as e:
            logger.exception(
                "async_batch_process_error",
                error=str(e),
                session_id=session_id,
                file=filepath,
            )
            if os.path.exists(filepath):
                os.remove(filepath)
            results.append({"file": filepath, "status": "error", "message": str(e)})

    return {"status": "batch_complete", "session_id": session_id, "results": results}


@celery_app.task(bind=True)
def process_files_batch_task(self, session_id: str, filepaths: List[str]):
    """
    Background task wrapper for batch processing.
    """
    return process_files_batch_logic(session_id, filepaths, task_id=self.request.id)
