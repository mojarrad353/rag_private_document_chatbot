"""
This module contains Celery tasks for background processing.
"""

import os
import structlog
from .celery_app import celery_app
from .rag import rag_service

logger = structlog.get_logger()


@celery_app.task(bind=True)
def process_file_task(self, session_id: str, filepath: str):
    """
    Background task to process an uploaded file.
    """
    logger.info("async_process_start", task_id=self.request.id, session_id=session_id)
    try:
        rag_service.process_file(session_id, filepath)
        # Cleanup file after processing
        if os.path.exists(filepath):
            os.remove(filepath)
        return {"status": "success", "session_id": session_id}
    except Exception as e:
        logger.exception("async_process_error", error=str(e), session_id=session_id)
        # Cleanup file on error too
        if os.path.exists(filepath):
            os.remove(filepath)
        raise e
