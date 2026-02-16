"""
This module configures the Celery application.
"""

from celery import Celery
from .config import settings


def make_celery(app_name: str) -> Celery:
    """
    Creates and configures a Celery application instance.
    """
    celery = Celery(
        app_name,
        backend=settings.REDIS_URL,
        broker=settings.REDIS_URL,
        include=["src.tasks"],
    )
    celery.conf.update(
        result_expires=3600,
    )
    return celery


celery_app = make_celery("rag_chatbot")
