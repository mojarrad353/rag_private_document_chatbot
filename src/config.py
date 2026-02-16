"""
This module contains the configuration settings for the application.
"""

import os
from typing import Any
from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    OPENAI_API_KEY: str
    OPENAI_MODEL_NAME: str = "gpt-5-mini"
    UPLOAD_FOLDER: str = "uploads"
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"

    # LangSmith Configuration
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str | None = None
    LANGCHAIN_PROJECT: str = "rag-private-document-chatbot"

    # Redis / Celery Configuration
    REDIS_URL: str = "redis://redis:6379/0"

    @classmethod
    @field_validator("LANGCHAIN_TRACING_V2", mode="before")
    def empty_str_to_bool_false(cls, v: Any) -> Any:
        """Converts empty strings to False for boolean fields."""
        if v == "":
            return False
        return v


settings = Settings()  # type: ignore[call-arg]

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_FOLDER, exist_ok=True)
