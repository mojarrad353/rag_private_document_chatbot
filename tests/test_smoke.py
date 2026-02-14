import sys
import os

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import settings


def test_config_defaults():
    """Test that default settings are loaded correctly."""
    assert settings.OPENAI_MODEL_NAME == "gpt-4o-mini"
    assert settings.UPLOAD_FOLDER == "uploads"
