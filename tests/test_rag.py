import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.rag import RAGService
import os


@pytest.fixture
def rag_service():
    # Patch the lazy properties to return mocks and avoid actual imports/init
    with (
        patch("src.rag.RAGService.embeddings", new_callable=PropertyMock) as mock_emb,
        patch("src.rag.RAGService.llm", new_callable=PropertyMock) as mock_llm,
    ):
        service = RAGService()
        mock_emb.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        yield service


@patch("src.rag.PyPDFLoader")
@patch("src.rag.RecursiveCharacterTextSplitter")
@patch("src.rag.Chroma")
def test_process_file(mock_chroma, mock_splitter, mock_loader, rag_service):
    """Test processing a PDF file."""
    # Setup mocks
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load.return_value = [MagicMock()]

    mock_splitter_instance = mock_splitter.return_value
    mock_splitter_instance.split_documents.return_value = [MagicMock()]

    with patch("os.path.exists", return_value=False):
        rag_service.process_file("session_1", "dummy.pdf")

    mock_chroma.from_documents.assert_called_once()
    # Verify it uses a persist_directory
    args, kwargs = mock_chroma.from_documents.call_args
    assert "persist_directory" in kwargs
    assert "session_1" in kwargs["persist_directory"]


@patch("src.rag.ConversationalRetrievalChain")
@patch("src.rag.Chroma")
@patch("src.rag.RedisChatMessageHistory")
@patch("src.rag.ConversationBufferMemory")
@patch("src.rag.get_openai_callback")
def test_get_answer_success(
    mock_cb, mock_memory, mock_redis, mock_chroma, mock_chain, rag_service
):
    """Test getting an answer successfully."""
    with patch("os.path.exists", return_value=True):
        # Mock Context Manager
        mock_cb_instance = mock_cb.return_value.__enter__.return_value
        mock_cb_instance.prompt_tokens = 10
        mock_cb_instance.completion_tokens = 20
        mock_cb_instance.total_tokens = 30
        mock_cb_instance.total_cost = 0.01

        mock_chain_instance = mock_chain.from_llm.return_value
        mock_chain_instance.invoke.return_value = {
            "answer": "The answer",
            "source_documents": [MagicMock(metadata={"source": "doc.pdf"})],
        }

        answer = rag_service.get_answer("session_1", "Question")

        assert answer == "The answer"
        mock_chroma.assert_called_once()
        mock_redis.assert_called_once()
        mock_chain_instance.invoke.assert_called_once()


def test_get_answer_no_session(rag_service):
    """Test getting answer without session directory existing."""
    with patch("os.path.exists", return_value=False):
        answer = rag_service.get_answer("unknown_session", "Question")
        assert "Please upload a PDF file first" in answer


@patch("src.rag.PyPDFLoader")
@patch("src.rag.RecursiveCharacterTextSplitter")
def test_process_file_no_text(mock_splitter, mock_loader, rag_service):
    """Test process_file raises ValueError when no text found."""
    mock_loader.return_value.load.return_value = [MagicMock()]
    mock_splitter.return_value.split_documents.return_value = []

    with pytest.raises(ValueError, match="No text found in document"):
        rag_service.process_file("session_error", "empty.pdf")


@patch("shutil.rmtree")
@patch("src.rag.RedisChatMessageHistory")
def test_clear_session(mock_redis, mock_rmtree, rag_service):
    """Test clearing a session."""
    with patch("os.path.exists", return_value=True):
        rag_service.clear_session("session_to_clear")
        mock_rmtree.assert_called_once()
        mock_redis.return_value.clear.assert_called_once()
