import pytest
from unittest.mock import MagicMock, patch
from src.rag import RAGService


@pytest.fixture
def rag_service():
    return RAGService()


@patch("src.rag.PyPDFLoader")
@patch("src.rag.RecursiveCharacterTextSplitter")
@patch("src.rag.Chroma")
@patch("src.rag.ConversationBufferMemory")
def test_process_file(
    mock_memory, mock_chroma, mock_splitter, mock_loader, rag_service
):
    """Test processing a PDF file."""
    # Setup mocks
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load.return_value = ["doc1"]

    mock_splitter_instance = mock_splitter.return_value
    mock_splitter_instance.split_documents.return_value = ["text1"]

    rag_service.process_file("session_1", "dummy.pdf")

    assert "session_1" in rag_service.sessions
    assert "vector_store" in rag_service.sessions["session_1"]
    assert "memory" in rag_service.sessions["session_1"]
    mock_chroma.from_documents.assert_called_once()


@patch("src.rag.ConversationalRetrievalChain")
def test_get_answer_success(mock_chain, rag_service):
    """Test getting an answer successfully."""
    # Setup session
    rag_service.sessions["session_1"] = {
        "vector_store": MagicMock(),
        "memory": MagicMock(),
    }

    mock_chain_instance = mock_chain.from_llm.return_value
    mock_chain_instance.invoke.return_value = {"answer": "The answer"}

    answer = rag_service.get_answer("session_1", "Question")

    assert answer == "The answer"


def test_get_answer_no_session(rag_service):
    """Test getting answer without session setup."""
    answer = rag_service.get_answer("unknown_session", "Question")
    assert "Please upload a PDF file first" in answer


@patch("src.rag.PyPDFLoader")
@patch("src.rag.RecursiveCharacterTextSplitter")
def test_process_file_no_text(mock_splitter, mock_loader, rag_service):
    """Test process_file raises ValueError when no text found."""
    mock_loader.return_value.load.return_value = ["doc1"]
    mock_splitter.return_value.split_documents.return_value = []

    with pytest.raises(ValueError, match="No text found in document"):
        rag_service.process_file("session_error", "empty.pdf")


def test_clear_session(rag_service):
    """Test clearing a session."""
    rag_service.sessions["session_to_clear"] = {"data": "test"}
    rag_service.clear_session("session_to_clear")
    assert "session_to_clear" not in rag_service.sessions


def test_clear_session_non_existent(rag_service):
    """Test clearing a non-existent session."""
    rag_service.clear_session("non_existent")
    # Should not raise error
