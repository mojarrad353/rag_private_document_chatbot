"""
Unit tests for src/rag.py — RAGService core logic.

All external dependencies (Chroma, Redis, HuggingFace, OpenAI, LangGraph)
are mocked so that these tests run fast without any network or GPU access.
"""

import os
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
from langchain_core.documents import Document

from src.rag import RAGService, AgentState, RetrieveState

# ---------------------------------------------------------------------------
# Shared fixture: a RAGService instance with lazy properties pre-mocked
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """Return a RAGService whose lazy properties are replaced with MagicMocks."""
    with (
        patch("src.rag.RAGService.embeddings", new_callable=PropertyMock) as mock_emb,
        patch("src.rag.RAGService.llm", new_callable=PropertyMock) as mock_llm,
        patch(
            "src.rag.RAGService.redis_client", new_callable=PropertyMock
        ) as mock_redis,
    ):
        mock_emb.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_redis.return_value = MagicMock()
        svc = RAGService()
        yield svc


# ---------------------------------------------------------------------------
# Lazy property initialisation
# ---------------------------------------------------------------------------


def test_lazy_redis_client():
    """Redis client is created lazily on first access."""
    svc = RAGService()
    with patch("redis.from_url") as mock_redis_from_url:
        mock_redis_from_url.return_value = MagicMock()
        _ = svc.redis_client
        mock_redis_from_url.assert_called_once()
        # Second access should NOT call from_url again
        _ = svc.redis_client
        mock_redis_from_url.assert_called_once()


def test_lazy_embeddings():
    """HuggingFaceEmbeddings is initialised lazily."""
    svc = RAGService()
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_hf:
        mock_hf.return_value = MagicMock()
        _ = svc.embeddings
        mock_hf.assert_called_once()
        _ = svc.embeddings
        mock_hf.assert_called_once()


def test_lazy_llm():
    """ChatOpenAI is initialised lazily."""
    svc = RAGService()
    with patch("langchain_openai.ChatOpenAI") as mock_openai:
        mock_openai.return_value = MagicMock()
        _ = svc.llm
        mock_openai.assert_called_once()
        _ = svc.llm
        mock_openai.assert_called_once()


# ---------------------------------------------------------------------------
# sanitize_query
# ---------------------------------------------------------------------------


def test_sanitize_query_truncates():
    """Queries longer than MAX_QUERY_LENGTH are truncated."""
    long_query = "a" * 3000
    result = RAGService.sanitize_query(long_query)
    assert len(result) == 2000


def test_sanitize_query_strips_control_chars():
    """Control characters are stripped from the query."""
    query = "hello\x00world\x01test\x07"
    result = RAGService.sanitize_query(query)
    assert "\x00" not in result
    assert "\x01" not in result
    assert "\x07" not in result
    assert "hello" in result
    assert "world" in result


def test_sanitize_query_strips_whitespace():
    """Leading/trailing whitespace is stripped."""
    result = RAGService.sanitize_query("  hello  ")
    assert result == "hello"


def test_sanitize_query_preserves_newlines():
    """Newlines are preserved (they aid readability)."""
    result = RAGService.sanitize_query("line1\nline2")
    assert "\n" in result


# ---------------------------------------------------------------------------
# sanitize_output
# ---------------------------------------------------------------------------


def test_sanitize_output_clean(service):
    """Clean output is passed through unchanged."""
    result = service.sanitize_output("This is a normal answer.")
    assert result == "This is a normal answer."


def test_sanitize_output_blocks_url(service):
    """Output containing a URL is replaced with a safe message."""
    result = service.sanitize_output("See https://evil.com for details.")
    assert "safe answer" in result


def test_sanitize_output_blocks_code_fence(service):
    """Output containing code fences is rejected."""
    result = service.sanitize_output("Here is code: ```python print('x')```")
    assert "safe answer" in result


def test_sanitize_output_blocks_system_prompt_leak(service):
    """Output echoing ABSOLUTE RULES is rejected."""
    result = service.sanitize_output("ABSOLUTE RULES are: ...")
    assert "safe answer" in result


def test_sanitize_output_blocks_context_delimiter(service):
    """Output containing <context> tags is rejected."""
    result = service.sanitize_output("Here is <context>the document</context>.")
    assert "safe answer" in result


def test_sanitize_output_blocks_untrusted_leak(service):
    """Output containing 'UNTRUSTED document text' is rejected."""
    result = service.sanitize_output("Remember this is UNTRUSTED document text.")
    assert "safe answer" in result


# ---------------------------------------------------------------------------
# _get_session_files_node
# ---------------------------------------------------------------------------


def test_get_session_files_node_returns_unique_files(service):
    """Node returns deduplicated list of files from vector store metadata."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "metadatas": [
            {"source": "file_a.pdf"},
            {"source": "file_b.pdf"},
            {"source": "file_a.pdf"},  # duplicate
        ]
    }
    state: AgentState = {
        "question": "What is revenue?",
        "chat_history": [],
        "documents": [],
        "generation": "",
        "files": [],
    }
    result = service._get_session_files_node(state, mock_vs)
    assert set(result["files"]) == {"file_a.pdf", "file_b.pdf"}


def test_get_session_files_node_empty_collection(service):
    """Node returns empty list when vector store has no metadata."""
    mock_vs = MagicMock()
    mock_vs.get.return_value = {"metadatas": []}
    state: AgentState = {
        "question": "test",
        "chat_history": [],
        "documents": [],
        "generation": "",
        "files": [],
    }
    result = service._get_session_files_node(state, mock_vs)
    assert result["files"] == []


# ---------------------------------------------------------------------------
# _retrieve_file_node
# ---------------------------------------------------------------------------


def test_retrieve_file_node_returns_documents(service):
    """Node returns documents from the cross-encoder pipeline."""
    retrieved_docs = [
        Document(page_content="Revenue was $10B", metadata={"source": "q3.pdf"})
    ]

    mock_retriever = MagicMock()
    state: RetrieveState = {"question": "What is revenue?", "file_source": "q3.pdf"}

    with (
        patch("langchain_community.cross_encoders.HuggingFaceCrossEncoder") as mock_ce,
        patch(
            "langchain.retrievers.document_compressors.CrossEncoderReranker"
        ) as mock_reranker_cls,
        patch("langchain.retrievers.ContextualCompressionRetriever") as mock_ccr_cls,
    ):
        mock_ccr = MagicMock()
        mock_ccr.invoke.return_value = retrieved_docs
        mock_ccr_cls.return_value = mock_ccr
        mock_reranker_cls.return_value = MagicMock()
        mock_ce.return_value = MagicMock()

        result = service._retrieve_file_node(state, mock_retriever)

    assert result["documents"] == retrieved_docs
    assert mock_retriever.search_kwargs["filter"]["source"] == "q3.pdf"


# ---------------------------------------------------------------------------
# _generate_node
# ---------------------------------------------------------------------------


def test_generate_node_no_documents(service):
    """Node returns fallback message when no documents are retrieved."""
    state: AgentState = {
        "question": "What is revenue?",
        "chat_history": [],
        "documents": [],
        "generation": "",
        "files": [],
    }
    result = service._generate_node(state)
    assert "no such information" in result["generation"].lower()


def test_generate_node_with_documents(service):
    """Node calls the LLM chain and returns the generated answer."""
    docs = [
        Document(
            page_content="Revenue was $10B.",
            metadata={"source": "/data/q3.pdf", "page": 0},
        )
    ]
    state: AgentState = {
        "question": "What is revenue?",
        "chat_history": [],
        "documents": docs,
        "generation": "",
        "files": [],
    }

    with patch("src.rag.get_openai_callback") as mock_cb_ctx:
        mock_cb = MagicMock()
        mock_cb.prompt_tokens = 50
        mock_cb.completion_tokens = 20
        mock_cb.total_tokens = 70
        mock_cb.total_cost = 0.001
        mock_cb_ctx.return_value.__enter__.return_value = mock_cb

        with patch("src.rag.QA_PROMPT") as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Revenue was $10B."
            mock_prompt.__or__ = MagicMock(return_value=mock_prompt)
            mock_prompt.__or__.return_value.__or__ = MagicMock(return_value=mock_chain)

            result = service._generate_node(state)

    assert "generation" in result


def test_generate_node_empty_llm_response(service):
    """Node returns fallback message when LLM returns an empty string."""
    docs = [
        Document(
            page_content="Some content.", metadata={"source": "/data/q3.pdf", "page": 0}
        )
    ]
    state: AgentState = {
        "question": "test?",
        "chat_history": [],
        "documents": docs,
        "generation": "",
        "files": [],
    }

    with patch("src.rag.get_openai_callback") as mock_cb_ctx:
        mock_cb = MagicMock()
        mock_cb.prompt_tokens = 10
        mock_cb.completion_tokens = 0
        mock_cb.total_tokens = 10
        mock_cb.total_cost = 0.0
        mock_cb_ctx.return_value.__enter__.return_value = mock_cb

        # Patch the full chain object returned by QA_PROMPT | llm | parser
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = ""  # empty response
        with patch("src.rag.QA_PROMPT") as mock_prompt:
            # Make every __or__ call return the same mock_chain
            mock_prompt.__or__ = lambda self, other: mock_chain
            mock_chain.__or__ = lambda self, other: mock_chain

            result = service._generate_node(state)

    assert "couldn't generate" in result["generation"].lower()


def test_generate_node_exception(service):
    """Node catches exceptions and returns an error message."""
    docs = [Document(page_content="content", metadata={"source": "f.pdf", "page": 0})]
    state: AgentState = {
        "question": "test?",
        "chat_history": [],
        "documents": docs,
        "generation": "",
        "files": [],
    }

    with patch("src.rag.get_openai_callback", side_effect=RuntimeError("API down")):
        result = service._generate_node(state)

    assert "error occurred" in result["generation"].lower()


# ---------------------------------------------------------------------------
# process_file
# ---------------------------------------------------------------------------


@patch("src.rag.PyMuPDFLoader")
@patch("src.rag.Chroma")
@patch("src.rag.LocalFileStore")
@patch("src.rag.create_kv_docstore")
@patch("src.rag.ParentDocumentRetriever")
def test_process_file_success(
    mock_pdr_cls, mock_kv, mock_fs, mock_chroma, mock_loader, service
):
    """process_file loads PDF and calls add_documents under a Redis lock."""
    mock_loader_inst = mock_loader.return_value
    mock_loader_inst.load.return_value = [
        Document(page_content="page 1", metadata={"source": "test.pdf"})
    ]

    mock_pdr_inst = MagicMock()
    mock_pdr_cls.return_value = mock_pdr_inst

    mock_lock = MagicMock()
    mock_lock.__enter__ = MagicMock(return_value=None)
    mock_lock.__exit__ = MagicMock(return_value=False)
    service.redis_client.lock.return_value = mock_lock

    service.process_file("session_abc", "test.pdf")

    mock_loader.assert_called_once_with("test.pdf")
    mock_pdr_inst.add_documents.assert_called_once()


@patch("src.rag.PyMuPDFLoader")
def test_process_file_empty_document_raises(mock_loader, service):
    """process_file raises ValueError when the PDF has no pages."""
    mock_loader.return_value.load.return_value = []
    with pytest.raises(ValueError, match="empty"):
        service.process_file("session_abc", "empty.pdf")


# ---------------------------------------------------------------------------
# get_answer
# ---------------------------------------------------------------------------


def test_get_answer_no_session_dir(service):
    """get_answer returns early when no vector store directory exists."""
    with patch("os.path.exists", return_value=False):
        result = service.get_answer("no_session", "question?")
    assert "upload" in result.lower()


@patch("src.rag.StateGraph")
@patch("src.rag.Chroma")
@patch("src.rag.LocalFileStore")
@patch("src.rag.create_kv_docstore")
@patch("src.rag.ParentDocumentRetriever")
@patch("src.rag.RedisChatMessageHistory")
def test_get_answer_full_workflow(
    mock_history, mock_pdr_cls, mock_kv, mock_fs, mock_chroma, mock_graph_cls, service
):
    """get_answer compiles and invokes the LangGraph workflow."""
    mock_history.return_value.messages = []

    mock_compiled = MagicMock()
    mock_compiled.invoke.return_value = {"generation": "Revenue was $10B."}
    mock_graph_cls.return_value.compile.return_value = mock_compiled

    mock_pdr_cls.return_value = MagicMock()

    with patch("os.path.exists", return_value=True):
        result = service.get_answer("session_xyz", "What is revenue?")

    assert result == "Revenue was $10B."
    mock_compiled.invoke.assert_called_once()
    mock_history.return_value.add_user_message.assert_called_once_with(
        "What is revenue?"
    )
    mock_history.return_value.add_ai_message.assert_called_once_with(
        "Revenue was $10B."
    )


# ---------------------------------------------------------------------------
# clear_session
# ---------------------------------------------------------------------------


@patch("shutil.rmtree")
@patch("src.rag.RedisChatMessageHistory")
def test_clear_session_removes_directory(mock_history, mock_rmtree, service):
    """clear_session removes the Chroma directory and clears Redis history."""
    with patch("os.path.exists", return_value=True):
        service.clear_session("session_to_clear")

    mock_rmtree.assert_called_once()
    rmtree_path = mock_rmtree.call_args[0][0]
    assert "session_to_clear" in rmtree_path
    mock_history.return_value.clear.assert_called_once()


@patch("shutil.rmtree")
@patch("src.rag.RedisChatMessageHistory")
def test_clear_session_no_directory(mock_history, mock_rmtree, service):
    """clear_session skips rmtree when directory does not exist."""
    with patch("os.path.exists", return_value=False):
        service.clear_session("ghost_session")

    mock_rmtree.assert_not_called()
    mock_history.return_value.clear.assert_called_once()
