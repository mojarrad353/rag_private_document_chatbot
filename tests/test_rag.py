import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from src.rag import RAGService, AgentState, RetrieveState
import os
from langchain_core.documents import Document


@pytest.fixture
def rag_service():
    # Patch the lazy properties to return mocks and avoid actual imports/init
    with (
        patch("src.rag.RAGService.embeddings", new_callable=PropertyMock) as mock_emb,
        patch("src.rag.RAGService.llm", new_callable=PropertyMock) as mock_llm,
        patch(
            "src.rag.RAGService.redis_client", new_callable=PropertyMock
        ) as mock_redis,
    ):
        service = RAGService()
        mock_emb.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_redis.return_value = MagicMock()
        yield service


def test_lazy_properties():
    """Test lazy initialization of complex properties."""
    service = RAGService()

    with (
        patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_hf,
        patch("langchain_openai.ChatOpenAI") as mock_openai,
        patch("redis.from_url") as mock_redis_from,
    ):
        # Trigger lazy loads
        _ = service.embeddings
        _ = service.llm
        _ = service.redis_client

        mock_hf.assert_called_once()
        mock_openai.assert_called_once()
        mock_redis_from.assert_called_once()


@patch("src.rag.PyMuPDFLoader")
@patch("src.rag.RecursiveCharacterTextSplitter")
@patch("src.rag.Chroma")
def test_process_file(mock_chroma, mock_splitter, mock_loader, rag_service):
    """Test processing a PDF file."""
    mock_loader_instance = mock_loader.return_value
    mock_loader_instance.load.return_value = [MagicMock()]
    mock_splitter_instance = mock_splitter.return_value
    mock_splitter_instance.split_documents.return_value = [MagicMock()]

    with patch("os.path.exists", return_value=False):
        rag_service.process_file("session_1", "dummy.pdf")

    mock_chroma.from_documents.assert_called_once()
    args, kwargs = mock_chroma.from_documents.call_args
    assert "session_1" in kwargs["persist_directory"]


def test_nodes_logic(rag_service):
    """Test the internal logic of individual LangGraph nodes."""
    # 1. Test _get_session_files_node
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "metadatas": [{"source": "file1.pdf"}, {"source": "file2.pdf"}]
    }
    state: AgentState = {
        "question": "q",
        "chat_history": [],
        "documents": [],
        "generation": "",
        "files": [],
    }

    res = rag_service._get_session_files_node(state, mock_vs)
    assert len(res["files"]) == 2
    assert "file1.pdf" in res["files"]

    # 2. Test _retrieve_file_node
    retrieve_state: RetrieveState = {"question": "q", "file_source": "file1.pdf"}
    mock_vs.as_retriever.return_value.invoke.return_value = [
        Document(page_content="hit", metadata={"source": "file1.pdf"})
    ]

    res = rag_service._retrieve_file_node(retrieve_state, mock_vs)
    assert len(res["documents"]) == 1
    assert res["documents"][0].page_content == "hit"

    # 3. Test _generate_node (Success)
    state["documents"] = [
        Document(
            page_content="answer here", metadata={"source": "file1.pdf", "page": 0}
        )
    ]
    with patch("src.rag.get_openai_callback") as mock_cb:
        mock_cb_inst = mock_cb.return_value.__enter__.return_value
        mock_cb_inst.prompt_tokens = 1
        mock_cb_inst.completion_tokens = 1
        mock_cb_inst.total_cost = 0.0

        # Patch the base prompt's pipe logic to avoid real LangChain piping in unit tests
        with patch("src.rag.QA_PROMPT", new=MagicMock()) as mock_prompt:
            mock_chain = MagicMock()
            mock_chain.invoke.return_value = "Generated response"
            # Mock the chain created by prompts | llm | parser
            mock_prompt.__or__.return_value.__or__.return_value = mock_chain

            res = rag_service._generate_node(state)
            assert res["generation"] == "Generated response"


@patch("src.rag.StateGraph")
@patch("src.rag.Chroma")
@patch("src.rag.RedisChatMessageHistory")
@patch("src.rag.get_openai_callback")
def test_get_answer_workflow(mock_cb, mock_redis, mock_chroma, mock_graph, rag_service):
    """Test the end-to-end get_answer workflow orchestration."""
    with patch("os.path.exists", return_value=True):
        mock_compiled = mock_graph.return_value.compile.return_value
        mock_compiled.invoke.return_value = {"generation": "Workflow answer"}

        answer = rag_service.get_answer("session_1", "Question")
        assert answer == "Workflow answer"


def test_get_answer_no_session(rag_service):
    """Test getting answer without session directory existing."""
    with patch("os.path.exists", return_value=False):
        answer = rag_service.get_answer("unknown_session", "Question")
        assert "Please upload a PDF file first" in answer


@patch("shutil.rmtree")
@patch("src.rag.RedisChatMessageHistory")
def test_clear_session(mock_redis, mock_rmtree, rag_service):
    """Test clearing a session."""
    with patch("os.path.exists", return_value=True):
        rag_service.clear_session("session_to_clear")
        mock_rmtree.assert_called_once()
        mock_redis.return_value.clear.assert_called_once()
