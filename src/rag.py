"""
This module contains the RAG (Retrieval Augmented Generation) service.
It handles document loading, splitting, vector storage, and retrieval.
"""

import os
import structlog
from typing import Dict, Any, List
from prometheus_client import Counter

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import SecretStr

from .config import settings

logger = structlog.get_logger()

# Prometheus Metrics
rag_tokens_total = Counter("rag_tokens_total", "Total tokens used by RAG", ["type"])
rag_cost_total = Counter("rag_cost_total", "Total cost of RAG operations in USD")

# Custom Prompt
CUSTOM_TEMPLATE = """You are a helpful assistant designed to answer \
questions based solely on the provided documents.

Instructions:
1. Use ONLY the context provided below to answer the user's question.
2. If the answer is not present in the context, state clearly that you \
do not know based on the document.
3. Do not use outside knowledge, assumptions, or hallucinate information.
4. If the context is empty or irrelevant to the question, inform the user \
that the document does not contain the necessary information.

Context:
{context}

Question:
{question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=CUSTOM_TEMPLATE, input_variables=["context", "question"]
)


class RAGService:
    """
    Service class for RAG operations.
    Manages user sessions, document processing, and query retrieval.
    """

    def __init__(self) -> None:
        """Initialize the RAG service."""
        self._embeddings = None
        self._llm = None

    @property
    def embeddings(self):
        """Lazy initialization of embeddings."""
        if self._embeddings is None:
            # pylint: disable=import-outside-toplevel
            from langchain_huggingface import HuggingFaceEmbeddings

            self._embeddings = HuggingFaceEmbeddings()
        return self._embeddings

    @property
    def llm(self):
        """Lazy initialization of LLM."""
        if self._llm is None:
            # pylint: disable=import-outside-toplevel
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_NAME,
                temperature=1,
                max_completion_tokens=256,  # type: ignore[call-arg]
                api_key=SecretStr(settings.OPENAI_API_KEY),
            )
        return self._llm

    def process_file(self, session_id: str, filepath: str) -> None:
        """
        Loads a PDF, splits it, and creates a vector store for the session.

        Args:
            session_id (str): The unique session identifier.
            filepath (str): Path to the uploaded PDF file.
        """
        logger.info("process_file_start", session_id=session_id, filepath=filepath)

        # 1. Load PDF
        loader = PyPDFLoader(filepath)
        documents = loader.load()
        logger.debug("pdf_loaded", page_count=len(documents), session_id=session_id)

        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)
        logger.debug("text_split", chunk_count=len(texts), session_id=session_id)

        if not texts:
            logger.error("no_text_found", session_id=session_id)
            raise ValueError("No text found in document.")

        # 3. Create Vector Store (Persisted to disk per session)
        persist_directory = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, session_id)
        Chroma.from_documents(
            texts,
            self.embeddings,
            persist_directory=persist_directory,
        )

        logger.info("process_file_complete", session_id=session_id)

    def get_answer(self, session_id: str, query: str) -> str:
        """
        Generates an answer for a given session and query.

        Args:
            session_id (str): The unique session identifier.
            query (str): The user's question.

        Returns:
            str: The generated answer from the LLM.
        """
        # Check if vector store exists on disk
        persist_directory = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, session_id)
        if not os.path.exists(persist_directory):
            logger.warning("get_answer_no_session_dir", session_id=session_id)
            return "Please upload a PDF file first."

        # Load Vector Store
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        # Initialize Redis-backed Memory
        message_history = RedisChatMessageHistory(
            url=settings.REDIS_URL, ttl=3600, session_id=session_id
        )
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            chat_memory=message_history,
            return_messages=True,
            output_key="answer",
        )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        )

        logger.info("invoke_chain_start", session_id=session_id)

        with get_openai_callback() as cb:
            result = qa_chain.invoke({"question": query})

            # Record Token Metrics
            rag_tokens_total.labels(type="prompt").inc(cb.prompt_tokens)
            rag_tokens_total.labels(type="completion").inc(cb.completion_tokens)
            rag_cost_total.inc(cb.total_cost)

            logger.info(
                "invoke_chain_complete",
                session_id=session_id,
                total_tokens=cb.total_tokens,
                prompt_tokens=cb.prompt_tokens,
                completion_tokens=cb.completion_tokens,
                total_cost=cb.total_cost,
            )

        # Log Source Documents
        if "source_documents" in result:
            source_docs = result["source_documents"]
            logger.info(
                "retrieved_documents",
                session_id=session_id,
                doc_count=len(source_docs),
                sources=[
                    str(doc.metadata.get("source")) for doc in source_docs
                ],  # type: ignore
            )

        return str(result["answer"])

    def clear_session(self, session_id: str):
        """Clears the session data for a given session ID."""
        # Cleanup vector store directory
        import shutil

        persist_directory = os.path.join(settings.CHROMA_PERSIST_DIRECTORY, session_id)
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)

        # Cleanup Redis history
        message_history = RedisChatMessageHistory(
            url=settings.REDIS_URL, session_id=session_id
        )
        message_history.clear()

        logger.info("session_cleared", session_id=session_id)


rag_service = RAGService()
