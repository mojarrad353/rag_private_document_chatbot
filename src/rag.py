"""
This module contains the RAG (Retrieval Augmented Generation) service.
It handles document loading, splitting, vector storage, and retrieval.
"""

from typing import Dict, Any

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# pylint: disable=no-name-in-module
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import SecretStr

from .config import settings

# Initialize Embeddings once
embeddings = HuggingFaceEmbeddings()

# Initialize LLM once
llm = ChatOpenAI(
    model=settings.OPENAI_MODEL_NAME,
    temperature=0.3,  # Slightly creative but focused
    max_tokens=256,
    api_key=SecretStr(settings.OPENAI_API_KEY),
)

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

    def __init__(self):
        # In-memory session storage (simple map for now)
        # Session ID -> {'vector_store': Chroma, 'memory': Memory}
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def process_file(self, session_id: str, filepath: str) -> None:
        """
        Loads a PDF, splits it, and creates a vector store for the session.

        Args:
            session_id (str): The unique session identifier.
            filepath (str): Path to the uploaded PDF file.
        """

        # 1. Load PDF
        loader = PyPDFLoader(filepath)
        documents = loader.load()

        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100
        )
        texts = text_splitter.split_documents(documents)

        if not texts:
            raise ValueError("No text found in document.")

        # 3. Create Vector Store (Ephemeral for now, or per session)
        # We use a unique collection name per session or just separate instances
        docsearch = Chroma.from_documents(
            texts,
            embeddings,
        )

        # 4. Initialize Memory
        if session_id not in self.sessions:
            self.sessions[session_id] = {}

        self.sessions[session_id]["vector_store"] = docsearch
        self.sessions[session_id]["memory"] = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",  # Important for some chains
        )

    def get_answer(self, session_id: str, query: str) -> str:
        """
        Generates an answer for a given session and query.

        Args:
            session_id (str): The unique session identifier.
            query (str): The user's question.

        Returns:
            str: The generated answer from the LLM.
        """
        if (
            session_id not in self.sessions
            or "vector_store" not in self.sessions[session_id]
        ):
            return "Please upload a PDF file first."

        session_data = self.sessions[session_id]
        memory = session_data["memory"]
        vector_store = session_data["vector_store"]

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            memory=memory,
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        )

        result = qa_chain.invoke({"question": query})
        return str(result["answer"])

    def clear_session(self, session_id: str):
        """Clears the session data for a given session ID."""
        if session_id in self.sessions:
            del self.sessions[session_id]


rag_service = RAGService()
