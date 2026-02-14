"""
This module contains the original main script for testing the RAG pipeline locally.
"""

import os

# from dotenv import load_dotenv # Assuming loaded via settings or handled here

# pylint: disable=no-name-in-module
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    """
    Main function to run the RAG pipeline.
    """
    # pylint: disable=too-many-locals
    # ---------------------------------------------------------
    # 1. SETUP API KEY
    # ---------------------------------------------------------
    # OpenAI Config
    openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    openai_api_key = os.getenv("OPENAI_API_KEY", "")

    if not openai_api_key:
        print("Error: OPENAI_API_KEY cannot be empty. Update .env")
        return

    # ---------------------------------------------------------
    # 2. LOAD LOCAL PDF
    # ---------------------------------------------------------
    filename = "sample_restaurant_receipts.pdf"

    # Verify file exists
    if not os.path.isfile(filename):
        print(f"ERROR: The file '{filename}' was not found in the current directory.")
        return

    print(f"Loading file: {filename}")

    # Use PyPDFLoader for PDF file loading
    loader = PyPDFLoader(filename)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    print(f"Documents split into {len(texts)} chunks")

    # Embedding
    embeddings = HuggingFaceEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings)
    print("Document ingested")

    # ---------------------------------------------------------
    # 3. DEFINE OPENAI MODEL (GPT-4o-mini)
    # ---------------------------------------------------------
    llm = ChatOpenAI(model=openai_model, temperature=0.1, max_tokens=256)

    # ---------------------------------------------------------
    # 4. BASIC RETRIEVAL QA
    # ---------------------------------------------------------
    qa_basic = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=False,
    )

    print("\n--- Basic Query ---")
    query_basic = "What is the total amount of the receipt?"
    print(qa_basic.invoke(query_basic)["result"])

    print("\n--- Summarization Query ---")
    query_summary = "List all the items ordered."
    print(qa_basic.invoke(query_summary)["result"])

    # ---------------------------------------------------------
    # 5. CUSTOM PROMPT
    # ---------------------------------------------------------
    custom_prompt_template = """Use the information from the document to answer \
the question at the end.
    If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.

    {context}

    Question: {question}
    """

    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": prompt}

    qa_custom = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=False,
    )

    print("\n--- Custom Prompt Query ---")
    query_custom = "What is the name of the restaurant?"
    print(qa_custom.invoke(query_custom)["result"])

    # ---------------------------------------------------------
    # 6. CONVERSATIONAL CHAIN WITH BUILT-IN MEMORY
    # ---------------------------------------------------------
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chat = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=False,
    )

    print("\n--- Chat Loop (Type 'quit' to exit) ---")

    while True:
        user_input = input("Question: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break

        result = qa_chat.invoke({"question": user_input})

        print("Answer: ", result["answer"])


if __name__ == "__main__":
    main()
