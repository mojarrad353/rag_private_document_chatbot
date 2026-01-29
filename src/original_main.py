#%%


import os
from langchain_community.document_loaders import PyPDFLoader # CHANGED: Import PDF Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter # CHANGED: Better for PDFs
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------
# 1. SETUP API KEY
# ---------------------------------------------------------
# OpenAI Config
openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
openai_api_key = os.getenv("OPENAI_API_KEY", "") 

assert openai_api_key, "Error: OPENAI_API_KEY cannot be empty. Update .env"

# ---------------------------------------------------------
# 2. LOAD LOCAL PDF
# ---------------------------------------------------------
filename = 'sample_restaurant_receipts.pdf'

# Verify file exists
if not os.path.isfile(filename):
    print(f"ERROR: The file '{filename}' was not found in the current directory.")
    # Stop execution if file is missing (in a script, you might return here)
else:
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
    print('Document ingested')

    # ---------------------------------------------------------
    # 3. DEFINE OPENAI MODEL (GPT-4o-mini)
    # ---------------------------------------------------------
    llm = ChatOpenAI(
        model=openai_model,
        temperature=0.1,
        max_tokens=256
    )

    # ---------------------------------------------------------
    # 4. BASIC RETRIEVAL QA
    # ---------------------------------------------------------
    qa_basic = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        return_source_documents=False
    )

    print("\n--- Basic Query ---")
    query = "What is the total amount of the receipt?" 
    print(qa_basic.invoke(query)['result'])

    print("\n--- Summarization Query ---")
    query = "List all the items ordered."
    print(qa_basic.invoke(query)['result'])

    # ---------------------------------------------------------
    # 5. CUSTOM PROMPT
    # ---------------------------------------------------------
    prompt_template = """Use the information from the document to answer the question at the end. 
    If you don't know the answer, just say that you don't know, definitely do not try to make up an answer.

    {context}

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}

    qa_custom = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        chain_type_kwargs=chain_type_kwargs, 
        return_source_documents=False
    )

    print("\n--- Custom Prompt Query ---")
    query = "What is the name of the restaurant?"
    print(qa_custom.invoke(query)['result'])

    # ---------------------------------------------------------
    # 6. CONVERSATIONAL CHAIN WITH BUILT-IN MEMORY
    # ---------------------------------------------------------
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

    qa_chat = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        chain_type="stuff", 
        retriever=docsearch.as_retriever(), 
        memory=memory, 
        return_source_documents=False
    )

    print("\n--- Chat Loop (Type 'quit' to exit) ---")

    while True:
        query = input("Question: ")
        
        if query.lower() in ["quit", "exit", "bye"]:
            print("Answer: Goodbye!")
            break
        
        result = qa_chat.invoke({"question": query})
        
        print("Answer: ", result["answer"])