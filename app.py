import os
import shutil
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# ---------------------------------------------------------
# GLOBALS AND INITIALIZATIONS
# ---------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. Validation
openai_api_key = os.getenv("OPENAI_API_KEY", "")
openai_model = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

if not openai_api_key:
    print("WARNING: OPENAI_API_KEY not found in .env")

# 2. Initialize Models (Shared across all users)
# initialize these once to save resources, but the VectorStore will be unique per user.
embeddings = HuggingFaceEmbeddings()

llm = ChatOpenAI(
    model=openai_model,
    temperature=0.3,
    max_tokens=256
)

# 3. Custom Prompt
custom_template = """You are a helpful assistant designed to answer questions based solely on the provided documents.

Instructions:
1. Use ONLY the context provided below to answer the user's question.
2. If the answer is not present in the context, state clearly that you do not know based on the document.
3. Do not use outside knowledge, assumptions, or hallucinate information.
4. If the context is empty or irrelevant to the question, inform the user that the document does not contain the necessary information.

Context:
{context}

Question:
{question}

Answer:"""

QA_PROMPT = PromptTemplate(
    template=custom_template, 
    input_variables=["context", "question"]
)


user_sessions = {}

# ---------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------

@app.route('/')
def home():
    """Renders the chat interface."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles PDF upload and processes it for the specific session."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    session_id = request.form.get('session_id')

    if not session_id:
        return jsonify({"error": "Session ID missing"}), 400
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        try:
            # A. Save File Temporarily
            filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")
            file.save(filepath)
            
            # B. Load and Split PDF
            loader = PyPDFLoader(filepath)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            texts = text_splitter.split_documents(documents)
            
            # C. Create Vector Store (Specific to this user)
            # create a fresh Chroma instance for this user's data
            docsearch = Chroma.from_documents(texts, embeddings)
            
            # D. Initialize Session Data
            if session_id not in user_sessions:
                user_sessions[session_id] = {}

            # Store the user's specific vector store and reset memory
            user_sessions[session_id]["vector_store"] = docsearch
            user_sessions[session_id]["memory"] = ConversationBufferMemory(
                memory_key="chat_history", 
                return_messages=True
            )
            
            # Cleanup: Delete the temp file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({"message": "File processed successfully!"})
            
        except Exception as e:
            print(f"Error processing file: {e}")
            return jsonify({"error": "Failed to process file."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat logic using the user's specific PDF data."""
    data = request.json
    user_query = data.get('message')
    session_id = data.get('session_id')

    if not user_query or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400

    # Check if user has uploaded a file
    if session_id not in user_sessions or "vector_store" not in user_sessions[session_id]:
        return jsonify({"answer": "Please upload a PDF file first."})

    try:
        # Retrieve user-specific objects
        session_data = user_sessions[session_id]
        memory = session_data["memory"]
        vector_store = session_data["vector_store"]

        # Create Chain dynamically using user's vector store and memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            chain_type="stuff", 
            retriever=vector_store.as_retriever(), 
            memory=memory, 
            return_source_documents=False,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )
        
        # Invoke the chain
        result = qa_chain.invoke({"question": user_query})
        
        return jsonify({"answer": result["answer"]})

    except Exception as e:
        print(f"Error during chat: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

if __name__ == '__main__':
    print("Starting Flask Server...")
    app.run(debug=True, port=5000)