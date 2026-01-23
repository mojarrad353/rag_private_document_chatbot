# 📄 RAG Private Document Chatbot

![Python](https://img.shields.io/badge/Python-3.x-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-orange)
![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey)

---

## 📌 Project Overview

The **RAG Private Document Chatbot** is a **Python-based** RAG with a web application that enables users to upload a PDF document and interact with it via a **conversational interface**.

The project demonstrates:
- Building a **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**.
- Processing and embedding PDF documents for semantic search using **ChromaDB**.
- Implementing **Conversational Memory** to handle follow-up questions effectively.
- Using **OpenAI's GPT-4o-mini** with strict custom prompting to prevent hallucinations.
- Providing a responsive Web UI using **Flask**.

---

## 🚀 Features

- 📂 **File Upload:** Upload and analyze the PDF file.
- 🧠 **Conversational Memory:** The bot remembers context from previous turns in the chat.
- 🔍 **Strict Contextual Answers:** Custom prompts ensure the AI answers *only* using the provided documents.
- 🔒 **Session Isolation:** Each user session maintains its own memory and vector store, ensuring data privacy between tabs.
- 💻 **Web Interface:** Clean, responsive HTML frontend served via Flask.

---

## 🧑‍💻 Skills & Technologies

- **Programming Language:** Python  
- **LLM Framework:** LangChain (Chains, Memory, Prompts)  
- **Model Provider:** OpenAI (GPT-4o-mini)  
- **Web Framework:** Flask  
- **Vector Database:** ChromaDB  
- **Embedding Model:** HuggingFace Embeddings (SentenceTransformers)  
- **Document Handling:** PyPDF  

---

## 📸 Example Usage

Below is a demonstration of the chatbot in action. The user uploads a restaurant receipt PDF and asks questions about the totals and specific items.
<img width="1595" height="1707" alt="Screenshot 2026-01-23 110006" src="https://github.com/user-attachments/assets/6c65edac-c28c-4f62-bb0e-99faac7804cb" />

---

## ▶️ How to Run the Project

### 1. Clone the repository
git clone https://github.com/mojarrad353/private_document_chatbot.git

### 2. Clone the repository
python run app.py (Note: add your OpenAI API key in .env)
