"""
This module contains the Flask application for the RAG chatbot.
It handles file uploads and chat interactions.
"""

import os
from flask import Flask, render_template, request, jsonify
from .config import settings
from .rag import rag_service

# Initialize Flask App
# We need to specify template folder because we moved app.py inside a package
# Assuming templates is in the same directory as this file
app = Flask(__name__, template_folder='templates')


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
            # Save File Temporarily
            original_filename = file.filename
            # Sanitize filename if needed, for now keep simple
            filepath = os.path.join(settings.UPLOAD_FOLDER, f"{session_id}_{original_filename}")
            file.save(filepath)

            # Process with RAG Service
            rag_service.process_file(session_id, filepath)

            # Cleanup: Delete the temp file
            if os.path.exists(filepath):
                os.remove(filepath)

            return jsonify({"message": "File processed successfully!"})

        except Exception as e:  # pylint: disable=broad-exception-caught
            # In production, use a proper logger
            print(f"Error processing file: {e}")
            return jsonify({"error": f"Failed to process file: {str(e)}"}), 500

    return jsonify({"error": "Unknown error"}), 500


@app.route('/chat', methods=['POST'])
def chat():
    """Handles the chat logic using the user's specific PDF data."""
    data = request.json
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    user_query = data.get('message')
    session_id = data.get('session_id')

    if not user_query or not session_id:
        return jsonify({"error": "Missing message or session_id"}), 400

    try:
        answer = rag_service.get_answer(session_id, user_query)
        return jsonify({"answer": answer})

    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"Error during chat: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500


if __name__ == '__main__':
    print("Starting Flask Server...")
    app.run(debug=True, port=5000)
