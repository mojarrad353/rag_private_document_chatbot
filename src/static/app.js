// --- Session Management ---
let sessionId = null;

async function initSession() {
    try {
        const res = await fetch('/session', { method: 'POST' });
        const data = await res.json();
        if (res.ok && data.session_id) {
            sessionId = data.session_id;
        } else {
            document.getElementById('status-msg').innerText = 'Failed to create session. Please refresh.';
            document.getElementById('status-msg').style.color = 'red';
            document.getElementById('upload-btn').disabled = true;
        }
    } catch (err) {
        document.getElementById('status-msg').innerText = 'Server unavailable. Please try again later.';
        document.getElementById('status-msg').style.color = 'red';
        document.getElementById('upload-btn').disabled = true;
    }
}

// --- Upload ---
async function uploadFiles() {
    const fileInput = document.getElementById('pdf-file');
    const files = fileInput.files;
    const statusMsg = document.getElementById('status-msg');
    const uploadBtn = document.getElementById('upload-btn');

    const MAX_FILES = 10;
    const MAX_FILE_SIZE_MB = 5;
    const MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024;

    if (files.length === 0) {
        alert("Please select at least one file first.");
        return;
    }

    if (files.length > MAX_FILES) {
        alert(`Maximum ${MAX_FILES} files per upload.`);
        return;
    }

    // Client-side validation (server enforces the same rules)
    for (let i = 0; i < files.length; i++) {
        if (!files[i].name.toLowerCase().endsWith('.pdf')) {
            alert(`Only PDF files are allowed: ${files[i].name}`);
            return;
        }
        if (files[i].size > MAX_FILE_SIZE) {
            alert(`File "${files[i].name}" exceeds ${MAX_FILE_SIZE_MB} MB limit.`);
            return;
        }
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('file', files[i]);
    }
    formData.append('session_id', sessionId);

    statusMsg.innerText = `Uploading ${files.length} file(s)...`;
    statusMsg.style.color = "#666";
    uploadBtn.disabled = true;

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (response.ok && result.task_ids) {
            const fileCount = result.file_count || files.length;
            statusMsg.innerText = `Processing ${fileCount} files... please wait.`;
            statusMsg.style.color = "#e67e22";
            // Poll for task completion
            pollTasksStatus(result.task_ids, fileCount);
        } else {
            statusMsg.innerText = "Error: " + (result.error || "Upload failed");
            statusMsg.style.color = "red";
            uploadBtn.disabled = false;
        }
    } catch (error) {
        console.error(error);
        statusMsg.innerText = "Upload failed. Please try again.";
        statusMsg.style.color = "red";
        uploadBtn.disabled = false;
    }
}

// --- Task Polling ---
function pollTasksStatus(taskIds, fileCount) {
    const statusMsg = document.getElementById('status-msg');
    const uploadBtn = document.getElementById('upload-btn');
    let completedTasks = new Set();
    let failed = false;

    const interval = setInterval(async () => {
        if (failed) {
            clearInterval(interval);
            return;
        }

        try {
            for (const taskId of taskIds) {
                if (completedTasks.has(taskId)) continue;

                const res = await fetch(`/status/${taskId}`);
                const data = await res.json();

                if (data.state === 'SUCCESS') {
                    completedTasks.add(taskId);
                } else if (data.state === 'FAILURE') {
                    failed = true;
                    clearInterval(interval);
                    statusMsg.innerText = `❌ Processing failed for a document: ${data.status || "Unknown error"}`;
                    statusMsg.style.color = "red";
                    uploadBtn.disabled = false;
                    return;
                }
            }

            if (completedTasks.size === taskIds.length) {
                clearInterval(interval);
                statusMsg.innerText = `✅ All ${fileCount} files ready! You can now chat.`;
                statusMsg.style.color = "green";
                document.getElementById('user-input').disabled = false;
                document.getElementById('send-btn').disabled = false;
                document.getElementById('user-input').focus();
                addMessage(`I have read ${fileCount} documents. I will provide citations in my answers. What would you like to know?`, 'bot-message');
            } else {
                statusMsg.innerText = `⏳ Processing documents (${completedTasks.size}/${taskIds.length})... please wait.`;
            }
        } catch (err) {
            console.error(err);
            // We'll retry on the next interval
        }
    }, 2000);
}

// --- Chat ---
async function sendMessage() {
    const inputField = document.getElementById("user-input");
    const sendBtn = document.getElementById("send-btn");
    const text = inputField.value.trim();

    if (!text) return;

    // 1. Show User Message
    addMessage(text, 'user-message');
    inputField.value = "";
    inputField.disabled = true;
    sendBtn.disabled = true;

    // Show typing indicator
    showTypingIndicator();

    try {
        // 2. Send to Backend
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });

        const data = await response.json();

        // Remove typing indicator before showing answer
        removeTypingIndicator();

        // 3. Show Bot Response
        if (data.answer !== undefined) {
            const botAnswer = data.answer || "I'm sorry, I couldn't generate an answer based on the document.";
            addMessage(botAnswer, 'bot-message');
        } else if (data.error) {
            addMessage("Error: " + data.error, 'bot-message');
        } else {
            addMessage("An unknown error occurred on the server.", 'bot-message');
        }

    } catch (error) {
        removeTypingIndicator();
        addMessage("Error: Could not reach server.", 'bot-message');
    } finally {
        inputField.disabled = false;
        sendBtn.disabled = false;
        inputField.focus();
    }
}

// --- Helpers ---
function addMessage(text, className) {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");
    div.className = `message ${className}`;
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showTypingIndicator() {
    const chatBox = document.getElementById("chat-box");
    const div = document.createElement("div");
    div.className = "typing-indicator";
    div.id = "typing-indicator";
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement("div");
        dot.className = "typing-dot";
        div.appendChild(dot);
    }
    
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById("typing-indicator");
    if (indicator) {
        indicator.remove();
    }
}

// --- Initialize ---
initSession();

document.getElementById("upload-btn").addEventListener("click", uploadFiles);
document.getElementById("send-btn").addEventListener("click", sendMessage);
document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") sendMessage();
});
