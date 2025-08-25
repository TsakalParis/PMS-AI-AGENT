# PMS AI Agent (Beds24)

This project demonstrates how to run an AI-powered Property Management System (PMS) Agent inside a Kaggle Notebook, leveraging Kaggle’s free GPU/CPU resources. The agent connects with the Beds24 API to automate tasks such as retrieving bookings, sending guest messages, and checking room availability.

🚀 Features

Beds24 API Integration

Secure authentication (invite code, refresh token, auth token management).

Booking retrieval with filters.

Sending and extracting guest messages.

Checking detailed and summarized room availability.

AI Agent with LangChain & LangGraph

Powered by Mistral-7B-Instruct (GGUF) via Hugging Face Hub.

Custom LangChain wrapper for llama-cpp-python.

Structured conversational memory and tool usage.

Interactive Gradio UI

Authenticate with Beds24 directly from the notebook.

Chat interface with the AI agent.

Real-time execution of Beds24 actions via natural language.

📂 Project Structure
├── beds24_invite_code.json      # Provide invite code for first-time setup
├── beds24_refresh_token.json    # Refresh token storage
├── beds24_auth_token.json       # Auth token storage
├── prompt.yaml                  # System and user prompt templates
├── hf_token.txt                 # Hugging Face access token
├── mock_messages/               # Mock booking messages for testing
└── notebook.ipynb               # Main Kaggle Notebook (AI Agent)

⚙️ Setup & Installation

Run the notebook in Kaggle. Required packages are installed automatically:

!pip install huggingface_hub
!pip install langchain==0.1.16 langchain_core==0.1.45 langgraph==0.0.33
!pip install -U langchain-community
!pip install llama-cpp-python --prefer-binary --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
!pip install gradio

🔑 Authentication

Place your Beds24 Invite Code inside beds24_invite_code.json in this format:

{
  "invite_code": "YOUR_INVITE_CODE",
  "created": "2025-08-25T00:00:00Z",
  "expiration": "2025-09-25T00:00:00Z"
}


Run the authentication process in the Gradio UI.

Tokens are automatically refreshed and stored in /kaggle/working/.

💬 Usage

Launch the Gradio UI from the notebook.

Authenticate with Beds24.

Start chatting with the AI agent. Example commands:

“Show me today’s bookings.”

“Send a message to booking ID 123456.”

“Check availability for Room 2 next weekend.”

🛠️ Tech Stack

Kaggle – Free GPU/CPU execution environment.

Beds24 API – PMS data integration.

LangChain & LangGraph – AI agent orchestration.

llama-cpp-python – Local inference of LLM models.

Hugging Face Hub – Model hosting and download.

Gradio – Interactive UI inside Kaggle notebooks.

Do you want me to also make a shorter, beginner-friendly README (like a quickstart version) for non-technical users, or keep it detailed for developers?
