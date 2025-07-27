# Intelligent RAG Agent with Observability and UI

This project delivers a complete, LLM-powered, context-aware support agent. It features a full Retrieval Augmented Generation (RAG) pipeline, an integrated observability stack with LangSmith, a user-friendly web interface built with Streamlit, and is fully containerized with Docker for easy deployment.

## Architecture Overview

The application follows a modern RAG architecture, orchestrated by LangChain and deployed as a self-contained service.

**Flow Diagram:**
```
User -> [Streamlit UI] -> [Docker Container] -> [LangChain Agent] -> [LangSmith (Observability)]
                                    |
                                    V
                    +--------------------------------+
                    |         RAG Pipeline           |
                    |                                |
                    | 1. Retrieve relevant docs      |
                    |    from Chroma Vector Store    |
                    |                                |
                    | 2. Augment prompt with context |
                    |                                |
                    | 3. Reason with LLM (OpenAI)    |
                    +--------------------------------+
                                    |
                                    V
                        [Response with Citations]
```

## Core Features

### 🧠 Intelligent Agent (RAG)
- Uses **LangChain** for robust agent orchestration
- Loads and processes text documents from a local `documents` folder
- Stores document embeddings in a persistent **ChromaDB** vector store
- The agent retrieves relevant context before answering a query, providing accurate, source-based answers

### 🔭 AI Observability
- Integrated with **LangSmith** out-of-the-box
- Automatically logs every user query, the retrieved documents, the final LLM-generated answer, and performance metrics like latency and token usage
- Provides a full trace of the agent's reasoning process for easy debugging and monitoring

### 🐳 Dockerized Deployment
- Uses `docker-compose` to run the entire application with a single command
- Ensures a consistent, reproducible environment for both development and production
- The ChromaDB database is persisted to the host machine via a Docker volume, so your vectorized data is not lost on restart

### 🖥️ Web Interface (Bonus)
- A clean, interactive chat interface built with **Streamlit**
- Features an input box for user queries, a real-time display of responses, and a persistent chat history for the session
- **Displays citations** for each response, showing the source document chunks used to generate the answer

## Project Structure

```
intelligent-agent-chatbot/
├── .env                      # Stores API keys and environment variables
├── app/
│   ├── main.py              # Core application: Streamlit UI + LangChain agent logic
│   └── documents/
│       └── project_docs.txt # Sample documents for the agent to learn from
├── Dockerfile               # Instructions to build the application container
├── docker-compose.yml       # Defines and runs the multi-container application
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Setup and Local Deployment

Follow these steps to get the chatbot running on your local machine.

### Prerequisites

- [Docker](https://www.docker.com/get-started) and Docker Compose
- [Git](https://git-scm.com/downloads/)
- An [OpenAI API Key](https://platform.openai.com/api-keys)
- A [LangSmith API Key](https://smith.langchain.com/) (create an account and an API key)

### Step 1: Clone the Project

Although the code is provided below, for a real setup you would clone a repository. For now, create the project folder structure as shown above.

```bash
mkdir -p intelligent-agent-chatbot/app/documents
cd intelligent-agent-chatbot
```
