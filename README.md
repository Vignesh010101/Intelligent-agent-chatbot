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

### Step2: Configure Environment Variables

Create a file named .env in the root of the project directory (intelligent-agent-chatbot/) and add your API keys.

File: .env

```python
# OpenAI API Configuration
OPENAI_API_KEY="sk-..."

# LangSmith Observability Configuration
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_API_KEY="ls__..."
LANGCHAIN_PROJECT="Intelligent Agent - RAG Demo"


LANGCHAIN_TRACING_V2="true": Enables LangSmith tracing.
```

LANGCHAIN_PROJECT: Groups all your runs under this project name in the LangSmith dashboard.

### Step 4: Build and Run with Docker

From the root of the project directory (intelligent-agent-chatbot/), run the following command:

``` bash
docker-compose up --build
```

This command will: Build the Docker image based on the Dockerfile.

Start the service defined in docker-compose.yml.

The first time it runs, the Python script will load project_docs.txt, create embeddings, and save them to the chroma_db directory.

### Step 5: Access the Application

Chatbot UI: Open your web browser and go to http://localhost:8501

Observability Dashboard: Go to LangSmith and log in. You will see your project "Intelligent Agent - RAG Demo" with detailed traces for every query.

### Screenshots
UI in Action

A screenshot would show the Streamlit interface. On the left, a chat history. On the right, the current conversation. A user asks "What is the pricing model?", and the agent responds with information from the documents, followed by an expandable "Citations" section showing the exact text snippets used.

<img width="1852" height="1000" alt="Screenshot from 2025-07-27 13-28-06" src="https://github.com/user-attachments/assets/5c39a45c-e0ff-4498-a9ed-312096a55572" />

<img width="1852" height="1000" alt="Screenshot from 2025-07-27 13-28-29" src="https://github.com/user-attachments/assets/43bdbb28-8b32-402b-b3fa-fa6b740b259c" />


### LangSmith Observability Dashboard

Few of screenshots attached below would show the LangSmith dashboard. A table lists recent runs, with columns for Name, Latency, Feedback, and Tokens. Clicking on a run would open a detailed trace view, showing the flow from ChatInput to Retriever to LLM and the final Output, with inputs and outputs for each step clearly visible.

<img width="1845" height="957" alt="Screenshot from 2025-07-27 13-42-01" src="https://github.com/user-attachments/assets/780dda46-ad5d-4ed9-9b55-95aed46dd981" />
<img width="1845" height="957" alt="image" src="https://github.com/user-attachments/assets/42a63274-e756-4d04-a953-8346dbad2244" />
<img width="1845" height="957" alt="image" src="https://github.com/user-attachments/assets/cabbf533-ac74-4c06-976f-46ad8c675221" />
<img width="1845" height="957" alt="image" src="https://github.com/user-attachments/assets/be9c8879-7981-4a1e-9903-e51fcccc9265" />



