import os
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
load_dotenv()

# Check for necessary API keys
if not os.getenv("OPENAI_API_KEY"):
    st.error("🚨 OpenAI API key not found. Please set it in your .env file.")
if not os.getenv("LANGCHAIN_API_KEY"):
    st.error("🚨 LangSmith API key not found. Please set it in your .env file.")

# Define constants
VECTOR_STORE_PATH = "chroma_db"
DOCUMENTS_PATH = "documents/project_docs.txt"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- PAGE SETUP ---
st.set_page_config(page_title="Intelligent Support Agent", page_icon="🤖")
st.title("🤖 Intelligent Support Agent")
st.caption("I am an expert on QuantumLeap AI products. Ask me anything!")


# --- CORE AGENT LOGIC (RAG CHAIN) ---

@st.cache_resource(show_spinner="Initializing RAG chain...")
def get_rag_chain():
    """
    Initializes and returns a Retrieval-Augmented Generation (RAG) chain.
    This function is cached to avoid re-initializing the model and vector store
    on every user interaction.
    """
    # 1. Initialize Embeddings Model
    embeddings = OpenAIEmbeddings()

    # 2. Setup Vector Store
    if not os.path.exists(VECTOR_STORE_PATH):
        st.info("Creating new vector store from documents...")
        # Load documents
        loader = TextLoader(DOCUMENTS_PATH)
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        splits = text_splitter.split_documents(docs)

        # Create and persist the vector store
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=VECTOR_STORE_PATH
        )
    else:
        st.info("Loading existing vector store...")
        vectorstore = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=embeddings
        )

    # 3. Create Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Define Prompt Templates
    rag_template = """
    You are an intelligent assistant for the company QuantumLeap AI.
    Answer the user's question based only on the following context.
    If you don't know the answer from the context, respond with exactly: "INSUFFICIENT_CONTEXT"
    If you can answer from the context, provide clear, concise answers and cite your sources.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    
    fallback_template = """
    You are a helpful and knowledgeable assistant. The user has asked a question, and you should provide a comprehensive and accurate answer based on your general knowledge. Be helpful, informative, and maintain a professional tone.

    QUESTION:
    {question}

    ANSWER:
    """
    
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    fallback_prompt = ChatPromptTemplate.from_template(fallback_template)

    # 5. Initialize LLMs
    rag_llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
    fallback_llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # 6. Build RAG Chain using LangChain Expression Language (LCEL)
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG chain for context-based answers
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | rag_llm
        | StrOutputParser()
    )
    
    # Fallback chain for general knowledge
    fallback_chain = (
        {"question": RunnablePassthrough()}
        | fallback_prompt
        | fallback_llm
        | StrOutputParser()
    )
    
    # Main function that handles RAG with intelligent fallback
    def get_answer_and_sources(question):
        try:
            # First, try to get documents from retriever
            docs = retriever.invoke(question)
            
            # Get the RAG answer
            rag_answer = rag_chain.invoke(question)
            
            # Check if RAG found sufficient context
            if "INSUFFICIENT_CONTEXT" in rag_answer:
                # Silently fall back to general knowledge
                fallback_answer = fallback_chain.invoke(question)
                return {
                    "answer": fallback_answer,
                    "sources": [],
                    "used_fallback": True
                }
            else:
                # Use RAG answer with sources
                return {
                    "answer": rag_answer,
                    "sources": [doc.page_content for doc in docs],
                    "used_fallback": False
                }
        except Exception as e:
            # If anything fails, fall back to general knowledge
            fallback_answer = fallback_chain.invoke(question)
            return {
                "answer": fallback_answer,
                "sources": [],
                "used_fallback": True
            }
    
    return get_answer_and_sources


# --- STREAMLIT UI & INTERACTION ---

# Get the initialized RAG chain
try:
    rag_chain_with_sources = get_rag_chain()
except Exception as e:
    st.error(f"Failed to initialize the RAG chain: {e}")
    st.stop()


# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you with QuantumLeap AI today?"}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about QuantumLeap AI features, pricing, or support..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Get response with intelligent fallback
            response = rag_chain_with_sources(prompt)
            answer = response["answer"]
            sources = response["sources"]
            used_fallback = response["used_fallback"]

            st.markdown(answer)
            
            # Only display citations if we have sources (i.e., used RAG)
            if sources and not used_fallback:
                with st.expander("Citations"):
                    for i, source in enumerate(sources):
                        st.info(f"Source {i+1}:\n\n" + source.replace("\n", "\n\n> "))
                        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
