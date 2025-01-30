import streamlit as st
import requests
import re
import json
import os
import torch
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings  # ✅ Updated import
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# ✅ Ensure GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Set page config at the very start
st.set_page_config(page_title="DeepSeek RAG Chatbot", layout="wide")

# Set the Ollama API endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDINGS_MODEL = "nomic-embed-text:latest"  # Lightweight embedding model

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # ✅ Ensures vector store persists
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False  # ✅ Flag to prevent reloading documents

# 📁 **Document processing function (Only Runs on Upload)**
def process_documents(uploaded_files):
    if st.session_state.documents_loaded:  # ✅ Prevents reprocessing if already done
        return

    st.session_state.processing = True
    documents = []
    
    for file in uploaded_files:
        file_path = os.path.join("temp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
            
        documents.extend(loader.load())
        os.remove(file_path)  # Clean up temp file

    # 🔹 **Reduce chunk size for faster retrieval**
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)  # Lowered for speed
    texts = text_splitter.split_documents(documents)
    
    # 🔹 **GPU-accelerated embeddings**
    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)

    # ✅ Use FAISS (GPU-accelerated vector search)
    st.session_state.vector_store = FAISS.from_documents(texts, embeddings)
    
    # ✅ Mark documents as processed (Prevents Reloading)
    st.session_state.documents_loaded = True
    st.session_state.processing = False
    return len(texts)

# 📁 **Sidebar: Document upload & settings**
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents for RAG (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        if not os.path.exists("temp"):
            os.makedirs("temp")
        
        with st.spinner("Processing documents... (It will take a while)"):
            num_chunks = process_documents(uploaded_files)
            st.success(f"✅ Processed {len(uploaded_files)} documents into {num_chunks} chunks!")
            st.session_state.rag_enabled = True
    
    st.markdown("---")
    st.session_state.rag_enabled = st.checkbox("Enable PDF-Chat Mode", value=st.session_state.rag_enabled)
    
    if st.button("Clear Documents"):
        st.session_state.vector_store = None
        st.session_state.rag_enabled = False
        st.session_state.documents_loaded = False  # ✅ Allows reprocessing if new files are uploaded
        st.success("✅ Documents cleared!")

# 🤖 **Main chat interface**
st.markdown("<h1 style='text-align: center; color: #00FF99;'>🤖 New DeepSeek RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #00AAFF;'>Chat with your PDF and Docs</p>", unsafe_allow_html=True)

# 💬 **Chat history display**
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✏️ **User input**
user_input = st.chat_input("Ask about your documents...")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🤖 **Generate response**
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 📝 **Optimized context retrieval**
        prompt = f"System: {st.session_state.get('system_message', 'You are a helpful AI assistant.')}\n"

        if st.session_state.rag_enabled and st.session_state.vector_store:
            # 🔹 **Retrieve only 1 most relevant document (Faster & less memory usage)**
            docs = st.session_state.vector_store.similarity_search(user_input, k=1)
            context = "\n".join([doc.page_content for doc in docs])
            prompt += f"Context from documents:\n{context}\n\nQuestion: {user_input}\nAnswer:"
        else:
            prompt += f"User: {user_input}"

        # ⚡ **Optimized Ollama request**
        payload = {
            "model": "deepseek-r1:7b",
            "prompt": prompt,
            "stream": True,
            "num_ctx": 1024,  # Optimize response speed
            "context": st.session_state.get("context", [])
        }

        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=600)  # 🔹 **Increased timeout**
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        if "response" in data:
                            # 🚀 **Real-time response streaming**
                            cleaned = re.sub(r"<think>.*?</think>", "", data["response"], flags=re.DOTALL)
                            cleaned = re.sub(r"\n+", " ", cleaned).strip()
                            if cleaned:
                                full_response += cleaned + " "
                                message_placeholder.markdown(full_response + "▌")
                    except json.JSONDecodeError:
                        pass

            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
