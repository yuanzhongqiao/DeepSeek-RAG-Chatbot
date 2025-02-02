import time
import streamlit as st
import requests
import re
import json
import os
import torch
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
# For BM25 and Ensemble Retrievers
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# ✅ Hardware configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

# ✅ Streamlit configuration
st.set_page_config(page_title="DeepSeek RAG Pro", layout="wide")

# Custom CSS
st.markdown("""
    <style>
        .stApp { background-color: #f4f4f9; }
        h1 { color: #00FF99; text-align: center; }
        .stChatMessage { border-radius: 10px; padding: 10px; margin: 10px 0; }
        .stChatMessage.user { background-color: #e8f0fe; }
        .stChatMessage.assistant { background-color: #d1e7dd; }
        .stButton>button { background-color: #00AAFF; color: white; }
    </style>
""", unsafe_allow_html=True)

# Constants
OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDINGS_MODEL = "nomic-embed-text:latest"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# 🚀 Initialize Cross-Encoder (Reranker) at the global level
reranker = None  # Declare globally
try:
    reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=device)
except Exception as e:
    st.error(f"Failed to load CrossEncoder model: {str(e)}")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_pipeline" not in st.session_state:
    st.session_state.retrieval_pipeline = None
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False



def ensemble_retrieve(query, bm25_retriever, vector_retriever, bm25_weight=0.4, vector_weight=0.6, top_k=5):
    bm25_results = bm25_retriever.get_relevant_documents(query)
    vector_results = vector_retriever.get_relevant_documents(query)
    
    combined_results = []
    for doc in bm25_results:
        combined_results.append((doc, bm25_weight))  # Assign BM25 weight
    for doc in vector_results:
        combined_results.append((doc, vector_weight))  # Assign vector search weight
    
    # Sort by weighted scores and return top results
    combined_results.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in combined_results[:top_k]]



def process_documents(uploaded_files):
    if st.session_state.documents_loaded:
        return

    st.session_state.processing = True
    documents = []
    
    # Create temp directory
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Process files
    for file in uploaded_files:
        try:
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
            os.remove(file_path)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
            return

    # Text splitting
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separator="\n"
    )
    texts = text_splitter.split_documents(documents)
    text_contents = [doc.page_content for doc in texts]

    # 🚀 Hybrid Retrieval Setup
    embeddings = OllamaEmbeddings(model=EMBEDDINGS_MODEL)
    
    # Vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # BM25 store
    bm25_retriever = BM25Retriever.from_texts(
        text_contents, 
        bm25_impl=BM25Okapi,
        preprocess_func=lambda text: re.sub(r"\W+", " ", text).lower().split()
    )

    # Ensemble retrieval
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_store.as_retriever(search_kwargs={"k": 5})
        ],
        weights=[0.4, 0.6]
    )

    # Store in session
    st.session_state.retrieval_pipeline = {
        "ensemble": ensemble_retriever,
        "reranker": reranker,  # Now using the global reranker variable
        "texts": text_contents
    }

    st.session_state.documents_loaded = True
    st.session_state.processing = False


# 🚀 Query Expansion with HyDE
def expand_query(query):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": "deepseek-r1:7b",
            "prompt": f"Generate a hypothetical answer to: {query}",
            "stream": False
        }).json()
        return f"{query}\n{response.get('response', '')}"
    except Exception as e:
        st.error(f"Query expansion failed: {str(e)}")
        return query

# 🚀 Advanced Retrieval Pipeline
def retrieve_documents(query):
    # Query expansion
    if st.session_state.enable_hyde:
        expanded_query = expand_query(query)
    else:
        expanded_query = query
    
    # First-stage retrieval
    docs = st.session_state.retrieval_pipeline["ensemble"].get_relevant_documents(expanded_query)
    
    # Reranking
    if st.session_state.enable_reranking:
        pairs = [[query, doc.page_content] for doc in docs]
        scores = st.session_state.retrieval_pipeline["reranker"].predict(pairs)
        ranked_docs = [doc for _, doc in sorted(zip(scores, docs), reverse=True)]
    else:
        ranked_docs = docs
    
    return ranked_docs[:st.session_state.max_contexts]

# 📁 Sidebar
with st.sidebar:
    st.header("📁 Document Management")
    uploaded_files = st.file_uploader(
        "Upload documents (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.documents_loaded:
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files)
            st.success("Documents processed!")
    
    st.markdown("---")
    st.header("⚙️ RAG Settings")
    
    st.session_state.rag_enabled = st.checkbox("Enable RAG", value=True)
    st.session_state.enable_hyde = st.checkbox("Enable HyDE", value=True)
    st.session_state.enable_reranking = st.checkbox("Enable Neural Reranking", value=True)
    st.session_state.max_contexts = st.slider("Max Contexts", 1, 5, 3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# 💬 Chat Interface
st.title("🤖 DeepSeek RAG Pro")
st.caption("Advanced RAG System with Hybrid Retrieval and Neural Reranking")

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # 🚀 Build context
        context = ""
        if st.session_state.rag_enabled and st.session_state.retrieval_pipeline:
            try:
                docs = retrieve_documents(prompt)
                context = "\n".join(
                    f"[Source {i+1}]: {doc.page_content}" 
                    for i, doc in enumerate(docs)
                )
            except Exception as e:
                st.error(f"Retrieval error: {str(e)}")
        
        # 🚀 Structured Prompt
        system_prompt = f"""Analyze the question and context through these steps:
1. Identify key entities and relationships
2. Check for contradictions between sources
3. Synthesize information from multiple contexts
4. Formulate a structured response

Context:
{context}

Question: {prompt}
Answer:"""
        
        # Stream response
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": "deepseek-r1:7b",
                "prompt": system_prompt,
                "stream": True,
                "options": {
                    "temperature": 0.3,
                    "num_ctx": 4096
                }
            },
            stream=True
        )
        
        try:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode())
                    token = data.get("response", "")
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                    
                    # Stop if we detect the end token
                    if data.get("done", False):
                        break
                        
            response_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Generation error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
