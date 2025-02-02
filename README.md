### 🚀 **DeepSeek RAG Chatbot 0.2 – Now with Hybrid Retrieval & Reranking!**  
**(100% Free, Private (No Internet), and Local PC Installation)**  

🔥 **DeepSeek + FAISS + BM25 + GPU = The Ultimate RAG Stack!**  

This chatbot enables **fast, accurate, and explainable retrieval of information** from PDFs, DOCX, and TXT files using **DeepSeek-7B**, **BM25**, **FAISS**, and **Neural Reranking (Cross-Encoder)**.  

---

## **🔹 New Features in This Version**
✅ **Hybrid Retrieval:** Combines **BM25 (keyword search) + FAISS (semantic search)** for **better accuracy**.  
✅ **Ensemble Retrieval:** Merges **BM25 & FAISS** results with weighting for **higher-quality answers**.  
✅ **Neural Reranking:** Uses **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)** to **rank retrieved documents** based on relevance.  
✅ **Query Expansion (HyDE):** Expands queries using **Hypothetical Document Embeddings** to **retrieve better matches**.  
✅ **Document Source Tracking:** Displays **which PDF/DOCX file** the retrieved answer comes from.  
✅ **Faster Processing:** Optimized **document chunking** and **GPU acceleration** for FAISS & Cross-Encoder.  

**In the Next Update** --> **Clickable PDF Links:** Users can **open the source document** to verify the response.  

---

## **🛠️ Installation & Setup**
### **1️⃣ Clone the Repository & Install Dependencies**
```bash
git clone https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot.git
cd DeepSeek-RAG-Chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### **2️⃣ Download & Set Up Ollama**
Ollama is required to run **DeepSeek-7B** and **Nomic Embeddings** locally.  
🔗 **Download Ollama** → [https://ollama.com/](https://ollama.com/)  

Then, pull the required models:
```bash
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
```

### **3️⃣ Run the Chatbot**
```bash
streamlit run app.py
```
---

## **📌 How It Works**
1️⃣ Upload PDFs, DOCX, or TXT files 📂  
2️⃣ **Hybrid Retrieval** (BM25 + FAISS) fetches the most relevant text 🔍  
3️⃣ **Neural Reranking** (Cross-Encoder) refines search results for higher accuracy 🏆  
4️⃣ **Query Expansion (HyDE)** improves recall by generating an expanded query 🔄  
5️⃣ **DeepSeek-7B** generates an answer based on the best-matched document chunks 💬 

6️⃣ **In the Next Update** --> **Sources are displayed** along with the response, with **clickable PDF links** 📑  

---

## **🔹 Why This Upgrade?**
| Feature | Old Version | New Version |
|---------|------------|------------|
| **Retrieval Method** | FAISS-only | BM25 + FAISS (Hybrid) |
| **Document Ranking** | No reranking | Cross-Encoder Reranking |
| **Query Expansion** | Basic queries only | HyDE Query Expansion |
| **Search Accuracy** | Moderate | **High** (Hybrid + Reranking) |

---

## **📌 Common Issues & Fixes**
💡 **Issue: OpenMP Conflict (OMP: Error #15)**  
✅ **Fix:** Remove Intel MKL conflicts & reinstall PyTorch  
```bash
pip uninstall intel-openmp mkl mkl-include
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

💡 **Issue: Slow Document Processing**  
✅ **Fix:** Reduce chunk size & optimize FAISS retrieval  
```python
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
st.session_state.vector_store = FAISS.from_documents(texts, embeddings)
```

---

## **📌 Contributing**
🚀 Want to improve this chatbot? Feel free to **fork this repo**, submit **pull requests**, or **report issues**!  

---

### **🔗 Connect & Share Your Thoughts!**
Got feedback or suggestions? Let’s discuss on **[Reddit](https://www.reddit.com/)**! 🚀💡
