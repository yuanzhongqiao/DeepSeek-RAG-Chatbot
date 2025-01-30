🚀 DeepSeek RAG Chatbot (100% Free, Private (No Internet) and Local PC Installation )

🔥 DeepSeek + FAISS + GPU = The Ultimate RAG Stack!

This chatbot enables fast, accurate retrieval of information from PDFs, DOCX, and TXT files using DeepSeek-7B, FAISS, and GPU acceleration.

🔹 Features

✅ Uploads & processes PDFs, DOCX, TXT files
✅ Uses FAISS for ultra-fast document search
✅ DeepSeek-7B generates responses based on document retrieval
✅ Streams responses in real-time
✅ Optimized for GPU acceleration

🛠️ Installation & Setup

1️⃣ Git Clone, Create Python env, activate it and Install Dependencies

git clone https://github.com/SaiAkhil066/DeepSeek-RAG-Chatbot.git

cd DeepSeek-RAG-Chatbot

python -m venv venv

venv/scripts/activate

pip install -r requirements.txt


2️⃣ Download ollama from its OFFICIAL WEBSITE, Pull the DeepSeek Model and NOMIC Model

https://ollama.com/

ollama pull deepseek-r1:7b

ollama pull nomic-embed-text


3️⃣ Run the Chatbot

streamlit run app.py

📌 How It Works

1️⃣ Upload PDFs, DOCX, or TXT files 📂
2️⃣ The chatbot embeds the content using FAISS 🔍
3️⃣ It retrieves the most relevant sections 📝
4️⃣ DeepSeek-7B generates a contextual response 💬
5️⃣ Streams the answer back in real-time 🚀

🔹 Why DeepSeek-7B?

DeepSeek-7B outperforms other models in Ollama for RAG tasks due to:
✔ Optimized for long-document comprehension
✔ Lower hallucination rate
✔ Faster inference with GPU acceleration
✔ Seamless FAISS integration for retrieval




📌 Common Issues & Fixes

💡 Issue: OpenMP Conflict (OMP: Error #15)
✅ Fix: Remove Intel MKL conflicts & reinstall PyTorch

pip uninstall intel-openmp mkl mkl-include
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

💡 Issue: Slow Document Processing
✅ Fix: Reduce chunk size & optimize FAISS retrieval

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
st.session_state.vector_store = FAISS.from_documents(texts, embeddings)

📌 Contributing

Want to improve this chatbot? Feel free to fork this repo, submit pull requests, or report issues!

📌 License

This project is open-source under the MIT License.
