# 📄 PDF RAG Assistant

A **Retrieval-Augmented Generation (RAG)** application that enables users to upload PDF documents and ask natural language questions. The application retrieves the most relevant document chunks using **ChromaDB** and generates context-aware answers with **Google Gemini** through **LangChain**.

---

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Automatic text chunking
- 🔍 Semantic search using vector embeddings
- 🧠 Context-aware question answering with Google Gemini
- 💾 Persistent vector database using ChromaDB
- 🌐 Interactive Streamlit web interface

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **LangChain**
- **Google Gemini**
- **ChromaDB**
- **HuggingFace Embeddings**
- **python-dotenv**
- **uv** (Package Manager)

---

## 📂 Project Structure

```
PDF-RAG-Assistant/
│── app.py                 # Streamlit application
│── create_database.py     # Creates the Chroma vector database
│── main.py                # RAG pipeline
│── requirements.txt
│── pyproject.toml
│── document_loaders/
│── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/2k23cs2313652-rishi/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
```

### 2. Create a virtual environment

Using `uv`:

```bash
uv venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=your_google_api_key
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## 📖 How It Works

1. Upload a PDF document.
2. The document is split into smaller chunks.
3. Embeddings are generated for each chunk.
4. Chunks are stored in ChromaDB.
5. When a user asks a question:
   - Relevant chunks are retrieved.
   - The retrieved context is sent to Google Gemini.
   - Gemini generates a context-aware answer.

---

## 🎯 Future Improvements

- Support multiple PDF uploads
- Chat history and conversation memory
- Source citations for retrieved content
- Hybrid search (Keyword + Semantic Search)
- Multiple LLM support
- Deploy on Streamlit Community Cloud

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Rishi Gupta**

GitHub: https://github.com/2k23cs2313652-rishi

---

⭐ If you found this project helpful, consider giving it a star!
