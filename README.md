# 📚 DocuMind — PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot that lets you upload multiple PDFs and ask questions about them in natural language, with source citations, streaming responses, and persistent chat history.

Built with **LangChain**, **ChromaDB**, **Mistral AI**, and **Streamlit**.

---

## 🚀 Live Demo

https://pdf-rag-assistant-ihw0.onrender.com

---

## ✨ Features

- **Multi-PDF upload** — process and query multiple documents at once
- **Chat history** — full conversation persists across the session, rendered as a real chat UI
- **Source citations** — every answer shows which PDF and page it came from
- **Streaming responses** — answers appear token-by-token instead of all at once
- **Persistent vector store** — embeddings are saved to disk via ChromaDB, so documents don't need re-processing on every run
- **Download conversation** — export the full chat as a `.txt` file
- **Duplicate detection** — re-uploading the same PDF skips re-embedding it

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | Mistral AI (`mistral-embed`) |
| LLM | Mistral AI (`mistral-small`) |
| Vector Store | ChromaDB (via `langchain-chroma`) |
| PDF Parsing | PyMuPDF |
| Text Splitting | LangChain `RecursiveCharacterTextSplitter` |

**Why Mistral embeddings instead of a local model?** Local embedding models (like HuggingFace's `bge-small`) load hundreds of MB of weights into memory, which exceeds free-tier hosting limits (e.g. Render's 512MB cap). Using Mistral's hosted embeddings API keeps the app's memory footprint small since the embedding computation happens on their servers.

---

## 📦 Setup

### 1. Clone the repo
```bash
git clone https://github.com/2k23cs2313652-rishi/PDF-RAG-Assistant.git
cd PDF-RAG-Assistant
```

### 2. Create a virtual environment and install dependencies

Using `uv` (recommended):
```bash
uv venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
uv pip install -r requirements.txt
```

Or using standard `pip`:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your API key

Create a `.env` file in the project root:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get a key from [console.mistral.ai](https://console.mistral.ai).

### 4. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🧠 How It Works

1. **Upload & Chunk** — PDFs are parsed page-by-page with PyMuPDF, then split into overlapping chunks (1500 chars, 300 overlap) for better context retention.
2. **Embed & Store** — Each chunk is embedded via Mistral's `mistral-embed` model and stored in a local ChromaDB collection.
3. **Retrieve** — On each question, the top relevant chunks are retrieved using **MMR (Maximal Marginal Relevance)** search, which balances relevance with diversity to avoid redundant context.
4. **Generate** — Retrieved chunks are passed as context to `mistral-small`, which streams back a grounded answer along with the source PDF/page it used.

---

## ⚠️ Known Limitations

- **Ephemeral storage on free hosting tiers**: Platforms like Render's free plan wipe the filesystem on redeploy or restart, so the vector store resets — documents need to be re-uploaded after that happens. For true persistence in production, use a paid persistent disk or an external hosted vector DB.
- **Duplicate detection is session-based**: file-hash tracking to skip re-embedding only lasts for the current session, not across app restarts.

---

## 📌 Future Improvements

- [ ] Add file-hash tracking that persists across restarts (e.g. store hashes in a small metadata file)
- [ ] Support additional file types (docx, txt)
- [ ] Add a reranking step for even better retrieval accuracy
- [ ] User authentication for multi-user deployments

---

## 👤 Author

Built by Rishi as part of a self-directed generative AI learning path covering LLMs, RAG pipelines, and LangChain.