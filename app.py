import os
import hashlib
import tempfile
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ==========================================================
# Config
# ==========================================================

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_rag_collection"

st.set_page_config(
    page_title="PDF RAG Chat",
    page_icon="📚",
    layout="wide"
)

# ==========================================================
# Cached resources (loaded once per session, not on every rerun)
# ==========================================================

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_llm():
    return ChatMistralAI(model="mistral-small-2506", streaming=True)


@st.cache_resource
def get_vectorstore(_embeddings):
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=_embeddings,
        collection_name=COLLECTION_NAME,
    )


embeddings = get_embeddings()
llm = get_llm()
vectorstore = get_vectorstore(embeddings)

# ==========================================================
# Prompt
# ==========================================================

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant answering questions about uploaded PDF documents.

Use only the provided context to answer the question. Be concise and accurate.

If the answer is not present in the context, say:
"I don't have enough information in the provided document(s) to answer that."

Where relevant, refer to the source documents by name.
"""
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

# ==========================================================
# Session state init
# ==========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {role, content, sources}

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # file hashes already embedded


def file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


def process_pdfs(uploaded_files):
    """Load, split, and embed newly uploaded PDFs. Skips duplicates."""
    new_chunks = []
    newly_added_names = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getbuffer()
        f_hash = file_hash(bytes(file_bytes))

        if f_hash in st.session_state.processed_files:
            continue  # already embedded in this session

        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(temp_dir, uploaded_file.name)
            with open(pdf_path, "wb") as f:
                f.write(file_bytes)

            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = uploaded_file.name

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=300
            )
            chunks = splitter.split_documents(docs)
            new_chunks.extend(chunks)

        st.session_state.processed_files.add(f_hash)
        newly_added_names.append(uploaded_file.name)

    if new_chunks:
        vectorstore.add_documents(new_chunks)
        try:
            vectorstore.persist()
        except Exception:
            pass  # newer Chroma versions auto-persist

    return newly_added_names


def format_sources(docs):
    """Build a de-duplicated, readable citation list from retrieved chunks."""
    seen = set()
    citations = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", None)
        label = f"{source}" + (f" — page {page + 1}" if isinstance(page, int) else "")
        if label not in seen:
            seen.add(label)
            citations.append(label)
    return citations


def build_conversation_text():
    lines = [f"PDF RAG Chat — exported {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for msg in st.session_state.messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
        if msg.get("sources"):
            lines.append(f"Sources: {', '.join(msg['sources'])}")
        lines.append("")
    return "\n".join(lines)

# ==========================================================
# Sidebar — upload & controls
# ==========================================================

with st.sidebar:
    st.header("📁 Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Process PDFs", type="primary"):
            with st.spinner("Processing PDF(s)..."):
                added = process_pdfs(uploaded_files)
            if added:
                st.success(f"Added: {', '.join(added)}")
            else:
                st.info("These files were already processed.")

    st.divider()

    total_chunks = vectorstore._collection.count() if vectorstore._collection else 0
    st.caption(f"📊 Vector store contains **{total_chunks}** chunks "
               f"(persisted at `{PERSIST_DIR}`)")

    st.divider()
    st.header("💬 Conversation")

    if st.session_state.messages:
        st.download_button(
            "⬇️ Download conversation",
            data=build_conversation_text(),
            file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.caption("No messages yet.")

# ==========================================================
# Main chat area
# ==========================================================

st.title("📚 Chat With Your PDFs")

if total_chunks == 0:
    st.info("👈 Upload and process at least one PDF to start chatting.")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

# Chat input
query = st.chat_input("Ask a question about your documents...")

if query:
    if total_chunks == 0:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5}
    )

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            docs = retriever.invoke(query)
            context = "\n\n".join(
                f"[{doc.metadata.get('source', 'Unknown')} - "
                f"page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
                for doc in docs
            )
            final_prompt = prompt.invoke({"context": context, "question": query})

        def stream_response():
            for chunk in llm.stream(final_prompt):
                if chunk.content:
                    yield chunk.content

        full_response = st.write_stream(stream_response)

        sources = format_sources(docs)
        if sources:
            with st.expander("📎 Sources"):
                for s in sources:
                    st.markdown(f"- {s}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
    })