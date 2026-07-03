from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
load_dotenv()


data = PyPDFLoader(
    "document_loaders/Deep+Learning+Ian+Goodfellow.pdf"
)
docs=data.load()

spilitter=RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)
chunks=spilitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
vectorstore=Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
print("Database created successfully!")
print("Total pages:", len(docs))
print("Total chunks:", len(chunks))