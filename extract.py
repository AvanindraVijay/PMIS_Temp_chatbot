import os
import logging
import torch
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document  # ✅ Updated import for compatibility

# ✅ File Paths
PDF_PATH = "E:\PMIS\Eng-data\Merged.pdf"  # Use forward slash for cross-platform compatibility
DB_FAISS_PATH = "vectorstore/db_faiss"

# ✅ Select Device
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    logging.warning("⚠️ No GPU found! Switching to CPU mode.")
logging.info(f"🔥 Using device: {device}")

# ✅ Load and Process PDF Data
def load_pdf_data():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

# ✅ Create FAISS Vector Database
def create_vector_db():
    pdf_docs = load_pdf_data()
    if not pdf_docs:
        logging.error("❌ No data loaded from PDF!")
        return

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
    )

    db = FAISS.from_documents(pdf_docs, embeddings)
    db.save_local(DB_FAISS_PATH)
    print(f"✅ Vector DB saved to {DB_FAISS_PATH}")

if __name__ == "__main__":
    create_vector_db()
