# rag/loader.py

from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

PDF_DATA_PATH = "data"


def load_pdf_documents(data_dir: str = PDF_DATA_PATH):
    """
    Load toàn bộ file PDF trong thư mục data_dir thành list Document.
    """
    loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def split_documents(documents, chunk_size: int = 512, chunk_overlap: int = 50):
    """
    Chunk list Document thành các đoạn nhỏ.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
