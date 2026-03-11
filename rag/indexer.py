# rag/indexer.py

from typing import Any
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS

VECTOR_DB_PATH = "vectorstores/db_faiss"
EMBEDDING_MODEL_FILE = "models/all-MiniLM-L6-v2-f16.gguf"


def get_embedding_model():
    """
    Tạo embedding model GPT4All từ file GGUF.
    """
    embedding_model = GPT4AllEmbeddings(model_file=EMBEDDING_MODEL_FILE)
    return embedding_model


def build_vector_store_from_chunks(chunks) -> Any:
    """
    Nhận list Document (đã chunk) và xây FAISS DB rồi lưu ra disk.
    """
    embedding_model = get_embedding_model()
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(VECTOR_DB_PATH)
    return db


def load_vector_store() -> Any:
    """
    Load FAISS DB đã lưu sẵn.
    """
    embedding_model = get_embedding_model()
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return db
