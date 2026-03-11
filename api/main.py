from fastapi import FastAPI
from pydantic import BaseModel

from rag.indexer import load_vector_store
from rag.qa import load_llm, create_qa_chain, answer_question

app = FastAPI(title="Local RAG API")

# Biến global để giữ chain trong bộ nhớ
qa_chain = None


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_event():
    """
    Khởi tạo vector store + LLM + qa_chain khi server start.
    """
    global qa_chain
    db = load_vector_store()
    llm = load_llm()
    qa_chain = create_qa_chain(llm, db)


@app.get("/")
def root():
    return {"service": "local-rag-api", "status": "running"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    Nhận câu hỏi, trả về câu trả lời từ RAG.
    """
    global qa_chain
    answer = answer_question(qa_chain, request.question)
    return AskResponse(answer=answer)

