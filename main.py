# main.py

from rag.loader import load_pdf_documents, split_documents
from rag.indexer import build_vector_store_from_chunks, load_vector_store
from rag.qa import load_llm, create_qa_chain, answer_question


def init_vector_db(rebuild: bool = False):
    """
    Nếu rebuild=True: đọc PDF + build lại FAISS.
    Nếu rebuild=False: chỉ load DB đã có.
    """
    if rebuild:
        docs = load_pdf_documents("data")
        chunks = split_documents(docs)
        db = build_vector_store_from_chunks(chunks)
    else:
        db = load_vector_store()
    return db


def main():
    print("Khởi động RAG bot với LLM local...")
    db = init_vector_db(rebuild=False)  # lần đầu có thể set True
    llm = load_llm()
    qa_chain = create_qa_chain(llm, db)

    print("RAG CLI sẵn sàng. Gõ 'exit' để thoát.\n")
    while True:
        question = input("Bạn hỏi: ")
        if question.strip().lower() in {"exit", "quit"}:
            break
        answer = answer_question(qa_chain, question)
        print(f"\nTrả lời: {answer}\n")


if __name__ == "__main__":
    main()
