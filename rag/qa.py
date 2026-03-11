# rag/qa.py

from langchain_community.llms import CTransformers
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate

MODEL_FILE = "models/vinallama-7b-chat_q5_0.gguf"


def load_llm(model_file: str = MODEL_FILE):
    """
    Tạo LLM local từ file GGUF (CTransformers).
    """
    llm = CTransformers(
        model=model_file,
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.1,
    )
    return llm


def create_prompt():
    """
    Prompt RAG giống QAbot.py nhưng gói thành hàm.
    """
    template = """<|im_start|>system
Sử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra câu trả lời.

{context}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )
    return prompt


def create_qa_chain(llm, db):
    """
    Tạo RetrievalQA chain từ llm + vector store.
    """
    prompt = create_prompt()
    retriever = db.as_retriever(
        search_kwargs={"k": 3},
        max_tokens_limit=1024,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


def answer_question(qa_chain, question: str) -> str:
    """
    Gọi chain để trả lời một câu hỏi. Không lặp lại cùng một cụm từ.

    """
    response = qa_chain.invoke({"query": question})
    # tuỳ vào kiểu trả về của RetrievalQA; nếu nó trả dict, ta lấy ['result']
    if isinstance(response, dict) and "result" in response:
        return response["result"]
    return str(response)
