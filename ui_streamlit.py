import streamlit as st

from rag.indexer import load_vector_store
from rag.qa import load_llm, create_qa_chain, answer_question

# Dùng session_state để không load lại mỗi lần bấm nút
if "qa_chain" not in st.session_state:
    st.write("⏳ Đang khởi tạo LLM + vector store...")
    db = load_vector_store()
    llm = load_llm()
    st.session_state.qa_chain = create_qa_chain(llm, db)

st.title("📚 Local RAG Chatbot")
st.write("Hỏi đáp dựa trên tài liệu PDF trong thư mục `data/`.")

question = st.text_input("Nhập câu hỏi của bạn:")

if st.button("Hỏi") and question.strip():
    with st.spinner("Đang suy nghĩ..."):
        answer = answer_question(st.session_state.qa_chain, question)
    st.markdown("**Trả lời:**")
    st.write(answer)
