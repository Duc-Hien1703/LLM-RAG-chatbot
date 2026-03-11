# Local RAG Chatbot (Vietnamese + PDF)

This project is a **local Retrieval-Augmented Generation (RAG) chatbot** that lets you chat in Vietnamese with your own PDF documents. 
It uses LangChain, FAISS and a GGUF LLaMA model to answer questions based on the content of the PDFs in the `data/` folder. 
Backend is built with FastAPI, with a simple Streamlit UI and CLI for testing. [web:9][web:64][web:112]

---

### Features

- Load all PDF files from the `data/` folder and split them into chunks for retrieval. [web:10]
- Create a FAISS vector store using local **GPT4All embeddings** (GGUF). [web:64]
- Use a local **VinaLLaMA 7B** GGUF model via CTransformers to generate answers.
- Three ways to interact:
  - **CLI**: ask questions directly in the terminal.
  - **API**: FastAPI endpoint `POST /ask` for programmatic access. [web:9][web:112]
  - **UI**: simple Streamlit chat interface.

---

## Project Structure

```text
.
├── main.py              # CLI entry point
├── api/
│   └── main.py          # FastAPI app (POST /ask)
├── rag/
│   ├── loader.py        # Load + split PDF documents
│   ├── indexer.py       # Build & load FAISS vector store
│   └── qa.py            # Load local LLM + create RetrievalQA chain
├── ui_streamlit.py      # Streamlit chat UI
├── data/                # Input PDFs (not tracked in git)
├── models/              # GGUF model files (ignored in git)
└── vectorstores/        # FAISS index (ignored in git)
