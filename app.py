import hashlib

import streamlit as st

from ollama_client import OllamaClient
from pdf_processor import MAX_FILE_BYTES, PDFError, PDFProcessor
from rag import RAGPipeline
from vector_store import VectorStore


def get_pipeline():
    """One pipeline (and one private vector store) per browser session."""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(PDFProcessor(), VectorStore(), OllamaClient())
        st.session_state.indexed = {}  # file hash -> (filename, chunk count)
        st.session_state.chat_history = []
    return st.session_state.pipeline


def main():
    st.set_page_config(page_title="PDF Q&A with RAG", page_icon="📚", layout="wide")
    pipeline = get_pipeline()

    st.title("PDF Q&A with Llama 3.1")
    st.caption("Upload PDFs, then ask questions. Answers come only from your documents and cite page numbers.")

    with st.sidebar:
        st.header("Documents")
        uploads = st.file_uploader(
            f"Upload PDF files (max {MAX_FILE_BYTES // (1024 * 1024)} MB each)",
            type="pdf",
            accept_multiple_files=True,
        )
        for upload in uploads or []:
            data = upload.getvalue()
            key = hashlib.sha256(data).hexdigest()
            # Streamlit reruns the script on every interaction; only index each file once
            if key in st.session_state.indexed:
                continue
            with st.spinner(f"Indexing {upload.name}..."):
                try:
                    count = pipeline.add_pdf(upload.name, data)
                except PDFError as e:
                    st.error(f"{upload.name}: {e}")
                    continue
            if count == 0:
                st.warning(f"{upload.name}: no text found (it may be a scanned image).")
            st.session_state.indexed[key] = (upload.name, count)

        for name, count in st.session_state.indexed.values():
            st.write(f"📄 {name}: {count} chunks")

        st.divider()
        llm = pipeline.llm
        if llm.is_available():
            st.success(f"Model: {llm.model} via Ollama")
        else:
            st.warning("Ollama is not running. Answers will show the best matching passage only.")
        retrieval = "ChromaDB embeddings" if pipeline.store.using_chromadb else "keyword search"
        st.caption(f"Retrieval: {retrieval}")

        if st.session_state.chat_history and st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])

    question = st.chat_input("Ask a question about your documents")
    if question:
        if not len(pipeline.store):
            st.warning("Please upload a PDF first.")
            return
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = pipeline.ask(question)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in result["sources"]:
                        meta = s["metadata"]
                        st.markdown(f"**{meta.get('filename')} p. {meta.get('page')}** (score {s['score']:.2f})")
                        st.text(s["text"][:500])
        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()
