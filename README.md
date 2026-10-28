# PDF Q&A System using RAG

![CI](https://github.com/SanfordJone/PDF-Based-QnA-System-using-RAG/actions/workflows/ci.yml/badge.svg)

Ask questions about your own PDF files and get answers that come only from those documents, with page numbers. The app uses **retrieval augmented generation (RAG)**: it finds the most relevant passages in your PDFs, then asks a local LLM (Llama 3.1 through [Ollama](https://ollama.com)) to answer from those passages.

Everything runs locally. Your documents are not sent to any cloud service.

## How it works

```
PDF upload ─► extract text per page ─► split into 1,000 char chunks (200 char overlap)
                                                │
                                                ▼
Question ─► find top 4 chunks (ChromaDB embeddings, cosine similarity) ─► prompt with
            excerpts + page numbers ─► Llama 3.1 via Ollama ─► answer with citations
```

| File | Role |
|---|---|
| `app.py` | Streamlit interface: upload PDFs, chat, see sources |
| `pdf_processor.py` | Validates uploads, extracts text per page, chunks with overlap |
| `vector_store.py` | Stores chunks and retrieves the most relevant ones (ChromaDB, with keyword fallback) |
| `rag.py` | The pipeline: retrieval, prompt building and answer generation |
| `ollama_client.py` | Talks to the Ollama API |
| `tests/` | pytest suite (29 tests) |

## Run it

**Requirements:** Python 3.11+ and [Ollama](https://ollama.com/download).

```bash
ollama pull llama3.1            # one time download of the model
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501, upload a PDF from the sidebar and ask a question. Each answer has a **Sources** section showing the file, page and similarity score of every passage used.

The first upload downloads ChromaDB's small embedding model. If that is not possible (for example offline), the app switches to keyword search automatically.

### With Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1   # first time only
```

### Settings

| Environment variable | Default |
|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `llama3.1` |

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests build small PDFs in memory and use a fake LLM, so they need no Ollama or internet access. GitHub Actions runs them on Python 3.11 and 3.12 for every push and also checks the Docker image builds.

## Security and privacy

I reviewed the app against the [OWASP Top 10 for LLM Applications](https://genai.owasp.org/llm-top-10/):

- **Prompt injection (LLM01):** a PDF can contain text like "ignore previous instructions". Retrieved text is wrapped in `<context>` tags, any such tags inside the document are removed, and the system prompt tells the model to treat excerpts as untrusted data. There is a test for this.
- **Sensitive information disclosure (LLM02):** each browser session has its own in-memory vector store, so one user's documents never appear in another user's answers, and uploads are not saved to disk. The model runs locally, so documents do not leave the machine.
- **Unbounded consumption (LLM10):** uploads are limited to 20 MB and 300 pages, password protected and non-PDF files are rejected, and every request to Ollama has a timeout.
- **Misinformation (LLM09):** the model is told to answer only from the excerpts and cite pages, and sources are shown so the user can check. When Ollama is offline the app says so instead of presenting a passage as a model answer.
- No API keys are needed. `.env` and local data are in `.gitignore` and `.dockerignore`.

Only upload documents you have the right to use.

## Limitations and next steps

- Scanned PDFs (images only) have no text to extract. OCR would fix this.
- Answer quality has not been measured yet. Next step: a small question and answer evaluation set to track retrieval hit rate and answer accuracy.
- Not deployed publicly yet, because it needs a machine that can run Ollama.

## History

First version built in April to May 2025. Reworked in October 2026 with help from an AI coding assistant (Claude): fixed the chunk overlap bug and the Ollama client, connected the app to a real RAG pipeline (the earlier version only used keyword matching), and added input validation, prompt injection defences, tests, CI and Docker. See the commit history for each change.
