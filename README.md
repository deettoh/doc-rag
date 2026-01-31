# DocRAG

**RAG-based PDF Summarizer + QnA Generator**

A portfolio-grade AI system demonstrating:
- Retrieval-Augmented Generation (RAG)
- Backend API design with FastAPI
- Vector search using PostgreSQL + pgvector
- LLM integration with strict boundaries
- Production-aligned engineering practices

## Features

- **PDF Upload**: Upload documents for processing
- **Document Summarization**: Generate concise summaries with page citations
- **QnA Generation**: Auto-generate questions from document content
- **Answer Evaluation**: Submit answers and receive AI-powered feedback
- **Document Chat**: Interactive Q&A with your documents

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.11, FastAPI, Uvicorn |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Database | PostgreSQL 15 + pgvector |
| PDF Processing | PyMuPDF |
| Frontend | Streamlit |
| Deployment | Fly.io, Streamlit Community Cloud |

## Getting Started

```bash
# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run locally
poetry run uvicorn app.main:app --reload
```

## License

MIT
