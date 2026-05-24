# RAGMind — Production RAG System

> Hybrid Retrieval-Augmented Generation · Mistral via Ollama · FAISS + ChromaDB · BM25 · LangChain · FastAPI

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **PDF Parsing** | PyMuPDF | Fast, page-level metadata |
| **Chunking** | LangChain RecursiveCharacterTextSplitter | Respects sentence/paragraph boundaries |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Local, free, 384-dim, runs on CPU |
| **Vector Store 1** | FAISS IndexFlatIP | Exact cosine similarity, fast in-process search |
| **Vector Store 2** | ChromaDB | Persistent storage, survives restarts |
| **Sparse Retrieval** | BM25 (rank-bm25) | Keyword frequency, exact term matching |
| **Hybrid Fusion** | Reciprocal Rank Fusion (RRF) | Scale-invariant rank combination |
| **LLM** | Mistral 7B via Ollama | Local, free, Apache 2.0, fast on CPU |
| **Orchestration** | LangChain LCEL | Composable chains, prompt management |
| **API** | FastAPI | Auto OpenAPI docs, async, fast |

---

## Setup

### Prerequisites
1. **Python 3.11** — https://python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
2. **Ollama** — https://ollama.com

### Install Mistral
```powershell
ollama pull mistral
ollama run mistral "Hello!"    # test it works
```

### Install Python packages
```powershell
py -3.11 -m pip install -r requirements.txt
```

### Run
```powershell
# Option A: Double-click start.bat

# Option B: Manual (2 terminals)
# Terminal 1:
ollama serve

# Terminal 2:
py -3.11 -m uvicorn backend.api:app --reload --port 8000
```

Open: **http://127.0.0.1:8000**
API docs: **http://127.0.0.1:8000/docs**

---

## Project Structure

```
ragmind/
├── backend/
│   ├── __init__.py
│   ├── rag_engine.py     ← Core pipeline (ingestion, retrieval, generation)
│   └── api.py            ← FastAPI REST server
├── frontend/
│   └── index.html        ← Full UI (drag-drop PDF, query, results)
├── data/                 ← Created automatically
│   ├── faiss_index/      ← FAISS binary index
│   ├── chroma_db/        ← ChromaDB persistent store
│   └── chunks.pkl        ← Document chunk cache
├── requirements.txt
├── start.bat
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Frontend UI |
| GET | `/health` | Server + model status |
| POST | `/ingest/text` | Ingest raw text |
| POST | `/ingest/file` | Upload PDF or TXT |
| POST | `/query` | Ask a question |
| GET | `/stats` | Knowledge base statistics |
| DELETE | `/reset` | Clear all data |

### Query request
```json
{
  "question": "What are the key findings?",
  "mode": "hybrid",
  "top_k": 6
}
```

---

## How Hybrid Retrieval Works

```
User Query
    │
    ├─── Embed with MiniLM ──→ FAISS cosine search ──→ Top-K semantic docs
    │                                                          │
    └─── Tokenise ───────────→ BM25 keyword search ──→ Top-K keyword docs
                                                               │
                                              Reciprocal Rank Fusion (RRF)
                                                               │
                                              score(d) = Σ 1/(rank + 60)
                                                               │
                                              Top-K fused docs → Mistral
                                                               │
                                                          Final Answer
```

**Why RRF instead of score averaging?**
FAISS returns cosine scores (0–1). BM25 returns TF-IDF scores (unbounded). They're incomparable. RRF uses only rank positions, which are scale-invariant — so it correctly combines both without bias.

---

## Interview Talking Points

**Q: Why use both FAISS and ChromaDB?**
FAISS is in-process (fastest possible search, ~1ms) but volatile — it's lost on restart. ChromaDB persists to disk automatically. Using both gives speed and durability.

**Q: When does BM25 beat semantic search?**
When the query contains exact terms: product codes (SKU-4821), proper nouns (specific person names), technical abbreviations (API, GDPR). Semantic search would match concepts but miss the exact string.

**Q: How does chunking strategy affect quality?**
Fixed-size chunking mid-sentence creates incoherent chunks that confuse retrieval. RecursiveCharacterTextSplitter tries paragraph → sentence → word splits in order, so boundaries are always at natural text breaks. Overlap (100 chars) ensures sentences straddling boundaries appear in at least one full chunk.

**Q: How did you reduce hallucinations?**
Three ways: (1) system prompt forbids answering outside context, (2) temperature=0.1 keeps generation near-deterministic, (3) hybrid retrieval maximises relevant context quality so the LLM has accurate grounding material.

**Q: Why Mistral over GPT-4?**
Mistral runs 100% locally — zero API cost, zero data privacy risk, no internet dependency. For a production system handling sensitive documents (legal, medical, financial), local inference is often a hard requirement.
