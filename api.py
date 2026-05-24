"""
RAGMind — FastAPI Server
========================
Run:  py -3.11 -m uvicorn backend.api:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Literal, Optional
from pathlib import Path
import traceback

from rag_engine import RAGPipeline, pdf_to_documents, txt_to_documents

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "RAGMind API",
    description = "Production RAG — Hybrid Semantic+BM25 retrieval · FAISS+Chroma · Mistral via Ollama",
    version     = "1.0.0",
    docs_url    = "/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
frontend_dir = Path(".")

rag: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup():
    global rag
    print("[API] Starting RAGMind...")
    rag = RAGPipeline()
    print("[API] Server ready ✓")


# ── Request / Response models ──────────────────────────────────────────────────

class IngestTextRequest(BaseModel):
    text:   str = Field(..., description="Raw text to ingest")
    source: str = Field("manual_input", description="Label for this document")

class QueryRequest(BaseModel):
    question: str   = Field(..., description="Question to ask")
    mode:     Literal["semantic", "bm25", "hybrid"] = Field("hybrid", description="Retrieval mode")
    top_k:    int   = Field(6, ge=1, le=20, description="Number of chunks to retrieve")


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the frontend UI."""
    index = Path("index.html")
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "RAGMind API running. Place index.html in /frontend folder."})


@app.get("/health", summary="Health check")
def health():
    s = rag.stats() if rag else {}
    return {
        "status":  "ok",
        "chunks":  s.get("total_chunks", 0),
        "sources": s.get("total_sources", 0),
        "model":   s.get("llm_model", "unknown"),
    }


@app.post("/ingest/text", summary="Ingest raw text")
def ingest_text(req: IngestTextRequest):
    """Chunk, embed, and index raw text into the knowledge base."""
    if not req.text.strip():
        raise HTTPException(400, "Text cannot be empty.")
    docs   = txt_to_documents(req.text, req.source)
    result = rag.ingest(docs)
    if result["status"] != "ok":
        raise HTTPException(422, result["message"])
    return result


@app.post("/ingest/file", summary="Upload PDF or TXT file")
async def ingest_file(file: UploadFile = File(...)):
    """
    Upload a PDF or TXT file.
    PDF: extracted page-by-page with page number metadata.
    TXT: ingested as a single document.
    """
    allowed = (".pdf", ".txt")
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(400, f"Only {allowed} files are supported.")

    contents = await file.read()

    try:
        if file.filename.lower().endswith(".pdf"):
            docs = pdf_to_documents(contents, file.filename)
        else:
            docs = txt_to_documents(contents.decode("utf-8", errors="ignore"), file.filename)
    except Exception as e:
        raise HTTPException(422, f"Failed to parse file: {str(e)}")

    if not docs:
        raise HTTPException(422, "No readable content found in the file.")

    result = rag.ingest(docs)
    if result["status"] != "ok":
        raise HTTPException(422, result["message"])
    return result


@app.post("/query", summary="Query the knowledge base")
def query(req: QueryRequest):
    """
    Ask a question against the ingested documents.

    Modes:
    - **semantic**: Dense vector similarity (FAISS cosine)
    - **bm25**: Sparse keyword search
    - **hybrid**: Reciprocal Rank Fusion of both (recommended)
    """
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        return rag.query(req.question, mode=req.mode, top_k=req.top_k)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Generation error: {str(e)}")


@app.get("/stats", summary="Knowledge base statistics")
def stats():
    """Return stats about the current knowledge base."""
    return rag.stats()


@app.delete("/reset", summary="Clear knowledge base")
def reset():
    """Wipe all ingested documents and vector indexes."""
    return rag.reset()
