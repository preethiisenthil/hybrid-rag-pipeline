"""
RAGMind — Production RAG Engine
================================
Stack:
  • PDF/TXT ingestion         — LangChain PyPDFLoader (page-level metadata)
  • Chunking                  — LangChain RecursiveCharacterTextSplitter
  • Embeddings                — HuggingFace all-MiniLM-L6-v2 (local, free)
  • Vector stores             — FAISS (fast exact search) + ChromaDB (persistent)
  • Sparse retrieval          — BM25 (keyword frequency)
  • Hybrid fusion             — Reciprocal Rank Fusion (RRF)
  • Generation                — Ollama Mistral (local, free, Apache 2.0)
  • Orchestration             — LangChain LCEL chains
"""

import os, time, hashlib, pickle, shutil, tempfile
from pathlib import Path
from typing import Literal

# LangChain PDF loader (uses pypdf under the hood — pure Python, no C deps)
from langchain_community.document_loaders import PyPDFLoader

# LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Embeddings — local HuggingFace model (no API key needed)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector stores
from langchain_community.vectorstores import FAISS, Chroma

# BM25 sparse retriever
from langchain_community.retrievers import BM25Retriever

# LLM — Ollama (local)
from langchain_ollama import OllamaLLM


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration  (override via .env or environment variables)
# ══════════════════════════════════════════════════════════════════════════════

EMBEDDING_MODEL  = "all-MiniLM-L6-v2"          # 384-dim, 22MB, runs on CPU
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",    "mistral")
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE       = int(os.getenv("CHUNK_SIZE",    "600"))
CHUNK_OVERLAP    = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K            = int(os.getenv("TOP_K",         "6"))

FAISS_PATH   = Path("data/faiss_index")
CHROMA_PATH  = Path("data/chroma_db")
DOCS_CACHE   = Path("data/chunks.pkl")


# ══════════════════════════════════════════════════════════════════════════════
#  Document extraction
# ══════════════════════════════════════════════════════════════════════════════

def pdf_to_documents(pdf_bytes: bytes, filename: str) -> list[Document]:
    """
    Extract text page-by-page using LangChain's PyPDFLoader.

    PyPDFLoader needs a file path so we write the uploaded bytes to a
    temporary file, load it, then delete the temp file immediately.

    Each page becomes one Document with metadata:
      source, page (1-indexed), doc_type, file_id
    """
    # Write bytes → temp file (PyPDFLoader requires a path, not bytes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages  = loader.load()          # returns list[Document], one per page
    finally:
        os.unlink(tmp_path)             # always clean up temp file

    # Enrich metadata — PyPDFLoader uses 0-indexed pages, make 1-indexed
    docs = []
    for page in pages:
        if len(page.page_content.strip()) > 40:   # skip blank / header-only pages
            page.metadata["source"]   = filename
            page.metadata["page"]     = page.metadata.get("page", 0) + 1
            page.metadata["doc_type"] = "pdf"
            page.metadata["file_id"]  = hashlib.md5(filename.encode()).hexdigest()[:8]
            docs.append(page)

    return docs


def txt_to_documents(content: str, filename: str) -> list[Document]:
    return [Document(
        page_content=content,
        metadata={"source": filename, "page": 1, "doc_type": "txt",
                  "file_id": hashlib.md5(filename.encode()).hexdigest()[:8]}
    )]


# ══════════════════════════════════════════════════════════════════════════════
#  Chunking
# ══════════════════════════════════════════════════════════════════════════════

def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    RecursiveCharacterTextSplitter tries splits in order:
      paragraph → sentence → word → character
    This preserves semantic boundaries far better than fixed-size splitting.

    Overlap (100 chars) ensures a sentence straddling two chunks
    appears in full in at least one of them.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_idx"] = i
        chunk.metadata["chunk_id"]  = hashlib.md5(
            f"{chunk.metadata['source']}_{i}_{chunk.page_content[:30]}".encode()
        ).hexdigest()[:12]

    return chunks


# ══════════════════════════════════════════════════════════════════════════════
#  RAG Pipeline
# ══════════════════════════════════════════════════════════════════════════════

class RAGPipeline:
    """
    Production-grade RAG pipeline.

    Retrieval modes
    ───────────────
    semantic  — FAISS dense vector search (cosine similarity)
                Best for: paraphrased questions, conceptual similarity
    bm25      — BM25 sparse keyword search (TF-IDF variant)
                Best for: exact terms, product codes, proper nouns
    hybrid    — Reciprocal Rank Fusion of semantic + BM25
                Best for: general use — catches what either alone misses

    Hallucination reduction
    ───────────────────────
    1. System prompt forbids answering outside the context
    2. Low temperature (0.1) keeps generation deterministic
    3. Hybrid retrieval surfaces more relevant context
    4. top_k=6 gives the LLM enough context without noise
    """

    def __init__(self):
        print("[RAG] Loading embedding model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},  # unit vectors → cosine similarity
        )

        print(f"[RAG] Connecting to Ollama ({OLLAMA_MODEL} @ {OLLAMA_BASE_URL})...")
        self.llm = OllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.1,          # near-deterministic — reduces hallucination
            num_predict=1024,         # max output tokens
        )

        self.faiss_store:  FAISS  | None = None
        self.chroma_store: Chroma | None = None
        self.all_chunks:   list[Document] = []

        Path("data").mkdir(exist_ok=True)
        self._load_from_disk()
        print("[RAG] Pipeline ready ✓")

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load_from_disk(self):
        if FAISS_PATH.exists():
            self.faiss_store = FAISS.load_local(
                str(FAISS_PATH), self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[RAG] FAISS loaded ({self.faiss_store.index.ntotal} vectors)")

        if CHROMA_PATH.exists():
            self.chroma_store = Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=self.embeddings,
            )
            print(f"[RAG] ChromaDB loaded")

        if DOCS_CACHE.exists():
            with open(DOCS_CACHE, "rb") as f:
                self.all_chunks = pickle.load(f)
            print(f"[RAG] {len(self.all_chunks)} chunks loaded from cache")

    def _persist(self):
        if self.faiss_store:
            self.faiss_store.save_local(str(FAISS_PATH))
        with open(DOCS_CACHE, "wb") as f:
            pickle.dump(self.all_chunks, f)

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def ingest(self, raw_docs: list[Document]) -> dict:
        t0     = time.time()
        chunks = chunk_documents(raw_docs)

        if not chunks:
            return {"status": "error", "message": "No usable content extracted from document."}

        # ── FAISS: dense semantic index ────────────────────────────────────────
        if self.faiss_store is None:
            self.faiss_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.faiss_store.add_documents(chunks)

        # ── ChromaDB: persistent vector store ─────────────────────────────────
        if self.chroma_store is None:
            self.chroma_store = Chroma.from_documents(
                chunks, self.embeddings,
                persist_directory=str(CHROMA_PATH),
            )
        else:
            self.chroma_store.add_documents(chunks)

        self.all_chunks.extend(chunks)
        self._persist()

        sources = list({c.metadata["source"] for c in chunks})
        return {
            "status":        "ok",
            "chunks_added":  len(chunks),
            "total_chunks":  len(self.all_chunks),
            "sources":       sources,
            "elapsed_s":     round(time.time() - t0, 2),
        }

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def _semantic(self, query: str, k: int) -> list[Document]:
        """FAISS cosine similarity search on normalised embeddings."""
        if not self.faiss_store:
            return []
        return self.faiss_store.similarity_search(query, k=k)

    def _bm25(self, query: str, k: int) -> list[Document]:
        """
        BM25 (Best Match 25) — probabilistic keyword relevance model.
        Scores documents by term frequency, penalising very long documents.
        Good at: exact matches, rare terms, codes, names.
        """
        if not self.all_chunks:
            return []
        retriever = BM25Retriever.from_documents(self.all_chunks, k=k)
        return retriever.invoke(query)

    def _hybrid_rrf(self, query: str, k: int) -> list[Document]:
        """
        Reciprocal Rank Fusion:  score(d) = Σ  1 / (rank(d) + 60)

        Why rank-based instead of score-based?
        FAISS scores (cosine 0-1) and BM25 scores are on completely
        different scales — you can't average them directly.
        RRF uses only the *rank position*, which is scale-invariant.
        Constant 60 prevents rank-1 from dominating too strongly.
        """
        sem_docs  = self._semantic(query, k=k)
        bm25_docs = self._bm25(query, k=k)

        scores:  dict[str, float]    = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(sem_docs):
            key = doc.page_content[:120]
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rank + 60)
            doc_map[key] = doc

        for rank, doc in enumerate(bm25_docs):
            key = doc.page_content[:120]
            scores[key]  = scores.get(key, 0.0) + 1.0 / (rank + 60)
            doc_map[key] = doc

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[k] for k, _ in ranked[:k]]

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(
        self,
        question:  str,
        mode:      Literal["semantic", "bm25", "hybrid"] = "hybrid",
        top_k:     int = TOP_K,
    ) -> dict:
        t0 = time.time()

        if not self.all_chunks:
            return {
                "answer":    "No documents in the knowledge base. Please upload a PDF or paste some text first.",
                "sources":   [],
                "chunks":    [],
                "mode":      mode,
                "latency_s": 0,
            }

        # 1. Retrieve
        if mode == "semantic":
            docs = self._semantic(question, k=top_k)
        elif mode == "bm25":
            docs = self._bm25(question, k=top_k)
        else:
            docs = self._hybrid_rrf(question, k=top_k)

        if not docs:
            return {"answer": "No relevant context found.", "sources": [], "chunks": [], "mode": mode, "latency_s": 0}

        # 2. Build context block with source citations
        context_parts = []
        for i, doc in enumerate(docs, 1):
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page",   "?")
            context_parts.append(
                f"[{i}] Source: {src}, Page {page}\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        # 3. Prompt — engineered to reduce hallucination
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a precise document assistant. Follow these rules strictly:
1. Answer ONLY using information from the provided context below.
2. If the answer is not in the context, say: "I couldn't find this in the uploaded documents."
3. Always cite your sources using the format [1], [2] etc. matching the context numbers.
4. Be concise and accurate.

Context:
{context}

Question: {question}

Answer:"""
        )

        # 4. Generate via LangChain LCEL chain
        chain  = prompt | self.llm
        answer = chain.invoke({"context": context, "question": question})

        return {
            "answer":    str(answer).strip(),
            "sources":   list({d.metadata.get("source", "?") for d in docs}),
            "chunks":    [
                {
                    "text":   d.page_content[:350],
                    "source": d.metadata.get("source", "?"),
                    "page":   d.metadata.get("page",   "?"),
                    "idx":    i + 1,
                }
                for i, d in enumerate(docs)
            ],
            "mode":      mode,
            "latency_s": round(time.time() - t0, 2),
        }

    # ── Stats ──────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        sources: dict[str, int] = {}
        for c in self.all_chunks:
            s = c.metadata.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1

        faiss_count = self.faiss_store.index.ntotal if self.faiss_store else 0

        return {
            "total_chunks":    len(self.all_chunks),
            "total_sources":   len(sources),
            "sources":         sources,
            "faiss_vectors":   faiss_count,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim":   384,
            "llm_model":       OLLAMA_MODEL,
            "chunk_size":      CHUNK_SIZE,
            "chunk_overlap":   CHUNK_OVERLAP,
            "top_k":           TOP_K,
            "vector_stores":   ["FAISS", "ChromaDB"],
            "retrieval_modes": ["semantic", "bm25", "hybrid"],
        }

    def reset(self) -> dict:
        self.faiss_store  = None
        self.chroma_store = None
        self.all_chunks   = []
        for p in [FAISS_PATH, CHROMA_PATH, DOCS_CACHE]:
            try:
                shutil.rmtree(p) if p.is_dir() else p.unlink()
            except FileNotFoundError:
                pass
        return {"status": "reset", "message": "Knowledge base cleared."}
