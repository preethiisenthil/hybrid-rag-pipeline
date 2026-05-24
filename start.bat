@echo off
title RAGMind — Production RAG
color 0F
echo.
echo  ╔══════════════════════════════════════╗
echo  ║   RAGMind — Production RAG System   ║
echo  ║   Mistral + FAISS + Chroma + BM25   ║
echo  ╚══════════════════════════════════════╝
echo.

cd /d "%~dp0"

echo [1/3] Starting Ollama (Mistral)...
start "" ollama serve
timeout /t 3 /nobreak >nul

echo [2/3] Starting FastAPI server...
start "" cmd /k "cd /d %~dp0 && py -3.11 -m uvicorn backend.api:app --reload --port 8000"
timeout /t 5 /nobreak >nul

echo [3/3] Opening RAGMind in browser...
start "" http://127.0.0.1:8000

echo.
echo  ✓ RAGMind is running at http://127.0.0.1:8000
echo  ✓ API docs at         http://127.0.0.1:8000/docs
echo.
echo  Press any key to exit this window (server keeps running)
pause >nul
