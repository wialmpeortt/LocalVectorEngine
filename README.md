# AtomizeNews - Verifiable News via Claim Atomization and Local Vector Search

**Course:** Software Engineering - H-Farm College  
**Author:** William  
**Timeline:** 4 Weeks (Agile Sprint)

## Problem Statement
Current AI models struggle with hallucinations and provide summaries that miss critical nuances. This project builds an offline-first verification engine that atomizes news articles into discrete factual claims and verifies them against primary sources using local vector search.

## How it works
1. **Atomization** - A local LLM (Llama 3.2 via Ollama) extracts individual factual claims from a news article
2. **Indexing** - Primary source documents are embedded and stored in a local BLite vector database (HNSW index)
3. **Verification** - Each claim is matched against the most semantically similar source chunks

## Tech Stack
- .NET 8
- Ollama (Llama 3.2) — Local LLM
- ONNX Runtime (all-MiniLM-L6-v2) — Vector embeddings
- BLite — Embedded vector database with HNSW index

## Project Structure
