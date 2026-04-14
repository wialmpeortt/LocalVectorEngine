# LocalVectorEngine — Local Vector Search Engine for RAG

**Course:** Software Engineering — H-Farm College
**Author:** William
**Project:** Zucchetti-7 — local vector search engine consumed by the *AI Document Q&A (RAG)* and *BLite Mobile + AI* projects
**Timeline:** 4 weeks (Agile sprint)

## Problem

LLMs have a knowledge cut-off and don't know about *your* documents. The **RAG** (Retrieval-Augmented Generation) pattern solves this in two phases: first you index the documents by turning them into embedding vectors; then, for every question, you retrieve the most similar chunks and pass them to the LLM as grounded context.

This repository provides **the vector search engine** — the "retrieval" half of RAG — as a reusable library. It is designed to be consumed by two separate projects:

- *AI Document Q&A* — a RAG application that answers questions over PDF/TXT/MD documents
- *BLite Mobile + AI* — a .NET MAUI app that indexes documents locally on a phone

## Architecture

The repository is organised as **one shared library** plus **one console demo** that exercises it end-to-end.

```
src/
├── LocalVectorEngine.Core/          ← public library (the "product")
│   ├── Interfaces/
│   │   ├── IChunkingService.cs      ← splits a document into chunks
│   │   ├── IEmbeddingService.cs     ← text → float[] (vector)
│   │   └── IVectorStore.cs          ← persistence + HNSW search
│   └── Models/
│       ├── DocumentChunk.cs         ← immutable chunk record
│       └── SearchResult.cs          ← chunk + similarity score
│
└── LocalVectorEngine.Demo/          ← console app that wires the pipeline
    └── Program.cs
```

### Core contracts

```csharp
public interface IChunkingService
{
    IEnumerable<DocumentChunk> Chunk(string documentId, string text, string source);
}

public interface IEmbeddingService
{
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
}

public interface IVectorStore
{
    Task StoreChunkAsync(DocumentChunk chunk, float[] embedding, CancellationToken ct = default);
    Task<IReadOnlyList<SearchResult>> SearchAsync(float[] queryEmbedding, int topK, CancellationToken ct = default);
}

public record DocumentChunk(string DocumentId, int ChunkIndex, string Text, string Source);
public record SearchResult(DocumentChunk Chunk, float Score);
```

The async methods take a `CancellationToken` to match the Zucchetti-7 spec and to make timeout/abort behaviour explicit on both server and mobile.

## RAG pipeline

```
INDEXING (run once per document)

  Source document
        │
        ▼
  IChunkingService         ── split ~512 tokens, overlap ~50
        │
        ▼
  IEmbeddingService        ── text → float[384]
        │
        ▼
  IVectorStore.Store       ── persist into BLite (HNSW)


QUERY (run for every user question)

  Question
        │
        ▼
  IEmbeddingService        ── question → float[384]
        │
        ▼
  IVectorStore.Search      ── HNSW: top-K most similar chunks
        │
        ▼
  SearchResult[] (chunk + score)  ── ready to feed an LLM as context
```

## Tech stack

| Component | Technology | Notes |
|---|---|---|
| Runtime | .NET 10 | |
| Embedding | ONNX Runtime + `all-MiniLM-L6-v2` | 384-dim, runs locally |
| Vector DB | BLite 4.3.0 | Built-in HNSW index |
| Diagrams | PlantUML | Sources in `docs/` |

No cloud calls — all retrieval runs on-device.

## Implementation status

| Component | Interface | Implementation |
|---|---|---|
| Chunking | ✅ `IChunkingService` | ⏳ Issue #21 |
| Embedding | ✅ `IEmbeddingService` | ⏳ Issue #10 — `OnnxEmbeddingService` |
| Vector store | ✅ `IVectorStore` | ⏳ Issue #11 — `BLiteVectorStore` |
| Demo console | — | ⏳ end-to-end pipeline |

## How to run

Download the ONNX model **and** its tokenizer vocabulary into `models/` (both are gitignored):

```bash
# 86 MB ONNX model
curl -L -o models/all-MiniLM-L6-v2.onnx \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx

# 232 KB BERT WordPiece vocabulary
curl -L -o models/vocab.txt \
  https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
```

Then:

```bash
# Build everything (library + demo + tests)
dotnet build

# Run the console demo
dotnet run --project src/LocalVectorEngine.Demo

# Run the test suite (model-dependent tests auto-skip if models/ is empty)
dotnet test
```

Paths can be overridden via the `LVE_MODEL_PATH` and `LVE_VOCAB_PATH` environment variables.

## Repository layout

```
Exam/
├── src/                          source code
├── tests/                        tests (in progress)
├── docs/                         UML diagrams (PlantUML + PNG)
├── models/                       ONNX model (gitignored)
├── LocalVectorEngine.slnx        solution file
└── README.md
```

## Project documentation

- `docs/class_diagram.puml` — class diagram (Core + planned implementations)
- `docs/sequence_diagram.puml` — sequence diagrams of the indexing and query flows
- The Kanban board on GitHub Projects tracks sprint-by-sprint progress.
