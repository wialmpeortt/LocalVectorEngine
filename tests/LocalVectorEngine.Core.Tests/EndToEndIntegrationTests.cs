using LocalVectorEngine.Core.Indexing;
using LocalVectorEngine.Core.Retrieval;
using LocalVectorEngine.Core.Services;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// End-to-end integration tests that wire the real services together:
/// <c>SlidingWindowChunkingService</c> → <c>OnnxEmbeddingService</c> → <c>BLiteVectorStore</c>.
///
/// These tests exercise the full RAG pipeline (indexing + retrieval) using the
/// actual ONNX model and a real BLite database. They are automatically skipped
/// when the ONNX model or vocab file is not present on disk, keeping CI green
/// on machines that don't have the 86 MB model downloaded.
/// </summary>
public sealed class EndToEndIntegrationTests : IDisposable
{
    private static readonly string RepoRoot = FindRepoRoot(AppContext.BaseDirectory);
    private static readonly string ModelPath = Path.Combine(RepoRoot, "models", "all-MiniLM-L6-v2.onnx");
    private static readonly string VocabPath = Path.Combine(RepoRoot, "models", "vocab.txt");
    private static bool ArtifactsAvailable => File.Exists(ModelPath) && File.Exists(VocabPath);

    private readonly string _dbPath = Path.Combine(
        Path.GetTempPath(), $"lve_e2e_{Guid.NewGuid():N}.db");

    public void Dispose()
    {
        if (File.Exists(_dbPath)) File.Delete(_dbPath);
    }

    // ── Indexing ────────────────────────────────────────────────────────────

    [SkippableFact]
    public async Task Index_document_stores_expected_chunk_count()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var embedder = new OnnxEmbeddingService(ModelPath, VocabPath);
        var chunker = new SlidingWindowChunkingService(chunkSizeWords: 50, overlapWords: 10);
        using var store = new BLiteVectorStore(_dbPath);
        var indexer = new DocumentIndexer(chunker, embedder, store);

        // ~30 words → fits in one chunk with chunkSize=50.
        const string text = "Retrieval-Augmented Generation is a pattern that combines retrieval " +
                            "and generation to ground LLM answers in external documents. " +
                            "It uses vector similarity search to find relevant passages.";

        int count = await indexer.IndexDocumentAsync("test-doc", text, "mem://test");

        Assert.Equal(1, count);
    }

    [SkippableFact]
    public async Task Index_long_document_produces_multiple_chunks()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var embedder = new OnnxEmbeddingService(ModelPath, VocabPath);
        var chunker = new SlidingWindowChunkingService(chunkSizeWords: 30, overlapWords: 5);
        using var store = new BLiteVectorStore(_dbPath);
        var indexer = new DocumentIndexer(chunker, embedder, store);

        // ~100 words → should produce multiple chunks with chunkSize=30.
        var words = Enumerable.Range(0, 100).Select(i => $"word{i}");
        var text = string.Join(' ', words);

        int count = await indexer.IndexDocumentAsync("big-doc", text, "mem://big");

        Assert.True(count >= 3, $"Expected at least 3 chunks from 100 words, got {count}");
    }

    // ── Retrieval ──────────────────────────────────────────────────────────

    [SkippableFact]
    public async Task Retrieve_finds_relevant_passage_after_indexing()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var embedder = new OnnxEmbeddingService(ModelPath, VocabPath);
        var chunker = new SlidingWindowChunkingService(chunkSizeWords: 50, overlapWords: 10);
        using var store = new BLiteVectorStore(_dbPath);
        var indexer   = new DocumentIndexer(chunker, embedder, store);
        var retriever = new RetrievalEngine(embedder, store);

        // Index two topically distinct passages.
        await indexer.IndexDocumentAsync("ai-doc",
            "Machine learning is a branch of artificial intelligence that enables " +
            "computers to learn patterns from data without being explicitly programmed.",
            "mem://ai");

        await indexer.IndexDocumentAsync("cooking-doc",
            "To make a perfect risotto, start by toasting the rice in butter, " +
            "then gradually add warm broth while stirring continuously.",
            "mem://cooking");

        // Ask an AI-related question.
        var results = await retriever.RetrieveAsync("What is machine learning?", topK: 2);

        Assert.NotEmpty(results);
        Assert.Equal("ai-doc", results[0].Chunk.DocumentId);
        Assert.True(results[0].Score > results[1].Score,
            "AI passage should score higher than cooking passage for an ML question.");
    }

    [SkippableFact]
    public async Task Retrieve_returns_results_sorted_by_descending_score()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var embedder = new OnnxEmbeddingService(ModelPath, VocabPath);
        var chunker = new SlidingWindowChunkingService(chunkSizeWords: 40, overlapWords: 10);
        using var store = new BLiteVectorStore(_dbPath);
        var indexer   = new DocumentIndexer(chunker, embedder, store);
        var retriever = new RetrievalEngine(embedder, store);

        // Index several passages on different topics.
        await indexer.IndexDocumentAsync("doc-a",
            "Vectors in machine learning represent data points in high-dimensional space.",
            "mem://a");
        await indexer.IndexDocumentAsync("doc-b",
            "The French Revolution began in 1789 and transformed French society.",
            "mem://b");
        await indexer.IndexDocumentAsync("doc-c",
            "Cosine similarity measures the angle between two vectors in embedding space.",
            "mem://c");

        var results = await retriever.RetrieveAsync("How does cosine similarity work?", topK: 3);

        // Scores should be in descending order.
        for (int i = 0; i < results.Count - 1; i++)
        {
            Assert.True(results[i].Score >= results[i + 1].Score,
                $"Result {i} (score={results[i].Score:F4}) should be >= result {i + 1} (score={results[i + 1].Score:F4})");
        }
    }

    [SkippableFact]
    public async Task Full_pipeline_index_file_then_query()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var embedder = new OnnxEmbeddingService(ModelPath, VocabPath);
        var chunker = new SlidingWindowChunkingService(chunkSizeWords: 50, overlapWords: 10);
        using var store = new BLiteVectorStore(_dbPath);
        var indexer   = new DocumentIndexer(chunker, embedder, store);
        var retriever = new RetrievalEngine(embedder, store);

        // Write a temp file and index it.
        var tempFile = Path.Combine(Path.GetTempPath(), $"lve_e2e_file_{Guid.NewGuid():N}.txt");
        const string content = "HNSW is a graph-based algorithm for approximate nearest-neighbour search. " +
                               "It builds a multi-layer proximity graph where each node represents a vector.";
        await File.WriteAllTextAsync(tempFile, content);

        try
        {
            int chunks = await indexer.IndexFileAsync(tempFile);
            Assert.True(chunks >= 1);

            var results = await retriever.RetrieveAsync("What is HNSW?", topK: 1);

            Assert.Single(results);
            Assert.Contains("HNSW", results[0].Chunk.Text);
        }
        finally
        {
            File.Delete(tempFile);
        }
    }

    // ── Helpers ─────────────────────────────────────────────────────────────

    private static string FindRepoRoot(string start)
    {
        var dir = new DirectoryInfo(start);
        while (dir is not null)
        {
            if (dir.GetFiles("*.slnx").Length > 0 || Directory.Exists(Path.Combine(dir.FullName, ".git")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return start;
    }
}
