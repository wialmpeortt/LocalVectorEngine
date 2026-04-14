using LocalVectorEngine.Core.Indexing;
using LocalVectorEngine.Core.Interfaces;
using LocalVectorEngine.Core.Models;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// DocumentIndexer is a pure orchestrator, so it's tested with lightweight
/// in-memory fakes for the three collaborators. No ONNX or BLite needed.
/// </summary>
public class DocumentIndexerTests
{
    // ─── Constructor ───────────────────────────────────────────────────────

    [Fact]
    public void Constructor_throws_on_null_chunker()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new DocumentIndexer(null!, new FakeEmbedder(), new FakeStore()));
    }

    [Fact]
    public void Constructor_throws_on_null_embedder()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new DocumentIndexer(new FakeChunker(), null!, new FakeStore()));
    }

    [Fact]
    public void Constructor_throws_on_null_store()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), null!));
    }

    // ─── IndexDocumentAsync — argument validation ──────────────────────────

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public async Task IndexDocumentAsync_throws_on_null_or_whitespace_documentId(string? id)
    {
        var indexer = new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), new FakeStore());

        // ArgumentException.ThrowIfNullOrWhiteSpace throws ArgumentNullException
        // for null and ArgumentException for whitespace — both derive from ArgumentException.
        await Assert.ThrowsAnyAsync<ArgumentException>(() =>
            indexer.IndexDocumentAsync(id!, "text", "src"));
    }

    [Fact]
    public async Task IndexDocumentAsync_throws_on_null_text()
    {
        var indexer = new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), new FakeStore());

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            indexer.IndexDocumentAsync("doc", null!, "src"));
    }

    [Fact]
    public async Task IndexDocumentAsync_throws_on_null_source()
    {
        var indexer = new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), new FakeStore());

        await Assert.ThrowsAsync<ArgumentNullException>(() =>
            indexer.IndexDocumentAsync("doc", "text", null!));
    }

    // ─── Happy path ────────────────────────────────────────────────────────

    [Fact]
    public async Task IndexDocumentAsync_embeds_and_stores_every_chunk_in_order()
    {
        var chunks = new[]
        {
            new DocumentChunk("doc", 0, "first",  "src"),
            new DocumentChunk("doc", 1, "second", "src"),
            new DocumentChunk("doc", 2, "third",  "src"),
        };
        var chunker  = new FakeChunker(chunks);
        var embedder = new FakeEmbedder();
        var store    = new FakeStore();
        var indexer  = new DocumentIndexer(chunker, embedder, store);

        var count = await indexer.IndexDocumentAsync("doc", "irrelevant", "src");

        Assert.Equal(3, count);
        Assert.Equal(3, embedder.CallCount);
        Assert.Equal(3, store.Stored.Count);

        // Chunks must be stored in the exact order produced by the chunker.
        Assert.Equal(new[] { 0, 1, 2 }, store.Stored.Select(p => p.Chunk.ChunkIndex));
        Assert.Equal(new[] { "first", "second", "third" }, store.Stored.Select(p => p.Chunk.Text));
    }

    [Fact]
    public async Task IndexDocumentAsync_returns_zero_when_chunker_yields_nothing()
    {
        var chunker  = new FakeChunker(Array.Empty<DocumentChunk>());
        var embedder = new FakeEmbedder();
        var store    = new FakeStore();
        var indexer  = new DocumentIndexer(chunker, embedder, store);

        var count = await indexer.IndexDocumentAsync("empty-doc", "", "src");

        Assert.Equal(0, count);
        Assert.Equal(0, embedder.CallCount);
        Assert.Empty(store.Stored);
    }

    [Fact]
    public async Task IndexDocumentAsync_pairs_each_chunk_with_its_own_embedding()
    {
        // Fake embedder returns a vector whose first element encodes the input length;
        // this lets us verify the indexer passes the right chunk text to the embedder.
        var chunks = new[]
        {
            new DocumentChunk("doc", 0, "a",   "src"),
            new DocumentChunk("doc", 1, "bb",  "src"),
            new DocumentChunk("doc", 2, "ccc", "src"),
        };
        var chunker  = new FakeChunker(chunks);
        var embedder = new FakeEmbedder(text => new float[] { text.Length, 0f, 0f });
        var store    = new FakeStore();
        var indexer  = new DocumentIndexer(chunker, embedder, store);

        await indexer.IndexDocumentAsync("doc", "irrelevant", "src");

        Assert.Equal(1f, store.Stored[0].Embedding[0]);
        Assert.Equal(2f, store.Stored[1].Embedding[0]);
        Assert.Equal(3f, store.Stored[2].Embedding[0]);
    }

    // ─── Cancellation ──────────────────────────────────────────────────────

    [Fact]
    public async Task IndexDocumentAsync_respects_pre_cancelled_token()
    {
        var indexer = new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), new FakeStore());
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            indexer.IndexDocumentAsync("doc", "text", "src", cts.Token));
    }

    [Fact]
    public async Task IndexDocumentAsync_stops_when_token_is_cancelled_mid_stream()
    {
        using var cts = new CancellationTokenSource();
        var chunks = new[]
        {
            new DocumentChunk("doc", 0, "first",  "src"),
            new DocumentChunk("doc", 1, "second", "src"),
            new DocumentChunk("doc", 2, "third",  "src"),
        };

        var store = new FakeStore();
        // Cancel during the first StoreChunkAsync call (before the chunk is added
        // to Stored). The first chunk still completes its store — cancellation
        // only sets a flag — but the indexer's next ct.ThrowIfCancellationRequested
        // on the next loop iteration will abort.
        store.OnStore = _ =>
        {
            if (store.Stored.Count == 0)
                cts.Cancel();
        };

        var indexer = new DocumentIndexer(new FakeChunker(chunks), new FakeEmbedder(), store);

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            indexer.IndexDocumentAsync("doc", "text", "src", cts.Token));

        Assert.Single(store.Stored); // only the first chunk made it in
    }

    // ─── Exception propagation ─────────────────────────────────────────────

    [Fact]
    public async Task IndexDocumentAsync_propagates_embedder_failure()
    {
        var chunks = new[] { new DocumentChunk("doc", 0, "t", "src") };
        var embedder = new FakeEmbedder(_ => throw new InvalidOperationException("boom"));
        var indexer = new DocumentIndexer(new FakeChunker(chunks), embedder, new FakeStore());

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            indexer.IndexDocumentAsync("doc", "text", "src"));
        Assert.Equal("boom", ex.Message);
    }

    [Fact]
    public async Task IndexDocumentAsync_propagates_store_failure()
    {
        var chunks = new[] { new DocumentChunk("doc", 0, "t", "src") };
        var store = new FakeStore { OnStore = _ => throw new InvalidOperationException("store kaput") };
        var indexer = new DocumentIndexer(new FakeChunker(chunks), new FakeEmbedder(), store);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            indexer.IndexDocumentAsync("doc", "text", "src"));
        Assert.Equal("store kaput", ex.Message);
    }

    // ─── IndexFileAsync ────────────────────────────────────────────────────

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public async Task IndexFileAsync_throws_on_null_or_whitespace_path(string? path)
    {
        var indexer = new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), new FakeStore());

        // Same caveat as IndexDocumentAsync: null yields ArgumentNullException,
        // whitespace yields ArgumentException; both derive from ArgumentException.
        await Assert.ThrowsAnyAsync<ArgumentException>(() => indexer.IndexFileAsync(path!));
    }

    [Fact]
    public async Task IndexFileAsync_throws_when_file_is_missing()
    {
        var indexer = new DocumentIndexer(new FakeChunker(), new FakeEmbedder(), new FakeStore());
        var nowhere = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".txt");

        await Assert.ThrowsAsync<FileNotFoundException>(() => indexer.IndexFileAsync(nowhere));
    }

    [Fact]
    public async Task IndexFileAsync_reads_file_and_indexes_it()
    {
        // The fake chunker just forwards the whole text as a single chunk,
        // which is enough to verify the file was actually loaded.
        var chunker  = new PassthroughChunker();
        var embedder = new FakeEmbedder();
        var store    = new FakeStore();
        var indexer  = new DocumentIndexer(chunker, embedder, store);

        var path = Path.Combine(Path.GetTempPath(), $"lve-indexer-{Guid.NewGuid():N}.txt");
        const string content = "hello from disk";
        await File.WriteAllTextAsync(path, content);

        try
        {
            var count = await indexer.IndexFileAsync(path);

            Assert.Equal(1, count);
            var stored = Assert.Single(store.Stored);
            Assert.Equal(content, stored.Chunk.Text);
            Assert.Equal(Path.GetFileNameWithoutExtension(path), stored.Chunk.DocumentId);
            Assert.Equal(Path.GetFullPath(path), stored.Chunk.Source);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Fakes
    // ═══════════════════════════════════════════════════════════════════════

    private sealed class FakeChunker : IChunkingService
    {
        private readonly IReadOnlyList<DocumentChunk> _chunks;

        public FakeChunker() : this(Array.Empty<DocumentChunk>()) { }

        public FakeChunker(IReadOnlyList<DocumentChunk> chunks) => _chunks = chunks;

        public IEnumerable<DocumentChunk> Chunk(string documentId, string text, string source)
            => _chunks;
    }

    /// <summary>Chunker that yields the full text as a single chunk.</summary>
    private sealed class PassthroughChunker : IChunkingService
    {
        public IEnumerable<DocumentChunk> Chunk(string documentId, string text, string source)
        {
            yield return new DocumentChunk(documentId, 0, text, source);
        }
    }

    private sealed class FakeEmbedder : IEmbeddingService
    {
        private readonly Func<string, float[]> _factory;
        public int CallCount { get; private set; }

        public FakeEmbedder() : this(_ => new float[] { 1f, 0f, 0f }) { }

        public FakeEmbedder(Func<string, float[]> factory) => _factory = factory;

        public Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
        {
            CallCount++;
            ct.ThrowIfCancellationRequested();
            return Task.FromResult(_factory(text));
        }
    }

    private sealed class FakeStore : IVectorStore
    {
        public List<(DocumentChunk Chunk, float[] Embedding)> Stored { get; } = new();

        /// <summary>Optional hook invoked just before a chunk is added to <see cref="Stored"/>.</summary>
        public Action<DocumentChunk>? OnStore { get; set; }

        public Task StoreChunkAsync(DocumentChunk chunk, float[] embedding, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            OnStore?.Invoke(chunk);
            Stored.Add((chunk, embedding));
            return Task.CompletedTask;
        }

        public Task<IReadOnlyList<SearchResult>> SearchAsync(float[] queryEmbedding, int topK, CancellationToken ct = default)
            => Task.FromResult<IReadOnlyList<SearchResult>>(Array.Empty<SearchResult>());
    }
}
