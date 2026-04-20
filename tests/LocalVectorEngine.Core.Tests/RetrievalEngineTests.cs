using LocalVectorEngine.Core.Interfaces;
using LocalVectorEngine.Core.Models;
using LocalVectorEngine.Core.Retrieval;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// RetrievalEngine is a thin orchestrator (embed question → search store),
/// so it's tested with in-memory fakes — no ONNX or BLite needed.
/// </summary>
public class RetrievalEngineTests
{
    // ─── Constructor ───────────────────────────────────────────────────────

    [Fact]
    public void Constructor_throws_on_null_embedder()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new RetrievalEngine(null!, new FakeStore()));
    }

    [Fact]
    public void Constructor_throws_on_null_store()
    {
        Assert.Throws<ArgumentNullException>(() =>
            new RetrievalEngine(new FakeEmbedder(), null!));
    }

    // ─── Argument validation ───────────────────────────────────────────────

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public async Task RetrieveAsync_throws_on_null_or_whitespace_question(string? question)
    {
        var engine = new RetrievalEngine(new FakeEmbedder(), new FakeStore());

        await Assert.ThrowsAnyAsync<ArgumentException>(() =>
            engine.RetrieveAsync(question!));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-100)]
    public async Task RetrieveAsync_throws_on_non_positive_topK(int topK)
    {
        var engine = new RetrievalEngine(new FakeEmbedder(), new FakeStore());

        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
            engine.RetrieveAsync("question", topK));
    }

    // ─── Happy path ────────────────────────────────────────────────────────

    [Fact]
    public async Task RetrieveAsync_embeds_question_and_returns_store_results()
    {
        var expected = new SearchResult[]
        {
            new(new DocumentChunk("doc", 0, "passage one",   "src"), 0.95f),
            new(new DocumentChunk("doc", 1, "passage two",   "src"), 0.82f),
            new(new DocumentChunk("doc", 2, "passage three", "src"), 0.71f),
        };

        var embedder = new FakeEmbedder();
        var store    = new FakeStore(expected);
        var engine   = new RetrievalEngine(embedder, store);

        var results = await engine.RetrieveAsync("my question", topK: 3);

        Assert.Equal(1, embedder.CallCount);
        Assert.Equal(3, results.Count);
        Assert.Equal(expected, results);
    }

    [Fact]
    public async Task RetrieveAsync_passes_topK_to_store()
    {
        var store  = new FakeStore();
        var engine = new RetrievalEngine(new FakeEmbedder(), store);

        await engine.RetrieveAsync("question", topK: 7);

        Assert.Equal(7, store.LastTopK);
    }

    [Fact]
    public async Task RetrieveAsync_uses_default_topK_when_not_specified()
    {
        var store  = new FakeStore();
        var engine = new RetrievalEngine(new FakeEmbedder(), store);

        await engine.RetrieveAsync("question");

        Assert.Equal(RetrievalEngine.DefaultTopK, store.LastTopK);
    }

    [Fact]
    public async Task RetrieveAsync_passes_embedding_from_embedder_to_store()
    {
        var questionEmbedding = new float[] { 0.1f, 0.2f, 0.3f };
        var embedder = new FakeEmbedder(_ => questionEmbedding);
        var store    = new FakeStore();
        var engine   = new RetrievalEngine(embedder, store);

        await engine.RetrieveAsync("anything");

        Assert.Same(questionEmbedding, store.LastQueryEmbedding);
    }

    [Fact]
    public async Task RetrieveAsync_returns_empty_list_when_store_is_empty()
    {
        var engine = new RetrievalEngine(new FakeEmbedder(), new FakeStore());

        var results = await engine.RetrieveAsync("question");

        Assert.Empty(results);
    }

    // ─── Cancellation ──────────────────────────────────────────────────────

    [Fact]
    public async Task RetrieveAsync_respects_pre_cancelled_token()
    {
        var engine = new RetrievalEngine(new FakeEmbedder(), new FakeStore());
        using var cts = new CancellationTokenSource();
        cts.Cancel();

        await Assert.ThrowsAsync<OperationCanceledException>(() =>
            engine.RetrieveAsync("question", ct: cts.Token));
    }

    // ─── Exception propagation ─────────────────────────────────────────────

    [Fact]
    public async Task RetrieveAsync_propagates_embedder_failure()
    {
        var embedder = new FakeEmbedder(_ => throw new InvalidOperationException("embed fail"));
        var engine   = new RetrievalEngine(embedder, new FakeStore());

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            engine.RetrieveAsync("question"));
        Assert.Equal("embed fail", ex.Message);
    }

    [Fact]
    public async Task RetrieveAsync_propagates_store_failure()
    {
        var store = new FakeStore { ThrowOnSearch = new InvalidOperationException("search fail") };
        var engine = new RetrievalEngine(new FakeEmbedder(), store);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            engine.RetrieveAsync("question"));
        Assert.Equal("search fail", ex.Message);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Fakes
    // ═══════════════════════════════════════════════════════════════════════

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
        private readonly IReadOnlyList<SearchResult> _results;

        public float[]? LastQueryEmbedding { get; private set; }
        public int LastTopK { get; private set; }

        /// <summary>If set, <see cref="SearchAsync"/> throws this exception.</summary>
        public Exception? ThrowOnSearch { get; set; }

        public FakeStore() : this(Array.Empty<SearchResult>()) { }

        public FakeStore(IReadOnlyList<SearchResult> results) => _results = results;

        public Task StoreChunkAsync(DocumentChunk chunk, float[] embedding, CancellationToken ct = default)
            => Task.CompletedTask;

        public Task<IReadOnlyList<SearchResult>> SearchAsync(float[] queryEmbedding, int topK, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            if (ThrowOnSearch is not null) throw ThrowOnSearch;

            LastQueryEmbedding = queryEmbedding;
            LastTopK           = topK;

            return Task.FromResult(_results);
        }
    }
}
