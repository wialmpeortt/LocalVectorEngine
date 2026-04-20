using LocalVectorEngine.Core.Models;
using LocalVectorEngine.Core.Services;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// Additional edge-case coverage for <see cref="BLiteVectorStore"/>.
/// </summary>
public sealed class BLiteVectorStoreEdgeCaseTests
{
    private const int Dim = BLiteVectorStore.EmbeddingDimension;

    private static float[] OneHot(int pos)
    {
        var v = new float[Dim];
        v[pos] = 1f;
        return v;
    }

    private static string TempDbPath([System.Runtime.CompilerServices.CallerMemberName] string name = "")
        => Path.Combine(Path.GetTempPath(), $"lve_blite_edge_{name}_{Guid.NewGuid():N}.db");

    [Fact]
    public async Task Search_topK_limits_results_even_when_more_are_stored()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            // Store 5 chunks in different directions.
            for (int i = 0; i < 5; i++)
                await store.StoreChunkAsync(new DocumentChunk("doc", i, $"chunk{i}", "src"), OneHot(i));

            var results = await store.SearchAsync(OneHot(0), topK: 2);

            Assert.Equal(2, results.Count);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_returns_fewer_than_topK_when_store_has_less()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);
            await store.StoreChunkAsync(new DocumentChunk("doc", 0, "only one", "src"), OneHot(0));

            var results = await store.SearchAsync(OneHot(0), topK: 10);

            Assert.Single(results);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task StoreChunk_allows_same_documentId_different_chunkIndex()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            await store.StoreChunkAsync(new DocumentChunk("doc-A", 0, "first",  "src"), OneHot(0));
            await store.StoreChunkAsync(new DocumentChunk("doc-A", 1, "second", "src"), OneHot(1));

            // Both should be searchable.
            var results = await store.SearchAsync(OneHot(1), topK: 2);
            Assert.Equal(2, results.Count);
            Assert.Equal("second", results[0].Chunk.Text); // closest
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_returns_mixed_results_from_multiple_documents()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            await store.StoreChunkAsync(new DocumentChunk("doc-A", 0, "alpha", "src-A"), OneHot(0));
            await store.StoreChunkAsync(new DocumentChunk("doc-B", 0, "beta",  "src-B"), OneHot(1));
            await store.StoreChunkAsync(new DocumentChunk("doc-C", 0, "gamma", "src-C"), OneHot(2));

            // Query a direction between OneHot(0) and OneHot(1).
            var query = new float[Dim];
            query[0] = 0.7f;
            query[1] = 0.7f;
            // Not normalized, but that's fine for testing ordering.

            var results = await store.SearchAsync(query, topK: 3);

            Assert.Equal(3, results.Count);
            // doc-A and doc-B should both appear in results.
            var docIds = results.Select(r => r.Chunk.DocumentId).ToList();
            Assert.Contains("doc-A", docIds);
            Assert.Contains("doc-B", docIds);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task StoreChunk_handles_long_text_content()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            var longText = new string('x', 10_000);
            var chunk = new DocumentChunk("big", 0, longText, "src");
            await store.StoreChunkAsync(chunk, OneHot(0));

            var results = await store.SearchAsync(OneHot(0), topK: 1);
            Assert.Single(results);
            Assert.Equal(10_000, results[0].Chunk.Text.Length);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task StoreChunk_handles_unicode_metadata()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            var chunk = new DocumentChunk("日本語ドキュメント", 0, "Le café résumé über straße", "ファイル://テスト");
            await store.StoreChunkAsync(chunk, OneHot(7));

            var results = await store.SearchAsync(OneHot(7), topK: 1);
            Assert.Single(results);
            Assert.Equal("日本語ドキュメント", results[0].Chunk.DocumentId);
            Assert.Equal("Le café résumé über straße", results[0].Chunk.Text);
            Assert.Equal("ファイル://テスト", results[0].Chunk.Source);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_scores_are_in_expected_range()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            await store.StoreChunkAsync(new DocumentChunk("d", 0, "t", "s"), OneHot(0));

            // Query with same direction → score ≈ 1
            var results = await store.SearchAsync(OneHot(0), topK: 1);
            Assert.InRange(results[0].Score, 0.99f, 1.01f);
        }
        finally { File.Delete(path); }
    }
}
