using LocalVectorEngine.Core.Models;
using LocalVectorEngine.Core.Services;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// Integration-ish unit tests for <see cref="BLiteVectorStore"/>.
///
/// Each test uses a unique temp <c>.db</c> file derived from the calling
/// method name and deletes it in a <c>try/finally</c>, so test runs are
/// hermetic and can execute in parallel.
/// </summary>
public sealed class BLiteVectorStoreTests
{
    private const int Dim = BLiteVectorStore.EmbeddingDimension; // 384

    // ---- test helpers -----------------------------------------------------

    /// <summary>Returns a one-hot unit vector of size <see cref="Dim"/>.</summary>
    private static float[] OneHot(int position)
    {
        var v = new float[Dim];
        v[position] = 1f;
        return v;
    }

    private static string TempDbPath([System.Runtime.CompilerServices.CallerMemberName] string name = "")
        => Path.Combine(Path.GetTempPath(), $"lve_blite_{name}_{Guid.NewGuid():N}.db");

    private static DocumentChunk Chunk(string docId, int idx, string text = "sample", string source = "mem://t")
        => new(docId, idx, text, source);

    // ---- constructor validation ------------------------------------------

    [Fact]
    public void Ctor_rejects_null_or_whitespace_path()
    {
        Assert.Throws<ArgumentException>(() => new BLiteVectorStore("   "));
        Assert.Throws<ArgumentException>(() => new BLiteVectorStore(""));
        Assert.Throws<ArgumentNullException>(() => new BLiteVectorStore(null!));
    }

    [Fact]
    public void Ctor_rejects_missing_directory()
    {
        var bogus = Path.Combine(Path.GetTempPath(), $"does_not_exist_{Guid.NewGuid():N}", "db.db");
        Assert.Throws<DirectoryNotFoundException>(() => new BLiteVectorStore(bogus));
    }

    // ---- StoreChunkAsync validation --------------------------------------

    [Fact]
    public async Task StoreChunk_rejects_wrong_dimension()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);
            await Assert.ThrowsAsync<ArgumentException>(() =>
                store.StoreChunkAsync(Chunk("d", 0), new float[Dim - 1]));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task StoreChunk_rejects_null_chunk_or_embedding()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                store.StoreChunkAsync(null!, OneHot(0)));
            await Assert.ThrowsAsync<ArgumentNullException>(() =>
                store.StoreChunkAsync(Chunk("d", 0), null!));
        }
        finally { File.Delete(path); }
    }

    // ---- SearchAsync validation ------------------------------------------

    [Fact]
    public async Task Search_rejects_wrong_dimension()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);
            await Assert.ThrowsAsync<ArgumentException>(() =>
                store.SearchAsync(new float[Dim + 1], 3));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_rejects_non_positive_topK()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                store.SearchAsync(OneHot(0), 0));
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() =>
                store.SearchAsync(OneHot(0), -1));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_on_empty_store_returns_empty()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);
            var results = await store.SearchAsync(OneHot(0), 5);
            Assert.Empty(results);
        }
        finally { File.Delete(path); }
    }

    // ---- round-trip behaviour --------------------------------------------

    [Fact]
    public async Task Search_returns_closest_chunk_first()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            await store.StoreChunkAsync(Chunk("doc-A", 0, "alpha"), OneHot(0));
            await store.StoreChunkAsync(Chunk("doc-B", 0, "beta"),  OneHot(1));
            await store.StoreChunkAsync(Chunk("doc-C", 0, "gamma"), OneHot(2));

            // Query = OneHot(1). Expected top-1: doc-B (score ~= 1),
            // the other two are orthogonal (score 0).
            var results = await store.SearchAsync(OneHot(1), topK: 3);

            Assert.NotEmpty(results);
            Assert.Equal("doc-B", results[0].Chunk.DocumentId);
            Assert.Equal("beta",  results[0].Chunk.Text);
            Assert.True(results[0].Score > 0.99f,
                $"expected cosine ~1.0 for identical one-hot, got {results[0].Score}");
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_preserves_chunk_metadata_roundtrip()
    {
        var path = TempDbPath();
        try
        {
            using var store = new BLiteVectorStore(path);

            var original = new DocumentChunk("doc-42", 7, "the quick brown fox", "file:///tmp/foo.md");
            await store.StoreChunkAsync(original, OneHot(3));

            var results = await store.SearchAsync(OneHot(3), topK: 1);

            Assert.Single(results);
            var got = results[0].Chunk;
            Assert.Equal(original.DocumentId, got.DocumentId);
            Assert.Equal(original.ChunkIndex, got.ChunkIndex);
            Assert.Equal(original.Text,       got.Text);
            Assert.Equal(original.Source,     got.Source);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task Search_survives_close_and_reopen()
    {
        var path = TempDbPath();
        try
        {
            using (var store = new BLiteVectorStore(path))
            {
                await store.StoreChunkAsync(Chunk("doc-X", 0, "persisted"), OneHot(5));
            } // dispose + flush

            using (var reopened = new BLiteVectorStore(path))
            {
                var results = await reopened.SearchAsync(OneHot(5), topK: 1);
                Assert.Single(results);
                Assert.Equal("doc-X", results[0].Chunk.DocumentId);
                Assert.Equal("persisted", results[0].Chunk.Text);
            }
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }

    // ---- lifecycle --------------------------------------------------------

    [Fact]
    public async Task Disposed_store_throws_on_further_use()
    {
        var path = TempDbPath();
        try
        {
            var store = new BLiteVectorStore(path);
            store.Dispose();

            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                store.StoreChunkAsync(Chunk("d", 0), OneHot(0)));
            await Assert.ThrowsAsync<ObjectDisposedException>(() =>
                store.SearchAsync(OneHot(0), 1));
        }
        finally { if (File.Exists(path)) File.Delete(path); }
    }
}
