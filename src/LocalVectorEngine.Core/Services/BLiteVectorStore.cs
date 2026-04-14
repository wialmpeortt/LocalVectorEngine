using BLite.Core;
using BLite.Core.Indexing;
using BLite.Core.Query;
using LocalVectorEngine.Core.Interfaces;
using LocalVectorEngine.Core.Models;
using LocalVectorEngine.Core.Persistence;

namespace LocalVectorEngine.Core.Services;

/// <summary>
/// <see cref="IVectorStore"/> backed by a local BLite database with a
/// built-in HNSW index over the 384-dim chunk embeddings.
///
/// The store is the persistence half of the RAG pipeline:
///   * <see cref="StoreChunkAsync"/> persists chunk + vector on indexing.
///   * <see cref="SearchAsync"/> runs an approximate k-NN cosine query.
///
/// <para>
/// BLite returns matching entities in similarity order but does not expose
/// per-result scores. Because <c>OnnxEmbeddingService</c> emits
/// L2-normalized vectors, cosine similarity is equivalent to the dot
/// product, which is computed here post-hoc so the caller gets a
/// meaningful score in <see cref="SearchResult.Score"/> (in <c>[-1, 1]</c>,
/// typically in <c>[0, 1]</c> for related text).
/// </para>
/// </summary>
public sealed class BLiteVectorStore : IVectorStore, IDisposable
{
    /// <summary>Embedding dimension enforced on every <c>StoreChunkAsync</c> call.</summary>
    public const int EmbeddingDimension = VectorStoreDbContext.EmbeddingDimension;

    private readonly VectorStoreDbContext _db;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private bool _disposed;

    /// <param name="databasePath">
    /// Path of the BLite <c>.db</c> file. The directory must exist;
    /// the file itself is created on first write.
    /// </param>
    public BLiteVectorStore(string databasePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(databasePath);

        var dir = Path.GetDirectoryName(Path.GetFullPath(databasePath));
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
            throw new DirectoryNotFoundException(
                $"Directory for BLite database does not exist: {dir}");

        _db = new VectorStoreDbContext(databasePath);
    }

    /// <inheritdoc />
    public async Task StoreChunkAsync(
        DocumentChunk chunk,
        float[] embedding,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(chunk);
        ArgumentNullException.ThrowIfNull(embedding);

        if (embedding.Length != EmbeddingDimension)
            throw new ArgumentException(
                $"Embedding has {embedding.Length} dims but the store expects {EmbeddingDimension}.",
                nameof(embedding));

        ct.ThrowIfCancellationRequested();

        // BLite's write path is not guaranteed to be safe under concurrent
        // writers on the same context, so we serialize Insert calls ourselves.
        await _writeLock.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            var entity = new VectorChunkEntity
            {
                DocumentId = chunk.DocumentId,
                ChunkIndex = chunk.ChunkIndex,
                Text       = chunk.Text,
                Source     = chunk.Source,
                Embedding  = embedding,
            };

            await _db.Chunks.InsertAsync(entity).ConfigureAwait(false);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<SearchResult>> SearchAsync(
        float[] queryEmbedding,
        int topK,
        CancellationToken ct = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(queryEmbedding);
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive.");
        if (queryEmbedding.Length != EmbeddingDimension)
            throw new ArgumentException(
                $"Query has {queryEmbedding.Length} dims but the store expects {EmbeddingDimension}.",
                nameof(queryEmbedding));

        ct.ThrowIfCancellationRequested();

        // HNSW KNN via the LINQ-shaped marker extension.
        var hits = await _db.Chunks
            .AsQueryable()
            .Where(x => x.Embedding.VectorSearch(queryEmbedding, topK))
            .ToListAsync()
            .ConfigureAwait(false);

        ct.ThrowIfCancellationRequested();

        // BLite does not surface per-result scores, so we recompute cosine
        // similarity ourselves. Embeddings are L2-normalized upstream, which
        // reduces cosine similarity to a plain dot product.
        var results = new List<SearchResult>(hits.Count);
        foreach (var hit in hits)
        {
            var score = DotProduct(queryEmbedding, hit.Embedding);
            var chunk = new DocumentChunk(hit.DocumentId, hit.ChunkIndex, hit.Text, hit.Source);
            results.Add(new SearchResult(chunk, score));
        }

        return results;
    }

    private static float DotProduct(float[] a, float[] b)
    {
        // Lengths are guaranteed equal by the ctor-time dimension check.
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return (float)sum;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _db.Dispose();
        _writeLock.Dispose();
        _disposed = true;
    }
}
