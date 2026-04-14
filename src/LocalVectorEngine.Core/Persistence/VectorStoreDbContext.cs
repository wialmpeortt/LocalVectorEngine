using BLite.Bson;
using BLite.Core;
using BLite.Core.Collections;
using BLite.Core.Indexing;
using BLite.Core.Metadata;

namespace LocalVectorEngine.Core.Persistence;

/// <summary>
/// BLite <see cref="DocumentDbContext"/> that hosts a single collection of
/// <see cref="VectorChunkEntity"/> backed by an HNSW vector index.
///
/// One context = one <c>.db</c> file on disk. Thread-safety and concurrency
/// semantics are those provided by BLite itself.
/// </summary>
public partial class VectorStoreDbContext : DocumentDbContext
{
    /// <summary>Embedding dimension of <c>all-MiniLM-L6-v2</c>.</summary>
    public const int EmbeddingDimension = 384;

    /// <summary>Logical name of the vector index (used by direct-API lookups).</summary>
    public const string VectorIndexName = "idx_chunk_embedding";

    public DocumentCollection<ObjectId, VectorChunkEntity> Chunks { get; set; } = null!;

    public VectorStoreDbContext(string databasePath) : base(databasePath)
    {
        InitializeCollections();
    }

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<VectorChunkEntity>()
            .ToCollection("document_chunks")
            .HasIndex(x => x.DocumentId)
            .HasVectorIndex(
                x => x.Embedding,
                dimensions: EmbeddingDimension,
                metric: VectorMetric.Cosine,
                name: VectorIndexName);
    }
}
