using BLite.Bson;

namespace LocalVectorEngine.Core.Persistence;

/// <summary>
/// BLite entity that backs a single <see cref="Models.DocumentChunk"/>
/// together with its embedding vector.
///
/// Kept separate from the public <c>DocumentChunk</c> record so that the
/// persistence layer owns its own identity type (<see cref="ObjectId"/>)
/// and storage-specific concerns do not leak into the domain model.
/// </summary>
public sealed class VectorChunkEntity
{
    public ObjectId Id { get; set; }

    /// <summary>Identifier of the source document this chunk belongs to.</summary>
    public string DocumentId { get; set; } = string.Empty;

    /// <summary>Sequential 0-based index of the chunk inside its document.</summary>
    public int ChunkIndex { get; set; }

    /// <summary>Raw text of the chunk.</summary>
    public string Text { get; set; } = string.Empty;

    /// <summary>Origin of the chunk (file path, URL, in-memory tag, …).</summary>
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// L2-normalized embedding vector. Must match
    /// <see cref="Services.OnnxEmbeddingService.EmbeddingDimension"/>.
    /// </summary>
    public float[] Embedding { get; set; } = Array.Empty<float>();
}
