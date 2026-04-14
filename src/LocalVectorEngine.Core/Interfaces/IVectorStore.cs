using LocalVectorEngine.Core.Models;

namespace LocalVectorEngine.Core.Interfaces;

/// <summary>
/// Persists chunks with their embeddings and performs vector similarity search.
/// </summary>
public interface IVectorStore
{
    /// <summary>
    /// Stores a chunk together with its precomputed embedding.
    /// </summary>
    Task StoreChunkAsync(DocumentChunk chunk, float[] embedding, CancellationToken ct = default);

    /// <summary>
    /// Returns the <paramref name="topK"/> chunks most similar to the query embedding,
    /// ordered from most to least similar.
    /// </summary>
    Task<IReadOnlyList<SearchResult>> SearchAsync(float[] queryEmbedding, int topK, CancellationToken ct = default);
}
