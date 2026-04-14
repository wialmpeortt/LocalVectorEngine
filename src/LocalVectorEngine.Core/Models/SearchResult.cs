namespace LocalVectorEngine.Core.Models;

/// <summary>
/// A single hit returned by <see cref="Interfaces.IVectorStore.SearchAsync"/>.
/// </summary>
/// <param name="Chunk">The retrieved chunk.</param>
/// <param name="Score">Similarity score (higher = more similar, typically cosine in [0,1]).</param>
public record SearchResult(DocumentChunk Chunk, float Score);
