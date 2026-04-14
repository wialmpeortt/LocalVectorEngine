using LocalVectorEngine.Core.Models;

namespace LocalVectorEngine.Core.Interfaces;

/// <summary>
/// Splits a document's raw text into overlapping chunks suitable for embedding.
/// </summary>
public interface IChunkingService
{
    /// <summary>
    /// Produces chunks for a single document.
    /// </summary>
    /// <param name="documentId">Stable identifier of the source document.</param>
    /// <param name="text">The full document text.</param>
    /// <param name="source">Human-readable source reference (file path, URL, citation).</param>
    /// <returns>Ordered, non-empty chunks with sequential <c>ChunkIndex</c> starting at 0.</returns>
    IEnumerable<DocumentChunk> Chunk(string documentId, string text, string source);
}
