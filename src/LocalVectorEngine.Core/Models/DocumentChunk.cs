namespace LocalVectorEngine.Core.Models;

/// <summary>
/// Immutable unit of indexed text. Produced by IChunkingService and stored by IVectorStore.
/// </summary>
/// <param name="DocumentId">Stable identifier of the source document.</param>
/// <param name="ChunkIndex">Zero-based position of this chunk inside the document.</param>
/// <param name="Text">The actual chunk content.</param>
/// <param name="Source">Human-readable source reference (file path, URL, citation).</param>
public record DocumentChunk(
    string DocumentId,
    int ChunkIndex,
    string Text,
    string Source);
