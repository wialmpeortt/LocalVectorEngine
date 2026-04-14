using LocalVectorEngine.Core.Interfaces;

namespace LocalVectorEngine.Core.Indexing;

/// <summary>
/// High-level orchestrator for the indexing half of the RAG pipeline.
///
/// <para>
/// Wires together the three core services so callers do not have to:
/// <list type="number">
///   <item><see cref="IChunkingService"/> splits the document text into chunks.</item>
///   <item><see cref="IEmbeddingService"/> turns each chunk into a 384-dim vector.</item>
///   <item><see cref="IVectorStore"/> persists chunk + embedding in the HNSW index.</item>
/// </list>
/// </para>
///
/// <para>
/// The class is a thin coordinator: no internal state, no caching, no retry.
/// All heavy work happens inside the injected services. Each chunk is embedded
/// and stored sequentially so that back-pressure on the store naturally throttles
/// the embedder; parallel indexing can be layered on top if ever needed.
/// </para>
/// </summary>
public sealed class DocumentIndexer
{
    private readonly IChunkingService _chunker;
    private readonly IEmbeddingService _embedder;
    private readonly IVectorStore _store;

    public DocumentIndexer(
        IChunkingService chunker,
        IEmbeddingService embedder,
        IVectorStore store)
    {
        ArgumentNullException.ThrowIfNull(chunker);
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentNullException.ThrowIfNull(store);

        _chunker  = chunker;
        _embedder = embedder;
        _store    = store;
    }

    /// <summary>
    /// Indexes an in-memory document.
    /// </summary>
    /// <param name="documentId">Stable identifier of the source document.</param>
    /// <param name="text">Full document text (may be empty).</param>
    /// <param name="source">Human-readable source reference (file path, URL, citation).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Number of chunks successfully stored.</returns>
    public async Task<int> IndexDocumentAsync(
        string documentId,
        string text,
        string source,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(documentId);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(source);

        ct.ThrowIfCancellationRequested();

        int count = 0;
        foreach (var chunk in _chunker.Chunk(documentId, text, source))
        {
            ct.ThrowIfCancellationRequested();

            var embedding = await _embedder
                .EmbedAsync(chunk.Text, ct)
                .ConfigureAwait(false);

            ct.ThrowIfCancellationRequested();

            await _store
                .StoreChunkAsync(chunk, embedding, ct)
                .ConfigureAwait(false);

            count++;
        }

        return count;
    }

    /// <summary>
    /// Indexes a document loaded from disk. The file name (without extension) is
    /// used as <c>documentId</c> and the absolute path as <c>source</c>.
    /// </summary>
    /// <param name="path">Path of a text file (.txt, .md, …).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>Number of chunks successfully stored.</returns>
    public async Task<int> IndexFileAsync(string path, CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Document file not found.", path);

        var absolute   = Path.GetFullPath(path);
        var documentId = Path.GetFileNameWithoutExtension(absolute);
        var text       = await File.ReadAllTextAsync(absolute, ct).ConfigureAwait(false);

        return await IndexDocumentAsync(documentId, text, absolute, ct).ConfigureAwait(false);
    }
}
