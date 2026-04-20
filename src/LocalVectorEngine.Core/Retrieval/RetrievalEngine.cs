using LocalVectorEngine.Core.Interfaces;
using LocalVectorEngine.Core.Models;

namespace LocalVectorEngine.Core.Retrieval;

/// <summary>
/// High-level orchestrator for the query (retrieval) half of the RAG pipeline.
///
/// <para>
/// Given a natural-language question the engine:
/// <list type="number">
///   <item>Embeds the question via <see cref="IEmbeddingService"/>.</item>
///   <item>Runs a k-NN search against <see cref="IVectorStore"/>.</item>
///   <item>Returns the top-K passages ranked by cosine similarity.</item>
/// </list>
/// The returned <see cref="SearchResult"/> list is ready to be injected
/// as grounded context into an LLM prompt (the LLM step itself is outside
/// this engine's scope).
/// </para>
/// </summary>
public sealed class RetrievalEngine
{
    /// <summary>Default number of passages returned when the caller does not specify <c>topK</c>.</summary>
    public const int DefaultTopK = 5;

    private readonly IEmbeddingService _embedder;
    private readonly IVectorStore _store;

    public RetrievalEngine(IEmbeddingService embedder, IVectorStore store)
    {
        ArgumentNullException.ThrowIfNull(embedder);
        ArgumentNullException.ThrowIfNull(store);

        _embedder = embedder;
        _store    = store;
    }

    /// <summary>
    /// Retrieves the most relevant passages for a natural-language question.
    /// </summary>
    /// <param name="question">The user's question (must not be null or whitespace).</param>
    /// <param name="topK">Maximum number of results to return (default: <see cref="DefaultTopK"/>).</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>
    /// Up to <paramref name="topK"/> results ordered from most to least similar.
    /// If the store contains fewer than <paramref name="topK"/> chunks the returned
    /// list may be shorter.
    /// </returns>
    public async Task<IReadOnlyList<SearchResult>> RetrieveAsync(
        string question,
        int topK = DefaultTopK,
        CancellationToken ct = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(question);
        if (topK <= 0)
            throw new ArgumentOutOfRangeException(nameof(topK), "topK must be positive.");

        ct.ThrowIfCancellationRequested();

        var queryEmbedding = await _embedder
            .EmbedAsync(question, ct)
            .ConfigureAwait(false);

        ct.ThrowIfCancellationRequested();

        return await _store
            .SearchAsync(queryEmbedding, topK, ct)
            .ConfigureAwait(false);
    }
}
