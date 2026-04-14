using LocalVectorEngine.Core.Interfaces;
using LocalVectorEngine.Core.Models;

namespace LocalVectorEngine.Core.Services;

/// <summary>
/// Word-based sliding-window <see cref="IChunkingService"/>.
///
/// Splits the input text on whitespace and emits overlapping chunks of
/// <c>ChunkSizeWords</c> words sliding by <c>(ChunkSizeWords - OverlapWords)</c>
/// at each step.
///
/// Word-based chunking is intentionally decoupled from the embedding
/// model's tokenizer: the chunker stays simple, deterministic, and
/// reusable across different embedding back-ends. A modest chunk size
/// (default 400 words) keeps every chunk comfortably below the 512-token
/// limit of <c>all-MiniLM-L6-v2</c> for typical English text.
/// </summary>
public sealed class SlidingWindowChunkingService : IChunkingService
{
    /// <summary>Default chunk length in words.</summary>
    public const int DefaultChunkSizeWords = 400;

    /// <summary>Default overlap between consecutive chunks, in words.</summary>
    public const int DefaultOverlapWords = 50;

    private readonly int _chunkSize;
    private readonly int _overlap;
    private readonly int _step;

    /// <param name="chunkSizeWords">Number of words per chunk. Must be &gt; 0.</param>
    /// <param name="overlapWords">
    /// Number of words shared between consecutive chunks.
    /// Must be in <c>[0, chunkSizeWords)</c>.
    /// </param>
    public SlidingWindowChunkingService(
        int chunkSizeWords = DefaultChunkSizeWords,
        int overlapWords = DefaultOverlapWords)
    {
        if (chunkSizeWords <= 0)
            throw new ArgumentOutOfRangeException(nameof(chunkSizeWords),
                "Chunk size must be positive.");
        if (overlapWords < 0)
            throw new ArgumentOutOfRangeException(nameof(overlapWords),
                "Overlap must be non-negative.");
        if (overlapWords >= chunkSizeWords)
            throw new ArgumentException(
                "Overlap must be strictly smaller than chunk size, otherwise the window cannot advance.",
                nameof(overlapWords));

        _chunkSize = chunkSizeWords;
        _overlap = overlapWords;
        _step = chunkSizeWords - overlapWords;
    }

    /// <inheritdoc />
    public IEnumerable<DocumentChunk> Chunk(string documentId, string text, string source)
    {
        ArgumentNullException.ThrowIfNull(documentId);
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(source);

        if (string.IsNullOrWhiteSpace(text))
            yield break;

        // Split on any run of whitespace (spaces, tabs, newlines) and drop empty entries.
        var words = text.Split(default(char[]), StringSplitOptions.RemoveEmptyEntries);
        if (words.Length == 0) yield break;

        int chunkIndex = 0;
        for (int start = 0; start < words.Length; start += _step)
        {
            int end = Math.Min(start + _chunkSize, words.Length);
            string chunkText = string.Join(' ', words, start, end - start);

            yield return new DocumentChunk(documentId, chunkIndex++, chunkText, source);

            // The chunk we just emitted already covers the tail of the document.
            if (end == words.Length) yield break;
        }
    }
}
