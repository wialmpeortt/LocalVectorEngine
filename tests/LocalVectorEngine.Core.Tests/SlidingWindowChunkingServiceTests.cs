using LocalVectorEngine.Core.Models;
using LocalVectorEngine.Core.Services;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

public class SlidingWindowChunkingServiceTests
{
    private const string DocId = "doc-1";
    private const string Source = "memory://test";

    // ---- constructor validation -------------------------------------

    [Theory]
    [InlineData(0, 0)]
    [InlineData(-1, 0)]
    public void Ctor_rejects_non_positive_chunk_size(int chunkSize, int overlap)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlidingWindowChunkingService(chunkSize, overlap));
    }

    [Fact]
    public void Ctor_rejects_negative_overlap()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            new SlidingWindowChunkingService(100, -1));
    }

    [Theory]
    [InlineData(100, 100)]
    [InlineData(100, 150)]
    public void Ctor_rejects_overlap_geq_chunk_size(int chunkSize, int overlap)
    {
        Assert.Throws<ArgumentException>(() =>
            new SlidingWindowChunkingService(chunkSize, overlap));
    }

    // ---- empty / trivial inputs -------------------------------------

    [Fact]
    public void Chunk_returns_nothing_for_empty_text()
    {
        var svc = new SlidingWindowChunkingService();
        Assert.Empty(svc.Chunk(DocId, "", Source));
    }

    [Fact]
    public void Chunk_returns_nothing_for_whitespace_only_text()
    {
        var svc = new SlidingWindowChunkingService();
        Assert.Empty(svc.Chunk(DocId, "   \n\t  ", Source));
    }

    [Fact]
    public void Chunk_returns_single_chunk_when_text_fits_in_window()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 10, overlapWords: 2);
        var text = "The quick brown fox jumps over the lazy dog";

        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Single(chunks);
        Assert.Equal(text, chunks[0].Text);
        Assert.Equal(0, chunks[0].ChunkIndex);
        Assert.Equal(DocId, chunks[0].DocumentId);
        Assert.Equal(Source, chunks[0].Source);
    }

    // ---- sliding behaviour ------------------------------------------

    [Fact]
    public void Chunk_emits_overlapping_chunks_with_sequential_indexes()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 4, overlapWords: 1);

        // 10 words -> step=3. Windows at offsets 0, 3, 6.
        // The window starting at 6 (w6..w9) already covers the tail of the
        // document, so no further chunk is emitted.
        var text = "w0 w1 w2 w3 w4 w5 w6 w7 w8 w9";

        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Equal(3, chunks.Count);

        Assert.Equal("w0 w1 w2 w3", chunks[0].Text);
        Assert.Equal("w3 w4 w5 w6", chunks[1].Text);
        Assert.Equal("w6 w7 w8 w9", chunks[2].Text);

        for (int i = 0; i < chunks.Count; i++)
            Assert.Equal(i, chunks[i].ChunkIndex);
    }

    [Fact]
    public void Chunk_overlap_is_exactly_overlap_words_between_consecutive_chunks()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 2);

        var text = string.Join(' ', Enumerable.Range(0, 20).Select(i => $"w{i}"));
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        for (int i = 0; i < chunks.Count - 1; i++)
        {
            var prevWords = chunks[i].Text.Split(' ');
            var nextWords = chunks[i + 1].Text.Split(' ');

            // Last 2 words of chunk i must equal first 2 words of chunk i+1
            // (only checked while the next chunk has at least overlap words,
            // i.e. it's not the trailing tail of the document).
            if (nextWords.Length < 2) continue;

            Assert.Equal(prevWords[^2], nextWords[0]);
            Assert.Equal(prevWords[^1], nextWords[1]);
        }
    }

    [Fact]
    public void Chunk_zero_overlap_produces_disjoint_chunks()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 3, overlapWords: 0);

        var text = "w0 w1 w2 w3 w4 w5 w6 w7 w8";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Equal(3, chunks.Count);
        Assert.Equal("w0 w1 w2", chunks[0].Text);
        Assert.Equal("w3 w4 w5", chunks[1].Text);
        Assert.Equal("w6 w7 w8", chunks[2].Text);
    }

    [Fact]
    public void Chunk_propagates_document_id_and_source_to_every_chunk()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 3, overlapWords: 1);

        var text = string.Join(' ', Enumerable.Range(0, 10).Select(i => $"w{i}"));
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.All(chunks, c =>
        {
            Assert.Equal(DocId, c.DocumentId);
            Assert.Equal(Source, c.Source);
        });
    }

    [Fact]
    public void Chunk_collapses_irregular_whitespace_to_single_spaces()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 0);

        var text = "alpha\tbeta\n\ngamma   delta";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Single(chunks);
        Assert.Equal("alpha beta gamma delta", chunks[0].Text);
    }
}
