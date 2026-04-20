using LocalVectorEngine.Core.Services;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// Additional edge-case coverage for <see cref="SlidingWindowChunkingService"/>.
/// </summary>
public class SlidingWindowChunkingEdgeCaseTests
{
    private const string DocId  = "edge";
    private const string Source = "mem://edge";

    [Fact]
    public void Chunk_single_word_returns_one_chunk()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 2);
        var chunks = svc.Chunk(DocId, "hello", Source).ToList();

        Assert.Single(chunks);
        Assert.Equal("hello", chunks[0].Text);
        Assert.Equal(0, chunks[0].ChunkIndex);
    }

    [Fact]
    public void Chunk_text_exactly_chunk_size_returns_one_chunk()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 2);
        var text = "one two three four five";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Single(chunks);
        Assert.Equal(text, chunks[0].Text);
    }

    [Fact]
    public void Chunk_text_one_word_longer_than_chunk_returns_two_chunks()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 2);
        // 6 words, step = 3 → windows at 0 and 3
        var text = "one two three four five six";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Equal(2, chunks.Count);
        Assert.Equal("one two three four five", chunks[0].Text);
        Assert.Equal("four five six", chunks[1].Text);
    }

    [Fact]
    public void Chunk_handles_unicode_text_correctly()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 3, overlapWords: 1);
        // 6 Unicode words → step = 2 → windows at 0, 2, 4(trailing)
        var text = "caffè résumé naïve über straße château";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.True(chunks.Count >= 2, $"Expected at least 2 chunks, got {chunks.Count}");
        Assert.Equal("caffè résumé naïve", chunks[0].Text);
    }

    [Fact]
    public void Chunk_handles_CJK_characters_as_words()
    {
        // CJK characters separated by spaces are treated as individual words.
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 2, overlapWords: 0);
        var text = "你好 世界 测试 数据";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        Assert.Equal(2, chunks.Count);
        Assert.Equal("你好 世界", chunks[0].Text);
        Assert.Equal("测试 数据", chunks[1].Text);
    }

    [Fact]
    public void Chunk_handles_very_long_document()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 100, overlapWords: 25);
        var words = Enumerable.Range(0, 1000).Select(i => $"word{i}");
        var text = string.Join(' ', words);

        var chunks = svc.Chunk(DocId, text, Source).ToList();

        // 1000 words, step=75 → ceil((1000-100)/75)+1 = 13 chunks
        Assert.True(chunks.Count >= 12, $"Expected many chunks, got {chunks.Count}");

        // All chunks should have sequential indexes.
        for (int i = 0; i < chunks.Count; i++)
            Assert.Equal(i, chunks[i].ChunkIndex);

        // No chunk text should be empty.
        Assert.All(chunks, c => Assert.False(string.IsNullOrWhiteSpace(c.Text)));
    }

    [Fact]
    public void Chunk_preserves_words_with_punctuation()
    {
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 0);
        var text = "Hello, world! This is a test.";
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        // Punctuation stays attached to words.
        Assert.Equal("Hello, world! This is a", chunks[0].Text);
    }

    [Fact]
    public void Chunk_with_overlap_one_less_than_chunk_size_produces_many_chunks()
    {
        // Extreme overlap: chunkSize=5, overlap=4, step=1.
        var svc = new SlidingWindowChunkingService(chunkSizeWords: 5, overlapWords: 4);
        var text = string.Join(' ', Enumerable.Range(0, 10).Select(i => $"w{i}"));
        var chunks = svc.Chunk(DocId, text, Source).ToList();

        // step=1, so (10-5)/1 + 1 = 6 chunks
        Assert.Equal(6, chunks.Count);
    }
}
