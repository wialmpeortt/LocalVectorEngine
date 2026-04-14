using LocalVectorEngine.Core.Services;
using Xunit;

namespace LocalVectorEngine.Core.Tests;

/// <summary>
/// Integration-style tests for <see cref="OnnxEmbeddingService"/>.
///
/// The tests are skipped automatically when the ONNX model and/or
/// tokenizer vocabulary are not present on disk — this keeps the
/// suite green on CI machines that do not have the 86 MB model
/// downloaded, while still running end-to-end locally.
/// </summary>
public class OnnxEmbeddingServiceTests
{
    private static readonly string RepoRoot = FindRepoRoot(AppContext.BaseDirectory);
    private static readonly string ModelPath = Path.Combine(RepoRoot, "models", "all-MiniLM-L6-v2.onnx");
    private static readonly string VocabPath = Path.Combine(RepoRoot, "models", "vocab.txt");

    private static bool ArtifactsAvailable => File.Exists(ModelPath) && File.Exists(VocabPath);

    [SkippableFact]
    public async Task EmbedAsync_returns_384_dimensional_vector()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var svc = new OnnxEmbeddingService(ModelPath, VocabPath);
        var vec = await svc.EmbedAsync("hello world");

        Assert.Equal(OnnxEmbeddingService.EmbeddingDimension, vec.Length);
    }

    [SkippableFact]
    public async Task EmbedAsync_returns_L2_normalized_vector()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var svc = new OnnxEmbeddingService(ModelPath, VocabPath);
        var vec = await svc.EmbedAsync("retrieval augmented generation");

        double norm = 0;
        foreach (var v in vec) norm += v * v;
        norm = Math.Sqrt(norm);

        Assert.InRange(norm, 0.999, 1.001);
    }

    [SkippableFact]
    public async Task EmbedAsync_semantically_similar_sentences_have_higher_cosine_than_unrelated()
    {
        Skip.IfNot(ArtifactsAvailable);

        using var svc = new OnnxEmbeddingService(ModelPath, VocabPath);

        var a = await svc.EmbedAsync("The cat sat on the mat.");
        var b = await svc.EmbedAsync("A feline rested on the rug.");
        var c = await svc.EmbedAsync("Quantum chromodynamics describes the strong interaction.");

        float similar   = Cosine(a, b);
        float unrelated = Cosine(a, c);

        Assert.True(similar > unrelated,
            $"Expected related sentences to be closer. sim={similar}, unrelated={unrelated}");
    }

    [Fact]
    public void Ctor_throws_FileNotFound_when_model_missing()
    {
        Assert.Throws<FileNotFoundException>(() =>
            new OnnxEmbeddingService("does/not/exist.onnx", VocabPath));
    }

    // ------------------------------------------------------------------
    private static float Cosine(float[] x, float[] y)
    {
        // Vectors are L2-normalized by the service.
        float dot = 0;
        for (int i = 0; i < x.Length; i++) dot += x[i] * y[i];
        return dot;
    }

    private static string FindRepoRoot(string start)
    {
        var dir = new DirectoryInfo(start);
        while (dir is not null)
        {
            if (dir.GetFiles("*.slnx").Length > 0 || Directory.Exists(Path.Combine(dir.FullName, ".git")))
                return dir.FullName;
            dir = dir.Parent;
        }
        return start;
    }
}
