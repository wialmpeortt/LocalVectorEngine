using LocalVectorEngine.Core.Services;

// LocalVectorEngine.Demo
// Proof-of-life host for the Core library. At this stage only the
// embedding service is wired up (Issue #10); chunking (#21) and the
// BLite vector store (#11) join later and the demo will grow into a
// full end-to-end RAG pipeline.

// ---------------------------------------------------------------
// Locate model artefacts. Defaults point at ../../models relative
// to the solution root, which is where `all-MiniLM-L6-v2.onnx`
// and `vocab.txt` are expected to live. Overridable via env vars.
// ---------------------------------------------------------------
string repoRoot = FindRepoRoot(AppContext.BaseDirectory);
string modelPath =
    Environment.GetEnvironmentVariable("LVE_MODEL_PATH")
    ?? Path.Combine(repoRoot, "models", "all-MiniLM-L6-v2.onnx");
string vocabPath =
    Environment.GetEnvironmentVariable("LVE_VOCAB_PATH")
    ?? Path.Combine(repoRoot, "models", "vocab.txt");

Console.WriteLine("LocalVectorEngine.Demo — OnnxEmbeddingService smoke test");
Console.WriteLine($"  model: {modelPath}");
Console.WriteLine($"  vocab: {vocabPath}");
Console.WriteLine();

if (!File.Exists(modelPath) || !File.Exists(vocabPath))
{
    Console.Error.WriteLine("Missing model or vocab file. See README for download instructions.");
    return 1;
}

using var embedder = new OnnxEmbeddingService(modelPath, vocabPath);

string[] samples =
{
    "Retrieval-Augmented Generation grounds LLM answers in source documents.",
    "Vector search finds semantically similar passages for a query.",
    "The quick brown fox jumps over the lazy dog.",
};

foreach (var s in samples)
{
    var vec = await embedder.EmbedAsync(s);
    double norm = 0;
    foreach (var v in vec) norm += v * v;
    norm = Math.Sqrt(norm);

    Console.WriteLine($"> \"{s}\"");
    Console.WriteLine($"  dim = {vec.Length}, ||v|| = {norm:F6}");
    Console.WriteLine($"  first 5 dims: [{string.Join(", ", vec.Take(5).Select(f => f.ToString("F4")))}]");
    Console.WriteLine();
}

// Quick cosine similarity check: the two RAG/vector-search sentences
// should be noticeably closer to each other than to the fox pangram.
var a = await embedder.EmbedAsync(samples[0]);
var b = await embedder.EmbedAsync(samples[1]);
var c = await embedder.EmbedAsync(samples[2]);

Console.WriteLine($"cos(RAG, vector search) = {Cosine(a, b):F4}");
Console.WriteLine($"cos(RAG, fox pangram)   = {Cosine(a, c):F4}");

return 0;

// ---------------------------------------------------------------
static float Cosine(float[] x, float[] y)
{
    // Vectors are already L2-normalized by the service, so cosine
    // similarity reduces to a simple dot product.
    float dot = 0;
    for (int i = 0; i < x.Length; i++) dot += x[i] * y[i];
    return dot;
}

static string FindRepoRoot(string start)
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
