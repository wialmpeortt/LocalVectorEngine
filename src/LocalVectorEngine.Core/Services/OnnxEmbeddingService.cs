using FastBertTokenizer;
using LocalVectorEngine.Core.Interfaces;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace LocalVectorEngine.Core.Services;

/// <summary>
/// <see cref="IEmbeddingService"/> backed by a local ONNX transformer model
/// (designed for <c>sentence-transformers/all-MiniLM-L6-v2</c>).
///
/// The service:
///   1. Tokenizes the input text with a BERT WordPiece tokenizer loaded from <c>vocab.txt</c>.
///   2. Runs a single forward pass through the ONNX model.
///   3. Mean-pools the <c>last_hidden_state</c> using the attention mask.
///   4. L2-normalizes the resulting vector so that cosine similarity reduces to a dot product.
///
/// The class is thread-safe for concurrent calls to <see cref="EmbedAsync"/>
/// because the underlying <see cref="InferenceSession"/> is thread-safe.
/// </summary>
public sealed class OnnxEmbeddingService : IEmbeddingService, IDisposable
{
    /// <summary>Embedding dimension produced by <c>all-MiniLM-L6-v2</c>.</summary>
    public const int EmbeddingDimension = 384;

    /// <summary>Maximum token length accepted by the model.</summary>
    public const int MaxSequenceLength = 512;

    private readonly InferenceSession _session;
    private readonly BertTokenizer _tokenizer;
    private bool _disposed;

    /// <param name="onnxModelPath">Path to the <c>.onnx</c> model file.</param>
    /// <param name="vocabPath">Path to the BERT tokenizer <c>vocab.txt</c>.</param>
    /// <param name="lowerCase">
    /// Whether the tokenizer lowercases input before tokenization.
    /// <c>true</c> matches the default <c>all-MiniLM-L6-v2</c> tokenizer.
    /// </param>
    public OnnxEmbeddingService(string onnxModelPath, string vocabPath, bool lowerCase = true)
    {
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException("ONNX model not found.", onnxModelPath);
        if (!File.Exists(vocabPath))
            throw new FileNotFoundException("Tokenizer vocabulary not found.", vocabPath);

        _session = new InferenceSession(onnxModelPath);

        _tokenizer = new BertTokenizer();
        using var vocabStream = File.OpenRead(vocabPath);
        _tokenizer.LoadVocabulary(vocabStream, convertInputToLowercase: lowerCase);
    }

    /// <inheritdoc />
    public Task<float[]> EmbedAsync(string text, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(text);
        ct.ThrowIfCancellationRequested();

        // ONNX inference is CPU-bound and synchronous; offload to a worker
        // thread so callers on the UI thread (e.g. MAUI) don't block.
        return Task.Run(() => Embed(text, ct), ct);
    }

    private float[] Embed(string text, CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        // 1. Tokenize. FastBertTokenizer pads/truncates to MaxSequenceLength.
        var (inputIds, attentionMask, tokenTypeIds) =
            _tokenizer.Encode(text, MaxSequenceLength);
        ct.ThrowIfCancellationRequested();

        int seqLen = inputIds.Length;
        var shape = new[] { 1, seqLen };

        // 2. Run the model.
        using var inputIdsTensor = OrtValue.CreateTensorValueFromMemory(inputIds, shape);
        using var attentionMaskTensor = OrtValue.CreateTensorValueFromMemory(attentionMask, shape);
        using var tokenTypeIdsTensor = OrtValue.CreateTensorValueFromMemory(tokenTypeIds, shape);

        var inputs = new Dictionary<string, OrtValue>
        {
            ["input_ids"]      = inputIdsTensor,
            ["attention_mask"] = attentionMaskTensor,
            ["token_type_ids"] = tokenTypeIdsTensor,
        };

        using var runOptions = new RunOptions();
        using var results = _session.Run(runOptions, inputs, _session.OutputNames);
        ct.ThrowIfCancellationRequested();

        // The first output is `last_hidden_state` with shape [1, seqLen, 384].
        var lastHidden = results[0].GetTensorDataAsSpan<float>();

        // 3. Mean pool with the attention mask.
        var pooled = new float[EmbeddingDimension];
        long validTokens = 0;
        for (int t = 0; t < seqLen; t++)
        {
            if (attentionMask[t] == 0) continue;
            validTokens++;
            int tokenOffset = t * EmbeddingDimension;
            for (int d = 0; d < EmbeddingDimension; d++)
                pooled[d] += lastHidden[tokenOffset + d];
        }
        if (validTokens == 0)
            throw new InvalidOperationException("Input produced zero valid tokens after tokenization.");

        float inv = 1f / validTokens;
        for (int d = 0; d < EmbeddingDimension; d++)
            pooled[d] *= inv;

        // 4. L2 normalize so cosine similarity = dot product.
        double sumSq = 0;
        for (int d = 0; d < EmbeddingDimension; d++)
            sumSq += pooled[d] * pooled[d];

        float norm = (float)Math.Sqrt(sumSq);
        if (norm > 1e-12f)
        {
            float invNorm = 1f / norm;
            for (int d = 0; d < EmbeddingDimension; d++)
                pooled[d] *= invNorm;
        }

        return pooled;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _session.Dispose();
        _disposed = true;
    }
}
