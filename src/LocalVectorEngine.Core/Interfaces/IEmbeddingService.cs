namespace LocalVectorEngine.Core.Interfaces;

/// <summary>
/// Converts raw text into a fixed-size dense vector (embedding).
/// Implementations may wrap a local ONNX model, a remote API, or a stub.
/// </summary>
public interface IEmbeddingService
{
    /// <summary>
    /// Produces an embedding vector for the given text.
    /// </summary>
    /// <param name="text">Input text. Must be non-null.</param>
    /// <param name="ct">Cancellation token.</param>
    /// <returns>A float array whose length matches the model's embedding dimension.</returns>
    Task<float[]> EmbedAsync(string text, CancellationToken ct = default);
}
