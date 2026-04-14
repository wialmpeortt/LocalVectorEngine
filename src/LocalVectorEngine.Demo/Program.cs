using LocalVectorEngine.Core.Interfaces;
using LocalVectorEngine.Core.Models;

// LocalVectorEngine.Demo
// Thin console host used to exercise the Core library end-to-end:
//   IChunkingService -> IEmbeddingService -> IVectorStore
//
// Real implementations (OnnxEmbeddingService, BLiteVectorStore, ...)
// will be wired in as they land in Core.

Console.WriteLine("LocalVectorEngine.Demo — RAG pipeline host");
Console.WriteLine("Core contracts available:");
Console.WriteLine($"  - {nameof(IChunkingService)}");
Console.WriteLine($"  - {nameof(IEmbeddingService)}");
Console.WriteLine($"  - {nameof(IVectorStore)}  (returns {nameof(SearchResult)})");
Console.WriteLine();
Console.WriteLine("No implementations wired yet. See issues #10, #11, #21.");
