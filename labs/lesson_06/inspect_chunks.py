from collections import Counter
from pathlib import Path

from supportops.knowledge_base import KnowledgeBaseLoader
from supportops.document_chunker import DocumentChunker

base_dir = Path(__file__).resolve().parent
root = base_dir.parent.parent / "data" / "knowledge_base"

loader = KnowledgeBaseLoader(
    root=root
)

documents = loader.load()

chunker = DocumentChunker(
    chunk_size=350,
    chunk_overlap=50,
)

chunks = chunker.split(documents)
total_source_docs = len(documents)
total_chunks = len(chunks)

chunk_sizes = [len(chunk.page_content) for chunk in chunks] if chunks else [0]
avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunks else 0
min_chunk_size = min(chunk_sizes) if chunks else 0
max_chunk_size = max(chunk_sizes) if chunks else 0

empty_chunks_count = sum(1 for size in chunk_sizes if size == 0)

chunks_per_source = Counter(chunk.metadata.get("source", "unknown") for chunk in chunks)

print("Chunking Report\n")
print(f"Source documents: {total_source_docs}")
print(f"Generated chunks: {total_chunks}\n")

print(f"Average chunk size: {avg_chunk_size:.1f} characters")
print(f"Minimum chunk size: {min_chunk_size} characters")
print(f"Maximum chunk size: {max_chunk_size} characters\n")

print("Chunks per source:")
for source_path, count in sorted(chunks_per_source.items()):
    print(f"  {source_path}: {count}")

print(f"\nEmpty chunks: {empty_chunks_count}")
print("-" * 40)


