from collections import Counter
from pathlib import Path

from langchain_core.documents import Document

from supportops.knowledge_base import KnowledgeBaseLoader

base_dir = Path(__file__).resolve().parent
root = base_dir.parent.parent / "data" / "knowledge_base"


def inspect_document(document: Document) -> None:
    source = document.metadata.get("source", "Unknown")

    text_length = len(document.page_content)

    preview = document.page_content[:100].replace("\n", " ")

    print(f"Source: {source}")
    print(f"Symbols count: {text_length}")
    print(f"Preview: {preview}...")
    print("-" * 50)


allowed_extensions = {".txt", ".md"}

for path in sorted(root.rglob("*")):
    if path.is_file() and path.suffix.lower() in allowed_extensions:
        current_doc = Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": path.relative_to(root).as_posix()},
        )

        inspect_document(current_doc)


def print_knowledge_base_report(documents: list[Document]) -> None:
    total_docs = len(documents)

    categories = []
    file_types = []
    empty_docs_count = 0

    for doc in documents:
        category = doc.metadata.get("category", "general")
        file_type = doc.metadata.get("file_type", "unknown")

        categories.append(category)

        file_types.append(file_type)

        if not doc.page_content or not doc.page_content.strip():
            empty_docs_count += 1

    category_counts = Counter(categories)
    file_type_counts = Counter(file_types)
    total_characters = sum(len(doc.page_content) for doc in documents)

    print("Knowledge Base Inspection\n")
    print(f"Documents loaded: {total_docs}\n")
    print(f"Total characters: {total_characters}\n")

    print("Categories:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")

    print("\nFile types:")
    for f_type, count in sorted(file_type_counts.items()):
        print(f"  {f_type}: {count}")

    print(f"\nEmpty documents: {empty_docs_count}")


loader = KnowledgeBaseLoader(
    root=root,
)

loaded_documents = loader.load()

print_knowledge_base_report(loaded_documents)
