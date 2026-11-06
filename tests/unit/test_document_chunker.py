import copy

import pytest
from langchain_core.documents import Document

from supportops.document_chunker import DocumentChunker
from supportops.knowledge_base import KnowledgeBaseLoader

pytestmark = pytest.mark.usefixtures("offline_langchain")


def test_txt_splitting_retains_content_metadata_and_input():
    paragraphs = [f"Section {i}: " + (f"token{i} " * 12) for i in range(4)]
    document = Document(
        page_content="\n\n".join(paragraphs),
        metadata={"source": "guide.txt", "file_type": "txt", "nested": {"tags": ["a"]}},
    )
    before = copy.deepcopy(document)

    chunks = DocumentChunker(chunk_size=80, chunk_overlap=15).split([document])

    assert len(chunks) > 1
    assert all(0 < len(chunk.page_content) <= 80 for chunk in chunks)
    assert all(
        any(f"token{i}" in chunk.page_content for chunk in chunks) for i in range(4)
    )
    assert all(chunk.metadata["nested"] == {"tags": ["a"]} for chunk in chunks)
    assert all(chunk.metadata["source"] == "guide.txt" for chunk in chunks)
    chunks[0].metadata["nested"]["tags"].append("changed")
    assert document == before
    assert all(chunk.metadata["nested"] == {"tags": ["a"]} for chunk in chunks[1:])


def test_markdown_headers_and_recursive_splitting_through_loader(tmp_path):
    content = (
        "# Handbook\n\nIntroduction marker.\n\n"
        "## Billing\n\nBilling marker.\n\n"
        "### Refunds\n\n"
        + " ".join(f"refund{i}" for i in range(35))
        + "\n\n## Technical\n\nTechnical marker.\n"
    )
    (tmp_path / "handbook.md").write_text(content, encoding="utf-8")
    documents = KnowledgeBaseLoader(tmp_path).load()
    before = copy.deepcopy(documents)

    chunks = DocumentChunker(chunk_size=90, chunk_overlap=10).split(documents)

    assert all(len(chunk.page_content) <= 90 for chunk in chunks)
    assert all(
        all(
            chunk.metadata[key] == value for key, value in documents[0].metadata.items()
        )
        for chunk in chunks
    )
    assert any(
        "Introduction marker" in c.page_content and c.metadata["header_1"] == "Handbook"
        for c in chunks
    )
    assert any(
        "Billing marker" in c.page_content and c.metadata["header_2"] == "Billing"
        for c in chunks
    )
    refunds = [c for c in chunks if c.metadata.get("header_3") == "Refunds"]
    assert len(refunds) > 1
    assert all(
        c.metadata["header_1"] == "Handbook" and c.metadata["header_2"] == "Billing"
        for c in refunds
    )
    assert all(
        any(f"refund{i}" in c.page_content.split() for c in refunds) for i in range(35)
    )
    assert any(
        "Technical marker" in c.page_content
        and c.metadata.get("header_2") == "Technical"
        and "header_3" not in c.metadata
        for c in chunks
    )
    assert documents == before


def test_indices_ids_and_order_are_deterministic():
    docs = [
        Document(
            page_content=" ".join(f"{source}{i}" for i in range(30)),
            metadata={"source": source, "file_type": "txt"},
        )
        for source in ("z.txt", "a.txt")
    ]
    chunker = DocumentChunker(chunk_size=45, chunk_overlap=5)
    chunks = chunker.split(docs)
    assert (
        chunks
        == chunker.split(docs)
        == DocumentChunker(45, 5).split(list(reversed(docs)))
    )
    sources = [c.metadata["source"] for c in chunks]
    assert sources == sorted(sources)
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    for source in ("a.txt", "z.txt"):
        group = [c for c in chunks if c.metadata["source"] == source]
        assert len(group) > 1
        assert [c.metadata["chunk_index"] for c in group] == list(range(len(group)))
        assert [c.metadata["chunk_id"] for c in group] == [
            f"{source}::chunk-{i:04d}" for i in range(len(group))
        ]


@pytest.mark.parametrize(
    "size, overlap", [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)]
)
def test_invalid_sizes_raise_value_error(size, overlap):
    with pytest.raises(ValueError):
        DocumentChunker(size, overlap)


@pytest.mark.parametrize(
    "size, overlap", [(True, 0), (10, False), (1.5, 0), (10, "1"), (None, 0)]
)
def test_non_integer_sizes_raise_type_error(size, overlap):
    with pytest.raises(TypeError):
        DocumentChunker(size, overlap)


@pytest.mark.parametrize("source", [None, "", "  ", 123])
def test_missing_or_invalid_source_raises_value_error(source):
    metadata = {"file_type": "txt"}
    if source is not None:
        metadata["source"] = source
    with pytest.raises(ValueError, match="source"):
        DocumentChunker().split([Document(page_content="content", metadata=metadata)])


@pytest.mark.parametrize("file_type", ["pdf", "", None, 123])
def test_unsupported_file_type_raises_value_error(file_type):
    document = Document(
        page_content="# Heading\n\nPlain content",
        metadata={"source": "file.pdf", "file_type": file_type},
    )
    with pytest.raises(ValueError, match="file_type"):
        DocumentChunker().split([document])


@pytest.mark.parametrize("file_type", ["txt", "md"])
@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_document_content_raises_value_error(file_type, content):
    with pytest.raises(ValueError, match="empty"):
        DocumentChunker().split(
            [
                Document(
                    page_content=content,
                    metadata={"source": "empty", "file_type": file_type},
                )
            ]
        )


def test_empty_input_produces_no_chunks():
    assert DocumentChunker().split([]) == []
