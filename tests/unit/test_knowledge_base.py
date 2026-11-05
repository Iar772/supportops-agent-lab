from pathlib import Path

import pytest
from langchain_core.documents import Document

from supportops.knowledge_base import KnowledgeBaseLoader

pytestmark = pytest.mark.usefixtures("offline_langchain")


@pytest.mark.parametrize("extension", ["md", "txt"])
def test_loads_document_with_content_and_root_metadata(tmp_path, extension):
    content = " \t\n# Support\n\n  Café — помощь  \nNext  line\n\n\t "
    filename = f"readme.{extension}"
    (tmp_path / filename).write_text(content, encoding="utf-8")

    documents = KnowledgeBaseLoader(tmp_path).load()

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, Document)
    assert document.page_content == "# Support\n\n  Café — помощь  \nNext  line"
    assert document.metadata == {
        "source": filename,
        "category": "general",
        "file_type": extension,
        "file_name": filename,
    }


@pytest.mark.parametrize(
    ("relative_path", "category", "file_type"),
    [
        ("billing/refunds.md", "billing", "md"),
        ("technical/api/errors.txt", "technical", "txt"),
        ("support/policies/urgent/escalation.md", "support", "md"),
    ],
)
def test_nested_documents_have_relative_source_and_top_level_category(
    tmp_path, relative_path, category, file_type
):
    path = tmp_path / relative_path
    path.parent.mkdir(parents=True)
    path.write_text("Support policy", encoding="utf-8")

    documents = KnowledgeBaseLoader(tmp_path).load()

    assert len(documents) == 1
    document = documents[0]
    assert document.page_content == "Support policy"
    assert document.metadata["source"] == relative_path
    assert document.metadata["category"] == category
    assert document.metadata["file_type"] == file_type


def test_deeply_nested_file_name_is_basename(tmp_path):
    path = tmp_path / "technical" / "api" / "errors.md"
    path.parent.mkdir(parents=True)
    path.write_text("API errors", encoding="utf-8")

    [document] = KnowledgeBaseLoader(tmp_path).load()

    assert document.metadata["file_name"] == "errors.md"


@pytest.mark.parametrize("extension", ["MD", "TXT", "Md", "Txt"])
def test_supported_extensions_are_case_insensitive(tmp_path, extension):
    filename = f"policy.{extension}"
    (tmp_path / filename).write_text("Policy", encoding="utf-8")

    documents = KnowledgeBaseLoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].page_content == "Policy"
    assert documents[0].metadata["source"] == filename


@pytest.mark.parametrize("reverse_traversal", [False, True])
def test_document_order_is_sorted_by_relative_path(
    tmp_path, monkeypatch, reverse_traversal
):
    expected_sources = [
        "alpha/zebra.txt",
        "middle.md",
        "zulu/aardvark.md",
        "zulu/nested/policy.txt",
    ]
    # Create files in reverse order, independently of the expected load order.
    for source in reversed(expected_sources):
        path = tmp_path / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")

    traversal = sorted(tmp_path.rglob("*"), reverse=reverse_traversal)
    # Control enumeration so filesystem-specific ordering cannot hide a regression.
    monkeypatch.setattr(Path, "rglob", lambda *_args, **_kwargs: iter(traversal))
    loader = KnowledgeBaseLoader(tmp_path)

    documents = loader.load()

    assert [document.metadata["source"] for document in documents] == expected_sources
    assert [document.page_content for document in documents] == expected_sources
    assert loader.load() == documents
    assert KnowledgeBaseLoader(tmp_path).load() == documents


def test_content_trimmed_before_storing(tmp_path):
    (tmp_path / "policy.md").write_text(
        "\n  Important rule.  \n",
        encoding="utf-8",
    )

    [document] = KnowledgeBaseLoader(tmp_path).load()

    assert document.page_content == "Important rule."


def test_missing_root_raises_file_not_found(tmp_path):
    root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError) as exc_info:
        KnowledgeBaseLoader(root)

    assert str(root) in str(exc_info.value)


def test_file_root_raises_value_error(tmp_path):
    root = tmp_path / "policy.md"
    root.write_text("Policy", encoding="utf-8")

    with pytest.raises(ValueError, match="not a directory") as exc_info:
        KnowledgeBaseLoader(root)

    assert str(root) in str(exc_info.value)


@pytest.mark.parametrize("extension", ["md", "txt"])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param("", id="empty"),
        pytest.param(" \t\n\r\n", id="whitespace-only"),
    ],
)
def test_empty_content_raises_with_relative_source(tmp_path, extension, content):
    (tmp_path / "a_valid.md").write_text("Valid policy", encoding="utf-8")
    path = tmp_path / "support" / f"empty.{extension}"
    path.parent.mkdir()
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="empty") as exc_info:
        KnowledgeBaseLoader(tmp_path).load()

    assert f"support/empty.{extension}" in str(exc_info.value)


@pytest.mark.parametrize("extension", ["md", "txt"])
def test_invalid_utf8_raises_value_error_with_relative_source(tmp_path, extension):
    (tmp_path / "a_valid.txt").write_text("Valid policy", encoding="utf-8")
    path = tmp_path / "technical" / f"invalid.{extension}"
    path.parent.mkdir()
    path.write_bytes(b"Invalid UTF-8: \xff\xfe")

    with pytest.raises(ValueError, match="Error decoding") as exc_info:
        KnowledgeBaseLoader(tmp_path).load()

    assert f"technical/invalid.{extension}" in str(exc_info.value)


@pytest.mark.parametrize(
    "filename", ["policy.pdf", "policy.json", "policy.html", "policy.md.bak", "README"]
)
def test_unsupported_files_are_skipped_without_decoding(tmp_path, filename):
    (tmp_path / filename).write_bytes(b"\xff\xfe")
    (tmp_path / "valid.md").write_text("Valid policy", encoding="utf-8")

    documents = KnowledgeBaseLoader(tmp_path).load()

    assert len(documents) == 1
    assert documents[0].metadata["source"] == "valid.md"
    assert documents[0].page_content == "Valid policy"


@pytest.mark.parametrize("layout", ["empty", "directories-only", "unsupported-only"])
def test_empty_knowledge_base_returns_empty_list(tmp_path, layout):
    if layout == "directories-only":
        (tmp_path / "support" / "empty.md").mkdir(parents=True)
        (tmp_path / "technical" / "empty.txt").mkdir(parents=True)
    elif layout == "unsupported-only":
        (tmp_path / "policy.pdf").write_bytes(b"\xff\xfe")

    assert KnowledgeBaseLoader(tmp_path).load() == []
