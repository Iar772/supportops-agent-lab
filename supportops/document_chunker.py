import copy

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int = 350,
        chunk_overlap: int = 50,
    ) -> None:
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
            raise TypeError("chunk_size must be a true integer")
        if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int):
            raise TypeError("chunk_overlap must be a true integer")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        headers_to_split_on = [
            ("#", "header_1"),
            ("##", "header_2"),
            ("###", "header_3"),
        ]
        self.md_header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )

    def split(self, documents: list[Document]) -> list[Document]:
        if not documents:
            return []

        for doc in documents:
            source = doc.metadata.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("document source must be a non-empty string")
            file_type = doc.metadata.get("file_type")
            if not isinstance(file_type, str) or file_type.lower() not in {"txt", "md"}:
                raise ValueError(f"unsupported file_type for {source}: {file_type!r}")
            if not doc.page_content.strip():
                raise ValueError(f"document content is empty: {source}")

        sorted_docs = sorted(documents, key=lambda d: d.metadata["source"])
        final_chunks: list[Document] = []

        for doc in sorted_docs:
            source = doc.metadata["source"]
            file_type = doc.metadata["file_type"].lower()
            blocks_to_split = []

            if file_type == "md":
                md_sections = self.md_header_splitter.split_text(doc.page_content)
                for section in md_sections:
                    section_metadata = section.metadata
                    blocks_to_split.append((section.page_content, section_metadata))
            else:
                blocks_to_split.append((doc.page_content, {}))

            chunk_index = 0

            for text_block, block_meta in blocks_to_split:
                sub_chunks = self.text_splitter.split_text(text_block)

                for chunk_text in sub_chunks:
                    new_metadata = copy.deepcopy(doc.metadata)
                    if block_meta:
                        new_metadata.update(block_meta)

                    new_metadata["chunk_index"] = chunk_index
                    new_metadata["chunk_id"] = f"{source}::chunk-{chunk_index:04d}"

                    new_doc = Document(page_content=chunk_text, metadata=new_metadata)
                    final_chunks.append(new_doc)
                    chunk_index += 1

        return final_chunks
