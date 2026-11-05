from pathlib import Path

from langchain_core.documents import Document


class KnowledgeBaseLoader:
    def __init__(self, root: Path) -> None:
        self.root = root
        if not self.root.exists():
            raise FileNotFoundError(f"{root} does not exist")
        self.is_directory(self.root)

    def is_directory(self, path: Path) -> bool:
        if path.is_dir():
            return True
        raise ValueError(f"{path} is not a directory")

    def load(self) -> list[Document]:
        documents = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".txt", ".md"):
                continue

            relative_path = path.relative_to(self.root)
            file_type = path.suffix.lower().lstrip(".")
            category_name = (
                relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
            )
            file_name = relative_path.name

            try:
                content = path.read_text(encoding="utf-8")
                cleaned_content = content.strip()
                if not cleaned_content:
                    raise ValueError(f"File {relative_path} is empty")

            except UnicodeDecodeError as e:
                raise ValueError(f"Error decoding {relative_path}: {e}")

            metadata = {
                "source": relative_path.as_posix(),
                "category": category_name,
                "file_type": file_type,
                "file_name": file_name,
            }
            document = Document(
                page_content=cleaned_content,
                metadata=metadata,
            )
            documents.append(document)
        return documents
