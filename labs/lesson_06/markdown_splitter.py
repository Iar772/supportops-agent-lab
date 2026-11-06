
from langchain_text_splitters import MarkdownHeaderTextSplitter


markdown = """
# Refund Policy

## Duplicate Charges

Verify both transaction IDs.

## Processing Time

Approved refunds take 5 business days.
""".strip()


header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "header_1"),
        ("##", "header_2"),
    ],
    strip_headers=False,
)

sections = header_splitter.split_text(markdown)

for section in sections:
    print("\nCONTENT:")
    print(section.page_content)

    print("METADATA:")
    print(section.metadata)
