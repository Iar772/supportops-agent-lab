
from langchain_text_splitters import RecursiveCharacterTextSplitter


# text = """
# The payment gateway is currently operational.
# All payment requests are being processed normally.
#
# If an HTTP 429 error occurs, check the
# rate-limit headers before retrying the request.
#
# If an HTTP 502 error occurs, verify the
# gateway logs and upstream service availability.
#
# High-priority production incidents must be
# escalated to the on-call engineer.
#
# """.strip()
#
#
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=80,
#     chunk_overlap=20,
#     length_function=len,
# )
#
# chunks = splitter.split_text(text)
#
# for index, chunk in enumerate(chunks):
#     print(f"\n--- CHUNK {index} ---")
#     print(chunk)
#     print("Length:", len(chunk))


from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


document = Document(
    page_content=(
        "The payment gateway is currently operational. All payment requests are being processed normally. "
        "If an HTTP 429 error occurs, check the rate-limit headers before retrying the request. "
        "If an HTTP 502 error occurs, verify the gateway logs and upstream service availability. "
        "High-priority production incidents must be escalated to the on-call engineer."
    ),
    metadata={
        "source": "technical/api_errors.md",
        "category": "technical",
    },
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40,
)

chunks = splitter.split_documents([document])

for index, chunk in enumerate(chunks):
    print(index)
    print(chunk.page_content)
    print(chunk.metadata)
    print(len(chunk.page_content))
