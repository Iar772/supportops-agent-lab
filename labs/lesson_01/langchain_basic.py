import os
from langchain_ollama import ChatOllama


model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
    temperature=0,
    reasoning=False,
    num_ctx=4096,
    num_predict=256,
)

messages = [
    (
        "system",
        "You are a technical support assistant. Answer in Russian.",
    ),
    (
        "human",
        "What does HTTP 429 mean?",
    ),
]

response = model.invoke(messages)

print("Answer:", response.content)
print("Usage:", response.usage_metadata)
print("Metadata:", response.response_metadata)
