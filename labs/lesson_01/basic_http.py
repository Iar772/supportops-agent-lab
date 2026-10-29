import os
import httpx


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")


def main() -> None:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a technical support assistant. "
                    "Answer in English. Be concise."
                ),
            },
            {
                "role": "user",
                "content": "What does HTTP 429 mean?",
            },
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": 4096,
            "num_predict": 256,
        },
    }

    response = httpx.post(
        url=OLLAMA_URL,
        json=payload,
        timeout=120.0,
    )
    response.raise_for_status()

    data = response.json()

    print("Answer:", data["message"]["content"])
    print("Input tokens:", data.get("prompt_eval_count"))
    print("Output tokens:", data.get("eval_count"))
    print(
        "Duration (s):",
        round(data["total_duration"] / 1_000_000_000, 2),
    )


if __name__ == "__main__":
    main()
