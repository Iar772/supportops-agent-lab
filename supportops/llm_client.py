
from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float



class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 256,
            },
        }
        response = httpx.post(
            url=self.base_url + "/api/chat",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid Ollama response: expected a JSON object")

        message = data.get("message")
        if not isinstance(message, dict):
            raise ValueError("Invalid Ollama response: 'message' must be an object")

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(
                "Invalid Ollama response: 'message.content' must be a non-empty string"
            )

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            duration_seconds=data.get("total_duration", 0) / 1_000_000_000
        )
