
import os
from typing import Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field


class ServiceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    service: Literal["billing", "gateway", "authentication", "other"]
    status: Literal["operational", "degraded", "outage", "unknown"]
    description: str = Field(min_length=5, max_length=200)


def main() -> None:
    response = httpx.post(
        "http://localhost:11434/api/chat",
        json={
            "model": os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Extract service status information. "
                        "Do not invent missing details. "
                        "Choose 'unknown' when the status is unclear."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Billing API is degraded. "
                        "Some payment requests are failing."
                    ),
                },
            ],
            "format": ServiceEvent.model_json_schema(),
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 512,
            },
        },
        timeout=120.0,
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    print("RAW JSON:", content)

    event = ServiceEvent.model_validate_json(content)

    print("PYDANTIC:", event)
    print("AS DICT:", event.model_dump())


if __name__ == "__main__":
    main()
