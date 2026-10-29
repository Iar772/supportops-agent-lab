from typing import List, Dict, Optional, Protocol
from dataclasses import dataclass
from copy import deepcopy
from supportops.llm_client import LLMResponse


class LLMClient(Protocol):
    def chat(self, messages: List[Dict[str, str]]) -> "LLMResponse":
        ...


class Conversation:
    def __init__(self, client: LLMClient, system_prompt: Optional[str] = None):
        self.client = client
        self._messages: List[Dict[str, str]] = []

        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> List[Dict[str, str]]:
        return deepcopy(self._messages)

    def ask(self, question: str) -> "LLMResponse":
        user_message = {"role": "user", "content": question}
        payload_messages = self._messages + [user_message]

        response = self.client.chat(payload_messages)

        self._messages.append(user_message)
        self._messages.append({"role": "assistant", "content": response.content})

        return response
