from copy import deepcopy
from typing import Any, Protocol

from supportops.llm_client import LLMResponse
from supportops.tool_runner import ToolRunner


class LLMClient(Protocol):
    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict] | None = None
    ) -> "LLMResponse": ...


class Conversation:
    def __init__(
        self, client: LLMClient, runner: ToolRunner, system_prompt: str | None = None
    ):
        self.client = client
        self.runner = runner
        self._messages: list[dict[str, Any]] = []

        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    @property
    def messages(self) -> list[dict[str, Any]]:
        return deepcopy(self._messages)

    def ask(self, question: str, tools: list[dict] | None = None) -> "LLMResponse":
        user_message = {"role": "user", "content": question}
        messages = deepcopy(self._messages) + [user_message]

        max_turns = 3

        for _ in range(max_turns):
            response = self.client.chat(deepcopy(messages), tools=tools)

            assistant_message = {"role": "assistant", "content": response.content}

            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {"function": {"name": tc.name, "arguments": deepcopy(tc.arguments)}}
                    for tc in response.tool_calls
                ]

            messages.append(assistant_message)

            if not response.tool_calls:
                self._messages = messages
                return response

            for tool_call in response.tool_calls:
                tool_result_message = self.runner.run(tool_call)

                messages.append(tool_result_message)

        raise RuntimeError(f"Max count requests to model exceeded ({max_turns}).")
