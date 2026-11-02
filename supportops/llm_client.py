from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    tool_calls: tuple[ToolCall, ...] = ()


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: int = 120):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout

    def chat(self, messages: list, tools: list[dict] | None = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools or [],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 256,
            },
        }
        response = httpx.post(
            url=self.base_url + "/api/chat", json=payload, timeout=self.timeout
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                "Invalid Ollama response: expected a JSON object"
            )

        message = data.get("message")
        if not isinstance(message, dict):
            raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                "Invalid Ollama response: 'message' must be an object"
            )

        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                "Invalid Ollama response: 'message.content' must be a string"
            )
        raw_tool_calls = message.get("tool_calls")
        if raw_tool_calls is None:
            raw_tool_calls = []
        if not isinstance(raw_tool_calls, list):
            raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                "Invalid Ollama response: 'message.tool_calls' must be an array"
            )
        parsed_tool_calls = []

        for tool_call in raw_tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                    f"Invalid tool call structure: expected a dictionary, got {type(tool_call).__name__}"
                )

            if "function" not in tool_call:
                raise ValueError(
                    "Invalid tool call structure: missing mandatory 'function' envelope"
                )

            func_data = tool_call.get("function")
            if not isinstance(func_data, dict):
                raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                    f"Invalid 'function' data: expected a dictionary, got {type(func_data).__name__}"
                )

            name = func_data.get("name")
            arguments = func_data.get("arguments")

            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Invalid or missing tool name in model payload: {name}"
                )

            if not isinstance(arguments, dict):
                raise ValueError(  # noqa: TRY004 - invalid response data, not a caller type error
                    f"Invalid arguments block format for tool '{name}': "
                    f"expected an object/dictionary, got {type(arguments).__name__}"
                )

            parsed_tool_calls.append(ToolCall(name=name, arguments=arguments))

        tool_calls_tuple = tuple(parsed_tool_calls)
        has_content = isinstance(content, str) and bool(content.strip())
        has_tool = len(tool_calls_tuple) > 0

        if not has_content and not has_tool:
            raise ValueError(
                "Invalid Ollama response: must contain either non-empty content or valid tool calls"
            )

        return LLMResponse(
            content=content,
            model=self.model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            duration_seconds=data.get("total_duration", 0) / 1_000_000_000,
            tool_calls=tool_calls_tuple,
        )
