import json
from collections.abc import Callable

from supportops.llm_client import ToolCall
from supportops.tools import get_support_policy, get_ticket_status


class ToolRunner:
    def __init__(self) -> None:
        self._tools: dict[str, Callable] = {
            "get_ticket_status": get_ticket_status,
            "get_support_policy": get_support_policy,
        }

    def run(self, tool_call: ToolCall) -> dict:
        if tool_call.name not in self._tools:
            raise ValueError(f"Tool {tool_call.name} not found.")

        tool_name = tool_call.name
        tool_arguments = tool_call.arguments
        print("\n--- Tool arguments ---")
        print(tool_arguments)
        print("\n--- Tool description ---")
        print(tool_name)
        tool = self._tools[tool_name]

        if not isinstance(tool_arguments, dict):
            raise ValueError(  # noqa: TRY004 - invalid model output, not a caller type error
                f"Tool arguments for '{tool_name}' must be a dictionary/object, got {type(tool_arguments).__name__}"
            )

        try:
            result = tool(**tool_arguments)
        except TypeError as e:
            raise ValueError(f"Invalid arguments for tool '{tool_name}': {e}")

        if not isinstance(result, dict):
            result = {"error": result}

        if "ticket_id" not in result and "ticket_id" in tool_arguments:
            result = {"ticket_id": tool_arguments["ticket_id"], **result}

        return {
            "role": "tool",
            "tool_name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        }
