import inspect
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
        tool = self._tools[tool_name]

        if not isinstance(tool_arguments, dict):
            raise ValueError(  # noqa: TRY004 - invalid model output, not a caller type error
                f"Tool arguments for '{tool_name}' must be a dictionary/object, "
                f"got {type(tool_arguments).__name__}"
            )

        sig = inspect.signature(tool)
        params = sig.parameters

        for arg_name in tool_arguments:
            if arg_name not in params:
                raise ValueError(
                    f"Unexpected argument '{arg_name}' for tool '{tool_name}'"
                )

        for param_name, param in params.items():
            if (
                param.default is inspect.Parameter.empty
                and param_name not in tool_arguments
            ):
                raise ValueError(
                    f"Missing required argument '{param_name}' for tool '{tool_name}'"
                )

            if param_name in tool_arguments:
                value = tool_arguments[param_name]
                expected_type = param.annotation

                # The registered tools use concrete int/str annotations; bool is not a ticket ID.
                if (
                    expected_type is not inspect.Parameter.empty
                    and type(value) is not expected_type
                ):
                    raise ValueError(
                        f"Invalid type for argument '{param_name}' in tool '{tool_name}'. "
                        f"Expected {expected_type.__name__}, got {type(value).__name__}"
                    )

        result = tool(**tool_arguments)

        if (
            isinstance(result, dict)
            and "ticket_id" not in result
            and "ticket_id" in tool_arguments
        ):
            result = {"ticket_id": tool_arguments["ticket_id"], **result}

        return {
            "role": "tool",
            "tool_name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        }
