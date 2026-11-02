import json
from unittest.mock import Mock

import pytest

from supportops.conversation import Conversation, LLMClient
from supportops.llm_client import LLMResponse, ToolCall
from supportops.tool_runner import ToolRunner


@pytest.mark.parametrize(
    ("question", "tool_call", "tool_result", "answer"),
    [
        pytest.param(
            "What is the status of ticket 123?",
            ToolCall("get_ticket_status", {"ticket_id": 123}),
            {"ticket_id": 123, "status": "open", "priority": "high"},
            "Ticket 123 is open with high priority.",
            id="ticket-status",
        ),
        pytest.param(
            "What are the support working hours?",
            ToolCall("get_support_policy", {"topic": "working_hours"}),
            {"response_time_hours": "Mon–Fri, 09:00–18:00 UTC"},
            "Support is available Mon–Fri, 09:00–18:00 UTC.",
            id="working-hours",
        ),
        pytest.param(
            "How quickly do you respond to urgent requests?",
            ToolCall("get_support_policy", {"topic": "high_priority"}),
            {"response_time_hours": "Initial response within 4 hours"},
            "The initial response to an urgent request is within 4 hours.",
            id="high-priority",
        ),
        pytest.param("Hi!", None, None, "Hello!", id="greeting"),
        pytest.param(
            "Check ticket 999",
            ToolCall("get_ticket_status", {"ticket_id": 999}),
            {"ticket_id": 999, "error": "not_found"},
            "Ticket 999 was not found.",
            id="ticket-not-found",
        ),
    ],
)
def test_support_scenarios(question, tool_call, tool_result, answer):
    """Script the LLM decision; exercise actual tool execution without a model."""
    final_response = LLMResponse(answer, "fake-model", 10, 5, 0.1)
    outcomes = []
    if tool_call is not None:
        outcomes.append(LLMResponse("", "fake-model", 10, 5, 0.1, (tool_call,)))
    outcomes.append(final_response)
    client = Mock(spec=LLMClient)
    client.chat.side_effect = outcomes
    runner = Mock(spec=ToolRunner, wraps=ToolRunner())
    conversation = Conversation(client, runner)
    schemas = [
        {"type": "function", "function": {"name": "get_ticket_status"}},
        {"type": "function", "function": {"name": "get_support_policy"}},
    ]

    result = conversation.ask(question, tools=schemas)
    model_calls = client.chat.call_args_list
    payloads = [call.args[0] for call in model_calls]

    user_message = {"role": "user", "content": question}
    assert result is final_response
    assert payloads[0] == [user_message]
    final_message = {"role": "assistant", "content": answer}
    if tool_call is None:
        runner.run.assert_not_called()
        assert payloads == [[user_message]]
        assert conversation.messages == [user_message, final_message]
    else:
        runner.run.assert_called_once_with(tool_call)
        assert len(payloads) == 2
        assert payloads[1][:2] == [
            user_message,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                        }
                    }
                ],
            },
        ]
        tool_message = payloads[1][2]
        assert tool_message["role"] == "tool"
        assert tool_message["tool_name"] == tool_call.name
        assert json.loads(tool_message["content"]) == tool_result
        assert conversation.messages == payloads[1] + [final_message]
    assert [call.kwargs["tools"] for call in model_calls] == [schemas] * len(outcomes)
    assert client.chat.call_count == len(outcomes)
