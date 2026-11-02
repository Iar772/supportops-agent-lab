import json

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from supportops.langchain_tools import get_support_policy, get_ticket_status

pytestmark = pytest.mark.usefixtures("offline_langchain")

SYSTEM_PROMPT = "You are a technical support assistant. Answer using the tool results."


@pytest.fixture
def agent_factory(scripted_model):
    def make_agent(responses):
        model = scripted_model(responses=responses)
        agent = create_agent(
            model=model,
            tools=[get_ticket_status, get_support_policy],
            system_prompt=SYSTEM_PROMPT,
        )
        return agent, model

    return make_agent


@pytest.mark.parametrize(
    ("question", "name", "arguments", "expected", "answer"),
    [
        pytest.param(
            "Check ticket 123",
            "get_ticket_status",
            {"ticket_id": 123},
            {"status": "open", "priority": "high"},
            "Ticket 123 is open with high priority.",
            id="ticket-status",
        ),
        pytest.param(
            "Check ticket 999",
            "get_ticket_status",
            {"ticket_id": 999},
            {"error": "not_found"},
            "Ticket 999 was not found.",
            id="ticket-not-found",
        ),
        pytest.param(
            "What are the support working hours?",
            "get_support_policy",
            {"topic": "working_hours"},
            {"response_time_hours": "Mon–Fri, 09:00–18:00 UTC"},
            "Support is available Mon–Fri, 09:00–18:00 UTC.",
            id="working-hours",
        ),
        pytest.param(
            "How quickly do you respond to urgent requests?",
            "get_support_policy",
            {"topic": "high_priority"},
            {"response_time_hours": "Initial response within 4 hours"},
            "The initial response is within 4 hours.",
            id="high-priority",
        ),
    ],
)
def test_agent_executes_one_tool_and_returns_result_to_model(
    agent_factory, backend_spies, question, name, arguments, expected, answer
):
    call = {"name": name, "args": arguments, "id": "support-call-1"}
    agent, model = agent_factory(
        [AIMessage(content="", tool_calls=[call]), AIMessage(content=answer)]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 10},
    )

    messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    assert messages[0].content == question
    assert messages[1].tool_calls == [{**call, "type": "tool_call"}]
    tool_message = messages[2]
    assert tool_message.tool_call_id == call["id"]
    assert tool_message.name == name
    assert tool_message.status == "success"
    assert json.loads(tool_message.content) == expected
    assert messages[-1].content == answer
    assert not messages[-1].tool_calls
    backend_spies[name].assert_called_once_with(*arguments.values())
    for other_name, spy in backend_spies.items():
        if other_name != name:
            spy.assert_not_called()

    assert len(model.calls) == 2
    assert isinstance(model.calls[0][0], SystemMessage)
    assert model.calls[0][0].content == SYSTEM_PROMPT
    assert isinstance(model.calls[0][1], HumanMessage)
    assert model.calls[0][1].content == question
    assert model.calls[1][-1] == tool_message
    assert model.tool_bindings
    for binding in model.tool_bindings:
        assert {tool["function"]["name"] for tool in binding} == {
            "get_ticket_status",
            "get_support_policy",
        }


def test_agent_executes_two_tools_before_requesting_final_answer(
    agent_factory, backend_spies
):
    calls = [
        {"name": "get_ticket_status", "args": {"ticket_id": 123}, "id": "ticket-call"},
        {
            "name": "get_support_policy",
            "args": {"topic": "high_priority"},
            "id": "policy-call",
        },
    ]
    answer = (
        "Ticket 123 is open with high priority. The initial response is within 4 hours."
    )
    agent, model = agent_factory(
        [AIMessage(content="", tool_calls=calls), AIMessage(content=answer)]
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Check ticket 123 and the urgent response policy",
                }
            ]
        },
        config={"recursion_limit": 10},
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 2
    assert {
        message.tool_call_id: (
            message.name,
            message.status,
            json.loads(message.content),
        )
        for message in tool_messages
    } == {
        "ticket-call": (
            "get_ticket_status",
            "success",
            {"status": "open", "priority": "high"},
        ),
        "policy-call": (
            "get_support_policy",
            "success",
            {"response_time_hours": "Initial response within 4 hours"},
        ),
    }
    backend_spies["get_ticket_status"].assert_called_once_with(123)
    backend_spies["get_support_policy"].assert_called_once_with("high_priority")
    assert len(model.calls) == 2
    assert [
        message for message in model.calls[1] if isinstance(message, ToolMessage)
    ] == tool_messages
    assert result["messages"][-1].content == answer
    assert not result["messages"][-1].tool_calls


def test_agent_answers_without_executing_tools(agent_factory, backend_spies):
    agent, model = agent_factory([AIMessage(content="Hello! How can I help?")])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Hi!"}]},
        config={"recursion_limit": 10},
    )

    assert [message.type for message in result["messages"]] == ["human", "ai"]
    assert result["messages"][-1].content == "Hello! How can I help?"
    assert not result["messages"][-1].tool_calls
    assert len(model.calls) == 1
    for spy in backend_spies.values():
        spy.assert_not_called()


@pytest.mark.parametrize(
    ("name", "arguments", "field"),
    [
        pytest.param(
            "get_ticket_status", {"ticket_id": "abc"}, "ticket_id", id="ticket-id-abc"
        ),
        pytest.param(
            "get_ticket_status", {"ticket_id": True}, "ticket_id", id="ticket-id-bool"
        ),
        pytest.param("get_support_policy", {"topic": 123}, "topic", id="topic-integer"),
    ],
)
def test_agent_returns_argument_validation_error_to_model(
    agent_factory, backend_spies, name, arguments, field
):
    agent, model = agent_factory(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": arguments, "id": "invalid-call"}],
            ),
            AIMessage(content="Please provide a valid argument."),
        ]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Check the support details"}]},
        config={"recursion_limit": 10},
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    error = tool_messages[0]
    assert error.tool_call_id == "invalid-call"
    assert error.name == name
    assert error.status == "error"
    assert field in error.content
    for spy in backend_spies.values():
        spy.assert_not_called()
    assert len(model.calls) == 2
    assert model.calls[1][-1] == error
    assert result["messages"][-1].content == "Please provide a valid argument."


def test_agent_can_correct_an_invalid_tool_call(agent_factory, backend_spies):
    agent, model = agent_factory(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_ticket_status",
                        "args": {"ticket_id": True},
                        "id": "invalid-ticket",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_ticket_status",
                        "args": {"ticket_id": 123},
                        "id": "corrected-ticket",
                    }
                ],
            ),
            AIMessage(content="Ticket 123 is open with high priority."),
        ]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Check ticket 123"}]},
        config={"recursion_limit": 10},
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert [(message.tool_call_id, message.status) for message in tool_messages] == [
        ("invalid-ticket", "error"),
        ("corrected-ticket", "success"),
    ]
    assert json.loads(tool_messages[1].content) == {
        "status": "open",
        "priority": "high",
    }
    backend_spies["get_ticket_status"].assert_called_once_with(123)
    backend_spies["get_support_policy"].assert_not_called()
    assert len(model.calls) == 3
    assert model.calls[1][-1] == tool_messages[0]
    assert model.calls[2][-1] == tool_messages[1]
    assert result["messages"][-1].content == "Ticket 123 is open with high priority."


def test_agent_reports_unknown_tool_without_executing_backends(
    agent_factory, backend_spies
):
    agent, model = agent_factory(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete_ticket",
                        "args": {"ticket_id": 123},
                        "id": "unknown-tool",
                    }
                ],
            ),
            AIMessage(content="That tool is unavailable."),
        ]
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Delete ticket 123"}]},
        config={"recursion_limit": 10},
    )

    tool_messages = [
        message for message in result["messages"] if isinstance(message, ToolMessage)
    ]
    assert len(tool_messages) == 1
    error = tool_messages[0]
    assert error.tool_call_id == "unknown-tool"
    assert error.name == "delete_ticket"
    assert error.status == "error"
    assert "delete_ticket" in error.content
    for spy in backend_spies.values():
        spy.assert_not_called()
    assert len(model.calls) == 2
    assert model.calls[1][-1] == error
    assert result["messages"][-1].content == "That tool is unavailable."


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_ticket_status", {"ticket_id": 123}),
        ("get_support_policy", {"topic": "working_hours"}),
    ],
)
def test_agent_propagates_unhandled_backend_errors(
    agent_factory, backend_spies, name, arguments
):
    error = RuntimeError("Support backend is unavailable")
    backend_spies[name].side_effect = error
    agent, model = agent_factory(
        [
            AIMessage(
                content="",
                tool_calls=[{"name": name, "args": arguments, "id": "backend-error"}],
            )
        ]
    )

    with pytest.raises(RuntimeError) as exc_info:
        agent.invoke(
            {"messages": [{"role": "user", "content": "Check the support details"}]},
            config={"recursion_limit": 10},
        )

    assert exc_info.value is error
    backend_spies[name].assert_called_once_with(*arguments.values())
    for other_name, spy in backend_spies.items():
        if other_name != name:
            spy.assert_not_called()
    assert len(model.calls) == 1
