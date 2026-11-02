from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from labs.lesson_03 import support_agent_demo
from supportops import langchain_tools

pytestmark = pytest.mark.usefixtures("offline_langchain")


@pytest.mark.parametrize(
    ("name", "field", "json_type"),
    [
        ("get_ticket_status", "ticket_id", "integer"),
        ("get_support_policy", "topic", "string"),
    ],
)
def test_adapter_exposes_tool_description_and_argument_schema(name, field, json_type):
    adapter = getattr(langchain_tools, name)
    schema = adapter.get_input_schema().model_json_schema()

    assert adapter.name == name
    assert adapter.description.strip()
    assert schema["required"] == [field]
    assert schema["properties"][field]["type"] == json_type


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        pytest.param(
            "get_ticket_status",
            {"ticket_id": 123},
            {"status": "open", "priority": "high"},
            id="open-ticket",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": 456},
            {"status": "closed", "priority": "low"},
            id="closed-ticket",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": 789},
            {"status": "in_progress", "priority": "medium"},
            id="in-progress-ticket",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": 999},
            {"error": "not_found"},
            id="missing-ticket",
        ),
        pytest.param(
            "get_support_policy",
            {"topic": "working_hours"},
            {"response_time_hours": "Mon–Fri, 09:00–18:00 UTC"},
            id="working-hours",
        ),
        pytest.param(
            "get_support_policy",
            {"topic": "high_priority"},
            {"response_time_hours": "Initial response within 4 hours"},
            id="high-priority",
        ),
        pytest.param(
            "get_support_policy",
            {"topic": "response_time"},
            {"response_time_hours": "Initial response within 24 hours"},
            id="response-time",
        ),
        pytest.param(
            "get_support_policy",
            {"topic": "unknown"},
            {"error": "not_found"},
            id="missing-policy",
        ),
    ],
)
def test_adapter_validates_and_forwards_valid_arguments(
    name, arguments, expected, backend_spies
):
    result = getattr(langchain_tools, name).invoke(arguments)

    assert result == expected
    backend_spies[name].assert_called_once_with(*arguments.values())
    for other_name, spy in backend_spies.items():
        if other_name != name:
            spy.assert_not_called()


@pytest.mark.parametrize(
    ("name", "arguments", "error_type", "field"),
    [
        pytest.param(
            "get_ticket_status",
            {"ticket_id": "abc"},
            "int_type",
            "ticket_id",
            id="ticket-id-abc",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": "123"},
            "int_type",
            "ticket_id",
            id="ticket-id-numeric-string",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": True},
            "int_type",
            "ticket_id",
            id="ticket-id-bool",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": 123.0},
            "int_type",
            "ticket_id",
            id="ticket-id-float",
        ),
        pytest.param(
            "get_support_policy",
            {"topic": 123},
            "string_type",
            "topic",
            id="topic-integer",
        ),
        pytest.param(
            "get_ticket_status",
            {"ticket_id": None},
            "int_type",
            "ticket_id",
            id="ticket-id-null",
        ),
        pytest.param(
            "get_ticket_status", {}, "missing", "ticket_id", id="ticket-id-missing"
        ),
        pytest.param(
            "get_support_policy",
            {"topic": None},
            "string_type",
            "topic",
            id="topic-null",
        ),
        pytest.param(
            "get_support_policy",
            {"topic": True},
            "string_type",
            "topic",
            id="topic-bool",
        ),
        pytest.param("get_support_policy", {}, "missing", "topic", id="topic-missing"),
    ],
)
def test_langchain_tools_reject_invalid_arguments(
    name, arguments, error_type, field, backend_spies
):
    with pytest.raises(ValidationError) as exc_info:
        getattr(langchain_tools, name).invoke(arguments)

    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["type"] == error_type
    assert errors[0]["loc"] == (field,)
    for spy in backend_spies.values():
        spy.assert_not_called()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_ticket_status", {"ticket_id": 123}),
        ("get_support_policy", {"topic": "high_priority"}),
    ],
)
def test_adapter_propagates_backend_errors(name, arguments, backend_spies):
    error = RuntimeError("Support backend is unavailable")
    backend_spies[name].side_effect = error

    with pytest.raises(RuntimeError) as exc_info:
        getattr(langchain_tools, name).invoke(arguments)

    assert exc_info.value is error
    backend_spies[name].assert_called_once_with(*arguments.values())


def test_support_agent_demo_runs_with_scripted_model(
    monkeypatch, capsys, scripted_model
):
    answer = (
        "Ticket 123 is open with high priority. The initial response is within 4 hours."
    )
    model = scripted_model(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "get_ticket_status",
                        "args": {"ticket_id": 123},
                        "id": "ticket-123",
                    },
                    {
                        "name": "get_support_policy",
                        "args": {"topic": "high_priority"},
                        "id": "policy-high-priority",
                    },
                ],
            ),
            AIMessage(content=answer),
        ]
    )
    model_factory = Mock(return_value=model)
    monkeypatch.setattr(support_agent_demo, "ChatOllama", model_factory)

    support_agent_demo.main()

    model_factory.assert_called_once()
    assert model.tool_bindings
    for binding in model.tool_bindings:
        assert {"get_ticket_status", "get_support_policy"} <= {
            tool["function"]["name"] for tool in binding
        }
    output = capsys.readouterr().out
    assert 'TOOL RESULT: {"status": "open", "priority": "high"}' in output
    assert (
        'TOOL RESULT: {"response_time_hours": "Initial response within 4 hours"}'
        in output
    )
    assert f"ASSISTANT: {answer}" in output
    assert len(model.calls) == 2
