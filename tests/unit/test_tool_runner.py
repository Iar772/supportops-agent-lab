import json

import pytest

from supportops.llm_client import ToolCall
from supportops.tool_runner import ToolRunner


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        (
            "get_ticket_status",
            {"ticket_id": 123},
            {"ticket_id": 123, "status": "open", "priority": "high"},
        ),
        (
            "get_ticket_status",
            {"ticket_id": 999},
            {"ticket_id": 999, "error": "not_found"},
        ),
        (
            "get_support_policy",
            {"topic": "working_hours"},
            {"response_time_hours": "Mon–Fri, 09:00–18:00 UTC"},
        ),
        (
            "get_support_policy",
            {"topic": "high_priority"},
            {"response_time_hours": "Initial response within 4 hours"},
        ),
        (
            "get_support_policy",
            {"topic": "unknown"},
            {"error": "not_found"},
        ),
    ],
)
def test_run_dispatches_tool_and_serializes_result(name, arguments, expected):
    message = ToolRunner().run(ToolCall(name, arguments))

    assert message["role"] == "tool"
    assert message["tool_name"] == name
    assert json.loads(message["content"]) == expected


def test_run_rejects_unknown_tool():
    with pytest.raises(ValueError, match="Tool unknown not found"):
        ToolRunner().run(ToolCall("unknown", {}))


@pytest.mark.parametrize("arguments", [None, [], "ticket_id=123", 123])
def test_run_rejects_non_object_arguments(arguments):
    with pytest.raises(ValueError, match="must be a dictionary/object"):
        ToolRunner().run(ToolCall("get_ticket_status", arguments))


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("get_ticket_status", {}),
        ("get_ticket_status", {"ticket_id": 123, "extra": True}),
        ("get_support_policy", {}),
        ("get_support_policy", {"ticket_id": 123}),
    ],
)
def test_run_rejects_missing_or_unexpected_arguments(name, arguments):
    with pytest.raises(ValueError, match=f"Invalid arguments for tool '{name}'"):
        ToolRunner().run(ToolCall(name, arguments))
