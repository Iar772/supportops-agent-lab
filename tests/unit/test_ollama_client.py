import json
from contextlib import ExitStack
from unittest.mock import Mock

import httpx
import pytest

from supportops.conversation import Conversation
from supportops.llm_client import LLMResponse, OllamaClient, ToolCall
from supportops.tool_runner import ToolRunner


@pytest.fixture
def mock_http(monkeypatch):
    with ExitStack() as stack:

        def install(handler):
            client = stack.enter_context(
                httpx.Client(transport=httpx.MockTransport(handler))
            )
            monkeypatch.setattr("supportops.llm_client.httpx.post", client.post)

        yield install


@pytest.fixture
def response_data():
    return {
        "model": "qwen3.5:4b",
        "message": {"role": "assistant", "content": "Too Many Requests."},
        "prompt_eval_count": 20,
        "eval_count": 8,
        "total_duration": 1_250_000_000,
    }


@pytest.mark.parametrize("timeout", [None, 7])
def test_chat_sends_request_and_parses_response(mock_http, response_data, timeout):
    messages = [{"role": "user", "content": "What does HTTP 429 mean?"}]
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_data)

    mock_http(handler)
    kwargs = {} if timeout is None else {"timeout": timeout}
    client = OllamaClient("http://ollama.test:11434", "qwen3.5:4b", **kwargs)

    response = client.chat(messages)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert str(request.url) == "http://ollama.test:11434/api/chat"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "model": "qwen3.5:4b",
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 256},
        "tools": [],
    }
    expected_timeout = 120 if timeout is None else timeout
    assert request.extensions["timeout"] == {
        phase: expected_timeout for phase in ("connect", "read", "write", "pool")
    }
    assert response == LLMResponse(
        content="Too Many Requests.",
        model="qwen3.5:4b",
        input_tokens=20,
        output_tokens=8,
        duration_seconds=1.25,
    )


@pytest.mark.parametrize(
    ("missing_key", "attribute"),
    [
        ("prompt_eval_count", "input_tokens"),
        ("eval_count", "output_tokens"),
        ("total_duration", "duration_seconds"),
    ],
)
def test_chat_defaults_missing_metrics_to_zero(
    mock_http, response_data, missing_key, attribute
):
    del response_data[missing_key]
    mock_http(lambda request: httpx.Response(200, json=response_data))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    response = client.chat([])

    assert getattr(response, attribute) == 0
    assert response.content == "Too Many Requests."


@pytest.mark.parametrize("data", [None, [], "unexpected", 123, True])
def test_chat_rejects_non_object_response(mock_http, data):
    mock_http(lambda request: httpx.Response(200, content=json.dumps(data)))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(
        ValueError, match="Invalid Ollama response: expected a JSON object"
    ):
        client.chat([])


@pytest.mark.parametrize(
    "data",
    [{}, {"message": None}, {"message": []}, {"message": "text"}, {"message": 123}],
)
def test_chat_rejects_missing_or_invalid_message(mock_http, data):
    mock_http(lambda request: httpx.Response(200, json=data))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(
        ValueError, match="Invalid Ollama response: 'message' must be an object"
    ):
        client.chat([])


@pytest.mark.parametrize(
    "message",
    [
        {},
        {"content": ""},
        {"content": " \t\n"},
    ],
)
def test_chat_rejects_missing_invalid_or_empty_content(
    mock_http, response_data, message
):
    response_data["message"] = message
    mock_http(lambda request: httpx.Response(200, json=response_data))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(
        ValueError,
        match="Invalid Ollama response: must contain either non-empty content or valid tool calls",
    ):
        client.chat([])


def test_chat_preserves_valid_content_whitespace(mock_http, response_data):
    response_data["message"]["content"] = "  Answer\n"
    mock_http(lambda request: httpx.Response(200, json=response_data))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    assert client.chat([]).content == "  Answer\n"


@pytest.mark.parametrize("status_code", [400, 429, 500])
def test_chat_raises_for_http_errors(mock_http, status_code):
    mock_http(lambda request: httpx.Response(status_code, json={"error": "failed"}))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        client.chat([])

    assert exc_info.value.response.status_code == status_code


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
def test_chat_propagates_transport_errors(mock_http, error_type):
    error = error_type("Ollama is unavailable")

    def handler(request):
        raise error

    mock_http(handler)
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(error_type) as exc_info:
        client.chat([])

    assert exc_info.value is error


def test_chat_rejects_invalid_json(mock_http):
    mock_http(lambda request: httpx.Response(200, text="invalid JSON"))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(json.JSONDecodeError):
        client.chat([])


@pytest.mark.parametrize("content", [None, "", "Checking the ticket."])
def test_chat_sends_schemas_and_parses_multiple_tool_calls(
    mock_http, response_data, content
):
    schemas = [{"type": "function", "function": {"name": "get_ticket_status"}}]
    expected_calls = (
        ToolCall("get_ticket_status", {"ticket_id": 123}),
        ToolCall("get_support_policy", {"topic": "high_priority"}),
    )
    message = {
        "tool_calls": [
            {"function": {"name": call.name, "arguments": call.arguments}}
            for call in expected_calls
        ]
    }
    if content is not None:
        message["content"] = content
    response_data["message"] = message
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_data)

    mock_http(handler)
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    response = client.chat([], tools=schemas)

    assert len(requests) == 1
    assert json.loads(requests[0].content)["tools"] == schemas
    assert response.content == (content or "")
    assert response.tool_calls == expected_calls


@pytest.mark.parametrize("content", [None, 123, False, [], {}])
@pytest.mark.parametrize("has_tool_calls", [False, True])
def test_chat_rejects_non_string_content_even_with_tools(
    mock_http, response_data, content, has_tool_calls
):
    message = {"content": content}
    if has_tool_calls:
        message["tool_calls"] = [
            {"function": {"name": "get_ticket_status", "arguments": {"ticket_id": 123}}}
        ]
    response_data["message"] = message
    mock_http(lambda request: httpx.Response(200, json=response_data))

    with pytest.raises(ValueError, match=r"'message\.content' must be a string"):
        OllamaClient("http://ollama.test", "qwen3.5:4b").chat([])


@pytest.mark.parametrize(
    "tool_calls", ["invalid", 123, False, 0, "", {}, {"function": {}}]
)
def test_chat_rejects_non_array_tool_calls(mock_http, response_data, tool_calls):
    response_data["message"]["tool_calls"] = tool_calls
    mock_http(lambda request: httpx.Response(200, json=response_data))

    with pytest.raises(ValueError, match=r"'message\.tool_calls' must be an array"):
        OllamaClient("http://ollama.test", "qwen3.5:4b").chat([])


@pytest.mark.parametrize(
    ("invalid_call", "error"),
    [
        pytest.param(
            None, "Invalid tool call structure: expected a dictionary", id="null-call"
        ),
        pytest.param(
            "invalid",
            "Invalid tool call structure: expected a dictionary",
            id="string-call",
        ),
        pytest.param(
            {}, "missing mandatory 'function' envelope", id="missing-function"
        ),
        pytest.param({"function": None}, "Invalid 'function' data", id="null-function"),
        pytest.param({"function": []}, "Invalid 'function' data", id="list-function"),
        pytest.param(
            {"function": "invalid"}, "Invalid 'function' data", id="string-function"
        ),
        pytest.param(
            {"function": {}}, "Invalid or missing tool name", id="missing-name"
        ),
        pytest.param(
            {"function": {"name": 123, "arguments": {}}},
            "Invalid or missing tool name",
            id="non-string-name",
        ),
        pytest.param(
            {"function": {"name": "", "arguments": {}}},
            "Invalid or missing tool name",
            id="empty-name",
        ),
        pytest.param(
            {"function": {"name": " \t", "arguments": {}}},
            "Invalid or missing tool name",
            id="blank-name",
        ),
        pytest.param(
            {"function": {"name": "get_ticket_status"}},
            "Invalid arguments block format",
            id="missing-arguments",
        ),
        pytest.param(
            {"function": {"name": "get_ticket_status", "arguments": None}},
            "Invalid arguments block format",
            id="null-arguments",
        ),
        pytest.param(
            {"function": {"name": "get_ticket_status", "arguments": []}},
            "Invalid arguments block format",
            id="list-arguments",
        ),
        pytest.param(
            {"function": {"name": "get_ticket_status", "arguments": "invalid"}},
            "Invalid arguments block format",
            id="string-arguments",
        ),
    ],
)
@pytest.mark.parametrize(
    ("content", "valid_call_position"),
    [
        pytest.param("", None, id="only-invalid-call"),
        pytest.param("Checking the ticket.", "before", id="valid-call-before-invalid"),
        pytest.param("", "after", id="valid-call-after-invalid"),
    ],
)
def test_chat_rejects_malformed_tool_calls(
    mock_http, response_data, invalid_call, error, content, valid_call_position
):
    calls = [invalid_call]
    valid_call = {
        "function": {"name": "get_ticket_status", "arguments": {"ticket_id": 123}}
    }
    if valid_call_position == "before":
        calls.insert(0, valid_call)
    elif valid_call_position == "after":
        calls.append(valid_call)
    response_data["message"] = {"content": content, "tool_calls": calls}
    mock_http(lambda request: httpx.Response(200, json=response_data))
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")

    with pytest.raises(ValueError, match=error):
        client.chat([])


@pytest.mark.parametrize("tool_calls", [None, []])
def test_chat_accepts_text_without_tool_calls(mock_http, response_data, tool_calls):
    response_data["message"]["tool_calls"] = tool_calls
    mock_http(lambda request: httpx.Response(200, json=response_data))

    response = OllamaClient("http://ollama.test", "qwen3.5:4b").chat([])

    assert response.content == "Too Many Requests."
    assert response.tool_calls == ()


@pytest.mark.parametrize(
    "invalid_call_index",
    [
        pytest.param(0, id="invalid-first"),
        pytest.param(1, id="invalid-between-valid-calls"),
        pytest.param(2, id="invalid-last"),
    ],
)
def test_mixed_tool_calls_do_not_execute_any_tool(
    mock_http, response_data, invalid_call_index
):
    calls = [
        {"function": {"name": "get_ticket_status", "arguments": {"ticket_id": 123}}},
        {
            "function": {
                "name": "get_support_policy",
                "arguments": {"topic": "working_hours"},
            }
        },
    ]
    calls.insert(invalid_call_index, {"function": None})
    response_data["message"] = {
        "content": "Checking the ticket and support hours.",
        "tool_calls": calls,
    }
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response_data)

    mock_http(handler)
    client = OllamaClient("http://ollama.test", "qwen3.5:4b")
    runner = Mock(spec=ToolRunner, wraps=ToolRunner())
    conversation = Conversation(
        client, runner, system_prompt="You are a support assistant."
    )
    history_before = conversation.messages

    with pytest.raises(ValueError, match="Invalid 'function' data"):
        conversation.ask("Check ticket 123 and support working hours")

    runner.run.assert_not_called()
    assert len(requests) == 1
    assert conversation.messages == history_before
