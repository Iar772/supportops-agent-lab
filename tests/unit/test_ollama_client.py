import json
from contextlib import ExitStack

import httpx
import pytest

from supportops.llm_client import LLMResponse, OllamaClient


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
        {"content": None},
        {"content": 123},
        {"content": False},
        {"content": []},
        {"content": {}},
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
        match=r"Invalid Ollama response: 'message\.content' must be a non-empty string",
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
