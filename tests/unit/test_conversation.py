from collections import deque
from copy import deepcopy

import pytest

from supportops.conversation import Conversation
from supportops.llm_client import LLMResponse


class FakeLLMClient:
    def __init__(self, *outcomes: LLMResponse | Exception):
        self.outcomes = deque(outcomes)
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> LLMResponse:
        self.calls.append(deepcopy(messages))
        outcome = self.outcomes.popleft()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def response():
    return LLMResponse(
        content="Check the upstream service.",
        model="fake-model",
        input_tokens=10,
        output_tokens=5,
        duration_seconds=0.25,
    )


@pytest.mark.parametrize("system_prompt", [None, "", "You are a support assistant."])
def test_initial_history(system_prompt):
    client = FakeLLMClient()

    conversation = Conversation(client, system_prompt=system_prompt)

    expected = [{"role": "system", "content": system_prompt}] if system_prompt else []
    assert conversation.messages == expected
    assert client.calls == []


def test_ask_returns_response_and_records_exchange(response):
    client = FakeLLMClient(response)
    conversation = Conversation(client)

    result = conversation.ask("Why am I getting HTTP 502?")

    user_message = {"role": "user", "content": "Why am I getting HTTP 502?"}
    assert result is response
    assert client.calls == [[user_message]]
    assert conversation.messages == [
        user_message,
        {"role": "assistant", "content": response.content},
    ]


def test_ask_sends_system_prompt_and_previous_exchanges(response):
    second_response = LLMResponse("Check its logs.", "fake-model", 20, 4, 0.5)
    client = FakeLLMClient(response, second_response)
    conversation = Conversation(client, system_prompt="You are a support assistant.")
    system_message = {"role": "system", "content": "You are a support assistant."}
    first_question = {"role": "user", "content": "Why HTTP 502?"}
    first_answer = {"role": "assistant", "content": response.content}
    second_question = {"role": "user", "content": "What should I check next?"}

    conversation.ask(first_question["content"])
    result = conversation.ask(second_question["content"])

    assert result is second_response
    assert client.calls == [
        [system_message, first_question],
        [system_message, first_question, first_answer, second_question],
    ]
    assert conversation.messages == [
        system_message,
        first_question,
        first_answer,
        second_question,
        {"role": "assistant", "content": second_response.content},
    ]


@pytest.mark.parametrize("has_previous_exchange", [False, True])
def test_failed_ask_preserves_history_and_allows_retry(response, has_previous_exchange):
    error = RuntimeError("LLM is unavailable")
    outcomes = [response] if has_previous_exchange else []
    client = FakeLLMClient(*outcomes, error, response)
    conversation = Conversation(client, system_prompt="Be helpful.")
    if has_previous_exchange:
        conversation.ask("First question")
    history_before = conversation.messages
    question = {"role": "user", "content": "Retry this question"}

    with pytest.raises(RuntimeError) as exc_info:
        conversation.ask(question["content"])

    assert exc_info.value is error
    assert conversation.messages == history_before
    assert client.calls[-1] == history_before + [question]

    assert conversation.ask(question["content"]) is response
    assert client.calls[-1] == client.calls[-2]
    assert conversation.messages == history_before + [
        question,
        {"role": "assistant", "content": response.content},
    ]


def test_messages_returns_independent_list(response):
    conversation = Conversation(FakeLLMClient(response))
    conversation.ask("Why HTTP 502?")
    messages = conversation.messages

    messages.clear()
    messages.append({"role": "user", "content": "Unsent question"})

    assert conversation.messages == [
        {"role": "user", "content": "Why HTTP 502?"},
        {"role": "assistant", "content": response.content},
    ]


def test_messages_returns_independent_dicts(response):
    conversation = Conversation(FakeLLMClient(response))
    conversation.ask("Why HTTP 502?")
    messages = conversation.messages

    messages[0]["content"] = "Changed question"
    messages[1]["content"] = "Changed answer"

    assert conversation.messages == [
        {"role": "user", "content": "Why HTTP 502?"},
        {"role": "assistant", "content": response.content},
    ]


def test_conversations_have_independent_histories(response):
    client = FakeLLMClient(response, response)
    first = Conversation(client, system_prompt="First prompt")
    second = Conversation(client, system_prompt="Second prompt")

    first.ask("First question")
    first_history = first.messages
    second.ask("Second question")

    assert first.messages == first_history
    assert client.calls[1] == [
        {"role": "system", "content": "Second prompt"},
        {"role": "user", "content": "Second question"},
    ]
    assert second.messages == client.calls[1] + [
        {"role": "assistant", "content": response.content},
    ]
