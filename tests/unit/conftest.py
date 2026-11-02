import socket
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any
from unittest.mock import Mock

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from supportops import langchain_tools


class ScriptedChatModel(BaseChatModel):
    """Return a finite script and record the real agent's messages and bound tools."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = Field(default_factory=list)
    tool_bindings: list[list[dict[str, Any]]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-test-model"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> "ScriptedChatModel":
        self.tool_bindings.append(
            deepcopy([convert_to_openai_tool(tool) for tool in tools])
        )
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = len(self.calls)
        if index >= len(self.responses):
            raise AssertionError(
                "Unexpected model call: the response script is exhausted"
            )
        self.calls.append(deepcopy(messages))
        return ChatResult(
            generations=[
                ChatGeneration(message=self.responses[index].model_copy(deep=True))
            ]
        )


@pytest.fixture
def scripted_model():
    return ScriptedChatModel


@pytest.fixture
def offline_langchain(monkeypatch):
    """Block outbound connections and disable tracing for opted-in tests."""
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING"):
        monkeypatch.setenv(name, "false")

    def deny_network(*args, **kwargs):
        pytest.fail("Network access is forbidden in LangChain unit tests")

    monkeypatch.setattr(socket, "create_connection", deny_network)
    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)


@pytest.fixture
def backend_spies(monkeypatch):
    spies = {}
    for name in ("get_ticket_status", "get_support_policy"):
        alias = name.replace("get_", "lookup_", 1)
        spy = Mock(wraps=getattr(langchain_tools, alias))
        monkeypatch.setattr(langchain_tools, alias, spy)
        spies[name] = spy
    return spies
