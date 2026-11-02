import os

from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from supportops.langchain_tools import get_support_policy, get_ticket_status


def main() -> None:
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        temperature=0,
        reasoning=False,
        num_ctx=4096,
        num_predict=512,
    )

    agent = create_agent(
        model=model,
        tools=[get_ticket_status, get_support_policy],
        system_prompt=(
            "You are a technical support assistant. "
            "Answer in English. "
            "Use the available tools to look up ticket statuses and support policies. "
            "Never invent ticket statuses, priorities, or policy details."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Check the status of ticket 123 and explain how quickly high-priority tickets receive an initial response"
                    ),
                }
            ]
        },
        config={"recursion_limit": 10},
    )

    print("\n--- AGENT EXECUTION ---")

    for message in result["messages"]:
        if isinstance(message, AIMessage) and message.tool_calls:
            print("TOOL CALL:", message.tool_calls)

        elif isinstance(message, ToolMessage):
            print("TOOL RESULT:", message.content)

        elif isinstance(message, AIMessage):
            print("ASSISTANT:", message.content)


if __name__ == "__main__":
    main()
