import os

from langchain.agents import create_agent
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import tool
from langchain_ollama import ChatOllama


@tool
def get_service_status(service: str) -> dict:
    """Get the operational status of a named service.

    Args:
        service: Service name, such as billing or gateway.
    """
    statuses = {
        "billing": "degraded",
        "gateway": "operational",
    }

    return {
        "service": service,
        "status": statuses.get(service, "unknown"),
    }


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
        tools=[get_service_status],
        system_prompt=(
            "You are a technical support assistant. "
            "Answer in English. "
            "Use the available tool to check service statuses. "
            "Never invent operational statuses."
        ),
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "What is the current status "
                        "of the billing service?"
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
