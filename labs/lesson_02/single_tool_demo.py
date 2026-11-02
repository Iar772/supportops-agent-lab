import json
import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3.5:4b"


def get_service_status(service: str) -> dict:
    """Return the mocked status of a service."""
    statuses = {
        "gateway": "operational",
        "billing": "degraded",
    }

    return {
        "service": service,
        "status": statuses.get(service, "unknown"),
    }


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_service_status",
            "description": "Get the current status of a named service.",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service name, such as billing or gateway.",
                    }
                },
                "required": ["service"],
                "additionalProperties": False,
            },
        },
    }
]


def chat(messages: list[dict], tools: list[dict] | None = None) -> dict:
    response = httpx.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "tools": tools or [],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_ctx": 4096,
                "num_predict": 512,
            },
        },
        timeout=120.0,
    )

    response.raise_for_status()
    return response.json()["message"]


messages = [
    {
        "role": "user",
        "content": "Check the current status of the billing service.",
    }
]

# Step 1: Let the LLM choose a tool.
assistant_message = chat(messages, tools)
print("LLM response:", assistant_message)

calls = assistant_message.get("tool_calls") or []

if len(calls) != 1:
    raise RuntimeError(
        f"Expected one tool call, received {len(calls)}: {assistant_message}"
    )

# Step 2: Validate and execute the requested tool.
function = calls[0]["function"]
name = function["name"]
arguments = function["arguments"]

if name != "get_service_status":
    raise ValueError(f"Unknown tool: {name}")

if not isinstance(arguments, dict):
    raise ValueError("Tool arguments must be an object")

service = arguments.get("service")

if not isinstance(service, str) or not service.strip():
    raise ValueError("Invalid service argument")

tool_result = get_service_status(service)

# Step 3: Send the tool result back to the model.
messages.append(
    {
        "role": "assistant",
        "content": assistant_message.get("content", ""),
        "tool_calls": calls,
    }
)

messages.append(
    {
        "role": "tool",
        "tool_name": name,
        "content": json.dumps(tool_result),
    }
)

# For this single-shot demo, no further tools are offered.
final_message = chat(messages)

print("Final answer:", final_message["content"])


SERVICE_STATUSES = {
    "gateway": "operational",
    "billing": "degraded",
}