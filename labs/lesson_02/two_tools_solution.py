from supportops.conversation import Conversation
from supportops.llm_client import OllamaClient
from supportops.tool_runner import ToolRunner

TOOLS_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Get the current status and priority of a ticket by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "integer", "description": "The unique ticket ID, e.g., 123"}
                },
                "required": ["ticket_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_support_policy",
            "description": "Get information about the support SLA and working policy for a specific topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The policy topic. Options: response_time, high_priority, working_hours"
                    }
                },
                "required": ["topic"],
                "additionalProperties": False,
            },
        },
    }
]


client = OllamaClient(base_url="http://localhost:11434", model="qwen3.5:4b")
runner = ToolRunner()

conv2 = Conversation(client=client, runner=runner)
question = "whats the status of request 123"
response2 = conv2.ask(question, tools=TOOLS_SCHEMAS)
print(question)
print('----')
print("final response:", response2.content)
