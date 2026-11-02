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


# 2. Инициализируем компоненты
client = OllamaClient(base_url="http://localhost:11434", model="qwen3.5:4b")
runner = ToolRunner()

# =====================================================================
# ТЕСТ СЦЕНАРИЙ 1: Запрос статуса заявки
# =====================================================================
# conv1 = Conversation(client=client, runner=runner)
# response1 = conv1.ask("Проверь статус заявки 123.", tools=TOOLS_SCHEMAS)
#
# print("--- Тест 1 ---")
# print("Финальный текст ответа:", response1.content)


# =====================================================================
# ТЕСТ СЦЕНАРИЙ 2: Запрос политики поддержки
# =====================================================================
conv2 = Conversation(client=client, runner=runner)
question = "Какой статус у заявки 123 и как быстро вы отвечаете на срочные обращения?"
response2 = conv2.ask(question, tools=TOOLS_SCHEMAS)
print(question)
print('----')
print("Финальный текст ответа:", response2.content)
