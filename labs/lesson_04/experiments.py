import os

from langchain_ollama import ChatOllama

from supportops.schemas import TicketAnalysis
from supportops.ticket_classifier import TicketClassifier


def main() -> None:
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        temperature=0,
        reasoning=False,
        num_ctx=4096,
        num_predict=512,
    )

    structured_model = model.with_structured_output(
        TicketAnalysis,
        method="json_schema",
    )

    classifier = TicketClassifier(model=structured_model)

    ticket_text = (
        "What are your support working hours?"
    )

    result = classifier.classify(ticket_text)

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()