
import os

from langchain_ollama import ChatOllama

from labs.lesson_04.raw_structured_output import ServiceEvent


def main() -> None:
    model = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
        temperature=0,
        reasoning=False,
        num_ctx=4096,
        num_predict=512,
    )

    structured_model = model.with_structured_output(
        ServiceEvent,
        method="json_schema",
        include_raw=True
    )
    event = structured_model.invoke(
        "Billing API is degraded. Some payment requests are failing."
    )

    print(event)
    print(type(event))

    assert isinstance(event, ServiceEvent)


if __name__ == "__main__":
    main()
