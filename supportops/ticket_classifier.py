from supportops.prompts import TICKET_CLASSIFICATION_PROMPT
from supportops.schemas import TicketAnalysis


class TicketClassifier:
    def __init__(self, model):
        self.model = model
        self.chain = TICKET_CLASSIFICATION_PROMPT | self.model

    def classify(self, ticket_text: str) -> TicketAnalysis:
        result = self.chain.invoke(
            {
                "ticket_text": (
                    ticket_text
                )
            }
        )
        return result
