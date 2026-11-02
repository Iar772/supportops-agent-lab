from langchain.tools import tool
from pydantic import StrictInt

from supportops.tools import (
    get_support_policy as lookup_support_policy,
)
from supportops.tools import (
    get_ticket_status as lookup_ticket_status,
)


@tool
def get_ticket_status(ticket_id: StrictInt) -> dict:
    """Look up a support ticket's status and priority by its integer ID."""
    return lookup_ticket_status(ticket_id)


@tool
def get_support_policy(topic: str) -> dict:
    """Look up support policy: working_hours, high_priority, or response_time."""
    return lookup_support_policy(topic)
