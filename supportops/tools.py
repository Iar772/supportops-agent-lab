def get_ticket_status(ticket_id: int) -> dict:
    status_priorities = {
        123: {"status": "open", "priority": "high"},
        456: {
            "status": "closed",
            "priority": "low",
        },
        789: {"status": "in_progress", "priority": "medium"},
    }
    return status_priorities.get(ticket_id, {"error": "not_found"})


def get_support_policy(topic: str) -> dict:
    topics = {
        "response_time": {"response_time_hours": "Initial response within 24 hours"},
        "high_priority": {"response_time_hours": "Initial response within 4 hours"},
        "working_hours": {"response_time_hours": "Mon–Fri, 09:00–18:00 UTC"},
    }
    return topics.get(topic, {"error": "not_found"})
