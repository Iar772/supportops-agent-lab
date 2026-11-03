import pytest
from pydantic import ValidationError

from supportops.schemas import TicketAnalysis


@pytest.fixture
def valid_data():
    return {
        "category": "technical",
        "priority": "high",
        "summary": "Production API returns HTTP 502 for all customers.",
        "needs_clarification": False,
        "missing_information": [],
    }


@pytest.mark.parametrize("category", ["technical", "billing", "account", "other"])
@pytest.mark.parametrize("priority", ["low", "medium", "high"])
def test_accepts_allowed_categories_and_priorities(valid_data, category, priority):
    data = {**valid_data, "category": category, "priority": priority}

    analysis = TicketAnalysis.model_validate(data)

    assert isinstance(analysis, TicketAnalysis)
    assert analysis.model_dump() == data


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("category", "unknown"),
        ("category", "Technical"),
        ("priority", "urgent"),
        ("priority", "HIGH"),
    ],
)
def test_rejects_invalid_categories_and_priorities(valid_data, field, value):
    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate({**valid_data, field: value})

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == "literal_error"


@pytest.mark.parametrize("length", [10, 240])
def test_summary_is_stripped_before_checking_length_boundaries(valid_data, length):
    summary = "x" * length

    analysis = TicketAnalysis.model_validate(
        {**valid_data, "summary": f" \t{summary}\n "}
    )

    assert analysis.summary == summary


@pytest.mark.parametrize(
    ("summary", "error_type"),
    [
        ("", "value_error"),
        (" \t\n", "value_error"),
        ("x" * 9, "string_too_short"),
        ("  " + "x" * 9 + "  ", "string_too_short"),
        ("x" * 241, "string_too_long"),
        (None, "string_type"),
        (123, "string_type"),
    ],
)
def test_rejects_invalid_summaries(valid_data, summary, error_type):
    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate({**valid_data, "summary": summary})

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("summary",)
    assert error["type"] == error_type


def test_missing_information_items_are_stripped_at_length_boundaries(valid_data):
    analysis = TicketAnalysis.model_validate(
        {
            **valid_data,
            "needs_clarification": True,
            "missing_information": [" \tID\n", " " + "x" * 100 + " "],
        }
    )

    assert analysis.needs_clarification is True
    assert analysis.missing_information == ["ID", "x" * 100]


@pytest.mark.parametrize(
    ("item", "error_type"),
    [
        ("", "string_too_short"),
        (" \t\n", "string_too_short"),
        (" x ", "string_too_short"),
        ("x" * 101, "string_too_long"),
        (123, "string_type"),
        (None, "string_type"),
    ],
)
def test_rejects_invalid_missing_information_items(valid_data, item, error_type):
    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate(
            {
                **valid_data,
                "needs_clarification": True,
                "missing_information": ["Steps to reproduce", item],
            }
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("missing_information", 1)
    assert error["type"] == error_type


@pytest.mark.parametrize("missing_information", [None, "Error code", 123])
def test_rejects_non_list_missing_information(valid_data, missing_information):
    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate(
            {**valid_data, "missing_information": missing_information}
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("missing_information",)
    assert error["type"] == "list_type"


@pytest.mark.parametrize(
    ("needs_clarification", "missing_information", "message"),
    [
        (True, [], "missing_information cannot be empty"),
        (False, ["Error code"], "missing_information must be empty"),
    ],
)
def test_clarification_flag_must_agree_with_missing_information(
    valid_data, needs_clarification, missing_information, message
):
    with pytest.raises(ValidationError, match=message) as exc_info:
        TicketAnalysis.model_validate(
            {
                **valid_data,
                "needs_clarification": needs_clarification,
                "missing_information": missing_information,
            }
        )

    assert exc_info.value.errors()[0]["loc"] == ()


@pytest.mark.parametrize("needs_clarification", ["true", "false", 1, 0, None])
def test_clarification_requires_a_strict_boolean(valid_data, needs_clarification):
    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate(
            {**valid_data, "needs_clarification": needs_clarification}
        )

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("needs_clarification",)
    assert error["type"] == "bool_type"


@pytest.mark.parametrize(
    "field",
    ["category", "priority", "summary", "needs_clarification", "missing_information"],
)
def test_all_fields_are_required(valid_data, field):
    del valid_data[field]

    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate(valid_data)

    error = exc_info.value.errors()[0]
    assert error["loc"] == (field,)
    assert error["type"] == "missing"


def test_extra_fields_are_forbidden(valid_data):
    with pytest.raises(ValidationError) as exc_info:
        TicketAnalysis.model_validate({**valid_data, "confidence": 0.99})

    error = exc_info.value.errors()[0]
    assert error["loc"] == ("confidence",)
    assert error["type"] == "extra_forbidden"
