from typing import Literal, Annotated

from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator, StrictBool, StringConstraints

CleanedString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=100)
]


class TicketAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: Literal["technical", "billing", "account", "other"]
    priority: Literal["low", "medium", "high"]
    summary: str = Field(min_length=10, max_length=240)
    needs_clarification: StrictBool
    missing_information: list[CleanedString]

    @field_validator("summary", mode="before")
    @classmethod
    def strip_summary_before_length_check(cls, value: str) -> str:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("Summary cannot be empty or consist only of spaces")
            return cleaned
        return value

    @model_validator(mode="after")
    def validate_clarification_and_info(self) -> "TicketAnalysis":
        if self.needs_clarification and not self.missing_information:
            raise ValueError(
                "missing_information cannot be empty when needs_clarification is True"
            )

        if not self.needs_clarification and self.missing_information:
            raise ValueError(
                f"missing_information must be empty when needs_clarification is False. "
                f"Found: {self.missing_information}"
            )

        return self