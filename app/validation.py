import json
import re

from pydantic import ValidationError

from app.schemas import LLMTurnResult


class MalformedLLMOutput(Exception):
    # ######################################################
    # Function name : __init__
    # Description   : Store the parsing failure reason and a bounded raw response.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def __init__(self, reason: str, raw: str) -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw = raw[:500]


# ######################################################
# Function name : parse_llm_output
# Description   : Parse and validate a model response against the turn schema.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def parse_llm_output(raw: str) -> LLMTurnResult:
    if not isinstance(raw, str):
        raise MalformedLLMOutput("output must be a string", str(raw))

    text = raw.strip()
    if not text:
        raise MalformedLLMOutput("output is empty", raw)

    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).replace("```", "")
    text = text.strip()
    first_object = text.find("{")
    last_object = text.rfind("}")
    if first_object >= 0 and last_object >= first_object:
        text = text[first_object : last_object + 1]

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        raise MalformedLLMOutput("not valid JSON", raw) from None

    try:
        return LLMTurnResult.model_validate(data)
    except ValidationError as error:
        summaries = [
            f"{'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors()[:2]
        ]
        summary = "; ".join(summaries)
        reason = "does not match expected schema"
        if summary:
            reason = f"{reason}: {summary}"
        raise MalformedLLMOutput(reason, raw) from None