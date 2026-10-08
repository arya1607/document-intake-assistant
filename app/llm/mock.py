import json
import re

from app.llm.base import LLMClient, LLMRequest
from app.schemas import FieldUpdate, LLMTurnResult


QUESTIONS = {
    "full_name": "What is your full legal name?",
    "home_address": "What is your home address?",
    "covers_worldwide_assets": "Should this document cover your worldwide assets?",
    "has_children": "Do you have any children?",
    "children_names": "What are your children's names?",
    "executor.name": "Who would you like to name as executor?",
    "executor.relationship": "What is your relationship to your executor?",
    "gifts": "Would you like to leave any gifts?",
    "additional_wishes": "Do you have any additional wishes?",
}

_RELATIONSHIPS = {
    "brother",
    "sister",
    "mother",
    "father",
    "son",
    "daughter",
    "husband",
    "wife",
    "partner",
    "friend",
}
_YES_WORDS = {"yes", "yeah", "yep", "y", "sure", "definitely", "correct"}
_NO_WORDS = {"no", "nope", "nah", "n"}


class FixtureMockLLMClient(LLMClient):
    """Returns pre-recorded model text for tests."""

    # ######################################################
    # Function name : __init__
    # Description   : Initialize queued responses and request tracking for tests.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.requests: list[LLMRequest] = []

    # ######################################################
    # Function name : generate
    # Description   : Return the next queued response or raise its queued exception.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def generate(self, request: LLMRequest) -> str:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError(
                "FixtureMockLLMClient was called more times than queued responses."
            )
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RuleBasedMockLLMClient(LLMClient):
    """A deterministic demo stand-in; this is not real natural-language understanding."""

    # ######################################################
    # Function name : generate
    # Description   : Produce deterministic updates from the user's message and targets.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def generate(self, request: LLMRequest) -> str:
        user_message = next(
            (
                message.content
                for message in reversed(request.messages)
                if message.role == "user"
            ),
            "",
        ).strip()
        lowered = user_message.casefold()
        updates: list[FieldUpdate] = []
        needs_clarification: list[dict[str, str]] = []
        first_target = request.target_fields[0] if request.target_fields else None

        for target in request.target_fields:
            if target == "full_name":
                name = _extract_name(user_message)
                if name:
                    updates.append(_update(target, name))
            elif target == "home_address" and user_message:
                updates.append(_update(target, user_message))
            elif target in {"covers_worldwide_assets", "has_children"}:
                answer = _yes_no(lowered)
                if answer is not None:
                    updates.append(_update(target, answer))
                else:
                    needs_clarification.append(
                        {
                            "field": target,
                            "reason": "Please clarify whether the answer is yes or no.",
                        }
                    )
            elif target in {"executor.name", "executor.relationship"}:
                first_executor_target = next(
                    (
                        candidate
                        for candidate in request.target_fields
                        if candidate in {"executor.name", "executor.relationship"}
                    ),
                    None,
                )
                if target != first_executor_target:
                    continue
                name, relationship = _extract_executor(user_message)
                if name:
                    updates.append(_update("executor.name", name))
                if relationship:
                    updates.append(_update("executor.relationship", relationship))
            elif target == "children_names" and user_message:
                names = _split_names(user_message)
                if names:
                    updates.append(_update(target, names))
            elif target in {"gifts", "additional_wishes"} and user_message:
                value = [] if _is_none_answer(lowered) else [user_message]
                updates.append(_update(target, value))

        result = LLMTurnResult(
            updates=updates,
            needs_clarification=needs_clarification,
            assistant_message=(
                QUESTIONS[first_target]
                if first_target in QUESTIONS
                else "Thank you. Your draft is ready for review."
            ),
        )
        return result.model_dump_json()


# ######################################################
# Function name : _update
# Description   : Build a stated field update for the rule-based mock client.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _update(field: str, value: str | bool | list[str]) -> FieldUpdate:
    return FieldUpdate(field=field, value=value, confidence="stated")


# ######################################################
# Function name : _extract_name
# Description   : Extract a likely full name from a short user message.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _extract_name(message: str) -> str | None:
    patterns = (
        r"^\s*my name is\s+(.+?)\s*[.!?]*\s*$",
        r"^\s*i am\s+(.+?)\s*[.!?]*\s*$",
        r"^\s*i['’]m\s+(.+?)\s*[.!?]*\s*$",
    )
    for pattern in patterns:
        match = re.match(pattern, message, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    words = message.split()
    if 2 <= len(words) <= 4:
        return message.strip()
    return None


# ######################################################
# Function name : _yes_no
# Description   : Interpret a clear yes-or-no answer from user text.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _yes_no(message: str) -> bool | None:
    words = set(re.findall(r"[a-z]+", message))
    if {"not", "sure"} <= words:
        return None
    if words & _YES_WORDS:
        return True
    if words & _NO_WORDS:
        return False
    return None


# ######################################################
# Function name : _extract_executor
# Description   : Extract an executor's name and recognized relationship.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _extract_executor(message: str) -> tuple[str | None, str | None]:
    match = re.match(
        r"^\s*my\s+([a-z]+)\s+([A-Z][A-Za-z'-]*)\s*[.!?]*\s*$", message
    )
    if match and match.group(1).casefold() in _RELATIONSHIPS:
        return match.group(2), match.group(1).casefold()

    candidate = message.strip().rstrip(".!?")
    if candidate and len(candidate.split()) == 1:
        return candidate, None
    return None, None


# ######################################################
# Function name : _split_names
# Description   : Split a user response into individual names.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _split_names(message: str) -> list[str]:
    parts = re.split(r"\s*,\s*|\s+and\s+", message.strip(), flags=re.IGNORECASE)
    return [part.strip() for part in parts if part.strip()]


# ######################################################
# Function name : _is_none_answer
# Description   : Detect an explicit response indicating that there are none.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _is_none_answer(message: str) -> bool:
    answer = message.strip().rstrip(".!?")
    return answer in {
        "no",
        "no gifts",
        "no additional wishes",
        "none",
        "nothing",
        "nope",
        "nah",
    }