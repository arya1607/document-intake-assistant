import pytest
from pydantic import ValidationError

from app.schemas import (
    CollectedState,
    FieldStatus,
    LLMTurnResult,
    MessageRequest,
    TrackedField,
)


# ---------- State ----------

def test_new_state_is_entirely_missing():
    s = CollectedState()
    assert s.full_name.status == FieldStatus.MISSING
    assert s.full_name.value is None
    assert s.executor.name.status == FieldStatus.MISSING
    assert s.children_names.value is None


def test_two_states_do_not_share_data():
    a = CollectedState()
    b = CollectedState()
    a.full_name.value = "Jane"
    assert b.full_name.value is None


def test_none_is_different_from_missing():
    s = CollectedState()
    s.gifts = TrackedField[list[str]](value=[], status=FieldStatus.CONFIRMED)
    assert s.gifts.value == []
    assert s.gifts.status == FieldStatus.CONFIRMED


def test_state_round_trips_through_json():
    s = CollectedState()
    s.full_name = TrackedField[str](value="Jane Smith", status=FieldStatus.CONFIRMED)
    s.executor.name = TrackedField[str](value="James", status=FieldStatus.CONFIRMED)
    restored = CollectedState.model_validate(s.model_dump(mode="json"))
    assert restored == s


# ---------- API request ----------

def test_message_is_trimmed():
    assert MessageRequest(message="  hello  ").message == "hello"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 2001])
def test_bad_messages_are_rejected(bad):
    with pytest.raises(ValidationError):
        MessageRequest(message=bad)


# ---------- LLM contract ----------

def test_valid_llm_result_parses():
    raw = {
        "updates": [
            {"field": "executor.name", "value": "James", "confidence": "stated"},
            {"field": "executor.relationship", "value": "brother"},
        ],
        "needs_clarification": [],
        "assistant_message": "Thanks! Anything you'd like to leave as gifts?",
    }
    result = LLMTurnResult.model_validate(raw)
    assert result.updates[0].value == "James"
    assert result.updates[1].confidence == "stated"      # default
    assert result.updates[1].is_correction is False       # default


def test_llm_result_may_have_no_updates():
    result = LLMTurnResult.model_validate({"assistant_message": "Could you clarify?"})
    assert result.updates == []


def test_llm_result_requires_a_message():
    with pytest.raises(ValidationError):
        LLMTurnResult.model_validate({"updates": []})


def test_llm_result_rejects_unknown_confidence():
    raw = {
        "updates": [{"field": "full_name", "value": "Jane", "confidence": "certain"}],
        "assistant_message": "ok",
    }
    with pytest.raises(ValidationError):
        LLMTurnResult.model_validate(raw)


def test_llm_values_can_be_bool_or_list():
    raw = {
        "updates": [
            {"field": "has_children", "value": True},
            {"field": "children_names", "value": ["Tom", "Ana"]},
        ],
        "assistant_message": "Got it.",
    }
    result = LLMTurnResult.model_validate(raw)
    assert result.updates[0].value is True
    assert result.updates[1].value == ["Tom", "Ana"]