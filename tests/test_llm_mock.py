import pytest
from pydantic import ValidationError

from app.llm.base import ChatMessage, LLMRequest
from app.llm.mock import FixtureMockLLMClient, RuleBasedMockLLMClient
from app.schemas import LLMTurnResult
from tests.fixture_loader import load_fixture


def make_request(
    target_fields: list[str], user_message: str
) -> LLMRequest:
    return LLMRequest(
        system_prompt="Test prompt",
        target_fields=target_fields,
        messages=[
            ChatMessage(role="assistant", content="Previous assistant prompt."),
            ChatMessage(role="user", content=user_message),
        ],
    )


def test_fixture_mock_returns_responses_in_order_and_records_requests():
    first_request = make_request(["full_name"], "Jane Smith")
    second_request = make_request(["home_address"], "10 Example Street")
    client = FixtureMockLLMClient(["first raw response", "second raw response"])

    assert client.generate(first_request) == "first raw response"
    assert client.generate(second_request) == "second raw response"
    assert client.requests == [first_request, second_request]


def test_fixture_mock_raises_queued_exception():
    expected = RuntimeError("fixture failure")
    client = FixtureMockLLMClient([expected])

    with pytest.raises(RuntimeError, match="fixture failure"):
        client.generate(make_request([], "hello"))


def test_fixture_mock_fails_clearly_when_exhausted():
    client = FixtureMockLLMClient([])

    with pytest.raises(AssertionError, match="more times than queued responses"):
        client.generate(make_request([], "hello"))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "valid_multi_field.json",
        "valid_executor_james.json",
        "correction.json",
        "ambiguous.json",
        "contradiction.json",
    ],
)
def test_valid_fixtures_parse_as_llm_turn_results(fixture_name):
    result = LLMTurnResult.model_validate_json(load_fixture(fixture_name))

    assert isinstance(result, LLMTurnResult)


def test_schema_invalid_fixture_is_rejected():
    with pytest.raises(ValidationError):
        LLMTurnResult.model_validate_json(load_fixture("schema_invalid.json"))


def test_rule_mock_extracts_full_name_from_phrase():
    raw = RuleBasedMockLLMClient().generate(
        make_request(["full_name"], "My name is Jane Smith")
    )

    result = LLMTurnResult.model_validate_json(raw)
    assert [(item.field, item.value) for item in result.updates] == [
        ("full_name", "Jane Smith")
    ]


def test_rule_mock_extracts_short_full_name_reply():
    raw = RuleBasedMockLLMClient().generate(
        make_request(["full_name"], "Jane Smith")
    )

    result = LLMTurnResult.model_validate_json(raw)
    assert result.updates[0].value == "Jane Smith"


def test_rule_mock_uses_entire_address_reply():
    raw = RuleBasedMockLLMClient().generate(
        make_request(["home_address"], "10 Example Street, Sampletown")
    )

    result = LLMTurnResult.model_validate_json(raw)
    assert result.updates[0].value == "10 Example Street, Sampletown"


@pytest.mark.parametrize(
    ("target", "answer", "expected"),
    [
        ("has_children", "Yeah", True),
        ("covers_worldwide_assets", "Nope", False),
    ],
)
def test_rule_mock_detects_yes_no(target, answer, expected):
    result = LLMTurnResult.model_validate_json(
        RuleBasedMockLLMClient().generate(make_request([target], answer))
    )

    assert result.updates[0].value is expected


def test_rule_mock_extracts_executor_without_inventing_surname():
    result = LLMTurnResult.model_validate_json(
        RuleBasedMockLLMClient().generate(
            make_request(["executor.name", "executor.relationship"], "my brother James")
        )
    )

    assert [(item.field, item.value) for item in result.updates] == [
        ("executor.name", "James"),
        ("executor.relationship", "brother"),
    ]


def test_rule_mock_splits_children_names():
    result = LLMTurnResult.model_validate_json(
        RuleBasedMockLLMClient().generate(
            make_request(["children_names"], "Tom, Ana and Lee")
        )
    )

    assert result.updates[0].value == ["Tom", "Ana", "Lee"]


def test_rule_mock_interprets_no_gifts_as_empty_list():
    result = LLMTurnResult.model_validate_json(
        RuleBasedMockLLMClient().generate(make_request(["gifts"], "No gifts"))
    )

    assert result.updates[0].value == []


def test_rule_mock_unclear_yes_no_has_no_update_and_asks_again():
    result = LLMTurnResult.model_validate_json(
        RuleBasedMockLLMClient().generate(
            make_request(["has_children"], "I'm not sure")
        )
    )

    assert result.updates == []
    assert result.needs_clarification
    assert result.assistant_message == "Do you have any children?"


def test_rule_mock_closes_when_there_are_no_target_fields():
    result = LLMTurnResult.model_validate_json(
        RuleBasedMockLLMClient().generate(make_request([], "Thanks"))
    )

    assert result.assistant_message
    assert result.updates == []
