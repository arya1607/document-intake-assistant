import pytest

from app.schemas import LLMTurnResult
from app.validation import MalformedLLMOutput, parse_llm_output
from tests.fixture_loader import load_fixture


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
def test_valid_fixtures_parse(fixture_name):
    assert isinstance(parse_llm_output(load_fixture(fixture_name)), LLMTurnResult)


def test_markdown_fences_and_chatter_are_removed():
    result = parse_llm_output(load_fixture("malformed_fenced.txt"))

    assert result.assistant_message == "Thank you."


@pytest.mark.parametrize(
    "fixture_name",
    [
        "malformed_truncated.txt",
        "malformed_not_json.txt",
        "schema_invalid.json",
    ],
)
def test_malformed_fixtures_raise(fixture_name):
    with pytest.raises(MalformedLLMOutput):
        parse_llm_output(load_fixture(fixture_name))


def test_empty_output_raises():
    with pytest.raises(MalformedLLMOutput) as error:
        parse_llm_output(" \n\t")

    assert error.value.reason == "output is empty"


def test_validation_error_reason_includes_short_schema_summary():
    with pytest.raises(MalformedLLMOutput) as error:
        parse_llm_output(load_fixture("schema_invalid.json"))

    assert "does not match expected schema" in error.value.reason
    assert len(error.value.raw) <= 500


def test_non_string_output_is_rejected():
    with pytest.raises(MalformedLLMOutput):
        parse_llm_output(None)  # type: ignore[arg-type]


def test_raw_output_is_truncated_for_logging():
    raw = "x" * 600

    with pytest.raises(MalformedLLMOutput) as error:
        parse_llm_output(raw)

    assert len(error.value.raw) == 500
