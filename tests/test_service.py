import json

import pytest

from app.llm.base import LLMConfigError, LLMTimeoutError, LLMUnavailableError
from app.llm.mock import FixtureMockLLMClient, RuleBasedMockLLMClient
from app.schemas import FieldStatus
from app.service import GREETING, ConversationService
from app.state_manager import get_field, is_complete
from app.store import InMemoryStore, SessionNotFound
from app.validation import MalformedLLMOutput
from tests.fixture_loader import load_fixture


def make_service(responses):
    store = InMemoryStore()
    llm = FixtureMockLLMClient(responses)
    return ConversationService(store, llm), store, llm


def test_start_session_returns_greeting_and_empty_state():
    service, store, _ = make_service([])

    response = service.start_session()

    assert "fictional" in response.assistant_message.casefold()
    assert response.assistant_message == GREETING
    assert response.warnings == []
    assert all(
        get_field(response.state, path).status is FieldStatus.MISSING
        for path in (
            "full_name",
            "home_address",
            "covers_worldwide_assets",
            "has_children",
            "children_names",
            "executor.name",
            "executor.relationship",
            "gifts",
            "additional_wishes",
        )
    )
    session = store.get(response.session_id)
    assert session is not None
    assert session.history[0].role == "assistant"
    assert session.history[0].content == GREETING


def test_valid_turn_updates_state_history_and_document_and_persists():
    service, store, _ = make_service(
        [
            load_fixture("valid_multi_field.json"),
            load_fixture("valid_executor_james.json"),
        ]
    )
    started = service.start_session()

    first = service.handle_turn(started.session_id, "My name is Jane Smith")
    after_first = store.get(started.session_id)
    assert first.state.full_name.value == "Jane Smith"
    assert "Jane Smith" in first.document
    assert after_first is not None
    assert len(after_first.history) == 3
    assert after_first.history[-2].content == "My name is Jane Smith"

    second = service.handle_turn(started.session_id, "My brother James")
    persisted = service.get_snapshot(started.session_id)
    session = store.get(started.session_id)

    assert second.state.full_name.value == "Jane Smith"
    assert second.state.executor.name.value == "James"
    assert persisted.state.full_name.value == "Jane Smith"
    assert session is not None
    assert len(session.history) == 5


def test_james_fixture_adds_no_surname_to_state_or_document():
    service, _, _ = make_service([load_fixture("valid_executor_james.json")])
    started = service.start_session()

    response = service.handle_turn(started.session_id, "My brother James")

    assert response.state.executor.name.value == "James"
    assert response.state.executor.relationship.value == "brother"
    assert response.state.full_name.value is None
    assert "James, my brother" in response.document
    assert "James Smith" not in response.document


def test_multi_field_response_applies_all_updates():
    service, _, _ = make_service([load_fixture("valid_multi_field.json")])
    started = service.start_session()

    response = service.handle_turn(started.session_id, "Jane Smith lives at 10 Example Street")

    assert response.state.full_name.value == "Jane Smith"
    assert response.state.home_address.value == "10 Example Street"
    assert response.state.has_children.value is True


def test_correction_turn_changes_no_children_to_yes_and_sets_names():
    no_children = json.dumps(
        {
            "updates": [
                {
                    "field": "has_children",
                    "value": False,
                    "confidence": "stated",
                }
            ],
            "needs_clarification": [],
            "assistant_message": "Would you like to correct that later?",
        }
    )
    service, _, _ = make_service(
        [no_children, load_fixture("correction.json")]
    )
    started = service.start_session()

    first = service.handle_turn(started.session_id, "No, I have no children")
    assert first.state.has_children.value is False

    corrected = service.handle_turn(
        started.session_id, "I need to correct that: I have children Tom and Ana"
    )

    assert corrected.state.has_children.value is True
    assert corrected.state.children_names.value == ["Tom", "Ana"]


def test_child_name_contradiction_is_stored_and_included_in_next_request():
    no_children = json.dumps(
        {
            "updates": [
                {"field": "has_children", "value": False, "confidence": "stated"}
            ],
            "needs_clarification": [],
            "assistant_message": "Do you have children?",
        }
    )
    service, store, llm = make_service(
        [
            no_children,
            load_fixture("contradiction.json"),
            json.dumps(
                {
                    "updates": [],
                    "needs_clarification": [],
                    "assistant_message": "Which answer is correct?",
                }
            ),
        ]
    )
    started = service.start_session()
    service.handle_turn(started.session_id, "No children")

    response = service.handle_turn(started.session_id, "My child Tom")
    session = store.get(started.session_id)

    assert response.state.children_names.value is None
    assert response.warnings
    assert session is not None
    assert session.conflicts == ["has_children"]

    service.handle_turn(started.session_id, "Please clarify")
    assert len(llm.requests) == 3
    assert "ask which is correct" in llm.requests[-1].messages[0].content
    assert "has_children" in llm.requests[-1].messages[0].content


def test_malformed_once_then_valid_succeeds_after_one_retry():
    service, _, llm = make_service(
        [load_fixture("malformed_not_json.txt"), load_fixture("valid_multi_field.json")]
    )
    started = service.start_session()

    response = service.handle_turn(started.session_id, "Jane Smith")

    assert response.state.full_name.value == "Jane Smith"
    assert len(llm.requests) == 2


def test_malformed_twice_raises_and_keeps_session_unchanged():
    malformed = load_fixture("malformed_not_json.txt")
    service, store, llm = make_service([malformed, malformed])
    started = service.start_session()
    before = store.get(started.session_id)

    with pytest.raises(MalformedLLMOutput):
        service.handle_turn(started.session_id, "Jane Smith")

    after = store.get(started.session_id)
    assert before is not None and after == before
    assert len(llm.requests) == 2


def test_timeout_once_then_valid_succeeds():
    service, _, llm = make_service(
        [LLMTimeoutError("temporary timeout"), load_fixture("valid_multi_field.json")]
    )
    started = service.start_session()

    response = service.handle_turn(started.session_id, "Jane Smith")

    assert response.state.full_name.value == "Jane Smith"
    assert len(llm.requests) == 2


def test_unavailable_once_then_valid_succeeds():
    service, _, llm = make_service(
        [
            LLMUnavailableError("provider unavailable"),
            load_fixture("valid_multi_field.json"),
        ]
    )
    started = service.start_session()

    response = service.handle_turn(started.session_id, "Jane Smith")

    assert response.state.full_name.value == "Jane Smith"
    assert len(llm.requests) == 2


def test_timeout_twice_raises_and_keeps_session_unchanged():
    service, store, llm = make_service(
        [LLMTimeoutError("timeout one"), LLMTimeoutError("timeout two")]
    )
    started = service.start_session()
    before = store.get(started.session_id)

    with pytest.raises(LLMTimeoutError):
        service.handle_turn(started.session_id, "Jane Smith")

    after = store.get(started.session_id)
    assert before is not None and after == before
    assert len(llm.requests) == 2


def test_config_error_is_not_retried():
    service, store, llm = make_service(
        [LLMConfigError("missing config"), load_fixture("valid_multi_field.json")]
    )
    started = service.start_session()
    before = store.get(started.session_id)

    with pytest.raises(LLMConfigError):
        service.handle_turn(started.session_id, "Jane Smith")

    after = store.get(started.session_id)
    assert before is not None and after == before
    assert len(llm.requests) == 1


def test_in_memory_store_returns_deep_copies():
    store = InMemoryStore()
    created = store.create()
    created.state.full_name.value = "Unpersisted"
    stored = store.get(created.id)

    assert stored is not None
    assert stored.state.full_name.value is None

    stored.state.full_name.value = "Jane"
    store.save(stored)
    stored.state.full_name.value = "Changed outside store"
    persisted = store.get(created.id)

    assert persisted is not None
    assert persisted.state.full_name.value == "Jane"


def test_schema_invalid_fixture_is_retried_then_raises_if_repeated():
    schema_invalid = load_fixture("schema_invalid.json")
    service, store, llm = make_service([schema_invalid, schema_invalid])
    started = service.start_session()
    before = store.get(started.session_id)

    with pytest.raises(MalformedLLMOutput):
        service.handle_turn(started.session_id, "Jane Smith")

    after = store.get(started.session_id)
    assert before is not None and after == before
    assert len(llm.requests) == 2


def test_unknown_session_raises_session_not_found():
    service, _, _ = make_service([])

    with pytest.raises(SessionNotFound):
        service.get_snapshot("missing-session")
    with pytest.raises(SessionNotFound):
        service.handle_turn("missing-session", "hello")


def test_rule_based_mock_completes_a_conversation_without_repeating_questions():
    service = ConversationService(InMemoryStore(), RuleBasedMockLLMClient())
    started = service.start_session()

    answers = [
        "My name is Jane Smith",
        "yes",
        "Tom and Ana",
        "my brother James",
        "none",
    ]
    questions = []
    session_id = started.session_id
    response = None
    for answer in answers:
        response = service.handle_turn(session_id, answer)
        questions.append(response.assistant_message)

    assert response is not None
    assert is_complete(response.state)
    assert len(questions) == len(set(questions))
