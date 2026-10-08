import json

from fastapi.testclient import TestClient

from app.llm.base import LLMConfigError
from app.llm.mock import FixtureMockLLMClient
from app.main import create_app
from app.service import ConversationService
from app.store import InMemoryStore
from tests.fixture_loader import load_fixture


DOCUMENT_BANNER = "*** FICTIONAL EXAMPLE DOCUMENT - NOT LEGAL ADVICE ***"


def result(updates, assistant_message="Thanks. What would you like to add?"):
    return json.dumps(
        {
            "updates": updates,
            "needs_clarification": [],
            "assistant_message": assistant_message,
        }
    )


def make_client(responses):
    llm = FixtureMockLLMClient(responses)
    service = ConversationService(InMemoryStore(), llm)
    client = TestClient(create_app(service=service))
    return client, llm


def start_session(client):
    response = client.post("/api/sessions")
    assert response.status_code == 201
    return response.json()


def send(client, session_id, message):
    return client.post(
        f"/api/sessions/{session_id}/messages",
        json={"message": message},
    )


def test_executor_james_does_not_gain_a_surname_in_state_or_document():
    client, _ = make_client([load_fixture("valid_executor_james.json")])
    session = start_session(client)

    response = send(client, session["session_id"], "my brother James")
    body = response.json()

    assert response.status_code == 200
    assert body["state"]["executor"]["name"]["value"] == "James"
    assert body["state"]["executor"]["relationship"]["value"] == "brother"
    assert body["state"]["full_name"]["value"] is None
    assert "James, my brother" in body["document"]
    assert "James Smith" not in body["document"]


def test_first_answer_can_update_name_address_and_worldwide_assets():
    client, _ = make_client(
        [
            result(
                [
                    {"field": "full_name", "value": "Jane Smith"},
                    {"field": "home_address", "value": "10 Example Street"},
                    {"field": "covers_worldwide_assets", "value": True},
                ]
            )
        ]
    )
    session = start_session(client)

    response = send(
        client,
        session["session_id"],
        "My name is Jane Smith, I live at 10 Example Street, and include worldwide assets.",
    )
    body = response.json()

    assert response.status_code == 200
    assert body["state"]["full_name"]["value"] == "Jane Smith"
    assert body["state"]["home_address"]["value"] == "10 Example Street"
    assert body["state"]["covers_worldwide_assets"]["value"] is True


def test_correcting_no_children_updates_names_and_document():
    no_children = result(
        [{"field": "has_children", "value": False}],
        "Understood. What else would you like to share?",
    )
    client, _ = make_client([no_children, load_fixture("correction.json")])
    session = start_session(client)

    first = send(client, session["session_id"], "I have no children.")
    assert first.json()["state"]["has_children"]["value"] is False
    assert "I have no children." in first.json()["document"]

    corrected = send(
        client,
        session["session_id"],
        "I need to correct that: I have children Tom and Ana.",
    )
    body = corrected.json()

    assert body["state"]["has_children"]["value"] is True
    assert body["state"]["children_names"]["value"] == ["Tom", "Ana"]
    assert "I have children:" in body["document"]
    assert "- Tom" in body["document"]
    assert "- Ana" in body["document"]
    assert "I have no children." not in body["document"]


def test_child_names_conflicting_with_confirmed_no_are_not_rendered_and_prompted():
    no_children = result([{"field": "has_children", "value": False}])
    follow_up = result([], "Which answer is correct?")
    client, llm = make_client(
        [no_children, load_fixture("contradiction.json"), follow_up]
    )
    session = start_session(client)
    session_id = session["session_id"]
    send(client, session_id, "No children.")

    conflict = send(client, session_id, "My child Tom.")
    conflict_body = conflict.json()
    assert conflict_body["warnings"]
    assert conflict_body["state"]["children_names"]["value"] is None
    assert "Tom" not in conflict_body["document"]
    assert "I have no children." in conflict_body["document"]

    send(client, session_id, "Please clarify.")
    assert len(llm.requests) == 3
    prompt_context = llm.requests[2].messages[0].content
    assert "has_children" in prompt_context
    assert "ask which is correct" in prompt_context


def test_ambiguous_answer_stays_unconfirmed_and_document_marks_it():
    client, _ = make_client([load_fixture("ambiguous.json")])
    session = start_session(client)

    response = send(client, session["session_id"], "Maybe include everything?")
    field = response.json()["state"]["covers_worldwide_assets"]

    assert field["status"] == "unconfirmed"
    assert field["value"] is True
    assert "[Awaiting confirmation]" in response.json()["document"]


def test_malformed_output_twice_returns_502_without_changing_session():
    malformed = load_fixture("malformed_not_json.txt")
    client, _ = make_client([malformed, malformed])
    session = start_session(client)
    session_id = session["session_id"]
    before = client.get(f"/api/sessions/{session_id}").json()

    response = send(client, session_id, "My name is Jane Smith.")
    after = client.get(f"/api/sessions/{session_id}").json()

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_BAD_RESPONSE"
    assert after == before


def test_confirmed_field_is_not_targeted_again_on_consecutive_requests():
    client, llm = make_client(
        [
            result(
                [{"field": "full_name", "value": "Jane Smith"}],
                "What is your address?",
            ),
            result(
                [{"field": "home_address", "value": "10 Example Street"}],
                "Thanks.",
            ),
        ]
    )
    session = start_session(client)

    send(client, session["session_id"], "My name is Jane Smith.")
    send(client, session["session_id"], "10 Example Street")

    assert "full_name" in llm.requests[0].target_fields
    assert "full_name" not in llm.requests[1].target_fields


def test_document_contains_fictional_banner_at_every_step():
    client, _ = make_client(
        [
            result([{"field": "full_name", "value": "Jane Smith"}]),
            result([{"field": "home_address", "value": "10 Example Street"}]),
        ]
    )
    session = start_session(client)

    assert DOCUMENT_BANNER in session["document"]
    for message in ("My name is Jane Smith.", "10 Example Street"):
        response = send(client, session["session_id"], message)
        assert response.status_code == 200
        assert DOCUMENT_BANNER in response.json()["document"]


def test_missing_gemini_configuration_starts_and_returns_friendly_error(
    monkeypatch,
):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    client = TestClient(create_app())

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["llm_configured"] is False

    session = start_session(client)
    response = send(client, session["session_id"], "My name is Jane Smith.")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_NOT_CONFIGURED"
