import pytest
from fastapi.testclient import TestClient

from app.llm.base import (
    LLMConfigError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.mock import FixtureMockLLMClient
from app.main import create_app
from app.service import ConversationService
from app.store import InMemoryStore
from tests.fixture_loader import load_fixture


def make_client(responses=None):
    service = ConversationService(
        InMemoryStore(),
        FixtureMockLLMClient(list(responses or [])),
    )
    return TestClient(create_app(service=service))


def test_create_session_returns_201_greeting_and_session_id():
    response = make_client().post("/api/sessions")

    assert response.status_code == 201
    body = response.json()
    assert body["session_id"]
    assert "fictional" in body["assistant_message"].casefold()


def test_message_happy_path_has_expected_response_fields():
    client = make_client([load_fixture("valid_multi_field.json")])
    session = client.post("/api/sessions").json()

    response = client.post(
        f"/api/sessions/{session['session_id']}/messages",
        json={"message": "My name is Jane Smith"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["assistant_message"]
    assert body["state"]["full_name"]["value"] == "Jane Smith"
    assert body["document"]
    assert body["warnings"] == []


@pytest.mark.parametrize(
    "payload",
    [{"message": ""}, {"message": "x" * 2001}],
)
def test_invalid_message_maps_to_validation_error(payload):
    client = make_client()
    session = client.post("/api/sessions").json()

    response = client.post(
        f"/api/sessions/{session['session_id']}/messages", json=payload
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_unknown_session_is_404_on_get_and_post():
    client = make_client()

    get_response = client.get("/api/sessions/unknown")
    post_response = client.post(
        "/api/sessions/unknown/messages", json={"message": "Hello"}
    )

    assert get_response.status_code == 404
    assert get_response.json()["error"]["code"] == "SESSION_NOT_FOUND"
    assert post_response.status_code == 404
    assert post_response.json()["error"]["code"] == "SESSION_NOT_FOUND"


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (LLMTimeoutError("private timeout detail"), 504, "LLM_TIMEOUT"),
        (LLMUnavailableError("private provider detail"), 503, "LLM_UNAVAILABLE"),
        (LLMConfigError("private config detail"), 503, "LLM_NOT_CONFIGURED"),
    ],
)
def test_llm_errors_map_to_friendly_http_errors(error, status, code):
    client = make_client([error, error])
    session = client.post("/api/sessions").json()

    response = client.post(
        f"/api/sessions/{session['session_id']}/messages",
        json={"message": "Jane Smith"},
    )

    assert response.status_code == status
    body = response.json()
    assert body["error"]["code"] == code
    assert "Traceback" not in response.text
    assert "private" not in body["error"]["message"]


def test_malformed_twice_maps_to_502_and_does_not_change_session():
    malformed = load_fixture("malformed_not_json.txt")
    client = make_client([malformed, malformed])
    session = client.post("/api/sessions").json()
    session_id = session["session_id"]
    before = client.get(f"/api/sessions/{session_id}").json()

    response = client.post(
        f"/api/sessions/{session_id}/messages",
        json={"message": "Jane Smith"},
    )
    after = client.get(f"/api/sessions/{session_id}").json()

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "LLM_BAD_RESPONSE"
    assert after == before


def test_health_reports_provider_and_configuration(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "")
    client = TestClient(create_app())

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_provider": "mock",
        "llm_configured": True,
    }


def test_gemini_without_key_still_starts_and_reports_unconfigured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")

    client = TestClient(create_app())
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "llm_provider": "gemini",
        "llm_configured": False,
    }


def test_validation_error_is_human_readable_and_has_no_traceback():
    client = make_client()
    session = client.post("/api/sessions").json()
    response = client.post(
        f"/api/sessions/{session['session_id']}/messages",
        json={"message": ""},
    )

    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert "Please check your request" in response.json()["error"]["message"]


def test_root_serves_static_index():
    response = make_client().get("/")

    assert response.status_code == 200
