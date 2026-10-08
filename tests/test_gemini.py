from types import SimpleNamespace

import httpx
import httpx2
import pytest
from google.genai.errors import APIError

from app.config import Settings
from app.llm.base import (
    ChatMessage,
    LLMConfigError,
    LLMRequest,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.factory import _UnavailableClient, build_llm_client
from app.llm.gemini import GeminiClient


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.call = None

    def generate_content(self, **kwargs):
        self.call = kwargs
        if self.error is not None:
            raise self.error
        return self.response


class FakeSDKClient:
    def __init__(self, response=None, error=None):
        self.models = FakeModels(response=response, error=error)


def make_request():
    return LLMRequest(
        system_prompt="System instruction for the test.",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi"),
        ],
    )


def make_gemini(response=None, error=None):
    fake_client = FakeSDKClient(response=response, error=error)
    return (
        GeminiClient(
            api_key="test-key",
            model="test-model",
            timeout_seconds=12,
            client=fake_client,
        ),
        fake_client,
    )


def test_generate_maps_roles_passes_system_and_requests_json():
    gemini, fake_client = make_gemini(SimpleNamespace(text='{"updates":[]}'))

    text = gemini.generate(make_request())
    call = fake_client.models.call

    assert text == '{"updates":[]}'
    assert call["model"] == "test-model"
    assert call["contents"] == [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi"}]},
    ]
    config = call["config"]
    assert config.system_instruction == "System instruction for the test."
    assert config.response_mime_type == "application/json"
    assert config.temperature == 0.2
    assert config.automatic_function_calling.disable is True


def test_empty_response_text_raises_unavailable():
    gemini, _ = make_gemini(SimpleNamespace(text="  "))

    with pytest.raises(LLMUnavailableError, match="returned no text"):
        gemini.generate(make_request())


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (httpx.ReadTimeout("private timeout"), LLMTimeoutError),
        (httpx2.ReadTimeout("private httpx2 timeout"), LLMTimeoutError),
        (APIError(429, {"message": "private rate detail"}), LLMUnavailableError),
        (APIError(500, {"message": "private server detail"}), LLMUnavailableError),
        (APIError(401, {"message": "private auth detail"}), LLMConfigError),
        (APIError(403, {"message": "private permission detail"}), LLMConfigError),
        (httpx.ConnectError("private connection detail"), LLMUnavailableError),
        (
            httpx2.ConnectError("private httpx2 connection detail"),
            LLMUnavailableError,
        ),
    ],
)
def test_sdk_errors_map_to_typed_errors_without_leaking_details(error, expected):
    gemini, _ = make_gemini(error=error)

    with pytest.raises(expected) as raised:
        gemini.generate(make_request())

    assert "private" not in str(raised.value)
    assert "test-key" not in str(raised.value)


def test_api_error_logs_only_the_http_status(caplog):
    gemini, _ = make_gemini(
        error=APIError(400, {"message": "private prompt or credential detail"})
    )

    with caplog.at_level("WARNING"):
        with pytest.raises(LLMUnavailableError):
            gemini.generate(make_request())

    assert "HTTP status 400" in caplog.text
    assert "private" not in caplog.text


@pytest.mark.parametrize(
    ("api_key", "model"),
    [("", "test-model"), ("test-key", ""), ("  ", "test-model")],
)
def test_missing_key_or_model_raises_config_error(api_key, model):
    with pytest.raises(LLMConfigError):
        GeminiClient(api_key, model, timeout_seconds=5, client=FakeSDKClient())


def test_factory_builds_working_gemini_client_from_settings(monkeypatch):
    import app.llm.gemini as gemini_module

    fake_client = FakeSDKClient(response=SimpleNamespace(text='{"updates":[]}'))
    created = {}

    def make_fake_sdk_client(*, api_key, http_options):
        created["api_key"] = api_key
        created["timeout"] = http_options.timeout
        return fake_client

    monkeypatch.setattr(gemini_module.genai, "Client", make_fake_sdk_client)
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="test-key",
        gemini_model="test-model",
        llm_timeout_seconds=7,
    )

    client = build_llm_client(settings)

    assert isinstance(client, GeminiClient)
    assert client.generate(make_request()) == '{"updates":[]}'
    assert created == {"api_key": "test-key", "timeout": 7000}


def test_factory_returns_safe_client_when_configuration_missing():
    settings = Settings(
        llm_provider="gemini",
        gemini_api_key="",
        gemini_model="test-model",
    )

    client = build_llm_client(settings)

    assert isinstance(client, _UnavailableClient)
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        client.generate(make_request())


def test_factory_returns_unavailable_client_if_gemini_constructor_rejects_config(
    monkeypatch,
):
    import app.llm.gemini as gemini_module

    def fail_constructor(**kwargs):
        raise LLMConfigError("configuration rejected")

    monkeypatch.setattr(gemini_module, "GeminiClient", fail_constructor)
    client = build_llm_client(
        Settings(
            llm_provider="gemini",
            gemini_api_key="test-key",
            gemini_model="test-model",
        )
    )

    assert isinstance(client, _UnavailableClient)
    with pytest.raises(LLMConfigError, match="configuration rejected"):
        client.generate(make_request())
