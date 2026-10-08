import logging
from typing import Any

import httpx
from google import genai
from google.genai import errors, types

try:
    import httpx2
except ImportError:
    httpx2 = None

from app.llm.base import (
    ChatMessage,
    LLMClient,
    LLMConfigError,
    LLMRequest,
    LLMTimeoutError,
    LLMUnavailableError,
)

logger = logging.getLogger(__name__)

_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (
    (httpx.TimeoutException, TimeoutError)
    if httpx2 is None
    else (httpx.TimeoutException, httpx2.TimeoutException, TimeoutError)
)
_CONNECTION_ERRORS: tuple[type[BaseException], ...] = (
    (httpx.ConnectError, OSError)
    if httpx2 is None
    else (httpx.ConnectError, httpx2.ConnectError, OSError)
)


class GeminiClient(LLMClient):
    # ######################################################
    # Function name : __init__
    # Description   : Validate Gemini settings and configure the provider client.
    # Date          : 08/10/2026
    # Author        : Arya Dere
    # ######################################################
    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_seconds: float,
        client: Any | None = None,
    ) -> None:
        if not api_key or not api_key.strip():
            raise LLMConfigError("GEMINI_API_KEY must be configured.")
        if not model or not model.strip():
            raise LLMConfigError("GEMINI_MODEL must be configured.")

        self._model = model.strip()
        self._client = (
            client
            if client is not None
            else genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=int(timeout_seconds * 1000),
                ),
            )
        )

    # ######################################################
    # Function name : generate
    # Description   : Send the conversation to Gemini and return its response text.
    # Date          : 08/10/2026
    # Author        : Arya Dere
    # ######################################################
    def generate(self, request: LLMRequest) -> str:
        contents = [_to_gemini_content(message) for message in request.messages]
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=request.system_prompt,
                    response_mime_type="application/json",
                    temperature=0.2,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(
                        disable=True
                    ),
                ),
            )
        except _TIMEOUT_ERRORS:
            raise LLMTimeoutError("The Gemini request timed out.") from None
        except errors.APIError as error:
            logger.warning("Gemini API returned HTTP status %s.", error.code)
            if error.code in {401, 403}:
                raise LLMConfigError("API key rejected") from None
            if error.code in {408, 504}:
                raise LLMTimeoutError("The Gemini request timed out.") from None
            raise LLMUnavailableError(
                "Gemini is temporarily unavailable."
            ) from None
        except _CONNECTION_ERRORS:
            raise LLMUnavailableError(
                "Gemini is temporarily unavailable."
            ) from None
        except Exception as error:
            logger.warning(
                "Unexpected Gemini SDK error (%s).",
                type(error).__name__,
            )
            raise LLMUnavailableError(
                "Gemini is temporarily unavailable."
            ) from None

        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMUnavailableError("Gemini returned no text.")
        return text


# ######################################################
# Function name : _to_gemini_content
# Description   : Convert an application chat message to Gemini content format.
# Date          : 08/10/2026
# Author        : Arya Dere
# ######################################################
def _to_gemini_content(message: ChatMessage) -> dict[str, object]:
    role = "model" if message.role == "assistant" else "user"
    return {"role": role, "parts": [{"text": message.content}]}