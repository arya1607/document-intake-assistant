from app.config import Settings
from app.llm.base import LLMClient, LLMConfigError, LLMRequest
from app.llm.mock import RuleBasedMockLLMClient


class _UnavailableClient(LLMClient):
    # ######################################################
    # Function name : __init__
    # Description   : Store the configuration error for an unavailable client.
    # Date          : 08/10/2026
    # Author        : Arya Dere
    # ######################################################
    def __init__(self, message: str) -> None:
        self._message = message

    # ######################################################
    # Function name : generate
    # Description   : Raise the stored configuration error when generation is requested.
    # Date          : 08/10/2026
    # Author        : Arya Dere
    # ######################################################
    def generate(self, request: LLMRequest) -> str:
        raise LLMConfigError(self._message)


# ######################################################
# Function name : build_llm_client
# Description   : Create the configured mock or Gemini language-model client.
# Date          : 08/10/2026
# Author        : Arya Dere
# ######################################################
def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "mock":
        return RuleBasedMockLLMClient()
    if settings.llm_provider != "gemini":
        return _UnavailableClient(
            f"Unknown LLM provider '{settings.llm_provider}'."
        )
    if not settings.gemini_api_key or not settings.gemini_model:
        return _UnavailableClient(
            "Gemini provider requires GEMINI_API_KEY and GEMINI_MODEL."
        )

    try:
        from app.llm.gemini import GeminiClient
    except (ImportError, AttributeError):
        return _UnavailableClient("Gemini client not available")
    try:
        return GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    except LLMConfigError as error:
        return _UnavailableClient(str(error))
    except Exception:
        return _UnavailableClient("Gemini client not available")
