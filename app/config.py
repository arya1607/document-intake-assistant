import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    llm_provider: str = "mock"
    gemini_api_key: str = ""
    gemini_model: str = ""
    llm_timeout_seconds: float = 30.0


# ######################################################
# Function name : get_settings
# Description   : Load application settings from environment variables.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def get_settings() -> Settings:
    load_dotenv()
    try:
        timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    except ValueError:
        timeout = 30.0
    return Settings(
        llm_provider=os.getenv("LLM_PROVIDER", "mock").strip().lower(),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "").strip(),
        llm_timeout_seconds=timeout,
    )


# ######################################################
# Function name : llm_configured
# Description   : Check whether the selected language model is configured.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def llm_configured(settings: Settings) -> bool:
    if settings.llm_provider == "mock":
        return True
    if settings.llm_provider == "gemini":
        return bool(settings.gemini_api_key and settings.gemini_model)
    return False