import sys
from pathlib import Path
import sys

# Running this file directly puts scripts/, not the project root, on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.llm.base import LLMError
from app.llm.factory import build_llm_client
from app.service import ConversationService
from app.store import InMemoryStore


# ######################################################
# Function name : main
# Description   : Run a smoke check against the configured Gemini assistant.
# Date          : 08/10/2026
# Author        : Arya Dere
# ######################################################
def main() -> None:
    settings = get_settings()
    service = ConversationService(InMemoryStore(), build_llm_client(settings))
    session = service.start_session()
    try:
        response = service.handle_turn(
            session.session_id,
            "My name is Alex Example.",
        )
    except LLMError as error:
        print(f"Gemini smoke test failed: {error}")
        sys.exit(1)
    print("Assistant:", response.assistant_message)
    print("State:", response.state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
