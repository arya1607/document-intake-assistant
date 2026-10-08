import logging

from app.document import render
from app.llm.base import (
    ChatMessage,
    LLMClient,
    LLMConfigError,
    LLMRequest,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.prompts import build_request
from app.schemas import CreateSessionResponse, SessionSnapshot, TurnResponse
from app.schemas import LLMTurnResult
from app.state_manager import apply_updates, next_targets
from app.store import Session, SessionNotFound, SessionStore
from app.validation import MalformedLLMOutput, parse_llm_output


logger = logging.getLogger(__name__)

GREETING = (
    "Welcome! This is a fictional example, not legal advice. "
    "What is your full name?"
)


class ConversationService:
    # ######################################################
    # Function name : __init__
    # Description   : Set up the session store and language-model client.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def __init__(self, store: SessionStore, llm: LLMClient) -> None:
        self.store = store
        self.llm = llm

    # ######################################################
    # Function name : start_session
    # Description   : Create a session and return its initial greeting and draft.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def start_session(self) -> CreateSessionResponse:
        session = self.store.create()
        session.history.append(
            ChatMessage(role="assistant", content=GREETING)
        )
        self.store.save(session)
        return CreateSessionResponse(
            session_id=session.id,
            assistant_message=GREETING,
            state=session.state,
            document=render(session.state),
            warnings=[],
        )

    # ######################################################
    # Function name : get_snapshot
    # Description   : Return the current state and rendered draft for a session.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def get_snapshot(self, session_id: str) -> SessionSnapshot:
        session = self.store.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)
        return SessionSnapshot(
            session_id=session.id,
            state=session.state,
            document=render(session.state),
        )

    # ######################################################
    # Function name : handle_turn
    # Description   : Process a user message and persist the updated conversation.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def handle_turn(self, session_id: str, message: str) -> TurnResponse:
        session = self.store.get(session_id)
        if session is None:
            raise SessionNotFound(session_id)

        targets = next_targets(session.state, session.conflicts)
        request = build_request(
            session.state,
            session.history,
            message,
            targets,
            session.conflicts,
        )
        parsed = self._generate_and_parse(request)
        result = apply_updates(session.state, parsed.updates)
        document = render(result.state)

        updated_session = Session(
            id=session.id,
            state=result.state,
            history=[
                *session.history,
                ChatMessage(role="user", content=message),
                ChatMessage(role="assistant", content=parsed.assistant_message),
            ],
            conflicts=result.conflicts,
        )
        self.store.save(updated_session)

        for warning in result.warnings:
            logger.warning("State update warning: %s", warning)
        return TurnResponse(
            assistant_message=parsed.assistant_message,
            state=result.state,
            document=document,
            warnings=result.warnings,
        )

    # ######################################################
    # Function name : _generate_and_parse
    # Description   : Generate model output and retry once if parsing fails.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def _generate_and_parse(self, request: LLMRequest) -> LLMTurnResult:
        for parse_attempt in range(2):
            raw = self._generate_with_retry(request)
            try:
                return parse_llm_output(raw)
            except MalformedLLMOutput:
                if parse_attempt == 0:
                    logger.info("Malformed LLM output; retrying once.")
                else:
                    raise
        raise AssertionError("Unreachable response parsing state.")

    # ######################################################
    # Function name : _generate_with_retry
    # Description   : Retry model generation once after transient provider errors.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def _generate_with_retry(self, request: LLMRequest) -> str:
        for attempt in range(2):
            try:
                return self.llm.generate(request)
            except LLMConfigError:
                raise
            except (LLMTimeoutError, LLMUnavailableError):
                if attempt == 1:
                    raise
                logger.info("Transient LLM error; retrying once.")
        raise AssertionError("Unreachable LLM generation state.")