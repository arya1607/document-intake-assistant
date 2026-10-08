import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings, llm_configured
from app.llm.base import (
    LLMClient,
    LLMConfigError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.factory import build_llm_client
from app.schemas import (
    CreateSessionResponse,
    ErrorResponse,
    MessageRequest,
    SessionSnapshot,
    TurnResponse,
)
from app.service import ConversationService
from app.store import InMemoryStore, SessionNotFound
from app.validation import MalformedLLMOutput


logger = logging.getLogger(__name__)
_STATIC_DIRECTORY = Path(__file__).resolve().parent.parent / "static"


def create_app(service: ConversationService | None = None) -> FastAPI:
    app_settings = get_settings()
    if service is None:
        store = InMemoryStore()
        llm: LLMClient = build_llm_client(app_settings)
        conversation_service = ConversationService(store, llm)
    else:
        conversation_service = service

    application = FastAPI(title="Document Intake Assistant")
    application.state.conversation_service = conversation_service
    application.state.settings = app_settings

    @application.exception_handler(RequestValidationError)
    async def request_validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            422,
            "VALIDATION_ERROR",
            "Please check your request and try again.",
        )

    @application.exception_handler(SessionNotFound)
    async def session_not_found_handler(
        request: Request, exc: SessionNotFound
    ) -> JSONResponse:
        return _error_response(
            404, "SESSION_NOT_FOUND", "That conversation could not be found."
        )

    @application.exception_handler(LLMTimeoutError)
    async def llm_timeout_handler(request: Request, exc: LLMTimeoutError) -> JSONResponse:
        return _error_response(
            504, "LLM_TIMEOUT", "The assistant took too long to respond. Please try again."
        )

    @application.exception_handler(LLMUnavailableError)
    async def llm_unavailable_handler(
        request: Request, exc: LLMUnavailableError
    ) -> JSONResponse:
        return _error_response(
            503,
            "LLM_UNAVAILABLE",
            "The assistant is temporarily unavailable. Please try again shortly.",
        )

    @application.exception_handler(LLMConfigError)
    async def llm_config_handler(request: Request, exc: LLMConfigError) -> JSONResponse:
        return _error_response(
            503,
            "LLM_NOT_CONFIGURED",
            "The assistant is not configured yet. Please contact the administrator.",
        )

    @application.exception_handler(MalformedLLMOutput)
    async def malformed_output_handler(
        request: Request, exc: MalformedLLMOutput
    ) -> JSONResponse:
        return _error_response(
            502,
            "LLM_BAD_RESPONSE",
            "The assistant returned an invalid response. Please try again.",
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unexpected API error")
        return _error_response(
            500, "INTERNAL_ERROR", "Something went wrong. Please try again."
        )

    @application.post(
        "/api/sessions",
        response_model=CreateSessionResponse,
        status_code=201,
    )
    def create_session() -> CreateSessionResponse:
        return conversation_service.start_session()

    @application.post(
        "/api/sessions/{session_id}/messages",
        response_model=TurnResponse,
    )
    def post_message(session_id: str, request: MessageRequest) -> TurnResponse:
        return conversation_service.handle_turn(session_id, request.message)

    @application.get(
        "/api/sessions/{session_id}",
        response_model=SessionSnapshot,
    )
    def get_session(session_id: str) -> SessionSnapshot:
        return conversation_service.get_snapshot(session_id)

    @application.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "llm_provider": app_settings.llm_provider,
            "llm_configured": llm_configured(app_settings),
        }

    application.mount(
        "/",
        StaticFiles(directory=_STATIC_DIRECTORY, html=True),
        name="static",
    )
    return application


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = ErrorResponse(error={"code": code, "message": message})
    return JSONResponse(status_code=status_code, content=body.model_dump())


app = create_app()