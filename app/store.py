from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Lock
from uuid import uuid4

from app.llm.base import ChatMessage
from app.schemas import CollectedState


@dataclass
class Session:
    id: str
    state: CollectedState = field(default_factory=CollectedState)
    history: list[ChatMessage] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


class SessionNotFound(Exception):
    """Raised when a session ID does not exist in the store."""


class SessionStore(ABC):
    @abstractmethod
    def create(self) -> Session:
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> Session | None:
        raise NotImplementedError

    @abstractmethod
    def save(self, session: Session) -> None:
        raise NotImplementedError


class InMemoryStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    def create(self) -> Session:
        session = Session(id=uuid4().hex)
        with self._lock:
            self._sessions[session.id] = deepcopy(session)
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return deepcopy(session) if session is not None else None

    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = deepcopy(session)