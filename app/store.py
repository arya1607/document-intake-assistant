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
    # ######################################################
    # Function name : create
    # Description   : Define creation of a new conversation session.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    @abstractmethod
    def create(self) -> Session:
        raise NotImplementedError

    # ######################################################
    # Function name : get
    # Description   : Define retrieval of a session by its identifier.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    @abstractmethod
    def get(self, session_id: str) -> Session | None:
        raise NotImplementedError

    # ######################################################
    # Function name : save
    # Description   : Define persistence of an updated conversation session.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    @abstractmethod
    def save(self, session: Session) -> None:
        raise NotImplementedError


class InMemoryStore(SessionStore):
    # ######################################################
    # Function name : __init__
    # Description   : Initialize the in-memory session map and synchronization lock.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = Lock()

    # ######################################################
    # Function name : create
    # Description   : Create and store a new isolated conversation session.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def create(self) -> Session:
        session = Session(id=uuid4().hex)
        with self._lock:
            self._sessions[session.id] = deepcopy(session)
        return session

    # ######################################################
    # Function name : get
    # Description   : Return a copy of a stored session when it exists.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return deepcopy(session) if session is not None else None

    # ######################################################
    # Function name : save
    # Description   : Store a copy of the updated conversation session.
    # Date          : 07/10/2026
    # Author        : Arya Dere
    # ######################################################
    def save(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = deepcopy(session)