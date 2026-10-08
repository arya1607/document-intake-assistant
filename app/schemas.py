"""Data models for the Document Intake Assistant.

Three groups live here:
  1. State   - the structured facts collected so far (the source of truth)
  2. LLM     - the contract the model must follow when it answers us
  3. API     - request/response shapes between the web page and the backend
"""

from enum import Enum
from typing import Generic, Literal, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------


class FieldStatus(str, Enum):
    MISSING = "missing"          # nothing known yet
    UNCONFIRMED = "unconfirmed"  # heard something, but vague/guessed/contradicted
    CONFIRMED = "confirmed"      # user clearly stated it


T = TypeVar("T")


class TrackedField(BaseModel, Generic[T]):
    """A value together with how sure we are about it.

    value=None + MISSING      -> never asked / never answered
    value=[] + CONFIRMED      -> asked, and the user said "none"
    value="x" + UNCONFIRMED   -> heard something but need to double check
    """

    value: Optional[T] = None
    status: FieldStatus = FieldStatus.MISSING


class ExecutorState(BaseModel):
    name: TrackedField[str] = Field(default_factory=TrackedField[str])
    relationship: TrackedField[str] = Field(default_factory=TrackedField[str])


class CollectedState(BaseModel):
    """Everything we collect for the Personal Wishes Document."""

    full_name: TrackedField[str] = Field(default_factory=TrackedField[str])
    home_address: TrackedField[str] = Field(default_factory=TrackedField[str])
    covers_worldwide_assets: TrackedField[bool] = Field(default_factory=TrackedField[bool])
    has_children: TrackedField[bool] = Field(default_factory=TrackedField[bool])
    children_names: TrackedField[list[str]] = Field(default_factory=TrackedField[list[str]])
    executor: ExecutorState = Field(default_factory=ExecutorState)
    gifts: TrackedField[list[str]] = Field(default_factory=TrackedField[list[str]])
    additional_wishes: TrackedField[list[str]] = Field(default_factory=TrackedField[list[str]])


# The only field paths the model is allowed to update.
# (Enforced in the state manager, not here - see note in FieldUpdate.)
FIELD_PATHS: tuple[str, ...] = (
    "full_name",
    "home_address",
    "covers_worldwide_assets",
    "has_children",
    "children_names",
    "executor.name",
    "executor.relationship",
    "gifts",
    "additional_wishes",
)

# ---------------------------------------------------------------------------
# 2. LLM CONTRACT
# ---------------------------------------------------------------------------

FieldValue = Union[str, bool, list[str], None]


class FieldUpdate(BaseModel):
    """One change the model proposes. Our code decides whether to apply it.

    NOTE: `field` is a plain string on purpose. If the model invents a field
    name we want to reject only that update, not the whole response.
    """

    field: str
    value: FieldValue = None
    # stated    = user said it clearly
    # inferred  = model guessed from context
    # ambiguous = user's answer was unclear
    confidence: Literal["stated", "inferred", "ambiguous"] = "stated"
    # True when the user is changing something they said earlier
    is_correction: bool = False


class Clarification(BaseModel):
    field: str
    reason: str


class LLMTurnResult(BaseModel):
    """The only shape of answer we accept from the model."""

    updates: list[FieldUpdate] = Field(default_factory=list)
    needs_clarification: list[Clarification] = Field(default_factory=list)
    assistant_message: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# 3. API CONTRACT
# ---------------------------------------------------------------------------

MAX_MESSAGE_LENGTH = 2000


class MessageRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)


class TurnResponse(BaseModel):
    assistant_message: str
    state: CollectedState
    document: str
    warnings: list[str] = Field(default_factory=list)


class CreateSessionResponse(TurnResponse):
    session_id: str


class SessionSnapshot(BaseModel):
    session_id: str
    state: CollectedState
    document: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody