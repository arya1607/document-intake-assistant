"""Deterministic document generator: state -> draft Personal Wishes Document.

No LLM is involved here. The same state always produces the same text, and
only CONFIRMED values are printed, so the preview always matches the
latest confirmed state.
"""

from app.schemas import CollectedState, FieldStatus

BANNER = "*** FICTIONAL EXAMPLE DOCUMENT - NOT LEGAL ADVICE ***"
MISSING_TEXT = "[Not yet provided]"
UNCONFIRMED_TEXT = "[Awaiting confirmation]"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _is_known(field) -> bool:
    """True only if the user clearly told us this (status CONFIRMED)."""
    return field.status == FieldStatus.CONFIRMED and field.value is not None and field.value != ""


def _placeholder(field) -> str:
    return UNCONFIRMED_TEXT if field.status == FieldStatus.UNCONFIRMED else MISSING_TEXT


def _text(field) -> str:
    """Confirmed value as text, otherwise a placeholder."""
    return str(field.value) if _is_known(field) else _placeholder(field)


def _list_lines(field, empty_message: str) -> list[str]:
    """Render a list field as bullet lines."""
    if field.status == FieldStatus.CONFIRMED and field.value is not None:
        if len(field.value) == 0:
            return [empty_message]  # user explicitly said "none"
        return [f"- {item}" for item in field.value]
    return [_placeholder(field)]


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _personal_details(state: CollectedState) -> list[str]:
    return [
        f"Full name: {_text(state.full_name)}",
        f"Home address: {_text(state.home_address)}",
    ]


def _scope(state: CollectedState) -> list[str]:
    field = state.covers_worldwide_assets
    if not _is_known(field):
        return [_placeholder(field)]
    if field.value:
        return ["This document is intended to cover my assets worldwide."]
    return ["This document is not intended to cover assets worldwide."]


def _children(state: CollectedState) -> list[str]:
    has_children = state.has_children
    if not _is_known(has_children):
        return [_placeholder(has_children)]
    if has_children.value is False:
        return ["I have no children."]  # any stale child names are ignored

    names = state.children_names
    if names.status == FieldStatus.CONFIRMED and names.value:
        return ["I have children:"] + [f"- {name}" for name in names.value]
    return ["I have children.", f"Names: {_placeholder(names)}"]


def _executor(state: CollectedState) -> list[str]:
    name, relationship = state.executor.name, state.executor.relationship
    if _is_known(name) and _is_known(relationship):
        return [f"I appoint {name.value}, my {relationship.value}, as the executor of my wishes."]
    return [
        f"Executor name: {_text(name)}",
        f"Relationship to me: {_text(relationship)}",
    ]


def _gifts(state: CollectedState) -> list[str]:
    return _list_lines(state.gifts, "I have no specific gifts to leave.")


def _additional_wishes(state: CollectedState) -> list[str]:
    return _list_lines(state.additional_wishes, "I have no additional wishes.")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def render(state: CollectedState) -> str:
    """Build the draft document text from the collected state."""
    sections: list[tuple[str, list[str]]] = [
        ("PERSONAL DETAILS", _personal_details(state)),
        ("SCOPE", _scope(state)),
        ("CHILDREN", _children(state)),
        ("EXECUTOR", _executor(state)),
        ("SPECIFIC GIFTS", _gifts(state)),
        ("ADDITIONAL WISHES", _additional_wishes(state)),
    ]

    lines = [BANNER, "", "PERSONAL WISHES DOCUMENT (DRAFT)", ""]
    for title, body in sections:
        lines.append(title)
        lines.extend(body)
        lines.append("")
    lines.append(BANNER)
    return "\n".join(lines)