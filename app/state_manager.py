from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from app.schemas import (
    FIELD_PATHS,
    CollectedState,
    FieldStatus,
    FieldUpdate,
    TrackedField,
)


@dataclass
class ApplyResult:
    state: CollectedState
    warnings: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)


# ######################################################
# Function name : get_field
# Description   : Retrieve a tracked field using its dotted path.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def get_field(state: CollectedState, field_path: str) -> TrackedField[Any]:
    """Get a tracked field using its dotted path."""
    if field_path not in FIELD_PATHS:
        raise ValueError(f"Unknown field path: {field_path}")

    current: Any = state
    for part in field_path.split("."):
        current = getattr(current, part)
    return current


# ######################################################
# Function name : set_field
# Description   : Update a tracked field's value and confirmation status.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def set_field(
    state: CollectedState,
    field_path: str,
    value: str | bool | list[str] | None,
    status: FieldStatus,
) -> None:
    """Set a tracked field's value and status using its dotted path."""
    tracked_field = get_field(state, field_path)
    tracked_field.value = value
    tracked_field.status = status


# ######################################################
# Function name : _coerce_value
# Description   : Validate and normalize an update for its target field.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _coerce_value(field_path: str, value: object) -> str | bool | list[str]:
    if value is None:
        raise ValueError("a value cannot be blank")

    if field_path in {
        "full_name",
        "home_address",
        "executor.name",
        "executor.relationship",
    }:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected a non-empty string")
        return value.strip()

    if field_path in {"covers_worldwide_assets", "has_children"}:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"yes", "true"}:
                return True
            if normalized in {"no", "false"}:
                return False
        raise ValueError("expected yes, true, no, or false")

    if field_path in {"children_names", "gifts", "additional_wishes"}:
        items = [value] if isinstance(value, str) else value
        if not isinstance(items, list) or not all(
            isinstance(item, str) for item in items
        ):
            raise ValueError("expected a string or a list of strings")
        return [item.strip() for item in items if item.strip()]

    raise ValueError(f"Unknown field path: {field_path}")


# ######################################################
# Function name : _same_value
# Description   : Compare proposed and stored field values with normalization.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _same_value(
    field_path: str, current: object, proposed: str | bool | list[str]
) -> bool:
    if field_path in {
        "full_name",
        "home_address",
        "executor.name",
        "executor.relationship",
    }:
        return (
            isinstance(current, str)
            and isinstance(proposed, str)
            and current.casefold() == proposed.casefold()
        )
    return current == proposed


# ######################################################
# Function name : _add_conflict
# Description   : Add a field path to conflicts once.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _add_conflict(conflicts: list[str], field_path: str) -> None:
    if field_path not in conflicts:
        conflicts.append(field_path)


# ######################################################
# Function name : _merge_lists
# Description   : Merge list values while removing case-insensitive duplicates.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def _merge_lists(existing: object, proposed: list[str]) -> list[str]:
    merged = list(existing) if isinstance(existing, list) else []
    seen = {item.casefold() for item in merged if isinstance(item, str)}
    for item in proposed:
        key = item.casefold()
        if key not in seen:
            merged.append(item)
            seen.add(key)
    return merged


# ######################################################
# Function name : apply_updates
# Description   : Apply validated updates and report warnings and conflicts.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def apply_updates(
    state: CollectedState, updates: list[FieldUpdate]
) -> ApplyResult:
    result_state = deepcopy(state)
    warnings: list[str] = []
    conflicts: list[str] = []
    path_order = {path: index for index, path in enumerate(FIELD_PATHS)}
    ordered_updates = sorted(
        enumerate(updates),
        key=lambda item: (path_order.get(item[1].field, len(FIELD_PATHS)), item[0]),
    )

    for _, update in ordered_updates:
        field_path = update.field
        if field_path not in path_order:
            warnings.append(f"Rejected update for unknown field '{field_path}'.")
            continue

        try:
            proposed = _coerce_value(field_path, update.value)
        except ValueError as error:
            warnings.append(f"Rejected update for '{field_path}': {error}.")
            continue

        if (
            field_path == "children_names"
            and proposed
            and result_state.has_children.status is FieldStatus.CONFIRMED
            and result_state.has_children.value is False
        ):
            _add_conflict(conflicts, "has_children")
            warnings.append(
                "Children were listed, but the confirmed answer says there are no children."
            )
            continue

        tracked_field = get_field(result_state, field_path)
        existing_status = tracked_field.status
        existing_value = tracked_field.value
        new_status = (
            FieldStatus.CONFIRMED
            if update.confidence == "stated"
            else FieldStatus.UNCONFIRMED
        )

        if existing_status is FieldStatus.CONFIRMED:
            if _same_value(field_path, existing_value, proposed):
                continue

            if update.confidence != "stated":
                _add_conflict(conflicts, field_path)
                warnings.append(
                    f"Uncertain update cannot change confirmed field '{field_path}'."
                )
                continue

            if update.is_correction:
                set_field(result_state, field_path, proposed, new_status)
            elif isinstance(proposed, list):
                if not proposed:
                    _add_conflict(conflicts, field_path)
                    warnings.append(
                        f"An empty update cannot erase confirmed field '{field_path}'."
                    )
                    continue
                merged = _merge_lists(existing_value, proposed)
                if merged != existing_value:
                    set_field(result_state, field_path, merged, existing_status)
            else:
                _add_conflict(conflicts, field_path)
                warnings.append(
                    f"Conflicting answer for confirmed field '{field_path}'."
                )
                continue
        else:
            set_field(result_state, field_path, proposed, new_status)

        if field_path == "has_children":
            if proposed is False:
                set_field(
                    result_state,
                    "children_names",
                    None,
                    FieldStatus.MISSING,
                )
        elif (
            field_path == "children_names"
            and proposed
            and result_state.has_children.status is FieldStatus.MISSING
        ):
            set_field(
                result_state,
                "has_children",
                True,
                FieldStatus.UNCONFIRMED,
            )

    return ApplyResult(state=result_state, warnings=warnings, conflicts=conflicts)


# ######################################################
# Function name : next_targets
# Description   : Select the next fields that need an answer or clarification.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def next_targets(
    state: CollectedState, conflicts: tuple[str, ...] | list[str] = (), limit: int = 2
) -> list[str]:
    if limit <= 0:
        return []

    eligible_paths = [
        path
        for path in FIELD_PATHS
        if path != "children_names"
        or (
            state.has_children.status is FieldStatus.CONFIRMED
            and state.has_children.value is True
        )
    ]
    conflict_set = set(conflicts)
    targets: list[str] = []

    for status in (
        None,
        FieldStatus.UNCONFIRMED,
        FieldStatus.MISSING,
    ):
        for path in eligible_paths:
            if path in targets:
                continue
            if status is None:
                matches = path in conflict_set
            else:
                matches = get_field(state, path).status is status
            if matches:
                targets.append(path)
                if len(targets) >= limit:
                    return targets

    return targets


# ######################################################
# Function name : is_complete
# Description   : Check whether all applicable document fields are confirmed.
# Date          : 07/10/2026
# Author        : Arya Dere
# ######################################################
def is_complete(state: CollectedState) -> bool:
    for path in FIELD_PATHS:
        if path == "children_names" and not (
            state.has_children.status is FieldStatus.CONFIRMED
            and state.has_children.value is True
        ):
            continue
        if get_field(state, path).status is not FieldStatus.CONFIRMED:
            return False
    return True