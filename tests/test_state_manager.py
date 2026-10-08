import pytest

from app.schemas import CollectedState, FieldStatus, FieldUpdate, TrackedField
from app.state_manager import apply_updates, get_field, is_complete, next_targets


def update(
    field: str,
    value: object,
    confidence: str = "stated",
    is_correction: bool = False,
) -> FieldUpdate:
    return FieldUpdate.model_construct(
        field=field,
        value=value,
        confidence=confidence,
        is_correction=is_correction,
    )


def test_apply_updates_does_not_mutate_the_input_state():
    state = CollectedState()
    state.gifts = TrackedField[list[str]](value=["old"], status=FieldStatus.MISSING)
    original = state.model_copy(deep=True)

    result = apply_updates(state, [update("full_name", "  Jane Smith  ")])

    assert state == original
    assert result.state.full_name.value == "Jane Smith"
    assert result.state is not state


def test_applies_valid_multi_field_updates():
    result = apply_updates(
        CollectedState(),
        [
            update("full_name", "Jane Smith"),
            update("executor.name", "James"),
            update("executor.relationship", "brother"),
        ],
    )

    assert result.state.full_name.value == "Jane Smith"
    assert result.state.executor.name.value == "James"
    assert result.state.executor.relationship.value == "brother"
    assert result.warnings == []


def test_unknown_field_is_rejected_without_blocking_valid_updates():
    result = apply_updates(
        CollectedState(),
        [update("unknown.field", "ignored"), update("full_name", "Jane")],
    )

    assert result.state.full_name.value == "Jane"
    assert result.warnings
    assert "unknown field" in result.warnings[0]


@pytest.mark.parametrize(("answer", "expected"), [("yes", True), ("No", False)])
def test_bool_strings_are_coerced(answer, expected):
    result = apply_updates(
        CollectedState(), [update("covers_worldwide_assets", answer)]
    )

    assert result.state.covers_worldwide_assets.value is expected


def test_wrong_value_type_is_rejected():
    result = apply_updates(CollectedState(), [update("full_name", True)])

    assert result.state.full_name.value is None
    assert result.warnings


def test_none_value_is_rejected():
    result = apply_updates(CollectedState(), [update("full_name", None)])

    assert result.state.full_name.status is FieldStatus.MISSING
    assert result.warnings


def test_stated_update_confirms_field():
    result = apply_updates(CollectedState(), [update("full_name", "Jane")])

    assert result.state.full_name.status is FieldStatus.CONFIRMED


def test_inferred_update_is_unconfirmed():
    result = apply_updates(
        CollectedState(), [update("full_name", "Jane", confidence="inferred")]
    )

    assert result.state.full_name.status is FieldStatus.UNCONFIRMED


def test_stated_update_upgrades_unconfirmed_field():
    state = CollectedState()
    state.full_name = TrackedField[str](
        value="Jane", status=FieldStatus.UNCONFIRMED
    )

    result = apply_updates(state, [update("full_name", "Jane Smith")])

    assert result.state.full_name.value == "Jane Smith"
    assert result.state.full_name.status is FieldStatus.CONFIRMED


def test_confirmed_scalar_difference_without_correction_is_a_conflict():
    state = CollectedState()
    state.full_name = TrackedField[str](
        value="Jane Smith", status=FieldStatus.CONFIRMED
    )

    result = apply_updates(state, [update("full_name", "Jane Doe")])

    assert result.state.full_name.value == "Jane Smith"
    assert result.conflicts == ["full_name"]
    assert result.warnings


def test_correction_overwrites_confirmed_scalar():
    state = CollectedState()
    state.full_name = TrackedField[str](
        value="Jane Smith", status=FieldStatus.CONFIRMED
    )

    result = apply_updates(
        state, [update("full_name", "Jane Doe", is_correction=True)]
    )

    assert result.state.full_name.value == "Jane Doe"
    assert result.state.full_name.status is FieldStatus.CONFIRMED
    assert result.conflicts == []


def test_uncertain_update_cannot_overwrite_confirmed_value():
    state = CollectedState()
    state.full_name = TrackedField[str](
        value="Jane Smith", status=FieldStatus.CONFIRMED
    )

    result = apply_updates(
        state, [update("full_name", "Jane Doe", confidence="ambiguous")]
    )

    assert result.state.full_name.value == "Jane Smith"
    assert result.conflicts == ["full_name"]
    assert result.warnings


def test_confirmed_lists_merge_with_case_insensitive_deduplication():
    state = CollectedState()
    state.gifts = TrackedField[list[str]](
        value=["Books", "Painting"], status=FieldStatus.CONFIRMED
    )

    result = apply_updates(
        state, [update("gifts", ["books", "Jewelry", "  ", "JEWELRY"])]
    )

    assert result.state.gifts.value == ["Books", "Painting", "Jewelry"]
    assert result.state.gifts.status is FieldStatus.CONFIRMED


def test_list_correction_replaces_existing_items():
    state = CollectedState()
    state.gifts = TrackedField[list[str]](
        value=["Books", "Painting"], status=FieldStatus.CONFIRMED
    )

    result = apply_updates(
        state, [update("gifts", ["Jewelry"], is_correction=True)]
    )

    assert result.state.gifts.value == ["Jewelry"]


def test_empty_list_confirms_none_on_a_missing_field():
    result = apply_updates(CollectedState(), [update("gifts", [])])

    assert result.state.gifts.value == []
    assert result.state.gifts.status is FieldStatus.CONFIRMED


def test_empty_non_correction_list_conflicts_with_confirmed_list():
    state = CollectedState()
    state.gifts = TrackedField[list[str]](
        value=["Books"], status=FieldStatus.CONFIRMED
    )

    result = apply_updates(state, [update("gifts", [])])

    assert result.state.gifts.value == ["Books"]
    assert result.conflicts == ["gifts"]


def test_child_names_conflict_with_confirmed_no_children():
    state = CollectedState()
    state.has_children = TrackedField[bool](
        value=False, status=FieldStatus.CONFIRMED
    )

    result = apply_updates(state, [update("children_names", ["Tom"])])

    assert result.state.children_names.value is None
    assert result.conflicts == ["has_children"]
    assert result.warnings


def test_child_names_infer_has_children_when_it_is_missing():
    result = apply_updates(CollectedState(), [update("children_names", ["Tom"])])

    assert result.state.children_names.value == ["Tom"]
    assert result.state.has_children.value is True
    assert result.state.has_children.status is FieldStatus.UNCONFIRMED


def test_confirming_no_children_resets_child_names():
    state = CollectedState()
    state.children_names = TrackedField[list[str]](
        value=["Tom"], status=FieldStatus.CONFIRMED
    )

    result = apply_updates(state, [update("has_children", False)])

    assert result.state.has_children.value is False
    assert result.state.children_names.value is None
    assert result.state.children_names.status is FieldStatus.MISSING


def test_updates_are_processed_in_field_path_order():
    result = apply_updates(
        CollectedState(),
        [update("children_names", ["Tom"]), update("has_children", True)],
    )

    assert result.state.has_children.value is True
    assert result.state.has_children.status is FieldStatus.CONFIRMED
    assert result.state.children_names.value == ["Tom"]


def test_james_brother_does_not_invent_a_surname():
    result = apply_updates(
        CollectedState(),
        [
            update("executor.name", "James"),
            update("executor.relationship", "brother"),
        ],
    )

    assert result.state.executor.name.value == "James"
    assert result.state.executor.relationship.value == "brother"
    assert result.state.full_name.value is None


def test_dotted_path_helpers_get_nested_tracked_field():
    state = CollectedState()

    assert get_field(state, "executor.name") is state.executor.name


def test_next_targets_prioritizes_conflicts_then_unconfirmed_then_missing():
    state = CollectedState()
    state.has_children = TrackedField[bool](
        value=True, status=FieldStatus.CONFIRMED
    )
    state.home_address = TrackedField[str](
        value="Address", status=FieldStatus.UNCONFIRMED
    )

    targets = next_targets(
        state, conflicts=("gifts", "full_name"), limit=4
    )

    assert targets == [
        "full_name",
        "gifts",
        "home_address",
        "covers_worldwide_assets",
    ]


def test_next_targets_respects_limit_and_skips_children_without_confirmed_yes():
    state = CollectedState()
    state.has_children = TrackedField[bool](
        value=True, status=FieldStatus.UNCONFIRMED
    )

    assert next_targets(state, limit=2) == ["has_children", "full_name"]
    assert "children_names" not in next_targets(state, limit=20)


def test_next_targets_skips_children_after_confirmed_no():
    state = CollectedState()
    state.has_children = TrackedField[bool](
        value=False, status=FieldStatus.CONFIRMED
    )

    assert "children_names" not in next_targets(state, limit=20)


def test_is_complete_requires_every_applicable_field_confirmed():
    state = CollectedState()
    for field_path in (
        "full_name",
        "home_address",
        "covers_worldwide_assets",
        "has_children",
        "executor.name",
        "executor.relationship",
        "gifts",
        "additional_wishes",
    ):
        field = get_field(state, field_path)
        if field_path in {"covers_worldwide_assets", "has_children"}:
            field.value = False
        elif field_path in {"gifts", "additional_wishes"}:
            field.value = []
        else:
            field.value = "filled"
        field.status = FieldStatus.CONFIRMED

    assert is_complete(state)


def test_is_complete_is_false_when_a_required_field_is_unconfirmed():
    state = CollectedState()
    state.full_name = TrackedField[str](
        value="Jane", status=FieldStatus.UNCONFIRMED
    )

    assert not is_complete(state)


def test_is_complete_requires_names_when_children_are_confirmed():
    state = CollectedState()
    state.has_children = TrackedField[bool](
        value=True, status=FieldStatus.CONFIRMED
    )

    assert not is_complete(state)
