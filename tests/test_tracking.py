"""Live stabilisation and session-counter tests."""

from __future__ import annotations

from growize_vision.domain import InspectionResult, MediaContext, SizeClass, Status
from growize_vision.tracking import LiveSession, LiveStabilizer


def result(status: Status, size: SizeClass | None = None) -> InspectionResult:
    return InspectionResult(
        result_id="test",
        context=MediaContext(source_id="test"),
        status=status,
        size_class=size,
    )


def accepted(size: SizeClass) -> InspectionResult:
    return result(Status.ACCEPTED, size)


def test_stable_label_requires_three_of_five_agreeing_frames() -> None:
    stabilizer = LiveStabilizer()
    first = stabilizer.feed(accepted(SizeClass.SMALL))
    second = stabilizer.feed(accepted(SizeClass.SMALL))
    third = stabilizer.feed(accepted(SizeClass.SMALL))
    assert first["stable"] is False
    assert second["stable"] is False
    assert third["stable"] is True
    assert third["stable_label"] == "SMALL"
    assert third["instant_label"] == "SMALL"
    assert third["samples"] == 3


def test_borderline_frame_clears_the_stable_label() -> None:
    stabilizer = LiveStabilizer()
    for _ in range(3):
        stabilizer.feed(accepted(SizeClass.BIG))
    review = stabilizer.feed(result(Status.REVIEW_BORDERLINE))
    assert review["stable"] is False
    assert review["stable_label"] is None
    assert review["samples"] == 0
    assert "not an accepted grade" in review["message"]


def test_camera_failure_discards_stale_overlay_and_stability():
    session = LiveSession()
    for _ in range(3):
        session.observe(accepted(SizeClass.BIG))
    session.last_annotated_rgb = "previous BIG overlay"
    session.invalidate_current_frame()
    assert session.last_annotated_rgb is None
    assert session.frames_seen == 3
    assert session.eligible_frames == 3
    state = session.observe(accepted(SizeClass.BIG))
    assert state["stable"] is False
    assert state["samples"] == 1


def test_majority_wins_but_current_frame_is_always_separate() -> None:
    stabilizer = LiveStabilizer()
    stabilizer.feed(accepted(SizeClass.SMALL))
    stabilizer.feed(accepted(SizeClass.SMALL))
    stabilizer.feed(accepted(SizeClass.BIG))
    stabilizer.feed(accepted(SizeClass.BIG))
    final = stabilizer.feed(accepted(SizeClass.SMALL))
    assert final["stable"] is True
    assert final["stable_label"] == "SMALL"
    assert final["instant_label"] == "SMALL"


def test_unstable_split_never_fabricates_a_label() -> None:
    stabilizer = LiveStabilizer()
    stabilizer.feed(accepted(SizeClass.SMALL))
    stabilizer.feed(accepted(SizeClass.BIG))
    stabilizer.feed(accepted(SizeClass.SMALL))
    state = stabilizer.feed(accepted(SizeClass.BIG))
    assert state["stable"] is False
    assert state["stable_label"] is None


def test_stabilizer_window_is_bounded() -> None:
    stabilizer = LiveStabilizer(window=3, minimum_agree=2)
    assert stabilizer._history.maxlen == 3
    stabilizer.feed(accepted(SizeClass.SMALL))
    stabilizer.feed(accepted(SizeClass.SMALL))
    stabilizer.feed(accepted(SizeClass.SMALL))
    state = stabilizer.feed(accepted(SizeClass.SMALL))
    assert state["samples"] == 3


def test_live_session_counts_only_accepted_frames_as_eligible() -> None:
    session = LiveSession()
    session.observe(result(Status.NO_OBJECT))
    session.observe(accepted(SizeClass.BIG))
    session.observe(result(Status.REVIEW_BORDERLINE))
    session.observe(accepted(SizeClass.BIG))
    assert session.frames_seen == 4
    assert session.eligible_frames == 2
    assert session.last_accepted_utc is not None
