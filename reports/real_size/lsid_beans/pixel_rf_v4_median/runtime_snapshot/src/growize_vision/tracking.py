"""Shared live-frame stabilisation without hiding borderline results."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .domain import InspectionResult, Status


@dataclass
class LiveStabilizer:
    """Require 3 of the last 5 frames to agree before showing a stable label.

    The raw current decision is always shown separately.  A review/borderline
    frame clears the stable label immediately, so smoothing cannot conceal an
    ambiguous measurement.
    """

    window: int = 5
    minimum_agree: int = 3
    _history: deque[str] = field(default_factory=lambda: deque(maxlen=5))
    last_status: Status | None = None

    def __post_init__(self) -> None:
        self._history = deque(self._history, maxlen=max(1, self.window))

    def feed(self, result: InspectionResult) -> dict[str, object]:
        self.last_status = result.status
        if result.status != Status.ACCEPTED or result.size_class is None:
            self._history.clear()
            return {
                "instant_label": None,
                "stable_label": None,
                "stable": False,
                "samples": 0,
                "message": "Current frame is not an accepted grade; no stabilised label.",
            }
        label = result.size_class.value
        self._history.append(label)
        if not self._history:
            return {
                "instant_label": label,
                "stable_label": None,
                "stable": False,
                "samples": 0,
                "message": "Collecting eligible frames.",
            }
        counts: dict[str, int] = {}
        for value in self._history:
            counts[value] = counts.get(value, 0) + 1
        winner, count = max(counts.items(), key=lambda item: item[1])
        stable = count >= self.minimum_agree and len(self._history) >= self.minimum_agree
        return {
            "instant_label": label,
            "stable_label": winner if stable else None,
            "stable": stable,
            "samples": len(self._history),
            "message": (
                f"Stable {winner} across {count}/{len(self._history)} eligible frames."
                if stable
                else f"Collecting eligible frames ({count}/{self.minimum_agree} agree)."
            ),
        }


@dataclass
class LiveSession:
    stabilizer: LiveStabilizer = field(default_factory=LiveStabilizer)
    frames_seen: int = 0
    eligible_frames: int = 0
    last_accepted_utc: str | None = None
    last_annotated_rgb: object | None = field(default=None, repr=False)

    def invalidate_current_frame(self) -> None:
        self.last_annotated_rgb = None
        self.stabilizer = LiveStabilizer(
            window=self.stabilizer.window,
            minimum_agree=self.stabilizer.minimum_agree,
            last_status=Status.INVALID_INPUT,
        )

    def observe(self, result: InspectionResult) -> dict[str, object]:
        self.frames_seen += 1
        stats = self.stabilizer.feed(result)
        if result.status == Status.ACCEPTED:
            self.eligible_frames += 1
            self.last_accepted_utc = result.timestamp
        return stats
