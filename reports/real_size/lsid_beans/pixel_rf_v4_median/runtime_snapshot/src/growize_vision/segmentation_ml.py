"""Optional learned-segmentation hook.

No learned checkpoint is shipped.  The assessment does not have licensed,
physically measured training data or a compatible GPU, so training/evaluation
of a vegetable segmenter has not been performed.  The baseline remains the
default provider.  This module makes the absence explicit instead of silently
pretending a COCO detector recognises every vegetable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelUnavailable(RuntimeError):
    pass


@dataclass
class LearnedSegmentationProvider:
    checkpoint_path: Path | None = None
    license: str | None = None
    model_card: Path | None = None

    @property
    def available(self) -> bool:
        return bool(
            self.checkpoint_path
            and self.checkpoint_path.exists()
            and self.license
            and self.model_card
            and self.model_card.exists()
        )

    def require(self) -> None:
        if not self.available:
            raise ModelUnavailable(
                "No learned segmentation provider is available. Required: a "
                "crop-specific checkpoint, its license, a model card, audited masks "
                "and a sealed held-out evaluation. Use the OpenCV baseline."
            )

    def segment(self, frame_bgr: Any, context: Any = None) -> Any:
        self.require()
        raise ModelUnavailable(
            "The optional inference adapter is not enabled in this build. "
            "Do not add a provider without the data/licence/evaluation gates."
        )
