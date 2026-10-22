"""Build a three-panel failure gallery from synthetic refusal fixtures.

Panels are: original input, annotated decision, and the machine-readable reason.
Every panel is labelled SYNTHETIC.  No real-produce failure is implied.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from growize_vision.config import load_inference_settings, load_policies
from growize_vision.domain import MediaContext, SourceType
from growize_vision.evidence import PROJECT_ROOT, load_json, relative_to_project
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import load_image_file

DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "demo" / "demo_manifest.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "failure_gallery"
FAILURE_IDS = {
    "no_marker_synthetic",
    "empty_mat_synthetic",
    "blurred_synthetic",
    "edge_clipped_synthetic",
}


def wrap(text: str, width: int = 58) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def fit(image: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale == 1.0:
        return image.copy()
    return cv2.resize(
        image, (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = load_inference_settings()
    policies = load_policies()
    service = InferenceService(policies=policies, settings=settings)
    manifest = load_json(args.manifest)
    fixtures = [
        item for item in manifest["fixtures"] if item["fixture_id"] in FAILURE_IDS
    ]
    rows = []
    failures = []
    for fixture in fixtures:
        path = PROJECT_ROOT / fixture["path"]
        decoded = load_image_file(
            path,
            max_pixels=settings.max_image_pixels,
            max_bytes=settings.max_upload_bytes,
        )
        result = service.inspect_frame(
            decoded.bgr,
            MediaContext(
                source_type=SourceType.IMAGE,
                source_id=fixture["fixture_id"],
                device="failure-gallery",
            ),
            InferenceRequest(
                crop_key=fixture["crop"],
                marker_side_mm=float(fixture["marker_side_mm"]),
                expected_marker_id=0,
                demo_policy=True,
            ),
        )
        original = fit(decoded.bgr, 620, 520)
        annotated = fit(
            result.annotated_image if result.annotated_image is not None else decoded.bgr,
            620,
            520,
        )
        panel_w, panel_h = 620, 540
        canvas = np.full((panel_h, panel_w * 3, 3), 248, dtype=np.uint8)
        canvas[: original.shape[0], : original.shape[1]] = original
        canvas[: annotated.shape[0], panel_w : panel_w + annotated.shape[1]] = annotated
        cv2.line(canvas, (panel_w, 0), (panel_w, panel_h), (200, 200, 200), 1)
        cv2.line(canvas, (panel_w * 2, 0), (panel_w * 2, panel_h), (200, 200, 200), 1)

        y = 34
        cv2.putText(
            canvas, "SYNTHETIC - NOT REAL PRODUCE", (panel_w * 2 + 18, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 160), 1, cv2.LINE_AA,
        )
        y += 30
        lines = [
            f"EXPECTED: {fixture['expected_demonstration_class']}",
            f"ACTUAL:   {result.status.value}",
            *(f"  {reason}" for reason in result.reasons[:3]),
            f"objects: {len(result.instances)}",
            f"policy: {result.policy.approval_status if result.policy else 'none'}",
        ]
        for line in lines:
            for part in wrap(line):
                cv2.putText(
                    canvas, part, (panel_w * 2 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (45, 45, 45), 1, cv2.LINE_AA,
                )
                y += 20
        cv2.putText(
            canvas, "original", (12, panel_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "annotated", (panel_w + 12, panel_h - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA,
        )
        target = output_dir / f"{fixture['fixture_id']}.png"
        if not cv2.imwrite(str(target), canvas):
            raise SystemExit(f"Could not write gallery image: {target}")
        expected = fixture["expected_demonstration_class"]
        actual = (
            result.size_class.value if result.size_class else result.status.value
        )
        if actual != expected:
            failures.append(f"{fixture['fixture_id']}: {expected} -> {actual}")
        rows.append(
            {
                "fixture": fixture["fixture_id"],
                "expected": expected,
                "actual": actual,
                "reason": result.reasons[0] if result.reasons else "",
                "image": relative_to_project(target),
            }
        )

    lines = [
        "# Failure gallery",
        "",
        "**All cases below are synthetic fixtures.** They demonstrate the refusal",
        "contract and are not examples of real-produce failures or accuracy.",
        "",
        "| Fixture | Expected | Actual | Primary reason | Image |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['fixture']} | {row['expected']} | {row['actual']} | "
            f"{row['reason'].replace('|', '/')} | [PNG]({Path(row['image']).name}) |"
        )
    lines.extend(
        [
            "",
            "A real-produce failure gallery cannot be built until licensed, measured",
            "specimens and the ARL-approved policy are available. That item is",
            "`BLOCKED`, not passed.",
        ]
    )
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Failure gallery: {output_dir}")
    print(f"Cases: {len(rows)}; contract failures: {failures or 'none'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
