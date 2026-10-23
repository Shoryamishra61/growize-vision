"""Inventory local species imagery without inventing specimen or size ground truth."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = []
    failures = []
    for path in sorted((ROOT / "data/raw").rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        relative = path.relative_to(ROOT).as_posix()
        source = path.relative_to(ROOT / "data/raw").parts[0]
        try:
            with Image.open(path) as original:
                image = ImageOps.exif_transpose(original).convert("RGB")
                gray = image.convert("L").resize((9, 8))
                pixels = list(gray.getdata())
                bits = [pixels[y * 9 + x] > pixels[y * 9 + x + 1]
                        for y in range(8) for x in range(8)]
                dhash = sum(int(bit) << i for i, bit in enumerate(bits))
                row = {"path": relative, "source": source,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                       "dhash": f"{dhash:016x}", "width": image.width, "height": image.height,
                       "label": None, "original_split": None, "physical_truth": False}
                if source == "kaggle_veg":
                    row.update(label=path.parent.name, original_split=path.parent.parent.name)
                elif source in {"okra_pods", "harvested_okra", "cropped_okra_pods", "white_bg_okra"}:
                    row["label"] = "Okra"
                elif source == "extracted_onion":
                    row["label"] = "Onion"
                elif source == "spinach":
                    row["label"] = path.parent.name
                rows.append(row)
        except (OSError, ValueError) as exc:
            failures.append({"path": relative, "error": str(exc)})
    groups = defaultdict(list)
    for row in rows:
        groups[(row["width"], row["height"], row["pixel_sha256"])].append(row)
    duplicates = [group for group in groups.values() if len(group) > 1]
    cross_split = [group for group in duplicates
                   if len({r["original_split"] for r in group if r["original_split"]}) > 1]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "images": len(rows), "sources": dict(Counter(r["source"] for r in rows)),
        "labels": dict(Counter(r["label"] for r in rows)),
        "decode_errors": failures, "duplicate_pixel_groups": len(duplicates),
        "cross_original_split_duplicate_groups": cross_split,
        "limitations": ["No specimen IDs or measured physical ground truth inferred.",
                        "dHash is a discovery aid, not a guarantee against near-duplicate leakage.",
                        "Source rights require separate approval before new training.",
                        "The spinach folder contains five distinct leaf taxa, not one spinach class."],
    }
    out = ROOT / "reports/species_audit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "inventory.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cross_original_split_duplicate_groups"}, indent=2))
    print(f"Cross-split identical-pixel groups: {len(cross_split)}", flush=True)


if __name__ == "__main__":
    main()
