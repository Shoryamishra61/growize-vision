"""Non-inferred crop reference cards.

These cards explain the configured measurement convention and unit definition.
They do not claim sample-specific nutrition, soil, yield or defect information;
those require external records and are outside this assessment.
"""

from __future__ import annotations

from typing import Any

from .config import PolicyRegistry

GENERAL_NOTES = {
    "potato": "Round/oblong tuber. Current metric is projected minimum width; ARL must confirm diameter vs width convention.",
    "onion": "Single bulb. Current metric is projected minimum width; keep roots/paper skin within the visible mask if ARL defines them as part of the unit.",
    "tomato": "Single fruit. Projected width is an equatorial proxy only for a suitably flat pose; height and cultivar can bias it.",
    "okra": "Single pod. Current metric is the straight longest axis, not a curved centreline. Strongly bent pods should be reviewed.",
    "cucumber": "Single fruit. Current metric is the straight longest axis; strongly bent fruit is outside the supported pose.",
    "spinach": "Single flat-laid leaf blade only. A bundle is a different unit and must not inherit a single-leaf cutoff.",
    "coriander": "No defensible single-photo size definition without specifying single leaf vs standardized bunch and packing. Kept POLICY_REQUIRED.",
}


def crop_cards(policies: PolicyRegistry, demo: bool = False) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for key in policies.crop_keys:
        policy = policies.policy(key, demo=demo)
        cards.append(
            {
                "crop_key": key,
                "display_name": policy.crop_name,
                "physical_unit": policy.physical_unit,
                "metric": policy.metric,
                "metric_description": policy.metric_description,
                "threshold_mm": policy.threshold_mm,
                "threshold_status": policy.approval_status,
                "policy_version": policy.policy_version,
                "note": GENERAL_NOTES.get(key, ""),
                "not_measured": [
                    "soil pH/NPK/EC",
                    "farm yield",
                    "sample weight",
                    "internal defects",
                    "nutrition per specimen",
                ],
            }
        )
    return cards
