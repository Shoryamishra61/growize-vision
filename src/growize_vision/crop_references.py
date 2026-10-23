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
    "okra": "Single pod. Current metric is the curved centreline length (skeleton path with terminal-radius compensation); ambiguous branched skeletons are refused for review rather than measured with a straight box.",
    "cucumber": "Single fruit. Current metric is the straight longest axis; strongly bent fruit is outside the supported pose.",
    "spinach": "Single flat-laid leaf blade only. A bundle is a different unit and must not inherit a single-leaf cutoff.",
    "coriander": "No defensible single-photo size definition without specifying single leaf vs standardized bunch and packing. Kept POLICY_REQUIRED.",
    "brinjal": "Mixed type crop: round types grade by diameter, long types by length; max projected span is used until the buyer fixes the convention.",
    "capsicum": "Blocky fruit graded by diameter; US market grade edges (64/76 mm) used as bands.",
    "cabbage": "Whole head diameter proxy; wrapper leaves must be consistent between lots or the buyer must fix trimming.",
    "cauliflower": "Curd diameter proxy; jacket leaves are excluded by placing the curd flat and fully visible.",
    "beetroot": "Root diameter proxy; tops off before capture.",
    "turnip": "Root diameter proxy; weakest evidence tier - replace with a buyer specification when available.",
    "carrot": "Topped root length; UNECE/USDA minimums underpin the reference.",
    "green_chilli": "Curved pod length via skeleton centreline; export vs market types differ.",
    "bitter_gourd": "Curved fruit length via skeleton centreline; TNAU preferred market range anchors the cut.",
    "ridge_gourd": "Curved fruit length via skeleton centreline; strongly bent or folded fruit must be re-laid flat.",
    "bottle_gourd": "Curved fruit length; fruits longer than the mat are refused - capture on a larger station.",
    "green_bean": "Pod length via skeleton centreline; cluster/guar pods share the pod-length convention.",
    "ivy_gourd": "Very small tender fruit; keep the item fully inside the ROI with the marker visible.",
    "pointed_gourd": "Fusiform fruit length via skeleton centreline.",
    "sweet_potato": "Root length proxy; wide genotype spread (8.9-24.8 cm) makes the buyer spec decisive.",
    "pumpkin": "Refused at this station: fruit exceeds the measurement mat; larger station required.",
    "ash_gourd": "Refused at this station: fruit exceeds the measurement mat; larger station required.",
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
                "bands": [band.to_dict() for band in policy.bands] if policy.bands else None,
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
