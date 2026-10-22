from pathlib import Path

from growize_vision.domain import MediaContext, SourceType
from growize_vision.inference import InferenceRequest, InferenceService
from growize_vision.media import load_image_file

svc = InferenceService()
names = [
    "potato_small_synthetic.png",
    "potato_big_synthetic.png",
    "okra_small_synthetic.png",
    "okra_big_synthetic.png",
    "no_marker_synthetic.png",
    "empty_mat_synthetic.png",
    "blurred_synthetic.png",
    "edge_clipped_synthetic.png",
]
for name in names:
    path = Path("data/demo") / name
    crop = "okra" if name.startswith("okra") else "potato"
    decoded = load_image_file(path)
    result = svc.inspect_frame(
        decoded.bgr,
        MediaContext(source_type=SourceType.IMAGE, source_id=name),
        InferenceRequest(
            crop_key=crop,
            marker_side_mm=50.0,
            expected_marker_id=0,
            demo_policy=True,
        ),
    )
    measurement = (
        result.instances[0].measurement.value_mm
        if result.instances and result.instances[0].measurement
        else None
    )
    print(
        name,
        result.status.value,
        result.size_class.value if result.size_class else None,
        "n=",
        len(result.instances),
        "mm=",
        round(measurement, 2) if measurement else None,
        "|",
        result.reasons[:2],
    )
