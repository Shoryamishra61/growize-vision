# Failure gallery

**All cases below are synthetic fixtures.** They demonstrate the refusal
contract and are not examples of real-produce failures or accuracy.

| Fixture | Expected | Actual | Primary reason | Image |
|---|---|---|---|---|
| no_marker_synthetic | CALIBRATION_REQUIRED | CALIBRATION_REQUIRED | A valid marker with the user-measured outer side length is required. | [PNG](no_marker_synthetic.png) |
| empty_mat_synthetic | NO_OBJECT | NO_OBJECT | NO_OBJECT: no foreground vegetable instance was detected. | [PNG](empty_mat_synthetic.png) |
| blurred_synthetic | LOW_IMAGE_QUALITY | LOW_IMAGE_QUALITY | BLUR_LOW (Laplacian variance 0.3 < 38.0) | [PNG](blurred_synthetic.png) |
| edge_clipped_synthetic | OCCLUDED | OCCLUDED | Object silhouette touches the frame edge; physical extent cannot be inferred from the visible part. | [PNG](edge_clipped_synthetic.png) |

A real-produce failure gallery cannot be built until licensed, measured
specimens and the ARL-approved policy are available. That item is
`BLOCKED`, not passed.
