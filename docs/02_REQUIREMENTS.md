
# Requirements

## Functional
| ID | Requirement | Evidence |
|---|---|---|
| FR-01 | Read RGB video/camera frames | Frame reader with timestamps |
| FR-02 | Estimate dense monocular depth | Numeric maps and metadata |
| FR-03 | Extract person-region candidates | Region overlays and audit |
| FR-04 | Maintain track IDs | Track sequences and ID-switch evaluation |
| FR-05 | Extract depth features | Feature CSV with documented units |
| FR-06 | Build temporal windows | Labelled arrays and provenance |
| FR-07 | Train temporal classifier | Checkpoint, config and log |
| FR-08 | Predict Stage A | normal / hard_negative_motion / fall |
| FR-09 | Predict Stage B | slip_fall / stair_fall / unknown_fall |
| FR-10 | Emit event records | Timestamped event schema |
| FR-11 | Expose API | API contract tests |

## Non-functional
Reproducibility; privacy and retention controls; end-to-end runtime measurement; robustness to viewpoint/occlusion/crowding; feature/model versioning; authenticated API before deployment.

## Out of scope initially
Cross-camera re-identification, guaranteed detection of fully occluded people, clinical diagnosis, and production-grade safety certification.
