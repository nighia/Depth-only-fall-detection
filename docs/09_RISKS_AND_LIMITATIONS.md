
# Risks and limitations

| Risk | Impact | Mitigation |
|---|---|---|
| Depth map has no person semantics | Regions may include objects | Audit masks; compare segmentation |
| Static background assumption | Camera/scene change breaks extraction | Fixed-camera baseline; scene-change checks |
| Multiple people merge | Identity/feature corruption | Crowding stress test |
| ID switches | Temporal sequence corrupted | Tracker evaluation and uncertainty flags |
| Monocular depth domain shift | Feature drift | External-scene tests and calibration |
| Hard negatives | False alarms | Include sitting, bending, kneeling, lying, exercise |
| Stair subtype scarcity | Unsupported claims | Dedicated stair recordings |
| Occlusion | Missing/merged regions | Evaluate partial/complete occlusion separately |
| Camera angle | Geometry changes | Multi-angle experiment |
| Runtime | Real-time target may fail | End-to-end profiling |
| Label inconsistency/leakage | Inflated results | Annotation guide and subject/session manifests |

Monocular depth is inferred from RGB, not measured depth. A person lying down is not necessarily falling. Stair proximity is not equivalent to stair-fall mechanism. Fully occluded people cannot be reconstructed reliably.
