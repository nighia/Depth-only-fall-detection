
# Research brief

## Working title
**Monocular Depth-Based Human Fall Detection Using YOLO26 and Temporal Sequence Recognition**

## Main question
Can spatial and temporal information derived from monocular depth maps support fall recognition without explicit human pose estimation or skeleton keypoints?

## Research questions
- RQ1: Can YOLO26 monocular depth maps support fall detection without pose estimation?
- RQ2: Which depth-derived features best distinguish falls from normal and hard-negative motion?
- RQ3: Which temporal architecture offers the best accuracy/runtime trade-off?
- RQ4: Can depth-derived features distinguish stair falls from flat-ground falls?
- RQ5: How robust is the pipeline under multiple people, crossing, partial occlusion and viewpoint changes?

## Hypotheses to test
- H1: Temporal features outperform static per-frame descriptors for distinguishing falls from lying/sitting/bending.
- H2: Shape + vertical movement + depth distribution performs better than individual feature groups.
- H3: A compact GRU provides a useful baseline for feature sequences.
- H4: Stair subtype recognition is harder than binary fall detection and requires dedicated stair data.

## Novelty framing
Do not claim depth-only fall detection itself is novel. A defensible contribution may be a systematic evaluation of a modern monocular-depth pipeline, feature ablations, temporal-model comparisons and robustness tests for stairs and multi-person scenes.

## Success definition
Use event-level precision/recall/F1, false alarms per hour, latency, robustness and compute. Frame accuracy alone is insufficient.
