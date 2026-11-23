
# Evaluation protocol

Split by subject/session; keep all views of one event together. Fit preprocessing on train only. Freeze thresholds before test.

Recognition: event precision/recall/F1, class-wise scores, macro-F1, false alarms/hour, missed and duplicate events, latency from annotated onset. Define event matching tolerance before evaluation.

Subtype: slip/stair/unknown macro-F1 only if the test set has adequate independently labelled events.

Tracking: ID switches, fragmentation, track coverage, IDF1/HOTA where annotations permit.

Runtime: end-to-end FPS, p50/p95 stage latency, CPU/GPU, RAM/VRAM, dropped frames and sustained run time. Measure depth inference separately from downstream work. Vendor benchmark numbers are not project measurements.

Report subject/session/event/hour counts and representative false positives/negatives. Frame accuracy alone is insufficient.
