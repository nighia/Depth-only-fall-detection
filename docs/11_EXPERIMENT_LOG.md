
# Experiment log template

- Run ID/date:
- Experiment ID/hypothesis:
- Dataset/version and exact split manifest:
- Subject/session/event counts:
- Feature schema/model/config:
- Seed and software versions:
- Hardware:
- Event precision/recall/F1:
- Macro-F1:
- False alarms/hour:
- Missed/duplicate events:
- Median/p95 detection latency:
- FPS and p50/p95 stage latency:
- RAM/VRAM:
- False positives/false negatives:
- Tracking failures:
- Interpretation and next controlled experiment:


---
## Run V0 — depth conversion verification (sandbox, not a real camera)
- Experiment ID/hypothesis: Phase 1 exit check — YOLO26 Depth returns numeric metric maps through the adapter
- Dataset: one street photo (people + bus); 90-frame video synthesised from it with Gaussian noise (sigma 4/255) and 2% gain jitter. Simulated sensor noise, NOT a real camera.
- Feature schema/model/config: schema 0.2, yolo26n-depth.pt, imgsz 768, configs/base.yaml
- Software: ultralytics 8.4.216, torch 2.15.0 (CPU), numpy 2.5.4, cv2 5.0.0
- Result: float32 metres, shape == frame, 100% valid; median 4.4 m, p1 2.2 m, p99 14.5 m
- FPS/latency (sandbox CPU, nano, 768): p50 about 390 ms/frame; first call about 4 s
- Static-scene noise: per-pixel std p95 0.28 m; about 1-3% of range at every distance
- Finding: absolute 0.15 m threshold gave 27 false regions in 10 static frames; 0.05 m floor + 6% relative gave 5
- Interpretation / next controlled experiment: repeat on a real empty-room capture with the real camera; tune depth_difference_relative from rel_std_p95
