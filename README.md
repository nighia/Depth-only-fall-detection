
# Monocular Depth-Based Human Fall Detection

Research branch: YOLO26 Depth + temporal recognition. This is separate from the Pose + Depth V2 plan.

## Research boundary
A conventional RGB camera feeds YOLO26 Depth, which estimates a dense monocular depth map. The downstream fall classifier uses depth maps and depth-derived features, without explicit pose keypoints or RGB appearance embeddings. This is **not an RGB-free system**: depth is estimated from RGB.

## Pipeline
RGB video -> YOLO26 Depth -> numeric depth map -> depth-background foreground extraction -> connected components -> person-region tracking -> depth features -> temporal windows -> GRU -> Stage A classification -> Stage B subtype classification -> event/API.

A depth map does not identify people. Region extraction and tracking are essential, separate research components.

## First implementation
1. Verify YOLO26 Depth inference and numeric output.
2. Inspect depth maps and confirm output units/scale.
3. Run fixed-camera background subtraction and inspect regions.
4. Add IoU tracking and audit ID switches/merged people.
5. Export depth-derived feature sequences.
6. Build subject/session-separated windows.
7. Train GRU baseline and evaluate at event level.

## Verify camera + depth first
```bash
python scripts/verify_camera_depth.py --camera 0 --weights weights/yolo26n-depth.pt
python -m fall_detection.cli depth-flicker --depth-dir <depth-dir-of-an-EMPTY-static-scene>
```
Check `artifacts/verify/preview_first.png`, then set `regions.depth_difference_relative` in `configs/base.yaml` just above the reported `rel_std_p95`.

## Labels
Stage A: `normal`, `hard_negative_motion`, `fall`.
Stage B (falls only): `slip_fall`, `stair_fall`, `unknown_fall`.
Scene context is separate: `flat_ground`, `stairs`, `unknown`. A fall near stairs is not automatically a stair fall.

## Quick start
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python -m fall_detection.cli --help
```
Install an Ultralytics version that explicitly supports YOLO26 Depth and verify its API. Add model weights locally under `weights/`; do not commit them.

## Important limitations
Monocular depth is estimated from RGB; it is not measured sensor depth. Depth maps do not provide semantic person masks or IDs. Background subtraction assumes a static camera. Connected components can merge people or include furniture/stairs. Hard negatives such as sitting, bending, squatting and intentional lying down are essential. Stair subtype claims require dedicated labelled stair-fall data.

This is a research scaffold, not a validated or production-ready fall detector. No trained weights or performance claims are included.
