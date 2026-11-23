
# Implementation roadmap

## Phase 0 — Setup
- [ ] Create environment; record Python/PyTorch/CUDA/Ultralytics versions.
- [ ] Obtain permitted sample video and compatible weights.
- [ ] Create dataset manifest and access notes.
Exit: reproducible environment.

## Phase 1 — Depth inference
- [ ] Verify output shape, numeric type, units/scale.
- [ ] Save timestamps and model metadata.
- [ ] Inspect difficult frames.
Exit: reproducible numeric depth maps.

## Phase 2 — Regions/tracking
- [ ] Build static-camera background.
- [ ] Extract components and visualize boxes.
- [ ] Add IoU tracker and audit IDs/merges.
Exit: usable single-person track sequences.

## Phase 3 — Features/data
- [ ] Export feature CSV and validate units.
- [ ] Annotate events and map labels.
- [ ] Create subject/session split manifests.
- [ ] Build temporal windows.
Exit: leak-free train/validation/test arrays.

## Phase 4 — Baselines
- [ ] Implement E0 rules and E1 GRU.
- [ ] Tune event threshold on validation.
- [ ] Evaluate event metrics and false alarms/hour.
Exit: reproducible baseline report.

## Phase 5 — Comparisons
- [ ] Feature ablations, depth-model variants, viewpoint and occlusion tests.
- [ ] 3D CNN/ConvLSTM only if justified.
- [ ] Stair subtype evaluation only with adequate data.

## Phase 6 — Prototype
- [ ] Event manager, API, dashboard integration.
- [ ] Authentication, privacy/retention review and sustained runtime test.
