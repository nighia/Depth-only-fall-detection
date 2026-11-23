
# Experiment plan

| ID | Experiment | Purpose |
|---|---|---|
| E0 | Depth-feature rules | Mathematical baseline |
| E1 | Depth features + GRU | Temporal learning baseline |
| E2 | Depth maps + 3D CNN | Spatial-temporal comparison |
| E3 | Depth maps + ConvLSTM | Recurrent spatial comparison |
| E4 | Feature ablations | Identify useful feature groups |
| E5 | YOLO26 Depth variants | Accuracy/runtime trade-off |
| E6 | Camera angles | Viewpoint robustness |
| E7 | Crowding/occlusion | Multi-person robustness |
| E8 | Flat ground vs stairs | Subtype feasibility |

Ablations: shape only; position/movement only; depth distribution only; shape + movement; shape + movement + depth; full features including quality.

Order: validate depth output -> validate regions -> validate tracking -> E0 -> E1 -> E4 -> E5–E7 -> E2/E3 if justified -> E8 when stair data are sufficient.

Record dataset/split manifest, model/config, feature schema, seed, hardware, metrics, runtime and failure analysis for every run.
