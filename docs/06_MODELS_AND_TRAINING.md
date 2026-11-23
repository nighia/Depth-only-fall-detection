
# Models and training

## E0 — Rule baseline
Use transparent temporal rules based on vertical movement, aspect change and region area. Tune on training/validation only.

## E1 — Depth features + GRU
First learned baseline. Input is `[time, feature]` per track. Use a compact GRU and Stage A classification. Train Stage B separately on fall events only if data support it.

## E2 — Depth-map sequences + 3D CNN
Retains spatial depth structure; likely higher compute/data demand.

## E3 — Depth-map sequences + ConvLSTM
Models spatial changes over time; more complex.

## Training controls
Subject/session split; training-only scaler; fixed seeds; class-imbalance handling documented; early stopping on validation; save config/checkpoint; do not tune on test. Report support counts and confidence intervals where feasible.

Thresholding, temporal smoothing and cooldown are tuned on validation and frozen before test evaluation.
