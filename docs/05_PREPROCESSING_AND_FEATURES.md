
# Preprocessing and features

## Depth maps
Keep numeric floating-point maps, not colorized previews. Record output units/scale and model version. Keep raw and calibrated maps separate. Handle invalid values explicitly.

## Baseline region extraction
For a fixed camera: collect a no-foreground background period, compute depth difference, threshold, perform morphological cleanup, find connected components and filter by area/shape. This is not a semantic person detector. Audit masks and boxes visually.

## Feature candidates
| Group | Features |
|---|---|
| Shape | bbox width/height, aspect ratio, foreground area |
| Position | normalized centroid x/y |
| Depth | median, IQR, variance |
| Motion | vertical centroid velocity, area rate, aspect rate, median-depth rate |
| Quality | region confidence / valid-depth fraction |

These are geometric proxies, not body pose. They depend on camera angle, perspective and region quality.

## Temporal windows
Initial hypothesis: 2–3 seconds; tune on validation data. Sort by timestamp per track. Never cross video/session boundaries. Preserve timestamps or time deltas when frames drop. Define event-overlap labelling before training.

Fit scalers on training data only. Reuse them unchanged on validation, test and deployment.
