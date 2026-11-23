
# Test plan

Unit tests: numeric depth validation; invalid-value handling; region bounding boxes; empty/small crop handling; stable IoU tracking; temporal window boundaries; model input shape; API schema validation.

Integration: short video -> depth -> regions -> tracks -> features; features -> windows -> model; prediction -> event -> API.

Research validity: no subject/session overlap; all event views grouped; scaler trained on train only; test not used for threshold selection; numeric depth used rather than colorized previews.

Stress: background change, crossing, partial occlusion, stationary person, intentional lying/sitting, stairs, reflective surfaces and low light.
