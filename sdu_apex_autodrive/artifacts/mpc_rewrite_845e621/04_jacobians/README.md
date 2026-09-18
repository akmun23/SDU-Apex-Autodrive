# Phase 5 — branch-aware stage Jacobians

`mpc_model_linearize` calculates A/B/d from the exact seven-state stage map.
It uses central differences in smooth regions, wrapped heading-row
differences, and one-sided derivatives toward the nominal branch when a
perturbation crosses a saturation branch. The affected columns and count are
reported to the caller.

The heading perturbation is `1e-3 rad`, rather than the handoff's suggested
starting `1e-5 rad`: float-valued stage outputs near ±π quantized the smaller
step enough to produce a 2.5% derivative error. The larger step reduced the
observed wrap-boundary error below `0.02%` and kept all straight/corner/braking
directional derivative checks below `0.04%`. These are numerical unit-test
results, not closed-loop prediction accuracy.
