# Exact Unity open-plane drive excitation — 2026-09-16

This is a diagnostic-only capture from the disposable open-plane
identification player built from the competition vehicle configuration. The
track was disabled only in the temporary build scene; Unity physics, vehicle
parameters, and production code were not changed.

The player ran in batchmode with `-batchmode` and without `-nographics`. The
trace was split only to keep every Git file below 100 MB. Read the parts in
numeric order; each has the same 142-column header and no rows were dropped:

- `wheel_contact_trace_part01.csv` — rows 1–40,000
- `wheel_contact_trace_part02.csv` — rows 40,001–80,000
- `wheel_contact_trace_part03.csv` — rows 80,001–111,266

The complete sequence reached 15.36 m/s. It contains the Unity runtime
wheel pose, RPM, motor/brake torque, forward/sideways slip, `WheelHit.force`,
contact vectors, body state, and applied command at the 1-kHz fixed-step
sampling rate. The paired static exact-scene capture is the settled pose
reference for suspension analysis.

Use `tools/model_id/analyze_unity_suspension_trace.py` with all three CSV
parts in order. This data is offline evidence only; no fitted coefficient from
it is runtime-approved.
