# Exact Unity open-plane combined-slip capture

- Build: disposable `exact_open_combined_v3_20260916`
- Unity: `2022.3.52f1` Linux player
- Run mode: `-batchmode`; no `-nographics`
- Experiment: `combined_slip_matrix_v1`
- Duration: `70.027 s`
- Rows: `70,028` at the Unity `0.001 s` fixed step
- Maximum body speed: `15.325 m/s`
- Commands: diagnostic self-commanded normalized throttle/steering schedule;
  no bridge or dev container was required
- Physics: unchanged; the trace component is read-only and writes observations
  only

The capture is open-plane data for identifying the Unity WheelCollider contact,
slip, and load mechanisms. It is not a runtime or production-MPC parameter
source until a causal model passes blind validation.
