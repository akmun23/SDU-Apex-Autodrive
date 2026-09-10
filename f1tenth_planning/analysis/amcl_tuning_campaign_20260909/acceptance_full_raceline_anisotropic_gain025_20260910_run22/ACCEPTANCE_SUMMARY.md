# AMCL anisotropic correction acceptance

Run: `20260910_run22`

- Controller: pure pursuit on the mintime raceline.
- Simulator: compete graphical `-batchmode`, camera-off level, IPv4 loopback bridge.
- Speed: full trajectory speed; measured GT peak `6.26 m/s`.
- Completed: approximately `11.77` raceline laps.
- Collisions: `0`.
- Source-time AMCL XY error: p50 `0.0338 m`, p95 `0.0977 m`, max `0.2255 m`.
- Source-time `/current_map_pose` XY error: p50 `0.0341 m`, p95 `0.1016 m`, max `0.2279 m`.
- Pure-pursuit cross-track error: p95 `0.0402 m`, max `0.0942 m`.

The tested change was `local_scan_correction_along_track_gain: 0.25` with
`local_scan_correction_cross_track_only: false`. It preserves continuous
AMCL correction and attenuates only the raceline-tangent component that was
measured pulling the estimate onto the wrong upper-hairpin branch.

Authoritative artifacts:

- `sensor_record_20260910_031141.csv`
- `source_time_diagnostic.csv`
- `source_time_diagnostic.json`
- `ground_truth_amcl.csv`
