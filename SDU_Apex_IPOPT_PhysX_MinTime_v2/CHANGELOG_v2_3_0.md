# v2.3.0 — PhysX-generated IPOPT initial motion (single-folder in-place update)

## Why
The actual v2.2.9 practice run still returned the *identical* maximum
synthetic-seed dynamics defect (0.466543) after 800 IPOPT iterations, even
with 4,999 recognized variable bounds and a 3-braking-phase topology.
Further increasing IPOPT iterations would not address the lack of a physically
simulated initial trajectory.

## Changes
- **Default native PhysX motion seed (`--seed-backend native`)**: drive a
  conservative track follower through a warm-up and one flying lap using the
  actual repository's persistent 25 x 1 ms native PhysX vehicle engine.
  Reject incomplete, reverse-progress, invalid gate, or wall-violating laps.
  Independently try 2.4, 1.8 and 1.3 m/s speed targets. This is NOT a winning
  trajectory and is NOT substituted for IPOPT steering/throttle controls.
- Capture every returned native `unity_source_state` (18 state values),
  4 normal loads, actual path offset, heading error, wheel spins, physical
  steering and vehicle lateral/yaw motion. Resample only the completed flying
  lap onto the IPOPT mesh; initialize motor torque, steering-rate and force
  variables, while retaining captured suspension states and wheel speeds.
- Unwrap the track's periodic heading error properly: a completed lap winds
  the yaw angle by 2*pi. Do not interpolate unwrapped yaw as a periodic scalar.
  Differentiate in station with the measured ds/dt, not discontinuous lap time.
- Guard swept map/footprint wall clearance between the native 25ms frames,
  and validate gate location and direction before marking a warm-start lap
  complete. Report each failed trial to `warmstart_physx/native_seed_report.json`
  and `seed_trace_*.csv` rather than running IPOPT on an unsuitable substitute.
- Preserve the historical original `optimizer_unity_wheels.py` and the v2.2.9
  elastic dynamics IPOPT feasibility/time optimization unchanged. Add
  `--seed-backend synthetic` for exact old seed behavior and
  `--seed-backend source` for the checked-in source-model (non-native) motion
  seed. `--setup-check` remains simulator-free.
- No separate optimizer directory or environment. ZIP contents are rooted in
  the SAME `SDU_Apex_IPOPT_PhysX_MinTime_v2/` package. Does not overwrite
  `.venv`, `native_physx_vehicle_step`, `_restored_git_history`, or results.

## Run

```bash
cd ~/Documents/GitHub/SDU-Apex-Autodrive
unzip -o ~/Downloads/SDU_Apex_IPOPT_PhysX_MinTime_v2_3_0_IN_PLACE_UPDATE.zip -d .
cd SDU_Apex_IPOPT_PhysX_MinTime_v2
source .venv/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q tests/
python3 run_ipopt.py --repo-root .. --mesh-spacing .20 --brake-zones 3 \
  --seed-backend native --physx-binary ./native_physx_vehicle_step \
  --feasibility-iterations 800 --iterations 1200 \
  --output ./optimized_practice_v2_3_0 --allow-history-fetch
```

## Validation boundary
Unit tests validate code paths and mathematical resampling, NOT whether the
conservative follower can complete this particular practice map. An actual
native PhysX or Unity run has **not** been executed in the packaging
sandbox. No fully feasible IPOPT solution or sub-4.9s lap has been
established. If native motion seeding fails, run stops before IPOPT and emits
an informative report for diagnosing tracking or actual wheel dynamics.
