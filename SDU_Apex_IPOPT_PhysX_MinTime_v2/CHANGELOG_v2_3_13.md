# v2.3.13 — Native slip reconstruction audit (no OCP equation changes)

The v2.3.12 run established exact native callback parity **conditional on native slips** but degraded IPOPT Phase-I vs v2.3.10: initial strict max dynamics defect 1.064776 -> 1.428774; best defect 1.038374 -> 1.333186. A source-equivalent tire callback is not sufficient to match native contact-state calculations.

`diagnose_slip_v2313.sh` performs a safe 2.4 m/s flying lap with the unchanged native PhysX backend; samples actual native per-wheel slip and force, wheel speed, body velocities and center steering. Evaluates the exact patched OCP CasADi tire/slip equations on the *same recorded* values, and exports `native_slip_v2313/summary.json`, `per_wheel.csv` and `trial.json`, plus `native_slip_v2313.log`.

The `native_seed.py` recorder adds a `wheel_omega_radps` field only when contact records are requested; ordinary saved warm starts are unchanged. Historical solver, generated OCP constraints, native PhysX binary and physical constants remain unchanged.

**Time alignment caveat:** PxVehicle wheel contact slip is calculated before the 1 ms scene solve. The emitted body/wheel speeds are sampled after that solve. Significant reconstructed-slip differences can include this staggering, particularly for wheel speed with 0.00076 s damping time. Do not interpret this as exact instantaneous model error without further native instrumentation.

**No IPOPT Phase-I or minimum-time solve occurs.** If the lap is not wall-safe and complete, the script raises and does not claim successful audit. `--brake-zones 3` remains needed for any later racing solve.
