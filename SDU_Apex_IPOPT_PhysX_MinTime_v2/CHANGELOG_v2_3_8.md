# v2.3.8 — Native-to-symbolic force consistency audit

**User data:** v2.3.7 Phase I (50 iterations) began at raw max defect
1.0647761866538858 and saved 1.0627929937238791 after one hard-feasible
backtrack, still infeasible. The initial residual breakdown includes
lateral force balance max=1.064776 and RMS=0.509705 m/s², scaled spring/damper
load max=0.658572, and scaled roll-rate collocation max=0.604459.

**New diagnostic:** `bash diagnose_force_v238.sh` compares the exact scalar
single-node wheel force expression used by the IPOPT OCP with the native-state
acceleration and wheel loads. It produces 143-node CSV of lateral/longitudinal
force imbalance, native and source-predicted loads, per-wheel forces, and
longitudinal/lateral slips. The earlier batched CasADi projection is evaluated
only as an independent cross-check, with discrepancy reported. The seed
projector now takes its algebraic force values from the scalar form, not the
unverified batched form. The solver cache/overlay, objective, constraints,
physical constants, and strict feasibility gate are unchanged.

**No hidden tuning:** The diagnostic does not adjust tire forces, friction,
normal load transfer or vehicle parameters to fit the native trace. No IPOPT
solve or native replay is run, and no racing candidate is exported.

**Installation:** Unzip over the repository root into the existing
`SDU_Apex_IPOPT_PhysX_MinTime_v2/`. Existing `.venv`, generated historical
working cache, compiled PhysX executable and prior results are preserved.

**Tests:** 90 package tests (86 previous + four scalar audit checks).
