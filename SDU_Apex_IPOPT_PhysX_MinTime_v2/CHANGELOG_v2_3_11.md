# v2.3.11 — Direct native tire-contact parity audit

This release does not alter the IPOPT OCP, the physical equations, the wheel
configuration, the final zero-dynamics-defect requirement, or the native vehicle
integrator. It adds recording and inspection of existing output fields from
`physx_native_vehicle_step` during the original, wall-checked 2.4 m/s physical
warm-start controller. The full flying lap is required; failed laps do not
produce a successful force-audit report.

New output: `native_contact_v2311/summary.json`, `per_wheel.csv`, and
`trial.json`, plus `native_contact_v2311.log`.

For each 25 ms controller observation from the 1 ms native PhysX update,
record raw longitudinal/lateral slip and true callback tire longitudinal/lateral
forces at each wheel. Compare the callback against two source-derived models:
(1) native Unity C++ `curveResponse` followed by combined-slip scaling, then
curve stiffness, and (2) historical CasADi stiffness applied BEFORE the
combined-slip utilization computation. The native source sets friction pair
coefficient to 1.0; the script verifies this before running.

Both variants use the existing IPOPT source-derived tire load filter applied
to the emitted suspension load, which is not necessarily the same as the tire
load supplied to the native callback. We therefore also infer the effective
callback tire load from measured forces where sufficiently informative; the
comparison does not treat the tire load model as exact or fit tire parameters.

No IPOPT or native racing candidate is executed or exported. An incomplete
native warm-start lap is reported as a failure. The previous successful
`native_warmstart.npz` and native PhysX binary are preserved.

Note: Prior reported v2.3.10 correction improved the best strict dynamics
defect only from 1.064776 to 1.038374 after 40 IPOPT iterations, not to a
feasible lap. See the user's v2.3.10 report for measured results.
