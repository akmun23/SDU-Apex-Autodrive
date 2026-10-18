# v2.3.14 — Native PhysX lateral slip basis sign

v2.3.13's new 481-frame/1,924-wheel-sample native capture shows an almost exact lateral slip **sign reversal** on both steered front wheels when using the historic optimizer's forward/left/up contact axes. The PhysX/Unity native lateral slip is right-positive in its wheel frame. The OCP's `_physx_applied_torque_slips_casadi` previously fed a left-positive lateral contact velocity to `atan` without sign conversion.

**Only the GENERATED working-copy wheel model** changes: `lateral_slip = -atan(lateral_speed / (abs(long_speed)+1))`. All denominators, slip thresholds, calibrated prefab constants, tire curve, raw native tire load and exact PhysX combined-slip callback remain unchanged. The historical original is byte-preserved; exact-cache checks allow upgrade from original or v2.3.12/13, and refuse unknown local changes.

Measured **in the uploaded v2.3.13 2.4 m/s safe native lap**, re-evaluating the historical forces using the corrected lateral-slip sign with measured native wheel load reduces Fy RMS difference from 3.7620 N to 0.5767 N, and slip RMS difference from 0.0038279 to 0.0005468 rad. Fx RMS remains approximately 0.03768 N. These are contact-algebra comparisons, **not** complete vehicle dynamics parity or a certified minimum-time lap. Native contact slip is sampled before scene integration; wheel speeds/body velocities are sampled 1ms later, and rear-wheel residual remains.

Run `bash diagnose_slip_v2314.sh` to evaluate the **actual generated CasADi function** on the saved v2.3.13 native sample CSV with a strict baseline-improvement audit, without rerunning PhysX or IPOPT. Next, `bash diagnose_phase1_v2314.sh` performs a capped 40-iteration Phase-I consistency test only (drive-only, no braking zones). It does not export an invalid raceline.

A fully feasible, closed, no-defect IPOPT racing lap with 3 explicit CAWB braking zones and subsequent native/Unity validation is **still required**; no sub-4.9s claim is made.
