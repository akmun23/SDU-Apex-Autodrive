# v2.3.1 — Native-to-symbolic algebraic seed reconciliation and projected elastic checkpoints

**In-place update:** extracted contents overwrite the same `SDU_Apex_IPOPT_PhysX_MinTime_v2/` folder. `.venv`, native executable, recovered historical solver, and old result directories are not included or replaced.

## Evidence motivating the correction

The user's real v2.3.0 native warm start completed an approximately 12-second flying lap with wall clearance. It supplied wheel and body states from a genuine continuous PhysX rollout, but Phase-I IPOPT could not reduce the original-symbolic-model defect from 1.064776 after 800 iterations. Most errors were in lateral force balance. The native warm-start was obtained at continuously positive throttle, whereas `--brake-zones 3` put 12 discrete IPOPT nodes into a *different*, full-CAWB-braking mode, with nonzero native wheel speeds. That is not a mode-consistent initial trajectory.

## Changes

1. Add `seed_projection.py`: initialize the symbolic OCP's **normal-load and acceleration algebraic guesses from the same `suspension_loads_numpy()` and `tire_forces_casadi()` routines** used by the restored historical source. Retain the measured native position, heading, body velocity, yaw rate, steering, suspension and drive-mode wheel RPM. Log how far the source tire+spring force predictions differ from measured loads and finite-difference accelerations in `warmstart_physx/seed_projection_audit.json`. This does **not** reconcile wheel-speed updates or periodic spatial collocation and is not a Unity/PhysX physics-parity claim.
2. Explicitly identify artificial brake-mode discontinuities: the seed-generation tracking controller did not request full CAWB braking. At brake nodes the initializer uses locked-wheel RPM=0, not the inconsistent native drive RPM, while clearly warning that these remain unmeasured states. **Run `--brake-zones 0` first** to isolate physical-model compatibility with the actual captured driving mode. A later optimization with three brake phases remains necessary for an aggressive minimum-time solution.
3. Replace over-strict elastic callback acceptance with a **projected elastic candidate**: recompute temporary numerical slacks from each trial's original physical dynamics before comparing their merit, while enforcing every other track, steering, throttle, suspension-travel and safety constraint. The final IPOPT gate **still requires exact zero elastic slack and strict original dynamics**. Failed or otherwise unsafe trajectories are never exported.
4. Add `--seed-file` to reuse an existing `native_warmstart.npz` rather than re-running the 24-second preflight simulation on every diagnostic. The file must have a matching parent run's `provenance.json`, verified map and centerline hashes, and exact state dimensions for the IPOPT mesh.
5. Verified in-place upgrade of unchanged v2.3.0/v2.2.9 working solver overlay. The restored Git history source remains byte-for-byte unmodified.

## Recommended diagnostic run using the user's already-completed native lap

```bash
python3 run_ipopt.py \
  --repo-root .. --mesh-spacing 0.20 \
  --seed-backend native \
  --seed-file ./optimized_practice_v2_3_0/warmstart_physx/native_warmstart.npz \
  --brake-zones 0 --phase1-hessian limited-memory \
  --feasibility-iterations 500 --iterations 1200 \
  --physx-binary ./native_physx_vehicle_step \
  --output ./optimized_practice_v2_3_1_drive_bootstrap \
  --allow-history-fetch
```

## Remaining work

An **actual model-consistent complete IPOPT minimum-time solution** has not been achieved. Even if the algebraic seed is reconciled exactly, the symbolic model's tire/contact laws, wheel integration, body dynamics, and spatial collocation can disagree with the captured PhysX rollout. If v2.3.1 cannot reduce the residual on drive-only native mode, the next engineering task is to compare the symbolic `Fx`, `Fy`, wheel angular-momentum, spring loads, and yaw moment against **recorded native 1-ms transitions**, then correct the model equations rather than making more numerical solver changes. User's stated target is the other teams' <4.9 s Porto record, not a currently demonstrated result.

**Verification boundary:** unit and overlay-compatibility tests only; no new full PhysX, IPOPT, or Unity run was launched when packaging v2.3.1.
