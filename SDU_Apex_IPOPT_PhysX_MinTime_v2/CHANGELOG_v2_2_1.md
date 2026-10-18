# v2.2.1 — In-place source-parameter compatibility hotfix

- Fix `Cannot resolve kMaxSteeringRad from checked-in unity_source_plant.cpp` when the source-derived plant omits a steering symbol or declares a computed constexpr expression.
- Resolve present C++ numeric literals, `f` suffixes, and simple constexpr arithmetic safely; fail when an explicitly declared expression cannot be resolved.
- Derive missing steering rate/limit from current native `f1tenth_localization/src/physx_odom_runtime.cpp`, **not** an arbitrary hard-coded value.
- If both checked-in sources define the steering values, cross-check and reject inconsistent values rather than silently adopting either.
- Record steering parameter provenance and hash the native actuator source together with plant and reset configuration.
- Does not change IPOPT dynamics/collocation, feasibility constraints, PhysX, or any previous output directory.
- Unit tests cover missing constants, native fallback, mismatches and constexpr syntax. No claim of native PhysX or Unity lap validation.

Install over the existing `SDU_Apex_IPOPT_PhysX_MinTime_v2` directory with `unzip -o ... -d .` from the repository root. Existing `.venv`, native binary, restored solver, and output directories are not contained in this ZIP and remain intact.
