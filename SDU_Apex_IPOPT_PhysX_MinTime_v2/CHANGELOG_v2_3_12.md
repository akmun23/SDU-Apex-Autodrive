# v2.3.12 — Source-matched PhysX callback load and combined slip

**Observed user native data (v2.3.11):** 481 safe 25-ms samples, four wheels each
(1,924 measured callback pairs), taken from a native 1-ms PhysX simulator
flying lap at 2.4 m/s. The native tire-force shader is replicated with a
combined Fx/Fy RMS error of **1.344e-7 N** and max error **8.84e-7 N** when
using the measured suspension load directly and applying wheel friction-curve
stiffness *after* combined-slip normalization. The previously modeled tire
load filter overpredicted the callback's effective load by about 1.49 N
in the 1,086 observations with recoverable load. Using the old load
calculation, native-shader-predicted force RMS error was **0.3479 N**;
using native-measured suspension force plus old stiffness ordering still
produced **0.03278 N** RMS error. All are direct observed comparisons; no
friction fitting or coefficient tuning performed.

**Physical model correction:** In the GENERATED working historical
`unity_wheel_model.py` only, `tire_forces_casadi` now passes positive per-wheel
suspension force directly as native shader `tireLoad`; it does not apply the
additional `_physx_filtered_tire_load_casadi` transformation. The shader's
combined-slip normalization uses un-stiffened curve responses before applying
per-direction curve stiffness, matching the source C++ `unityCurveTireForce`.
Historical restored files remain byte-identical. Existing compiled native
PhysX code and source remain unchanged. Existing IPOPT solver overlay, Phase-I
checkpoint acceptance, track/actuator constraints, and strict zero-defect gate
are unchanged. The physical equation *does* change; this is an evidence-based
correction, **not** purely a numerical adjustment.

The revised working wheel model is checked against exact original/known
patched content before any overwrite; unexpected edits fail closed. Original
and generated wheel model SHA256 checks are reported in the solver provenance.

**Reproduction and next measured test:**

```bash
bash diagnose_callback_v2312.sh
bash diagnose_phase1_v2312.sh
```

The first command reads existing `native_contact_v2311/per_wheel.csv`,
compiles the actual generated CasADi shader, verifies all recorded native Fx/Fy
components, checks callback source anchors, and writes
`native_contact_v2312/summary.json`. It uses NO IPOPT and NO simulator.
The second performs at most 40 IPOPT feasibility iterations with the saved
native warm start and no native replay; it outputs
`optimized_practice_v2_3_12_native_tire_debug/ipopt_report_pass_00.json`
and `ipopt_v2312_debug.log`. Its `--brake-zones 0` setting is diagnostic only;
minimum-time race attempts must allow zero-throttle CAWB braking.

**Limits:** Source-callback parity is established only when using measured
native wheel slips and loads; native contact velocity/slip reconstruction from
OCP body states remains approximate, as do full scene contacts, recursive
suspension, and collocation intervals. This fix does not demonstrate an IPOPT
feasible lap, a native-replayed racing lap, Unity equivalence, or sub-4.9s.
