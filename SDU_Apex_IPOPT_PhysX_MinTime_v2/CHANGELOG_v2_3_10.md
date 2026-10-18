# v2.3.10 — Correct source-model suspension lateral frame (native evidence)

The v2.3.9 per-wheel candidate audit supplied by the user established that
native PhysX has left-higher loads in the relevant corners, while the previous
symbolic suspension load model predicted right-higher loads. Lateral load
transfer correlations at both axles were approximately -0.9993 and 97.9% of
nodes had opposite signs. Reflecting the **Unity-x-right mount lateral arm**
into the model's **left-positive roll-coordinate frame** reduced per-wheel
load RMS error from 2.173852 to 0.991981 N (54.4%) and maximum load error
from 6.585722 to 3.187344 N. Correlation became +0.9993 at both axles.

## Source-justified correction

`source_vehicle.current_model` now converts the *lateral only* wheel-mount
coordinate from Unity x-right into forward/left/up model coordinates before
constructing the historical suspension model. The source model and historical
IPOPT OCP equations are otherwise untouched. There are no fitted coefficients.
The change affects the source spring/damper algebraic loads, tire-force load
inputs, and the ensuing OCP feasibility equations.

The current native body-state roll and roll rate, original wheel index order,
original PhysX tire law, lateral-force equations, hard wall/actuator limits,
full-lap periodicity, time objective, and zero-defect gate are retained.
`source_vehicle` refuses unexpected source wheel ordering rather than guessing.
The existing v2.3.9 comparison is preserved as a legacy-vs-corrected audit.

## Important limits

The same audit showed the corrected load distribution **worsened** native vs
symbolic lateral acceleration RMS from 0.410383 to 0.486322 m/s², while the
symbolic *total* load remained ~3.217 N below the native mean. Thus this fix
reconciles a definite coordinate mismatch but does **not** reconcile full
native PhysX scene integration, tire slip, body-contact impulses, or the
remaining acceleration mismatch. There is NO verified feasible minimum-time
IPOPT lap or sub-4.9 s native/Unity lap from this change.

## Usage in existing v2 directory

Run `bash diagnose_model_v2310.sh` to audit loads/forces (no IPOPT), then
`bash diagnose_phase1_v2310.sh` for 40 IPOPT feasibility iterations using the
unprojected safe native seed (max 210 seconds, no replay, still drive-only).
The script prints and saves `ipopt_v2310_debug.log` and the usual strict
`ipopt_report_pass_00.json`. Do not run a long 1,200-iteration solver unless
Phase-I progress is materially better and exact zero dynamics becomes viable.
