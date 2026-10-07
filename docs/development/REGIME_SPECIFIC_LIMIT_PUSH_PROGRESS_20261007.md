# Regime-specific limit-push progress — 2026-10-07

This continues `SDU_APEX_REGIME_SPECIFIC_LIMIT_PUSH_HANDOFF_2026-10-07.md`.
The handoff's central rule remains: small mechanism-specific candidates,
explicit support, whole-run validation, exact runtime parity, and one isolated
track change at a time. No broad open-plane sweep was run or is planned.

## Current machine and protected state

- At the latest check, no simulator/controller/recorder/watcher process or
  running container was present. The Docker daemon itself remains running.
- The two trusted P0 parent sessions remain unchanged: 20 scored laps, best
  5.7378 s, pooled mean 5.7904 s, zero collisions. Their trajectory hash is
  `a3c9cda87f5f2d809cd1c684f98e2af6d1fa43a7530c1af2d8fa38290bef4910`.
- The P0 trajectory and simulator physics were not edited. The competition
  MPC now enables Y1 only inside its measured speed/steering/hold gates.
- Three short practice-track runs were made: a P0 screen, then matched
  high-steer control and Y1 treatment runs. No open-plane run was made.
- Test/final-test model partitions remain unopened.

## WP-R0 / Y1: fitted specialist, validation, and runtime candidate

The Y1 fit in
`live_runs/racing_model_diagnostics_20261007/y1_authority_loss_fit/` uses only
three training captures: `y1_targeted_retry`, `race_train_r02`, and
`race_train_r03`. It models a signed equilibrium yaw response at 2.5, 3.0,
and 3.5 m/s, interpolated only between those speed anchors, with the explicit
first-order transition

```text
r_next = r_eq(u, physical_steering) + (r - r_eq) * exp(-dt / tau)
```

The fitted hold time constant is 0.129314 s. It is active only for near-steady
steering: full through 0.05 rad/s and faded to baseline at 0.15 rad/s. The
turn-in (52 samples) and unwind (27 samples) tau fits are sparse and remain
diagnostic; production keeps its legacy transient response for those phases.

Two complete independent validation captures support the local one-step
improvement:

| Whole-run capture | Supported samples | Legacy yaw-rate RMSE | Y1 candidate RMSE | Reduction |
|---|---:|---:|---:|---:|
| `race_validation_r04` | 610 | 1.8724 rad/s | 0.2111 rad/s | 88.7% |
| `race_validation_r05` | 619 | 1.8664 rad/s | 0.1730 rad/s | 90.7% |

Neither run had a wrong-direction prediction in supported samples. These are
one-step scores—not recursive/full-lap accuracy or proven MPC convergence.
The fitted surface is now wired into the default MPC configuration at
`f1tenth_mpc/config/mpc_competition.yaml`, with the measured support gates
below. The optimizer runner now accepts the partial vehicle-model overlay in
`config/racing/y1_authority_loss_optimizer_candidate.yaml`, merges only its
`vehicle_model_overrides`, and records the overlay hash in resolved-config
provenance. The tracked surface CSV is included in the repository build
context. The optimizer's `--self-test` with this overlay succeeded and
reported Y1 enabled at the 2.5/3.0/3.5 m/s anchors; its 4.606043 s circle
result is a solver smoke result, **not** a practice-lap prediction.

### Lower steering edge and step-response question

The response below 0.30 rad is **not** constant. Training-only measurements
near 4 m/s show about 1.76 rad/s at 0.15 rad, 2.24 rad/s at 0.20 rad, then
1.23 rad/s at 0.21 rad. The change is speed-dependent and non-monotone. The
held-out r04/r05 captures contain strong support at 0.15 and 0.30/0.42 rad but
little or none at 0.20–0.25 rad, so the lower edge is not identified well
enough to activate this fit there.

The candidate therefore preserves the production model through 0.29 rad,
uses a narrow 0.29–0.30 smooth continuity taper, and is full-weight only from
0.30–0.42 rad (fading out by 0.45 rad). This taper is a support guard, not a
claim that the response is a step or a plateau. A step-response-like temporal
function is appropriate locally as a first-order recurrence with a
state/phase-dependent equilibrium and time constant. The current data justify
only the hold-phase tau; they do not justify using one temporal constant or a
hard steering threshold across all regimes.

### Runtime/offline parity and practice evidence

The compiled production C model, Python optimizer model, and CasADi model were
checked on 324 probes; maximum C/Python yaw difference was `1.195e-6 rad/s`,
Python/CasADi difference was zero, and the hold/tau gate map difference was
`2.403e-6`. The analytic-vs-finite-difference Jacobian maximum was
`1.675e-2` across eight steering-gate probes. Baseline behavior was unchanged
through 0.29 rad.

Existing practice-bag teacher-forced replay has sparse activation and does not
show a broad practice gain. On the Y1-disabled high-steer control capture,
only three samples were materially active: active-sample yaw RMSE improved
from 1.100 to 0.845 rad/s, but whole-pre-impact RMSE worsened from 0.445 to
0.542 rad/s across 283 samples. The wide `|steering| >= 0.30` group also
worsened (0.987 to 1.306 rad/s across 37 samples); most of that group is
outside Y1's joint speed/rate support, so it is not a score of the active
specialist alone. This is a useful warning against claiming a general transfer
gain.

The P0 screen completed one scored lap at 5.8388 s with no collision, but Y1
was materially active on zero of 559 truth samples. It therefore says nothing
about Y1's effect on P0 lap time.

The matched high-steer live control and Y1-treatment runs used the same frozen
trajectory (`5770f155…`) and pinned practice simulator. Control reached its
first collision at 7.5017 s (collision count +1); treatment reached its first
collision at 7.5731 s (count +2 was already present when the watcher observed
the event). Neither completed a lap. The roughly 0.07 s difference is not a
demonstrated improvement, and the treatment did not make this line driveable.
The offline failure timeline labels both as `ODOM_WHEEL_BURST` before impact;
that is a P0-p99 threshold/event signature, **not proof of the crash's root
cause**. The artifacts are in
`live_runs/racing_candidate_failure_atlas_y1_20261007/`.

The Y1-disabled control bag is the valid same-state/action replay baseline.
The Y1-treatment bag's logged baseline already contains Y1, so its candidate
versus logged-baseline replay is not a valid A/B comparison. No further
simulator run is justified until the current high-steer failure signature is
addressed; repeating this pair would not change the conclusion.

## GT-only multispeed yaw-fit check

The fit pipeline is explicitly teacher-labelled by simulator truth, not by the
derived estimator: practice body state and yaw labels come from
`/autodrive/roboracer_1/odom`, while runtime estimator `/odom` is not a fitting
label or a candidate input. The open-plane loader reads simulator rigid-body
truth from consecutive packet IDs. In practice captures, I changed the target
pairing to the next source-ordered truth row at the fixed 25 ms model step;
receipt timestamps now only associate a current truth row with its command.
Comparing the former receipt-time interpolation against this pairing on the
two trusted P0 bags gives label differences of 0.0024 and 0.0026 rad/s RMSE
(p95 absolute about 0.005 rad/s). On the collision-censored burst capture the
difference is much larger: 0.097 rad/s RMSE, p95 0.202 rad/s. That confirms
why timing jitter must not define the simulator step.

I fitted one deliberately broad GT residual as a diagnostic using four whole
training runs (27,735 rows): P0 r01, multispeed r04, and the two 7.5 m/s
high-steer training runs. Six complete held-out runs (48,106 rows) included P0
r02, multispeed r05/r07, and three 7.5 m/s validation captures. The candidate
and reports are under
`live_runs/racing_model_diagnostics_20261007/yaw_gt_multispeed_jointfit_r01/`.
It is not integrated into MPC.

The aggregate one-step held-out yaw RMSE looked better (`3.025 -> 1.444`
rad/s), but concealed major transfer failures. P0 r02 worsened from `0.065`
to `0.765` rad/s. Recursive replay on both independent multispeed holdouts
reproduced the same failure: at 25 ms yaw RMSE improved about `2.343 -> 1.448`
rad/s, but by 100 ms it worsened `2.872 -> 4.566`, and by 750 ms it worsened
`2.851 -> 7.35` rad/s. The 750 ms position and heading scores also worsened.
At 2.5 m/s and 0.30 rad, each held-out run's 750 ms yaw RMSE went from about
`0.02` to `17.57` rad/s. This candidate is rejected; a good one-step score is
not evidence of a usable recursive model.

The useful conclusion is not that GT fitting fails. It is that this one global
residual does not preserve the speed/steering-dependent transition behavior
when recursively rolled out. Keep the existing narrow Y1 and Y4 GT-derived
specialists as distinct comparators; do not enable this broad candidate. The
available matched-angle multispeed captures cover 2.5, 4.5, and 6.5 m/s, with
separate high-steer captures around 7.5 m/s. They still do not establish a
complete 0–12 m/s yaw model. No simulator was started for this analysis.

## WP-R1: odometry packet/source-order question

The P0 bags
`practice_r0_current_safe_r01_20261005` and
`practice_r0_current_safe_r02_20261005` were read offline in bag receipt order
for left encoder, right encoder, and IMU source stamps.

- r01: 3,092 complete packets; no within-topic reversal or duplicate; no
  earlier incomplete packet was passed while emitting later complete packets.
- r02: 3,060 complete packets; no within-topic reversal or duplicate. It has
  one right-encoder-only startup packet whose source stamp predates the first
  left-encoder and IMU samples. All 3,060 later complete packets are
  source-monotone; the partial packet never completes late.

Thus a strict-drop or bounded-reorder variant would not change the emitted
integration sequence on these clean P0 captures, except to discard that inert
startup fragment. The observed packet-order bug is **not reproduced on this
track**. Per the handoff stop rule, close the P0 source-order branch; no odom
code change or live odometry A/B is justified by current evidence. This does
not claim other tracks or transports cannot reorder packets.

## Throttle specialist status

The event-specific throttle action gate exists and is covered by focused
production-actuator behavior checks. It requires the measured speed, steering,
freshness, and `+0.08` requested throttle-increment support; it latches the
event rather than reclassifying every sample. The exact transfer is **not yet
accepted**: the saved `practice_throttle_slew_6p5_lowsteer_fast_ramp_r01`
capture recorded only 5/10 scored laps and two collisions, so it is a rejected
run/candidate result, not a successful L1 transfer. Preserve it as failure
diagnostic evidence. A run with an unknown overlay/config hash cannot be
counted as a valid A/B.

No additional L1 open-plane capture is justified. Before considering any
repeat, resolve the saved run's exact overlay provenance and identify whether
the collision was caused by the action policy or another factor. Any future
track test must be isolated, recorder/watcher-first, and stop at the first
collision.

## Remaining handoff work

1. Finish a source-attributed L1 decision from existing bag/replay evidence;
   reject it if the two-collision run used the event policy and the failure is
   attributable to it. Do not collect more L1 data.
2. Fit a compact Y3 steering-reversal/unwind specialist from existing
   train/validation captures, score complete windows at 25/100/250/400/750 ms,
   and retain it only if whole-capture results support it. Do not combine Y1
   and Y3 changes in one run.
3. Build P0 sector/regime occupancy from the two trusted 9g parent bags and
   their frozen geometry/speed references.
4. Generate a small frozen-geometry speed-schedule candidate, starting with
   straight/exit behavior. Report measured per-sector delta time, exit speed,
   wall margin, CTE, MPC rejects, and wheel mismatch. Treat schedule time as a
   proxy only.
5. Run the established batch simulator only after candidate/hash/static checks
   and recorder/watcher readiness. Abort immediately on one collision. Begin
   with the handoff's short screen; only a promising collision-free candidate
   receives the longer confirmation protocol.
6. Run constrained low-dimensional BO only after manual schedule perturbations
   establish usable, repeatable response. Keep P0 geometry frozen until the
   specialist and schedule gates pass.

No additional open-plane or broad test sweep is warranted at this point.
