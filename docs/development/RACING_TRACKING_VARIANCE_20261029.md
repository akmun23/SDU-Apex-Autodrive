# Closing execution loss and lap variance — 2026-10-29

## Objective and protected baseline

Reduce both mean lap time and the fast/slow-lap spread on the current 4.800 s
source-feasible path. Do not confuse a faster predicted reference with better
execution. Preserve the verified 50.293/50.287 s profiles and their original
v1 reference; warmup is excluded. All simulator tests are supervised and stop
on collision or controller fault. Truth is offline evaluation only.

Verified combined trials: **49.588 / 49.467 / 49.614 s for ten scored laps**,
warmup excluded, no collision/controller fault in any trial. Third trial uses
promoted production AMCL defaults, WITHOUT an AMCL overlay. Two isolated cycle
overruns across the three runs used the validated prior plan; do not claim a
strict <20 ms maximum. The <49.8 s ten-lap goal passed in all three trials.

The unattended raceline campaign finished all 16 configured starts at a
`plateau`, retaining 4.800 s. It is NOT still running, and no faster valid line
was produced. Do not restart the identical cases or label this a physical floor.

## Established evidence

Latest baseline mean/best/worst 5.028700/4.933800/5.131800 s, population standard
deviation 0.056300 s; 3/10 laps below 5 s. The preceding run has mean 5.029300 s,
spread 4.974800–5.115800 s and standard deviation 0.043087 s. A fast isolated lap
is not consistent sub-5 s racing.

Existing per-run `time_loss.json` uses controller-state kinematic accounting:
the latest 228.48 ms gap contains 197.33 ms outside-path travel, 15.47 ms forward
speed shortfall, 7.90 ms heading/lateral contribution and about 7.78 ms remaining
timing/boundary/clock terms. Progress 1–2 m accounts for 91.66 ms. This identifies
where to investigate, not whether localization or control causes the difference.

Exact-header-epoch localization comparison on BOTH baseline runs found:

| Run | Forward pose bias | Forward pose RMS | Lateral pose RMS |
| --- | ---: | ---: | ---: |
| path `_01` | -5.536 cm | 7.213 cm | 2.629 cm |
| path `_02` | -5.533 cm | 7.087 cm | 2.673 cm |

No timestamp matching offset was present. Thus the ~7.6 cm total AMCL position
RMS is mostly longitudinal lag/bias, NOT 7.6 cm sideways error. Do not introduce
an arbitrary pose offset or tune steering around an unproven lateral bias.
Native LiDAR acquisition timing vs assigned packet stamp is investigated below.
The existing AMCL already rejects repeated identical scans after lock.

## Implemented source-input chronology correction

The source witness provides physical rack AND held commanded target, but the
old steering-slew objective still compares input increments against PHYSICAL
rack increments. A correct source command could therefore be penalized for the
actuator's response. The command-history tracking term also compared the last
input against physical rack, rather than the actual previous planned target.

Implemented a v2 reference with one extra column: exact previous 25 ms steering
target, exported from the periodic saved command witness, including the lap
seam. This is offline reference DATA, not an optimizer/vehicle state. No new
physical calculation, fitted number or Unity live input is introduced.

- Physical rack cost stays matched to physical rack.
- Previous-input cost matches the previous planned input.
- Input-increment cost matches current minus previous PLANNED input.
- QP, exact nonlinear scoring and standalone stage scoring agree.
- Delay is NOT added inside the QP: the real two-packet timing is already
  handled by command-application-time prediction; `MPC_COMMAND_DELAY_STEPS=0`.
- v1/ordinary corridor behavior is unchanged, keeping the original baseline
  reproducible. New v2 files are explicitly marked; history is held, not
  interpolated across packet edges; malformed/mixed schemas fail closed.

On the exact 4.800 s plan, the old input/rack-slew discrepancy peaks at 0.05277
rad, generating 11.86 objective units summed over 192 source command intervals.
This is a diagnosed objective inconsistency, not proof of a lap-time gain.

Tests: MPC CTest 13/13, including v2 parser/history validation, exact zero
artificial slew cost for a planned input transition, and condensed/nonlinear
cost-direction parity for v2 in both turn directions. Export tests 11/11 pass.
The same saved witness independently re-certified with unchanged source defect
3.288e-7, periodic defect zero and full source-tick margin checks.

Controlled v2 trials (NOT promoted):

- `mpc_source_history4p800_20261029_01`: two scored laps 5.0528/5.0638 s,
  then a corridor controller fault; no collision.
- Fixed an additional chronology issue: future reference increments must use
  adjacent horizon targets, not repeatedly subtract a fixed 25 ms reference
  history when actual progress differs from reference speed.
- `mpc_source_history_horizon4p800_20261029_01`: six scored laps
  5.1308/5.1808/5.0808/5.0068/4.9558/5.0378 s, then a corridor fault.
  No collision; max gate time 25.32 ms. Still not a racing improvement.

Keep the v1 reference and protected baseline profile for localization trials.

## Localization cause investigation and runtime changes

Raw odometry on baseline `_02`, above 2 m/s: forward-speed bias -0.00372 m/s,
RMSE 0.01159 m/s; rear-lateral RMSE 0.00150 m/s. Gyro yaw-rate matches truth.
These errors do not by themselves explain the roughly 5.5 cm forward pose bias.
MPC already propagates source pose to command-application time: an inspected
cycle had 31 ms source age and propagated 50 ms / approximately 12.5 cm.
Do NOT add another forward offset to compensate that same known delay twice.

Exact-epoch scan corrections on BOTH protected baseline runs:

- Below 3 m/s, raw scan cluster reduces forward RMS from about 7.2–7.3 cm
  to 4.1–4.2 cm; useful in 87–90% of samples.
- Above 6 m/s, raw scan cluster increases forward RMS from about 7.5–7.7 cm
  to 9.1–9.5 cm; useful in only 25–28% of samples.
- Existing `max(slow_gain, fast_gain)` prevented lowering fast correction
  independently. Replaced with explicit speed-regime selection; defaults
  remain 0.03/0.03, so baseline behavior is preserved.
- Added a sealed AMCL overlay path through trial manifest, Docker entrypoint
  and launch, so experiments no longer require editing production defaults.
- Candidate 0.15 slow / 0.01 fast, threshold 3 m/s:
  `mpc_amcl_motion4p800_20261029_01` completed warmup + ten scored laps,
  50.235 s total, no collision/controller fault. Repeat `_02`: 50.023 s,
  mean 5.0023 s, standard deviation 0.0521 s, range 4.9388–5.1028 s;
  max MPC gate 19.7106 ms, no over-budget cycles or fault/collision. First trial
  overlapped a short offline scan analysis, so its timing is not an isolated
  scheduling benchmark; do not call it a deadline pass.

`tools/analyze_scan_timing.py` compares recorded scan endpoints against the map
at nearby truth epochs; no truth or estimated offset enters runtime. Both
baseline bags have their minimum aggregate residual at -17.5 ms. Fast-motion
endpoint RMS drops 4.99 -> 3.38 cm (`_01`) and 5.23 -> 3.56 cm (`_02`).
This supports older LiDAR acquisition, not proof of exact sensor age: map
resolution is 5 cm and extrinsic/map errors can also shift the optimum.
Unity `LIDAR.FixedUpdate` captures ranges periodically; `Socket.EmitTelemetry`
reads the held `CurrentRangeArray`. The ROS bridge stamps the whole packet at
receipt, with no acquisition timestamp in the permitted stock telemetry.
Thus scan acquisition and packet stamping are demonstrably different events.

Tests added: independent fast/slow gain selection, exact-epoch truth/path
joins, empty scoring rejection, scan endpoint map pixel-center/extrinsic
conventions. All targeted tests pass.

Motion-gain repeat `_02`: truth/path cross-track RMS 7.70 cm (baseline
`_02`: 10.65 cm); forward localization RMS 4.95 cm (baseline 7.09 cm), bias
-3.13 cm (baseline -5.53 cm), lateral RMS 2.63 cm (baseline 2.67 cm).
This is a measured localization/tracking improvement, not just a lap-time score.

Implemented explicit `scan_acquisition_delay_s` in the production AMCL path.
Subtract it from the original header ONCE, bracket old odometry, perform the
scan update and anchor map->odom at that SAME corrected epoch. Current pose
continues propagating normally; MPC prediction is not given a second forward
offset. Pending-scan retries retain the original message. Diagnostics append
original packet stamp and configured delay. Invalid calibration beyond odometry
history fails at startup. Default stayed zero during the initial validation;
promotion after the repeated combined trials is recorded below.

Age-only calibrated trial `mpc_amcl_scan_age4p800_20261029_02` (17.5 ms, old
0.03/0.03 gains): warmup then 4.9098/5.0788/4.8968 s; controller corridor
fault during the next lap, no collision. First three scored laps have forward
pose bias -1.35 cm/RMS 4.52 cm and truth/path RMS 6.51 cm. Whole moving capture
forward bias +0.64 cm, confirming the direction of the phase correction but
not reliable full-run racing. Do not promote age-only or report a completed
ten-lap result. Its `_01` predecessor never started driving due to a ROS Humble
Duration-constructor build error, fixed using an integer-nanosecond Time.

Next controlled trial combines the age correction with independently tested
motion gains: `mpc_amcl_age_motion4p800_20261029_01`. Same v1 path, MPC weights,
speed and safety margins. Completed warmup + ten scored laps, no collision or
controller fault:

| Metric | Protected baseline `_02` | Combined `_01` |
| --- | ---: | ---: |
| Ten scored laps | 50.287 s | 49.588 s |
| Mean lap | 5.0287 s | 4.9588 s |
| Lap population standard deviation | 56.30 ms | 42.94 ms |
| Best / worst lap | 4.9338 / 5.1318 s | 4.9158 / 5.0618 s |
| MPC gate maximum | 19.584 ms | 19.116 ms |
| Over-budget cycles | 0 | 0 |
| Truth/path cross-track RMS | 10.65 cm | 5.53 cm |
| Forward map-pose RMS | 7.09 cm | 3.19 cm |
| Forward map-pose bias | -5.53 cm | +0.25 cm |
| Lateral map-pose RMS | 2.67 cm | 2.67 cm |

Scored laps: 4.9388, 4.9428, 4.9708, 4.9718, 4.9248, 5.0038, 4.9248,
5.0618, 4.9158, 4.9328 s (8/10 below 5 s). Timing did not change by relaxing
the gate; all 2,230 recorded cycles stayed below 20 ms.

Controller-state time accounting now assigns 99.70 ms/lap to outside-path
travel vs baseline 197.33 ms; forward-speed loss rose 15.47 -> 58.58 ms.
This is still accounting, not an independent source-dynamics accuracy claim.
The shorter actual driven path, lower exact-epoch pose error and faster laps
are consistent with reducing localization phase error. Remaining speed and
path losses, plus two laps above 5 s, mean tracking is not perfect.

Repeat: `mpc_amcl_age_motion4p800_20261029_02`. No calibration or weight changes
between combined trials. Then promoted the same delay/gains into
`f1tenth_localization/config/gpu_amcl_cpp_params.yaml`, removed the two
superseded experimental overlay files (their sealed per-run copies remain),
and ran `mpc_amcl_prod4p800_20261029_01` WITHOUT an AMCL overlay. No new raceline,
speed multiplier, MPC weight change, tire fit or clearance relaxation was
needed for the measured improvement.

| Trial | Ten scored laps | Mean | Lap std | Best / worst | Max MPC gate | Deferred >20 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined `_01` | 49.588 s | 4.9588 s | 42.94 ms | 4.9158 / 5.0618 s | 19.116 ms | 0 |
| Combined `_02` | 49.467 s | 4.9467 s | 53.58 ms | 4.8758 / 5.0358 s | 20.714 ms | 1 |
| Production defaults `_01` | 49.614 s | 4.9614 s | 44.53 ms | 4.8888 / 5.0518 s | 21.426 ms | 1 |

Repeat `_02` forward pose RMS 3.04 cm/bias -0.18 cm, lateral RMS 2.70 cm,
truth/path RMS 5.53 cm. Production-default trial: forward pose RMS 2.92 cm/bias
+0.52 cm, lateral RMS 2.73 cm, truth/path RMS 6.17 cm. Each trial has 8/10
scored laps below 5 s. All 30 scored laps average 4.9556 s. Variance is modestly
improved against the latest baseline, NOT near zero, and not better than every
earlier baseline trial. A consistent <4.9 s mean and a strict scheduling-time
bound remain unachieved. No claim of a new physics/rollout accuracy certificate.

This supports scan acquisition/packet timestamp mismatch as a material cause
of the backward bias. The 17.5 ms value is an OFFLINE CALIBRATION for this
stock practice image / 40 Hz bridge setup, not an acquisition timestamp exposed
by Unity and not guaranteed portable to another sensor cadence or transport.
Ground truth was used only to diagnose/validate, never subscribed by AMCL,
odometry or MPC. The correction updates measurement association, not body
physics or an arbitrary map-position offset.

Reproduce the promoted setup (choose a NEW run ID):

```bash
SDU_APEX_TRIAL_TOTAL_LAPS=11 tools/run_mpc_practice_trial.sh NEW_RUN_ID \
  f1tenth_mpc/config/mpc_source_path_tracking_candidate.yaml \
  f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800_witness_candidate.csv
```

The MPC path profile remains explicit, preserving other-map competition
weights/reference choices. AMCL corrections are now in production defaults.
The simulator/recorder/controller were stopped after the final completed trial;
the completed raceline campaign was not restarted unchanged.

Verification: host MPC CTest 13/13; localization CTest 8/8; timing/path/manifest
Python unit tests 20/20; bridge/workspace tests 8/8; FULL retained tool tests
151/151 with ROS sourced (do not overwrite ROS's PYTHONPATH); optimizer source-command
tests 11/11 in its maintained `.venv`; shell/Python syntax and static runtime
topic policy pass. The stock system Python lacks optimizer `networkx`, so use
the maintained optimizer environment, not an accidental dependency workaround.
No extra runtime topic was added. Organizer approval of the existing 40 Hz
bridge wrapper remains unresolved; a topic-policy pass is not rules approval.

## Next gates

1. DONE: diagnose scan-time / motion-gain effects, compare whole-run truth/path
   tracking, repeat the winner, promote and verify the production-default path.
2. Remaining: explain remaining 150–160 ms/lap reference-to-execution loss
   (first combined trial: about 100 ms outside-path and 59 ms forward speed),
   and reduce both slow-lap tails and rare scheduling overruns. Do not undo the
   acquisition-time correction to tune around the old phase bias.
3. Keep source-v2 input-cost experiments separate until they beat this new
   repeated baseline. Repeated sub-49.8 s is established; sub-4.9 s mean is not.
