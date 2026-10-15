# Wheel-signal audit and odometry correction work — 2026-10-11

## Scope and gate

This work follows the requested sequence: resolve the encoder/body-speed
ambiguity offline first; evaluate an odometry-focused correction; then
diagnose MPC prediction before considering trajectory changes. The 2026-10-12
continuation changed only odometry observer parameters/source and its YAML;
MPC, trajectory, and simulator physics remain unchanged. No simulator was
launched.

The audit uses the fixed 25 ms sample grid and excludes non-contiguous packet
sequences and the explicitly invalid one- and three-packet actuator-response
events. It does not treat the encoder-derived speed as independent physical
wheel-speed ground truth.

## Step 1 — what the stored signals actually are

The two focused source captures are:

- [`lower feedback-gap capture`](../../live_runs/openplane_yaw_highsteer_wheel_feedbackgap_train_20261011_r01/)
- [`upper feedback-gap capture`](../../live_runs/openplane_yaw_highsteer_wheel_feedback_upper_train_20261011_r01/)

Each capture has its original ROS bag, fixed-grid source dataset, raw encoder
angle sidecar, and a sidecar manifest with source hashes. The manifests record
the bag hashes as:

| Capture | Bag SHA-256 | Valid adjacent-angle rates |
| --- | --- | ---: |
| Lower | `dca0b614b6d76a05d9d2d38cca77af73ca6cde373850c96a8543851d506de63d` | 2,033 / 2,071 frames |
| Upper | `8196e2623fb8f785b279e298af60422389ba9c313394e03216585026fece60a1` | 2,038 / 2,070 frames |

### Capture provenance and end-to-end source trace

Docker event history ties both captures to the pinned Explore simulator
`autodriveecosystem/autodrive_roboracer_sim@sha256:1a26bcc8b91fe845f2caafb95533bf3606092502430efca493709e167a5adb40`
and API image
`autodriveecosystem/autodrive_roboracer_api@sha256:ce081910948c3f30898322358d682b79cf165aa287a3dc27128dbacae99178c7`:

| Capture | Simulator start (UTC) | API capture container | API start (UTC) |
| --- | --- | --- | --- |
| Lower | 2026-10-11 17:35:35 | `sdu_apex_openplane_openplane_yaw_highsteer_wheel_feedbackgap_train_20261011_r01` | 2026-10-11 17:35:46 |
| Upper | 2026-10-11 17:47:49 | `sdu_apex_openplane_openplane_yaw_highsteer_wheel_feedback_upper_train_20261011_r01` | 2026-10-11 17:47:54 |

The simulator binary reports Unity `2022.3.52f1`; the downloaded public Unity
project is branch `AutoDRIVE-Simulator`, commit
`75b7df5216524a3c6b3150d698139190c6c44f8f` (2026-09-21), also Unity
`2022.3.52f1`. The image's build scene is `RoboRacer - Sim Racing -
Copy.unity`, whereas the available source branch has
`RoboRacer - Sim Racing - Explore.unity`. Engine-version agreement is not
source/build identity; source-only component behavior below remains
provisional for the pinned binary. The complete static source caveat is in
[`UNITY_SIMULATOR_ASSET_AUDIT.md`](UNITY_SIMULATOR_ASSET_AUDIT.md).

The public source-to-bag path is:

| Stage | Verified behavior | Limit |
| --- | --- | --- |
| Public `WheelEncoder.FixedUpdate` | Reads `WheelCollider.rpm`, accumulates revolutions using `Time.deltaTime`, quantizes to integer ticks using PPR × gear ratio, and derives cumulative angle from ticks. | `Wheel.rpm` and the encoder's internal speed fields are not serialized by the stock socket. The source behavior is not proven to be the pinned player's exact behavior. |
| Public F1TENTH prefab / `Socket` | Rear encoders use PPR 16 and gear ratio 120 (1,920 ticks/revolution). `V1 Encoder Ticks` and `V1 Encoder Angles` are formatted to four decimal places; no wheel-speed field is emitted. | The public Explore prefab is not the pinned Copy scene. |
| Pinned API image / ROS | Parses `V1 Encoder Angles` as two floats and publishes each unchanged as `JointState.position[0]`; `velocity` and `effort` are empty. | The ROS receipt timestamp is assigned by the bridge, not by the Unity encoder update. |
| Workspace 40 Hz bridge / offline dataset | The bridge's packet diagnostic parser converts the two angle tokens to floats without scaling. The derived circumferential-rate feature is `(Δangle × 0.059 m) / 0.025 s`; the trajectory timebase is fixed at 25 ms. | This is an angle-derived circumference rate, not an independent measurement of ground speed or contact slip. |

The public Explore scene binds `LeftWheelEncoder` and `RightWheelEncoder` to
the correspondingly named prefab components. In that prefab, the left
encoder is at local `x=-0.118 m` and references `RearLeftWheelCollider`; the
right encoder is at local `x=+0.118 m` and references
`RearRightWheelCollider`. Both use PPR 16 and gear ratio 120. This establishes
the named wheel mapping for the available prefab and scene, not necessarily
for the pinned Copy-scene binary.

The public `IMU.cs` source updates in `FixedUpdate`. It emits:

- orientation quaternion `[−q.z, q.x, −q.y, q.w]`;
- local Euler angles in radians, with the first component derived from Unity
  local Euler `z` and sign/wrap adjusted;
- body angular velocity `[−ω.z, ω.x, −ω.y]` in rad/s;
- body linear velocity `[v.z, −v.x, v.y]` in m/s;
- body linear acceleration from a finite difference of world Rigidbody
  velocity over `Time.deltaTime`, then inverse-transformed and axis-mapped to
  `[a.z, −a.x, a.y]` in m/s².

The prefab sets `GravityCompensation: 1`. In this implementation, `true`
selects `g=0`, so the acceleration calculation does not subtract gravity;
`false` selects `Physics.gravity` for subtraction. This is counterintuitive
from the field name and must not be silently interpreted as a gravity-free
accelerometer. The socket serializes these IMU arrays to four decimal places.
The exact pinned API image parses the orientation quaternion, angular-velocity
and acceleration arrays as floats and assigns their components to the
corresponding ROS `Imu` fields without scaling or axis changes. It sets
`frame_id="imu"` and all three covariance diagonals to `0.0025`. The workspace
40 Hz bridge's official-message wrapper likewise copies those fields; its
packet patch replaces the ROS stamp with that packet's receive stamp. This
confirms the API-side ROS mapping for these captures, but not whether the
public source's component settings match the pinned simulator binary.

This path was checked directly against both 2026-10-11 capture bags by
joining packet diagnostics and ROS messages on the bridge receive timestamp:

| Capture | Paired packets | Quaternion / gyro / acceleration component mismatches | Left/right encoder position mismatches |
| --- | ---: | ---: | ---: |
| Lower | 3,572 | 0 / 14,288; 0 / 10,716; 0 / 10,716 | 0 / 3,572 each |
| Upper | 3,670 | 0 / 14,680; 0 / 11,010; 0 / 11,010 | 0 / 3,670 each |

All encoder `JointState.velocity` arrays in these pairs are empty. Therefore
the API/workspace ROS conversion is not introducing an angle, IMU-axis, or
unit change in these captures. It does not establish the values that the
different pinned Unity build generated before serialization.

The bag packet diagnostics do not retain the socket's `V1 Encoder Ticks`
field, so raw integer tick counts cannot be checked sample-by-sample. The
serialized angle increments are nevertheless consistent with the public
1,920-tick/revolution quantum `2π/1920 = 0.00327249 rad`: across contiguous
packet-sequence pairs, the median distance to the nearest quantum multiple is
zero; p95 is 0.000061–0.000072 rad, and 99.5–99.9% of residuals are at most
0.0001 rad for the two rear encoders. This supports the documented tick
quantization plus four-decimal serialization, but is not a direct raw-tick
comparison and does not prove pinned-build source identity.

In the public source prefab, the `WheelCollider.radius` is 0.059 m, while
`VehicleController.WheelRadius` is 0.0325 m. The latter is used only in the
public `ExtendedDifferentialDrive` torque conversion; the former is used for
wheel geometry and the public encoder's RPM/tick speed calculations. This
resolves the source-code role of the two values: 0.0325 m is not an encoder
scale correction. It does not prove which drive branch the pinned player
uses.

The public prefab serializes `driveType: 5` (`SkidSteer`). In that source
branch the controller reports `CurrentSteeringAngle = angularInput * 100`
for skid steer, versus radians for its non-skid-steer branch. Both captures'
packet feedback values are in `[-0.5, 0.5]`, and the exact API forwards those
values unchanged to the ROS steering feedback topic. This mismatch means the
public source does not establish the pinned player's steering mode or
feedback units. It is recorded as a source/build semantic discrepancy, not
resolved by assuming the bag or public prefab is wrong. The public
`TimeManager.asset` says 1 ms fixed timestep, but that is likewise not
verified for the pinned binary; the 40 Hz ROS stream and fixed 25 ms offline
grid do not determine Unity's internal physics timestep.

The ROS `JointState.position[0]` values match the bridge packet's
`simulator_encoder_angles_left/right` values exactly in both bags. The ROS
`JointState.velocity` arrays are empty for all 7,242 encoder messages. Thus
the available raw source is cumulative encoder angle in radians—not raw
hardware counts and not an independent wheel angular-velocity measurement.
The dataset's `encoder_raw_surface_mps` is derived causally from adjacent
angles as

`(angle[k] - angle[k-1]) * 0.059 m/rad / 0.025 s`,

accepting source-stamp spacings of 15–35 ms only as an alignment/validity
check. The physical rate denominator remains the fixed 25 ms grid. Stored
`frames[:, 5:7]` are a distinct, approximately 100 ms trailing filtered view
derived from the same encoder-angle source; they are not independent wheel
truth. Reconstructing that trailing filter from the angles matches the stored
values with 0.0245/0.0237 m/s RMSE for the lower bag and approximately
0/0.0120 m/s for the upper bag (left/right).

The exact-two event classifier gives 16/16 eligible events in the lower
capture and 15/16 in the upper capture. The one upper event classified as
three-packet is excluded. All 31 retained events have contiguous packet IDs
and exact 25 ms sample spacing. Therefore this comparison does not use the
invalid one- or three-packet response cases.

### Packet-level trace

The following is the lower-capture positive-turn, 0.5 rad throttle-cut step
(`probe_yawfeedbackgap_r01_v6.80_a0.500_turn+1_step`). The command changes at
packet 846; the measured feedback changes two packets later, at packet 848.
Raw and filtered wheel values are left/right surface-speed estimates in m/s;
`u_GT` is simulator-truth chassis forward speed.

| Relative to command | Packet | Throttle command / feedback | Raw encoder-derived L/R | Filtered encoder L/R | `u_GT` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| −100 ms | 842 | 0.5 / 0.5 | 12.573 / 12.519 | 12.476 / 12.421 | 8.145 |
| −25 ms | 845 | 0.5 / 0.5 | 12.573 / 12.519 | 12.566 / 12.514 | 8.290 |
| 0 ms | 846 | 0.0 / 0.5 | 12.581 / 12.519 | 12.555 / 12.501 | 8.338 |
| +25 ms | 847 | 0.0 / 0.5 | 12.573 / 12.519 | 12.418 / 12.364 | 8.385 |
| +50 ms | 848 | 0.0 / 0.0 | 0.502 / 0.502 | 9.589 / 9.547 | 8.179 |
| +75 ms | 849 | 0.0 / 0.0 | 0.000 / 0.000 | 6.363 / 6.335 | 7.967 |
| +100 ms | 850 | 0.0 / 0.0 | 0.000 / 0.000 | 3.254 / 3.254 | 7.739 |
| +125 ms | 851 | 0.0 / 0.0 | 0.000 / 0.000 | 0.126 / 0.126 | 7.521 |
| +150 ms | 852 | 0.0 / 0.0 | 0.000 / 0.000 | 0.000 / 0.000 | 7.313 |

This is a large and repeatable **encoder-angle/chassis-motion decoupling**:
the angle-derived rate reaches zero while the chassis is still moving above
7 m/s. The filtered rate follows later, as expected from its trailing
window. The bridge/ROS alignment, packet cadence, and 100 ms filter
reconstruction checks do not support dropped packets, stale alignment, or a
wheel-radius conversion error as explanations for the abrupt event. The
existing low-slip wheel-scale diagnostic found only a few-percent effective
radius adjustment, far too small to explain a near-12.5-to-zero m/s change.

### Resolution and limit

The evidence rules out a simple ROS/bridge timing or fixed scale-conversion
artifact as the cause of this event. It establishes that both available
encoder-speed representations are derived from the simulator's cumulative
encoder-angle output, and that this output is not a reliable instantaneous
chassis-speed proxy through the throttle cut. The public Unity source says
that this angle is generated from WheelCollider RPM, but its source/build and
steering-mode discrepancy means that implementation detail cannot be
asserted for the pinned player. It does **not** establish that the physical
tire/wheel angular speed itself becomes zero: the simulator rigid-state
topic has chassis motion but no independent per-wheel angular-velocity truth,
while the bridge's `V1 Encoder Angles` field is the same source as the ROS
encoder position. Physical tire spin, spin-down/lock, and the pinned
simulator's encoder semantics therefore cannot be separated further using
these logs. Calling this “measured encoder/chassis decoupling” is supported;
calling the encoder rate physical wheel truth is not.

This closes Step 1 at the limit of available observations. No new capture is
needed to establish that the production estimator must not treat this raw
signal as ground-truth wheel speed. A future claim about physical wheel
rotation would require an independent per-wheel simulator truth channel; no
such channel exists in these bags.

## Step 2 — odometry correction

The actionable diagnosis is a measurement-semantics mismatch: cumulative
encoder angle/rate is an observed sensor feature, not a latent physical wheel
state. The candidate correction will therefore target odometry's body-motion
outputs from permitted causal sensors/actuator feedback, supervised by
simulator truth only offline. The yaw atlas remains frozen. Training data must
be training-split only, with exact 25 ms contiguous histories; validation may
select the correction, and the same held-out practice runs must be replayed
for the final paired comparison.

The acceptance report must give per-run errors for forward and lateral
velocity, yaw rate, raw encoder-derived rate, filtered encoder rate, and
integrated position/yaw. It must compare against the same baseline windows and
must not promote a candidate if an odometry output regresses. In particular,
an improvement to the prediction of the encoder measurement alone does not
count as an odometry improvement. MPC and raceline work remain gated on
completing this step.

### Same-held-out replay results

The current worktree odometry executable (diagnostic schema v7) was replayed
from the same two six-lap practice bags, with source sensor timing preserved.
The original bag's recorded estimator is schema v5. The replay process played
only left encoder, right encoder, and IMU into odometry; truth and command
topics were read offline by the scorer only. Both bags passed the existing
40 Hz, no-collision, and no-active-timing-fault gates.

| Held-out run | Scored samples | `u` RMSE v5 → v7 (m/s) | `v` RMSE v5 → v7 (m/s) | `r` RMSE | 0.775 s integrated position RMSE v5 → v7 (m) | endpoint RMSE v5 → v7 (m) | integrated yaw RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Practice r02 | 1,344 | 0.1014 → 0.0862 | 0.01038 → 0.01006 | both 4.35e-8 rad/s | 0.03490 → 0.02646 | 0.05313 → 0.04223 | both 0.00887 rad |
| Practice r03 | 1,312 | 0.1019 → 0.0801 | 0.01012 → 0.00982 | both 4.16e-8 rad/s | 0.03393 → 0.02302 | 0.05301 → 0.04044 | both 0.00862 rad |

Integrated position/yaw above are truth-reset *local 0.775 s motion windows*,
not full-run accumulated pose drift. The gyro yaw output is passed through;
the yaw atlas was not modified. Bias and 95th-percentile errors are retained
in each run's machine-readable score JSON.

An activation-probe revision of the same paired evaluator records direct
per-sample baseline/candidate output differences. On r02, it found 1,522
common aligned samples and the candidate changed 0; the scored set contained
1,344 samples, including 36 truth-defined high-yaw samples, with no changes
in either subset. On r03, it found 1,525 common aligned samples and changed
0; the scored set contained 1,312 samples, including 26 high-yaw samples,
again with no changes. Thus the identical practice scores are not a
metric-rounding artifact: the candidate made no published-state substitution
in either held-out practice capture.

The initial open-plane score included overlapping pre-phase context and was
provisional. It has been superseded by the corrected unique-sample replay
below; none of the provisional open-plane figures are used for decisions.

### Corrected paired replay — 2026-10-12

The evaluator now removes each phase's negative-time context, deduplicates
source packet stamps before making 32-sample windows, and saves aligned
baseline/candidate states. The corrected replay passed the fixed packet-grid
quality check on all three captures. This used the same source-built
odometry node and params, replayed at 4x, with only left/right encoder and
IMU topics supplied to the node. Ground truth remained offline scoring data.

| Capture | Scored samples | High-yaw samples (`|r_GT| ≥ 2.5`) | High-yaw `u` RMSE, baseline → candidate (m/s) | High-yaw `u` p95 abs, baseline → candidate (m/s) | All-sample `u` RMSE, baseline → candidate (m/s) | `v` RMSE, baseline = candidate (m/s) | 0.775 s local position RMSE, baseline → candidate (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Excitation train 1 | 32,256 | 3,842 | 17.144 → 12.066 | 41.586 → 27.142 | 16.233 → 15.591 | 0.957 | 7.038 → 6.769 |
| Excitation train 2 | 35,328 | 4,418 | 21.331 → 15.501 | 50.365 → 37.347 | 21.444 → 20.703 | 0.919 | 9.265 → 8.963 |
| Independent open-plane validation | 35,328 | 4,464 | 19.199 → 14.192 | 42.698 → 33.647 | 20.729 → 20.104 | 0.925 | 8.985 → 8.731 |

The high-yaw forward-speed RMSE improvement transfers to the independent
open-plane validation capture (about 26%); it is also present on both
training captures (about 30% and 27%). The improvement is real for that
extreme open-plane regime, but it does not make the estimate accurate: the
validation high-yaw `u` RMSE remains 14.19 m/s and p95 absolute error remains
33.65 m/s. The candidate changes only `u`; `v` and gyro `r` are identical by
construction. Local position RMSE over the 0.775 s windows improves only
8.985 to 8.731 m on validation. These errors are far too large for a useful
full-motion odometry estimate.

The new artifacts contain respectively 37,413, 40,971, and 40,947 unique
valid phase packets; 10,058, 11,011, and 11,004 repeated/context packets were
excluded, and the incomplete tails left 5,157, 5,643, and 5,619 valid packets
unscored. Paired source-state archives are saved beside each score JSON under
`live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/high_yaw_candidate_paired_source_replay_20261012/`.

The practice comparison remains the decisive transfer gate: the candidate
changed zero of 1,522 aligned output samples in r02 (including all 36
truth-defined high-yaw samples) and zero of 1,525 in r03 (including all 26
high-yaw samples). Thus the candidate improves a severe open-plane failure
mode but does not change the car's published estimate in either practice
run. It remains disabled and is **not promoted** for racing odometry.

The encoder/body mismatch remains large in these held-outs and is not
interpreted as true wheel-speed error:

| Held-out run | Fixed-25 ms raw angle-rate vs chassis `u` RMSE, L/R (m/s) | Existing 100 ms filtered angle-rate vs chassis `u` RMSE, L/R (m/s) |
| --- | ---: | ---: |
| Practice r02 | 2.207 / 2.205 | 1.981 / 1.981 |
| Practice r03 | 2.152 / 2.151 | 1.927 / 1.927 |

These signal/body residuals establish that neither angle-derived view can be
used as physical per-wheel ground truth. They are measurement-mismatch
diagnostics only. On the lower throttle-cut capture, v7 reports no wheel update
from the command through +150 ms; its instantaneous packet rate falls to about
0.5 m/s at +50 ms and zero thereafter while truth chassis `u` stays above
7.3 m/s. The burst/dropout logic therefore avoids the dangerous speed collapse.

There is, however, a regime caveat hidden by the overall improvement. On the
small `encoder mismatch >= 1 m/s` subset, v7 `u` RMSE changes 0.0676→0.0754
m/s on r02 and 0.0776→0.0716 m/s on r03. On the more focused high-mismatch,
high-throttle subset (18 samples on r02, 15 on r03), it changes 0.186→0.215
and 0.076→0.119 m/s, respectively. This is mixed/worse in exactly the region
under investigation, so the current v7 result is not yet accepted as the
final Step 2 correction despite improving full-run `u`, `v`, and local pose on
both runs.

The previously trained sensor-only plain GRU was also replay-scored on these
same held-outs and is rejected: `u` RMSE is 0.268/0.252 m/s and `v` RMSE is
0.0167/0.0174 m/s; its 0.775 s integrated position endpoint RMSE is
0.171/0.155 m versus 0.053/0.053 m for the recorded v5 odometry. It remains a
comparator only.

### Train-only residual candidates — rejected

All 58 clean training captures were replayed through the same current-source
odometry executable, using only left/right encoders and IMU as runtime inputs;
truth was read offline as a label. The source-stamp-deduplicated set contains
311,873 rows. Whole captures—not individual packets—were the cross-validation
groups. Three residual candidates were evaluated (static, 16-sample history,
and 16-sample history with causal actuator feedback/command features).

The latest 16-sample actuator-aware model selected ridge alpha 0.1. Its grouped
CV mean run RMSE fell from 0.723 to 0.334 m/s, but median run RMSE worsened
from 0.0929 to 0.1700 m/s and only 4/58 runs improved. On the sealed practice
captures, its forward-speed RMSE was 1.177 m/s (r02) and 1.206 m/s (r03),
versus 0.0862 and 0.0801 m/s for current-source replay. Its integrated endpoint
errors also grew to roughly 0.8 m. The static and history-only variants failed
similarly. **No learned residual has been integrated.** These fits are useful
negative evidence: the residual is strongly run/regime-dependent, and a
pooled correction is unsafe.

### Localized high-yaw / wheel-decoupling diagnosis

The two full-input excitation training captures contain the largest remaining
errors. In one contiguous packet trace, at approximately 8.8 m/s estimated
odom speed, the truth body state reaches `u=-1.92 m/s`, `v=+2.23 m/s`,
`r=-3.45 rad/s`, while both encoder-derived surface rates remain about
`9.20 m/s`; throttle feedback is 0.36 immediately before its command is cut.
This is a body-frame spin/sliding state with severe wheel/body decoupling, not
a timestamp or fixed-radius error. The encoders still are not independent
physical wheel-speed truth.

On the same fixed 25 ms grid, the rigid-body relation
`du/dt = ax + r^2*x_com + r*v` and its lateral counterpart
`dv/dt = ay - yaw_accel*x_com - r*u` explain the measured one-step
accelerations in the `|r| > 2.5 rad/s` subset: forward-acceleration RMSE is
0.43/0.53 m/s² and lateral-acceleration RMSE is 0.63/0.68 m/s² on the two
excitation runs. This supports a real model mechanism, but does not establish
that integrating the equations from the production observer's inaccurate
initial lateral state gives a stable estimate.

A conservative **offline-only** gate was evaluated: run a latent coupled
IMU integration only above `|r|=2.5 rad/s`, keep the existing odom output
unless mapped wheel speed exceeds the latent predicted `u` by more than
2 m/s, and leave published `v` unchanged. Across the 58 training captures this
changed only 2 runs (the two full-input excitation captures), improved their
whole-run `u` RMSE by 0.54 and 0.58 m/s, and left the other 56 run scores
unchanged. Pooled high-yaw `u` RMSE fell from 15.20 to 12.08 m/s, but remains
far too large to call the regime solved. On practice r02/r03, the gate made
zero replacements: all scored `u`, `v`, and local integrated-pose results were
identical to current-source replay. It has neither demonstrated a practice
gain nor regressed practice; it remains a research candidate, not a promoted
odometry change.

Turning on the existing full-turn IMU integration indiscriminately was also
rejected: it worsened lateral-velocity RMSE on high-yaw samples. The latent
`v` used by the spin gate must therefore remain separate from the published
`v` estimate unless a held-out test proves otherwise. No yaw-atlas parameter
or implementation was touched.

### Historical broad-candidate decision

The current-source v7 replay remains a useful comparator against the bag's
recorded v5 estimator: on both held-out practices it lowers full-run `u`/`v`
RMSE and 0.775 s integrated position error. However, in the 18/15-sample
high-wheel-mismatch/high-throttle subset it regresses `u` RMSE from
0.186 to 0.214 m/s on r02 and from 0.076 to 0.119 m/s on r03. The broad
v7/high-yaw candidates did not address that subset. Those candidates remain
rejected; the narrow gate below is a separate later correction.

The 2026-10-12 continuation identified the affected v7 decision path and
completed a training-selected, paired validation of a narrowly gated
high-speed correction. The previous broad pooled residual fits and the
high-yaw spin observer remain rejected/research-only. This gate is the only
new odometry parameter promoted by this audit; MPC diagnostics can now proceed
as a separate next stage.

#### High-speed yaw-bias lateral-demand gate — 2026-10-12

The localized practice regression is caused by applying the existing positive
high-speed yaw-rate correction in low-steering, high-throttle encoder/body
mismatch samples. In r02/r03, the 18/15 target samples have measured steering
at most 0.016/0.020 rad and median `|u_GT r_GT|` of 1.58/2.22 m/s^2. The
independent high-speed/high-steer validation captures instead have a median
proxy near 6.18 m/s^2 in their `speed >= 6 m/s, |steer| >= 0.30 rad` region.
Training full-input captures also contain substantial high-speed, low-steer
wheel mismatch, but their lateral-demand proxy is generally much higher
(median 7.48–8.10 m/s^2 in the throttle/mismatch subset). These values explain
why turning the correction off globally is unsafe and why the first candidate
gate (4–8 m/s^2) was too broad.

An opt-in smoothstep gate was added to the existing high-speed bias term,
using only causal mapped wheel speed and IMU yaw rate to form
`a_lat_proxy = speed * abs(yaw_rate)`. The 4–8 m/s^2 candidate was rejected:
it changed the hard-steering r01 capture's whole-run `u` RMSE from 0.4185 to
0.4828 m/s. The narrower 3–5 m/s^2 gate was then evaluated on the two
training-only full-input captures and selected before final validation:

| Training capture | All-sample `u` RMSE, ungated → gated (m/s) | High-yaw `u` RMSE (m/s) | Interpretation |
| --- | ---: | ---: | --- |
| Full-input train 1 | 16.23315 → 16.23378 | 17.14418 → 17.14509 | No meaningful change; extreme spin remains unsolved. |
| Full-input train 2 | 21.44382 → 21.44379 | 21.33123 → 21.33123 | No meaningful change; extreme spin remains unsolved. |

The high errors in these open-plane spin captures are not hidden by the gate;
it is deliberately a local correction, not a new high-yaw model. Paired
held-out practice replay with the 3–5 gate changes only 323/1,522 aligned
samples in r02 and 302/1,525 in r03, none of the truth `|r| >= 2.5 rad/s`
samples. Whole-run forward-speed RMSE changes 0.08621→0.08424 m/s (r02) and
0.08006→0.07956 m/s (r03); `v` and gyro `r` are unchanged to numerical
precision. The high-mismatch/high-throttle subset improves 0.2145→0.1999 m/s
(18 samples) and 0.1194→0.1017 m/s (15 samples), respectively.

The same fixed gate is effectively neutral on both held-out hard-turn
captures: `u` RMSE changes 0.418539→0.418413 m/s (r01) and
0.301921→0.301731 m/s (r02), with `v` unchanged to below 0.000001 m/s and
gyro yaw unchanged. The previous 4–8 gate and all intermediate candidate
artifacts are retained. On the independent full-input validation capture,
all-sample `u` RMSE is 20.728809→20.728786 m/s, high-yaw `u` RMSE is
19.198726→19.198715 m/s, and truth-reset local position RMSE is
8.984678→8.984667 m. These are practically neutral changes in a very poorly
estimated spin regime, not evidence that the gate solves it. The runtime YAML
now enables the fixed 3–5 gate after the config-driven r02 replay reproduced
the expected result (candidate `u` RMSE 0.084217 m/s; `v` and yaw unchanged).
Thus the gate improves the supported practice mismatch samples without a
material regression in the two training, two hard-turn validation, or
independent broad open-plane captures.

Source changes add the
`use_turn_speed_bias_lateral_accel_gate` option and threshold parameters. The
deployment YAML enables the validated 3–5 m/s² gate; the observer unit test
and targeted odometry package build pass. The replay evaluator accepts
repeated candidate parameter overrides. Artifacts for this work are under
`live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/turn_bias_lateral_gate_20261012/`.

The narrow 3–5 m/s² gate is the only odometry parameter promoted by this
audit. It uses only the existing wheel encoder and IMU inputs; simulator truth
remained an offline scoring label. The broad open-plane high-yaw error remains
large, so that regime is still unsolved and must not be represented as a
generally accurate observer. No EKF, AMCL, MPC, trajectory, or simulator
physics change was made.

Artifacts and evaluator:

- Paired held-out scores: `live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/production_replay_current_sqlite/practice_r02/practice_score_full.json` and the corresponding `practice_r03` file.
- Rejected train-only residual fits and replay caches are under `live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/` (`odom_u_residual_train_only_20261011`, `odom_u_history_residual_train_only_20261011`, and `odom_u_actuator_history_residual_train_only_20261011`).
- Reusable scoring: `tools/vehicle_dynamics_learning/score_practice_odom_replay.py`.
- Train-only replay/fit: `tools/vehicle_dynamics_learning/train_odom_u_residual.py`.
- Corrected paired replay: `tools/vehicle_dynamics_learning/evaluate_high_yaw_wheel_slip_odom.py`.

## Step 3 — MPC prediction with measured odometry initialization

The frozen 135-window, two-run practice benchmark was replayed with the
current production MPC C model and identical recorded controls in three modes:
simulator-truth `u/v/r` initialization, baseline-odometry initialization, and
the newly gated corrected-odometry initialization. Map pose and actuator
state were kept at recorded/truth values to isolate motion-state
initialization. The source-stamp join was validated as one-to-one and
monotonic: all matched samples were within 2.3 ms of their 25 ms dataset
sample, and no rollout failed.

At the 0.75 s horizon, macro-run RMSE (truth / baseline odom / corrected odom)
was `u` 0.1062 / 0.1062 / 0.1060 m/s, `v` 0.0170 / 0.0181 / 0.0181 m/s,
yaw rate 0.0582 / 0.0583 / 0.0579 rad/s, and lateral path error 0.0163 /
0.0178 / 0.0177 m. Thus the odometry initialization error is measurable but
small relative to the continuing model rollout error on this ordinary
practice domain; the new odometry gate does not materially change the MPC
forecast. This is not a high-speed/high-steering validation.

The stepwise report localizes the earliest differences. With truth
initialization, one-step forward-speed RMSE is 0.0374/0.0398 m/s on r02/r03
(equivalent one-step acceleration-error RMSE about 1.50/1.59 m/s²); by 250 ms
it is 0.1147/0.1147 m/s. The model holds `v` constant per stage; `v` RMSE
reaches about 0.032 m/s by 1 s. The yaw response remains the largest
rotational residual: one-step yaw-rate RMSE is about 0.031/0.033 rad/s and
reaches about 0.063/0.062 rad/s by 250 ms. These findings identify
longitudinal and lateral state transitions for targeted model work, not a
weight-tuning justification.

An exploratory affine longitudinal refit was fitted only to practice training
laps 1–8 and checked on training laps 9–12 and the sealed r02/r03 runs. It
lowered training acceleration RMSE but worsened one-step held-out speed
increment RMSE on both practice runs; it is rejected and not integrated. The
already-trained history-conditioned model is also not integrated: existing
recursive whole-run reports show its attractive one-step performance does
not remain accurate through rollout. No MPC equation or weight change is
currently justified by those candidates. The exact MPC build, scripts, and
stepwise metrics are under
`live_runs/derived_dynamics_learning_20260928/odom_wheel_signal_correction_20261011/mpc_prediction_analysis_20261012/`.

## Step 4 — recorded full-run command replay and raceline ranking

Not yet completed. Do not claim offline lap-time ranking or a new optimizer
candidate until full-run recorded-command replay has been compared per lap
and per corner. No reference trajectory or MPC weight has been changed, and
no simulator was launched. Continue using the existing bags before requesting
any new simulation data.
