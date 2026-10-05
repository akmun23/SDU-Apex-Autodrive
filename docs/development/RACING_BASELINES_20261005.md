# Racing baseline freeze and reproduction status — 2026-10-05

## Scope and order

This work follows `SDU_APEX_RESULTS_FIRST_RACING_PLAN_2026-10-05.md` in order.
The short practice circuit and the longer final circuit are separate test
targets. The sub-5-second objective applies to the short circuit. No vehicle
physics, runtime odometry, localization, MPC, or actuator code has been changed
in this work stage; current edits are confined to analysis/provenance tooling
and the debug recorder topic list.

## Frozen source state

- Repository revision at task start: `c85085634b0186fac0c1c0cd081b148fe25a7be2`.
- Worktree at task start: clean.
- Current runtime controller/localization/config files remain unchanged from
  that revision. The first worktree edits are post-run analysis and capture
  provenance helpers.
- Practice simulator image: `autodriveecosystem/autodrive_roboracer_sim` at
  digest `sha256:b4bbda41fdb1da7a2eadba4350ed3a0cb5020e7eb783454e78399ef5852dbd76`.
- Controller base image available in the project-selected rootless daemon:
  `sdu-apex-autodrive:dev-practice-odom-aggressive-20261005`, image ID
  `sha256:4261033717eafe110d9ce7b3fe32d2583828b2e1f5dd7ce071b12978c31289c0`.

The current practice baseline input selected for fresh reproduction is the
practice map plus the established speed-headroom trajectory. The final-track
inputs are the current default map and trajectory. Fresh run manifests record
hashes for each asset, MPC/odom/EKF/AMCL configuration, relevant runtime source,
and both container images.

| Frozen input | SHA-256 |
|---|---|
| Practice map YAML | `fcc4416fbab3c84aed6394e23d792f5533c914a777b743556b5f2b568807985d` |
| Practice map image | `0c39e17b3ae88ca70067c76c830c6de542af08c149f0616c06c9de227d79d016` |
| Practice baseline trajectory | `7cb54fee2c9c7a832592ed3d20372e84e31bfb725b89383902af688884e1258b` |
| Final map YAML | `e3ee2b6899d74fc3620832acdd108c26545f9b1d1d4af00b9178171c59f079a5` |
| Final map image | `b9f250c3bb1fc948616e5e14ee1671931cd70b94117958291c197589178801f2` |
| Final baseline trajectory | `86019b57460382e07022ea5914994dfe684d6e0958d72153e6051f0e8d362272` |
| MPC config | `06f639cb3ced1675a7c0e943808abc9c7ccab900912f41d30008d3ab7bdf423a` |
| Odom config | `99868087af2e2dd1f61a9babf784dcf123b89d47d4ef76bde789c476d91ead96` |
| EKF config | `31c8268f1ef5a6785a4f00a0ee03486c9d604ebb33e5570a2d8ff995792f93e4` |
| AMCL config | `afe718f8e25bab736c444a516e1f40186c68e247fdcf7979c6649b350339e550` |
| Vehicle planning profile | `254b532fbaf963dea30474cf7042009abeff9bcd25d9292a60967202d49291aa` |
| Path-tracking profile | `d5e20ee213d457f8d1471f51ccd953cd9d7ba5a947bf0d5ffe554a15f87d65a1` |

## Historical real-simulator evidence

Bag values below were independently re-read from their saved SQLite recordings
with `tools/analyze_practice_bag.py`. Historical artifacts do not contain a
complete configuration/image manifest, so they establish observed performance
but are not exact current-checkout reproductions.

| Track | Run | Scored laps | Best (s) | Mean (s) | Collisions | Status |
|---|---|---:|---:|---:|---:|---|
| Practice | `practice_speed_headroom_repeat_12lap_20260926_codex6` | 10/10 | 5.9638 | 6.0181 | 0 | Historical safe parent candidate |
| Practice | `practice_ntumethod_eval_12lap_20260926` | 10/10 | 6.0027 | 6.0212 | 0 | Independent historical safe run |
| Practice | `practice_speed_headroom_12lap_20260927_01` | 10/10 | 5.9688 | 6.0183 | 0 | Repeat historical safe run |
| Practice | `practice_speed105_12lap_20260927_01` | 1/10 | 5.8168 | n/a | 1 | Faster first scored lap, then collision |
| Final | `competition_diagnostics_20260924_1430` | 10/10 | 10.8087 | 10.9837 | 0 | Historical safe run |
| Final | `competition_odom_cal_20260924_1457` | 10/10 | 10.8007 | 11.0027 | 0 | Historical safe run |

The handoff's quoted final-track baseline of approximately 10.77 s best and
10.93 s mean was not found in these audited bags. The checked-in recorded
evidence supports approximately 10.80 s best and 10.98–11.00 s mean instead.
This discrepancy must be resolved by fresh reproduction, not assumed away.

## Recent short-track candidate evidence

| Run | Scored laps | Mean (s) | Collisions | Observation |
|---|---:|---:|---:|---|
| `practice_surface_hybrid_r02_20261005` | 2/10 | 6.0552 | 0 | Promising partial run; insufficient for promotion |
| `practice_legacy_yaw_r07_12lap_20261005` | 6/10 | 5.9346 | 1 | Faster partial run; collision before full validation |
| `practice_hybrid_yaw_r08_12lap_20261005` | 4/10 | 5.9266 | 1 | Collision-terminated; not a safe baseline |
| `practice_yaw_memory_r10_12lap_20261005` | 0/10 | n/a | 0 | No lap-count progress and 836 nonlinear-rollout rejections |
| `practice_lateral_velocity_r03_20261005` | 4/10 | 6.2790 | 0 | Partial, high variance including a 7.1627 s lap |

These results show that the recent high-demand branch has not yet produced a
repeatable safe improvement. They do not establish that the car cannot go
faster; the one-lap speed-profile probe did post 5.8168 s before colliding.

### Fresh final-track R0 reproduction attempt

The first current-checkout final-track run did not reproduce the historical
baseline and did not complete a lap. It had zero collisions, but the watchdog
stopped it after the car stopped making odometry progress. The bag records
1,081 LiDAR scans at 40.008 Hz (p50 interval 25.09 ms; p95 25.77 ms), so this
failure is not explained by a 20 Hz sensor stream.

| Run | Scored laps | Collisions | MPC optimal | MPC rejected nonlinear rollout | Odom short-run p95 | AMCL Frenet p95 |
|---|---:|---:|---:|---:|---:|---:|
| `final_r0_current_safe_r01_20261005` | 0 | 0 | 97 | 920 | 2.5 cm tangential / 0.8 cm normal | 2.61 m tangential / 11.4 cm normal |

The run ended after approximately 4.5 m of motion. The first saved nonlinear
failure is a stage-0 corridor failure: the predicted lateral state is 0.07646 m
against a 0.07078 m upper bound, exceeding the configured 5 mm tolerance by
about 0.7 mm. The controller then repeatedly published zero speed. This is a
measured failure signature, not yet a validated fix.

AMCL health reports show accepted local corrections with no rejected scans;
the repeated applied correction was about 3.3 cm per scan, predominantly in
the along-track direction. Over this short straight segment, AMCL accumulated
about 2.61 m of tangential error while odom remained within about 2.5 cm. That
makes scan-correction bias/observability a concrete localization hypothesis
to test. It does not yet establish whether changing the along-track gain alone
will recover the full safe lap.

For context, the historical complete final run `competition_diagnostics_20260924_1430`
has AMCL p95 error of 10.7 cm tangential and 3.4 cm normal, while its odom
drifts to multi-metre error over the full session. The fresh short capture has
excellent odom only over the initial 4.5 m; it cannot be extrapolated to a lap.
The fresh practice full run has AMCL p95 of 16.7 cm tangential / 4.4 cm normal
and odom p95 of 1.93 m tangential / 1.58 m normal over its ten scored laps.
These comparisons make longitudinal error the dominant observed localization
component, while also showing that the fresh final startup failure differs
from the historical full-run behavior.

## Ledger and next gate

`tools/racing/build_performance_ledger.py` scans direct practice/final run bags,
preserves failed/partial runs, and writes both the machine-readable
`live_runs/racing_performance_ledger.json` and a readable ledger. The first
historical scan found 75 of 76 bags scoreable, but none had complete
configuration and image provenance. Fresh runs carry `run_manifest.json`
sidecars.

### Fresh practice-track R0 reproduction

Two independent full runs used the frozen practice map and trajectory, the same
controller image, diagnostics enabled, and the official batchmode/Xvfb
simulator launcher. Each completed one warmup, ten timed laps, and one extra
lap with zero collisions. The controller reported only optimal MPC solves.
Lidar message-header cadence was about 39.96 Hz in both runs.

| Run | Ten timed laps (s) | Mean (s) | Collisions | MPC optimal | Header rate |
|---|---|---:|---:|---:|---:|
| `practice_r0_current_safe_r01_20261005` | 6.0647, 6.0717, 6.1047, 6.0417, 6.0747, 6.0957, 6.0737, 6.0837, 6.0757, 6.0257 | 6.0712 | 0 | 3005 | 39.964 Hz |
| `practice_r0_current_safe_r02_20261005` | 6.0557, 6.0967, 6.0667, 6.0947, 6.0417, 6.1077, 6.0087, 6.0527, 6.0827, 6.0207 | 6.0628 | 0 | 2967 | 39.953 Hz |

The two-run mean is 6.0670 s. Relative to the best older safe mean near 6.018 s,
the roughly 0.05 s difference is small; per user direction this is accepted as
the current reproducible optimization parent, not treated as a blocking
regression. It is not yet a sub-5-second result.

Practice R0 is complete. The automatic race report is now generated for both
fresh practice runs and includes per-lap/per-sector time, Frenet localization,
tracking, saturation, clearance, and one-step MPC prediction residuals. The
projection and Frenet tools also cover the failed current final run and a
historical complete final run. Control-time latency analysis is complete in
[`RACING_LATENCY_FINDINGS_20261005.md`](RACING_LATENCY_FINDINGS_20261005.md).

The current work is the handoff's R4 envelope stage (Tasks 8–10), using only
whole-run training for fitting, whole-run validation for scoring, and the two
fresh practice runs as an additional separate transfer check. No speed/line
change is promoted until the acceleration and sustained-curvature support is
audited by speed bin. The final-track startup failure remains a separate
localization/MPC diagnostic and is not being mixed into practice-track lap-time
claims.
