# Racing checkpoint and autonomous construction — 2026-10-29

> Superseded runtime result: acquisition-time / AMCL corrections completed
> **49.588 / 49.467 / 49.614 s** for ten scored laps, same 4.800 s path and MPC profile.
> See [RACING_TRACKING_VARIANCE_20261029.md](RACING_TRACKING_VARIANCE_20261029.md).
> The campaign below exhausted its 16 starts at `plateau`, retaining 4.800 s;
> it is NOT currently running. The rest of this file preserves the checkpoint.

## Earlier measured checkpoint

Warmup excluded in every row; real practice Unity simulator, production MPC,
permitted-sensor odometry and AMCL. No truth fed into autonomy.

| Target / profile | Ten scored laps | Mean lap | Max MPC gate |
| --- | ---: | ---: | ---: |
| Previous frozen v343 record | 51.030 s | 5.1030 s | 20.024 ms |
| 4.875 s source witness / frozen weights | 50.617 s | 5.0617 s | 19.545 ms |
| 4.800 s witness / frozen weights | 50.475 s | 5.0475 s | 21.202 ms |
| 4.800 s / path tracking profile, first | 50.293 s | 5.0293 s | 20.114 ms |
| Same path/profile, repeat | **50.287 s** | **5.0287 s** | **19.584 ms** |

All four new source-witness trials above completed warmup + ten scored laps
and the post-target collision watch without collision/controller fault. The
path-profile totals differ by 6 ms; individual lap population standard
deviations are 0.0431/0.0563 s, NOT zero variance. First path-profile run had
one >20 ms deferred cycle; repeat had none. Do not claim a universal timing
bound from these observations. The ten-lap goal remains unmet: 0.287 s to
sub-50, 0.487 s to 49.8 s, and 1.287 s to a 4.9 s mean.

Runs and complete histories:

- `live_runs/mpc_source_witness4p875_20261029_01/`
- `live_runs/mpc_source_witness4p800_20261029_02/`
- `live_runs/mpc_source_path4p800_20261029_01/`
- `live_runs/mpc_source_path4p800_20261029_02/`

Each has a sealed prelaunch manifest, original full parameters/overlay, bag,
supervisor log, run report JSON/CSV and decoded diagnostic JSONL. The `_4p800_01`
non-path run never launched (missing input path); do not count it as driving.

## Implemented changes and usable artifacts

The 4.800 s source witness is a NEW feasible result, not a fast infeasible
iterate. Fixing integer packet ownership restored exact IPOPT warm starts;
the previous resampling silently changed 15/21 packets in 195/192-packet
restarts. With four measured wall cuts the 4.800 s witness converged in 17
iterations / 21.14 s. Independent source replay checks the full perimeter at
all 4,800 physics ticks with 5 cm margin. Scaled defect 3.288e-7, periodic defect
zero, recursive full-lap XY mismatch 0.0132 mm, extra clearance 0.0358 mm.
Those tiny numbers compare our optimizer with OUR production plant, not Unity.

Saved, reproducible artifacts in the practice trajectory directory:

- `source_command_4p800_witness_candidate.csv`: 15-column schema-marked geometry,
  speed, source-state and held-input references. SHA-256
  `03433092578e36f2dc143b3272c6bce507bef10675c1a2b96989c7757653e03b`.
- `source_command_4p800_witness.npz`: initial/shooting state and command witness.
- `source_command_4p800_validation.json`: source certification, NOT a Unity
  certificate. Its original paths refer to the recovered run artifacts.

The verified path overlay is
`f1tenth_mpc/config/mpc_source_path_tracking_candidate.yaml`, applied on the full
`mpc_competition.yaml`. Compared with frozen weights: e_y 4000→6000, e_psi
1500→2000, steering slew 2000→1500; throttle/speed/slip costs unchanged. SQP
local radii remain 0.04 rad / 0.05 throttle. No odometry/tire equation changed.
It remains an explicit race profile, not a silent replacement of other-map
competition defaults or the frozen reference.

The change was motivated by measured time accounting: on the first 4.800 s
trial, 206.5 ms/lap is associated with wider path travel versus 26.6 ms with
forward-speed shortfall. This is controller-state kinematic accounting, not
independent truth-based causal proof. A prior joint fourfold speed-weight /
large damping reduction failed; its overlay was removed from active configs
and is retained only with that failed recording.

Production and offline MPC replay now share `mpc_trajectory_io.hpp`. Previously
the replay executable silently ignored the source-reference columns, testing
a different reference from the car. The duplicate production parser is gone;
malformed/mixed/unmarked source rows fail closed. This is startup I/O only,
not extra solver work. Python bag scoring also recognizes schema comments.

Verification: MPC CTest **13/13**, tool tests **147/147**, focused command
OCP/campaign/realizability tests **25/25**. No claim that all historical optional
optimizer tests pass; their previously documented failures remain separate.

## Automated construction: completed plateau

Service: `sdu-apex-raceline-campaign.service`.
Campaign: `live_runs/raceline_campaign_20261029/`.
Entry point: `command_campaign.py`, started by `tools/run_raceline_campaign.sh`.

This reproduces the loop that actually produced 4.800 s:

1. Re-certify the initial witness with the current compiled production plant.
2. Ask IPOPT to solve the production-command OCP for one fewer 25 ms interval.
   Optimize the periodic geometry, physical states and held steering/torque
   together. No pure-pursuit score, no learned physics or geometry multipliers.
3. Independently replay the complete lap from ONE start, inspecting every 1 ms
   source tick and the full vehicle perimeter plus 5 cm.
4. Feed measured clearance failures back as exact prefix constraints and restart
   from the unchanged saved primal. Never smooth a checked path afterward.
5. Re-certify a passing result independently before updating the campaign's
   best witness. Keep the prior valid result; never replace it with a faster
   infeasible iterate. Export schema-marked references only on passing gates.
6. Continue to the next shorter duration automatically. If necessary try eight
   cyclic packet phases and both uniform-resample / one-packet-gap initializers.
   They preserve the source equations; these are initial guesses, not new fits.
7. At a stalled attempt, save and move on. A restart is allowed when the best
   physical constraint residual has at least halved; otherwise do not recycle
   the same failed start. Stop honestly at `plateau` after all configured starts
   fail, or `target_reached` at 4.700 s. Neither proves a global lower bound.

All 16 available CPUs are permitted. No CPU quota, wall-time cap or memory cap.
BLAS remains single-threaded to avoid nested oversubscription; independent
source maps use 16 workers. Individual IPOPT calls cap at 300 iterations,
but a substantially improving saved primal may continue automatically. A
50-iteration lack of meaningful feasibility progress stops a call. Thousands
of iterations are NOT selected merely because compute is available.

Every improved sampled primal is atomically saved/fsynced, normally at most
five iterations apart. The campaign manifest is atomic/fsynced and a writer
lock prevents duplicate campaigns from corrupting it. Resume verifies seed,
map, model and settings fingerprints. Power loss can lose work since the last
checkpoint, not the certified best witness. Interrupted/incomplete output is
never overwritten. SIGTERM is checked outside CasADi's callback exception
conversion; otherwise the solver can swallow a stop request and continue.

The real solver stagnation smoke test stopped at iteration 25, retained its
best primal and reported `no_validated_witness`, not success. Service interruption
testing verified both forced-stop checkpoint recovery and subsequent graceful
signal handling. No simulator is running alongside the all-core search.

The service restarts after unexpected process failure, with a rate limit to
avoid endlessly retrying a broken setup. This is a transient user service,
NOT an enabled boot-time daemon. After reboot, start the same wrapper to resume.

```bash
tools/run_raceline_campaign.sh
```

Readable terminal progress:

```bash
watch -n 5 'SDU_Apex_IPOPT_PhysX_MinTime_v2/.venv/bin/python SDU_Apex_IPOPT_PhysX_MinTime_v2/command_campaign.py --status live_runs/raceline_campaign_20261029'
```

Detailed live solver log:

```bash
journalctl --user -fu sdu-apex-raceline-campaign.service
```

The campaign does NOT automatically overwrite production targets or claim
Unity validation. Its next faster source-feasible output needs the same
supervised MPC practice validation. It cannot guarantee an improving raceline
on every cycle or infer a physical floor from stagnation. The old spatial v354
run is not resumed: saved best 4.921756 s is already slower than 4.800 s.

## Reproduce the verified faster driving profile

```bash
SDU_APEX_TRIAL_TOTAL_LAPS=11 tools/run_mpc_practice_trial.sh UNIQUE_RUN_ID \
  f1tenth_mpc/config/mpc_source_path_tracking_candidate.yaml \
  f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800_witness_candidate.csv
```

Pause the all-core optimization while measuring uncontended timing. First lap
is warmup. Supervisor aborts immediately on collision/controller fault and
retains the post-target collision watch. Static allowed-topic policy passes;
organizer approval of the custom 40 Hz bridge remains unresolved. Do not claim
competition certification or treat privileged offline bridge diagnostics as
permitted online state inputs.
