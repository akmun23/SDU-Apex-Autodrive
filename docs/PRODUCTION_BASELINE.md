# Practice racing baseline — 2026-10-29

## Active setup

- Track: `f1tenth_planning/maps/autodrive_practice_20260924_b.yaml`
- Reference/commands: `f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv`
- Plant witness: adjacent `source_command_4p800_witness.npz`
- MPC practice profile: `f1tenth_mpc/config/mpc_practice_fastline.yaml`
- Production AMCL: `f1tenth_localization/config/gpu_amcl_cpp_params.yaml`
- General competition profile and track remain separate: `mpc_competition.yaml` and `autodrive_mintime_sim_5p0_dense/`.

## Verified result

Three complete supervised trials each ran one warmup plus ten scored laps, with
no collision or controller fault. Scored lap totals were 49.588 s, 49.467 s,
and 49.614 s (30 scored laps total: 148.669 s; mean 4.9556 s/lap). Each run
had eight of ten scored laps below 5 s. The best recorded scored lap was
4.8758 s. This beats the stated 49.8 s ten-lap practice target on all three
trials; it does not establish ten consecutive sub-5 s laps.

| Trial | Ten-lap total | Mean lap | Lap standard deviation | Best–worst | MPC maximum | Deferred cycles |
|---|---:|---:|---:|---:|---:|---:|
| `mpc_amcl_age_motion4p800_20261029_01` | 49.588 s | 4.9588 s | 42.94 ms | 4.9158–5.0618 s | 19.116 ms | 0 |
| `mpc_amcl_age_motion4p800_20261029_02` | 49.467 s | 4.9467 s | 53.58 ms | 4.8758–5.0358 s | 20.714 ms | 1 |
| `mpc_amcl_prod4p800_20261029_01` | 49.614 s | 4.9614 s | 44.53 ms | 4.8888–5.0518 s | 21.426 ms | 1 |

The last trial used the production AMCL defaults, without an AMCL overlay.
The 20 ms timing goal was not met in every cycle: two trials recorded one
deferred cycle each. Do not claim a strict maximum below 20 ms.

Map-pose forward RMSE was 3.19, 3.04, and 2.92 cm; lateral RMSE was 2.67,
2.70, and 2.73 cm. Truth-to-reference cross-track RMS was 5.53, 5.53, and
6.17 cm. These are practice-run measurements, not a general accuracy guarantee.

## Reproduction and retained evidence

Run the setup in [`STARTUP_GUIDE.md`](../STARTUP_GUIDE.md), with a new run ID.
The complete source/configuration hashes and run procedure are sealed by
`tools/run_mpc_practice_trial.sh`. The four local captures retained in
`live_runs/` are the three trials above and the 50.287 s comparison run
`mpc_source_path4p800_20261029_02`. Older run folders and their repetitive
screens were removed; all listed scores remain recorded here and in the
per-run artifacts.

The 4.800 s value is the reference/source-command duration, not a guaranteed
achieved lap time. The unattended line-search campaign reached its configured
plateau at this reference and did not produce a faster validated trajectory.
No simulator or optimizer job was running at the cleanup snapshot.

## Current components

See [`PRODUCTION_CODE_MAP.md`](PRODUCTION_CODE_MAP.md) for file ownership and
the active execution path. Competition-topic restrictions are listed in
[`RUNTIME_TOPIC_POLICY.md`](RUNTIME_TOPIC_POLICY.md).
