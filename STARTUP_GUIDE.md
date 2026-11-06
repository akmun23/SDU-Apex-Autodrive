# Running the MPC

One controller, two track configurations. The ROS launch selects the profile
from the map identity.

| Track | MPC profile | Raceline |
|---|---|---|
| Practice | `f1tenth_mpc/config/mpc_practice.yaml` | `f1tenth_planning/trajectories/autodrive_practice_20260924_b/source_command_4p800.csv` |
| Competition | `f1tenth_mpc/config/mpc_competition.yaml` | `f1tenth_planning/trajectories/autodrive_iros_2026_compete/ipopt_best_6p925_source_witness.csv` |

## Closed-loop trial against the Unity simulator

    tools/run_trial.sh practice RUN_ID
    tools/run_trial.sh competition RUN_ID

The trial builds the working tree inside the development image, starts the
controller, recorder, simulator and a supervisor that stops the run on a
collision, an MPC fault or a completed lap count, and writes everything to
`live_runs/RUN_ID`. Use a new RUN_ID every time.

Environment: `SDU_APEX_TRIAL_TOTAL_LAPS` (warm-up plus scored laps, default 6),
`SDU_APEX_MPC_PARAMETER_OVERLAY` (a YAML overlay inside the repository),
`SDU_APEX_MPC_NATIVE_TUNING=true` and `SDU_APEX_MPC_LTO=true` (build for the
host CPU).

    source /opt/ros/<distro>/setup.bash
    tools/summarize_trial.py live_runs/RUN_ID      # outcome, deferrals, timing, QP accuracy
    tools/evaluate_mpc_bag.py live_runs/RUN_ID/run # lap times

## Competition image

    tools/build_competition.sh
    tools/run_competition.sh

## Development

Frozen-problem solver benchmarks and the diagnostics code live in
`f1tenth_mpc/util/`; see `docs/MPC.md`.

    tools/stop_autodrive.sh      # stop leftover containers
