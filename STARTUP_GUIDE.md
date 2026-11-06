# Running the MPC: choose practice or competition

The maintained controller uses one of TWO standalone track configurations.
The ROS launcher selects the profile from the map identity. There is no
shared stack of old weight overlays in the normal runtime.

| Track | Complete MPC profile | Native command raceline |
|---|---|---|
| Practice | f1tenth_mpc/config/mpc_practice.yaml | autodrive_practice_20260924_b/source_command_4p800.csv |
| Competition | f1tenth_mpc/config/mpc_competition.yaml | autodrive_iros_2026_compete/ipopt_best_6p925_source_witness.csv |

The practice profile is the exact merged, qualified 4500/1450 parameter
set: five complete independent ten-scored-lap races, mean 49.1622 s,
zero collisions/controller failures or hard command deadlines. The
competition profile has INDEPENDENT weights and remains experimental.
The competition reference is the 6.925 s PhysX source-validated IPOPT
line, NOT a verified Unity/MPC ten-lap time.

## Choose a track

    tools/run_mpc_trial.sh show
    SDU_APEX_TRIAL_TOTAL_LAPS=11 tools/run_mpc_trial.sh practice practice_01
    SDU_APEX_IROS_TOTAL_LAPS=11 tools/run_mpc_trial.sh competition comp_01

Each supervised runner starts the controller, recorder, Unity simulator
and collision/lap watcher, saving results to live_runs/RUN_ID. Use a new
RUN_ID every time. The normal dispatcher refuses runtime parameter
overrides and thus cannot silently mix practice and competition weights.

For intentional development A/B runs only, call the lower-level
tools/run_mpc_practice_trial.sh or tools/run_iros_trial.sh directly
with its explicitly named MPC parameter overlay. Such overlays are
never required to run the production practice baseline.

## Competition image

    tools/build_competition.sh
    tools/run_competition.sh

This standalone launch has the same default IROS map and pinned IPOPT
raceline. Source-only validity is not a substitute for closed-loop
validation. The native source-plant physical corridor, controller
faults, and 20 ms command-publication checks remain active.

## Development

    AUTODRIVE_REBUILD=0 SDU_APEX_IMAGE=sdu-apex-autodrive:dev tools/start_dev.sh
    SDU_APEX_SIM_TRACK=practice SDU_APEX_SIM_MODE=batchmode tools/start_simulator.sh

Stop with tools/stop_autodrive.sh or Ctrl-C. Superseded experiment files
were preserved outside active config under
live_runs/mpc_profile_history_20261101.
