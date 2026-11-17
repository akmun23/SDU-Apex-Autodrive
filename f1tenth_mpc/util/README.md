# Frozen MPC diagnostics

These tools use the production plant, scorer, local model and QP solver. They
are excluded from the default controller build. Keep cases and generated
reports outside the package, for example under `live_runs/`.

```sh
# once: build and source sdu_apex_msgs in a ROS environment
source /opt/ros/humble/setup.bash
colcon build --merge-install --packages-up-to sdu_apex_msgs
source install/setup.bash
cmake -S f1tenth_mpc -B build/mpc_util -DCMAKE_BUILD_TYPE=Release \
  -DSDU_APEX_MPC_NATIVE_TUNING=ON -DCMAKE_PREFIX_PATH="$PWD/install"
cmake --build build/mpc_util --target mpc_export_case mpc_sqp_bench mpc_qp_check -j4
```

## Capture and export

From the repository root, run `tools/sim/run_trial.sh` with
`SDU_APEX_MPC_PARAMETER_OVERLAY` pointing to a YAML overlay containing
`capture_source_benchmark: true`. Decode the bag with
`tools/analysis/summarize_trial.py RUN --keep-export RUN/analysis`, where `RUN`
is the path to the run directory. Capture only changes diagnostics.

```sh
python3 f1tenth_mpc/util/export_cases.py \
  --exporter build/mpc_util/mpc_export_case \
  --output live_runs/cases \
  --source practice:live_runs/RUN/analysis/diag.jsonl:f1tenth_planning/trajectories/practice/raceline.csv
```

The default exports 30 cases per track. Sampling keeps standing starts,
infeasible starts, practice hairpins and the observed progress bins. The
manifest records input hashes and record indices. Cases are same-build POD
captures: regenerate them after a POD layout change. Current format is `MPCCASE6`
with `source-pod/v4` captures, validated by magic/size checks.

A case is a frozen problem, not a reconstruction of every trial of the live
cycle: it holds the first iteration's problem and the last attempted radius.

## Solver benchmark

```sh
build/mpc_util/mpc_sqp_bench solve CASE [ITERATIONS=4] [RADIUS=0.5]
build/mpc_util/mpc_sqp_bench drive CASE CYCLES [ITERATIONS=4] [DEADLINE_MS=22] [RADIUS=0.5] \
    [E_Y_SD=0] [E_PSI_SD=0] [SEED=0]
```

Environment: `MPC_OVERRIDE="name=value,..."` replaces captured weights and
margins; `MPC_THREADS` sets the Jacobian workers (use 1 with a very large
deadline and fixed iterations for a deterministic run); `SQP_PENALTY` and
`SQP_ACCURACY` set the violation penalty and `sqp_accuracy_tolerance`;
`NOISE_FILE` (lines of "e_y_error e_psi_error", a blank line between
recordings) replays recorded localisation errors instead of Gaussian noise.

`solve` prints per-iteration model-versus-exact data as JSON, including the
attempts and curvature corrections of each iteration and whether the accepted
step meets the accuracy tolerance. `drive` runs the controller closed loop on
the source plant itself. It applies the first command of each published plan
for 25 ms. A
deferred cycle holds the previous command, and the run ends at the first wall
contact. Contact uses zero reserve and zero nonlinear tolerance; the configured
reserve margin is logged separately. The benchmark counts physically rejected
horizons separately; the live supervisor aborts on them. It publishes only
horizons that pass the configured publication checks, including any far-stage
wall tolerance, and reports every accepted racing-step error.
This synchronous replay receives fresh source state every cycle; it does not
model ROS transport gaps or the node's epoch-gated plan continuation. The
reported baseline replay with recorded localisation noise does not fail;
its cases and noise recordings are absent from this checkout. Use this replay
for solver quality and compute cost, and closed-loop simulator runs for failure
rates and tuning (`tools/analysis/sweep_mpc.py`).

TODO(verify)

Wall-clock timing is real, so pin and isolate threads as `tools/sim/run_trial.sh`
does when timing matters.

## QP solver check

```sh
python3 f1tenth_mpc/util/check_qp_reference.py --solver build/mpc_util/mpc_qp_check
```

The script checks deterministic properties of the convex QP against OSQP
(`pip install osqp`).

## State-gap forensics

```sh
python3 tools/analysis/analyze_mpc_state_gaps.py live_runs/RUN --output live_runs/RUN/gaps.json
```

Run the trial summary export first. This aligns each stale diagnostic with the
last state receipt at the recorder, its source stamp, the controller snapshot,
and the first recorded collision or rejected cycle. It marks waits after
supervisor abort separately, and reports callback/synchronization timings.
Recorder receipt and controller callback receipt are independent clocks/events;
callback-entry analysis requires recorded per-message callback-entry timestamps.
