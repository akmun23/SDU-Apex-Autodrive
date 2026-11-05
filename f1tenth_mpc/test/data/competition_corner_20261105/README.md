# Competition corner, 5 November 2026

Portable inputs for `tools/competition_failure_benchmark.py`. Unity, ROS and
the ignored `live_runs` tree are unnecessary for replay. Extraction alone
requires the original bag and its ROS message types.

Source: `source_sqp_iros_viability_20261105_02`. All 22 recorded source-solver
epochs from 142.680016 to 144.154231 m are retained in receipt order. The upper
extraction bound is 144.16 m to include the epoch described as “144.15 m”.
Do not sort by progress: adjacent command-time predictions can move backwards
slightly when their publication leads differ.

| Role | Record index | Progress, m |
| --- | ---: | ---: |
| Last published feasible plan | 756 | 142.841566887 |
| First source-infeasible result | 757 | 142.992144702 |
| Reported terminal failure | 769 | 143.847507964 |
| Last recorded solve | 775 | 144.154231245 |

Files:

- `sequence.json.gz`: states, all 22 source internal channels, actuator queues,
  initialization flags and provenance, warm starts, published primals, frozen
  competition parameters, issued/applied host commands and epoch warnings.
- `raceline.csv.gz`: original complete racing reference, before the recorded
  curvature/braking reference envelope is applied.
- `physical_model.json.gz`: original physical source/configuration files,
  including all compiled tire, contact, geometry and substep constants.
- `oracle_baseline.json.gz`: frozen 31-sample objectives, terminal targets,
  complete model PODs, and reconstructed per-SQP primal/state histories.
- `investigation.json.gz`: parity, reachability and additional-iteration evidence.
- `manifest.json`: input provenance and SHA-256 checksums.

The archived SQP traces are **offline reconstructions**, not unrecorded live
iterates. Historic diagnostics used 12 significant digits, omitted initialization
flags and frozen schedules, and discarded rejected final primals. Flags are
recovered from exact-stamp bag state and the source predictor initialization
contract. Initial costs pass relative tolerance 1e-7; the three recorded final
primals independently pass cost/violation and clearance checks. Missing rejected
final primals cannot be independently certified by their summary scores alone.

Replay keeps the archived model and objective fixed across optimizer changes.
The same-build POD has explicit size guards; the original physical configuration
and readable reference samples remain available separately. Repeated runs must
have identical numerical results after removing wall-clock fields. Changed
optimizer results may improve feasibility; previous feasible states and stored
feasibility witnesses must remain feasible.

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
cmake -S f1tenth_mpc -B build/f1tenth_mpc
cmake --build build/f1tenth_mpc --target \
  mpc_source_live_fixture mpc_source_exact_offline_optimize \
  mpc_source_feasibility_witness_probe mpc_source_benchmark_capture_check -j 4

python3 tools/competition_failure_benchmark.py replay \
  --output artifacts/competition_corner_replay

python3 tools/competition_failure_benchmark.py witness \
  --replay-dir artifacts/competition_corner_replay \
  --roles first_infeasible last_feasible terminal_failure --steps 30 60

ctest --test-dir build/f1tenth_mpc --output-on-failure \
  -R test_competition_failure_benchmark
```

All witnesses enforce throttle >= 0.001. Exact-zero braking is excluded.
Collision clearance is hard at zero excess above the original 5 cm body margin;
the preferred extra 3 cm is elastic. Searches are offline diagnostic tools and
never publish commands or enter the controller runtime.

Future live captures can enable `capture_source_benchmark: true` in an isolated
ROS overlay. Diagnostics then retain exact frozen model/objective PODs, each
SQP primal warm start/trial/acceptance and rejected final controls. All state
channels are written with 17 significant digits. Capture is disabled by default
and its integrated timing impact is not qualified. This does not enable the
experimental terminal-speed viability inequality.
