# AutoDRIVE Exact-Car Minimum-Time Optimizer v1.3

This optimizer is designed for the **current AutoDRIVE simulated RoboRacer in
this repository**, not for a generic physical F1TENTH car.

## Vehicle sources

V1.3 reads the checked-out repository at runtime.

| Quantity | Source |
|---|---|
| yaw response, longitudinal response, brake envelope, steering/rate, target-speed slew, max command speed | `f1tenth_mpc/include/mpc_types.h` |
| actual car geometry | `f1tenth_planning/config/autodrive_sim_vehicle.yaml` |
| planning footprint + wall clearance | `f1tenth_planning/config/autodrive_sim_vehicle.yaml:mintime` |
| current validated lateral acceleration envelope | `f1tenth_control/config/path_tracking_autodrive.yaml:max_lateral_accel` |

The optimizer does **not** use Pacejka coefficients, TUM's double-track vehicle,
old cornering stiffness values, or a generic friction coefficient.

At the reviewed repository state the important values resolve to:

```text
physical body:            0.510 m x 0.273 m
planning footprint width: 0.300 m
required wall clearance:  0.150 m from the planning/physical side envelope
minimum aligned centre→wall distance: 0.300 m
steering:                  ±0.5235987756 rad
steering rate:             ±3.2 rad/s
validated lateral accel:   6.50 m/s²
command speed ceiling:     16.0 m/s
target-speed slew:         +3 / -8 m/s²
```

The yaw and longitudinal coefficients are printed from the MPC header at every
run so repository drift cannot be hidden.

## Safety rule

The wall constraint mirrors the existing `optimize_trajectory.py` semantics:

```text
0.5 * optimizer_width_m + wall_clearance_m
= 0.5 * 0.30 + 0.15
= 0.30 m minimum centre-to-wall distance when aligned.
```

V1.3 additionally enlarges the footprint when the real `0.510 x 0.273 m` body
is angled relative to the path.

After optimization it calls the repo's own `compute_wall_distances.py` against
the actual map. If either side measures less than the required `0.30 m`, the
final trajectory is rejected rather than silently promoted.

## Installation

The archive can remain as a nested directory inside the repository.

```bash
python3 -m pip install -r requirements_autodrive_mintime.txt
```

## Self test

From the extracted bundle's `f1tenth_planning/scripts` directory:

```bash
python3 run_autodrive_mintime.py --self-test
```

## Optimize the current track

```bash
python3 run_autodrive_mintime.py
```

The runner automatically uses:

- the parent SDU-Apex-Autodrive repository;
- the current AutoDRIVE map;
- the current production raceline as a warm start;
- the existing repo centerline/map preparation pipeline.

## Why V1.3 should no longer sit at 3000 iterations

The uploaded V1.2 run showed that the bad repaired continuation starts were the
problem, not the fine mesh itself. V1.3 orders starts using those measurements:

```text
coarse:     wave:pi/2 -> right -> warm -> center
refinement: production warm raceline -> previous solution -> center
```

It stops after the first successful refinement and uses bounded budgets:

```text
coarse max:     1400 iterations
refinement max: 450 iterations
```

On the uploaded real centerline, with the corrected car limits and safety
clearance, development testing converged in:

```text
0.25 m:  71 iterations
0.18 m:  61
0.14 m:  57
0.11 m: 100
0.08 m:  68
```

## Output

```text
autodrive_mintime_exact/
├── autodrive_mintime_raceline.csv
├── solution_nodes.csv
├── report.json
├── resolved_config.yaml
├── checkpoints/
│   ├── continuation_report.json
│   └── mesh_*/
└── track_prep/
```

The controller trajectory is now the **actual optimizer-node path** rather than
a 0.02 m re-spline. The old dense export was responsible for artificial
curvature spikes.

Send back the entire output directory or at minimum:

- `report.json`
- `resolved_config.yaml`
- `solution_nodes.csv`
- `autodrive_mintime_raceline.csv`
- `checkpoints/continuation_report.json`
- complete terminal output

The next tuning step should be based on the actual solved constraint usage and
closed-loop simulator trace, not on generic real-car parameters.
