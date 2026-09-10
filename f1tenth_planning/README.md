# Track planning

The active planning pipeline is the single `optimize_trajectory.py` entry
point. It extracts the corridor from the production map and runs the TUM
minimum-time optimizer using the AutoDRIVE vehicle profile in
`config/autodrive_sim_vehicle.yaml`.

The optimizer uses one constant-friction vehicle model. Variable-friction map
branches and their unused input files are not part of this repository.

```bash
python3 f1tenth_planning/scripts/optimize_trajectory.py
```

The current production inputs are the 2.5 cm FTG map and its generated mintime
raceline. The output CSV is consumed directly by Pure Pursuit. Intermediate
prepared tracks and visualizations are generated locally and are not part of
the clean baseline.

The exported trajectory contains arc length, position, heading, curvature,
target speed, acceleration, and left/right wall distances. The optimizer
verifies the vehicle steering curvature limit and wall clearance before the
CSV is usable by the controller.
