# AutoDRIVE Min-Time Optimizer v1.3

V1.3 is a corrective release based on the first full real-track V1.2 run.

## Repository-native car model only

V1.3 no longer carries duplicated vehicle numbers in its optimizer YAML.
It resolves the current checked-out project at runtime:

- `f1tenth_mpc/include/mpc_types.h`
  - held-out AutoDRIVE yaw response
  - held-out longitudinal target-speed response
  - measured braking envelope
  - steering angle/rate
  - target-speed slew limits
  - 16 m/s project command ceiling
- `f1tenth_planning/config/autodrive_sim_vehicle.yaml`
  - actual `0.510 m` length
  - actual `0.273 m` body width
  - `0.080 m` rear overhang
  - existing planning footprint `optimizer_width_m = 0.30 m`
  - existing `wall_clearance_m = 0.15 m`
- `f1tenth_control/config/path_tracking_autodrive.yaml`
  - current measured runtime `max_lateral_accel = 6.50 m/s^2`

The optimizer explicitly does **not** read TUM/Pacejka tire dynamics, `mu`,
BachelorProject cornering stiffness, or the physical-reference double-track
plant.

## Wall safety now matches the existing planning pipeline

The minimum aligned center-to-wall distance is:

`optimizer_width_m / 2 + wall_clearance_m = 0.30 / 2 + 0.15 = 0.30 m`.

The NLP therefore never treats a point at the wall as legal. Zero optimization
"wall slack" means the car is at the configured safety boundary: at least
`0.15 m` side-of-planning-footprint clearance remains.

When heading error makes the actual `0.510 x 0.273 m` body wider than the
`0.30 m` planning footprint, V1.3 uses the larger heading-aware physical
footprint.

After the final solve, the runner calls the repository's existing
`compute_wall_distances.py` against the real map and **rejects the trajectory**
if either side has raw center-to-wall distance below `0.30 m`.

## Convergence fix based on the real V1.2 run

The V1.2 output showed:

- coarse `wave:pi/2`: same optimum as the successful right-biased start,
  but only ~49 IPOPT iterations instead of ~1041;
- fine production-raceline warm starts:
  - 0.18 m: ~91 iterations
  - 0.14 m: ~85 iterations
  - 0.11 m: ~57 iterations;
- repaired/raw continuation starts repeatedly consumed 3000 iterations.

V1.3 therefore:

- tries `wave:pi/2` first on the coarse mesh;
- tries the production raceline first on every refinement mesh;
- stops immediately after the first converged refinement candidate;
- limits coarse starts to 1400 iterations;
- limits refinement starts to 450 iterations;
- keeps best-converged-mesh fallback/checkpointing.

On the uploaded real centerline with the corrected `6.50 m/s^2` envelope and
`0.15 m` wall clearance, the revised solver converged in testing at:

- 0.25 m: 71 iterations
- 0.18 m: 61 iterations
- 0.14 m: 57 iterations
- 0.11 m: 100 iterations
- 0.08 m: 68 iterations

No 3000-iteration attempt was required.

## Export fix

V1.2's 0.02 m dense respline could create artificial curvature spikes of tens
of `1/m` that were not present in the optimized states.

V1.3 exports the actual optimizer nodes directly. For a 0.08 m final mesh this
is already ~700 samples around the track and is dense enough for the 40 Hz MPC.
The exported heading and curvature are the states used by the OCP, while an
independent geometric spline is retained only as a diagnostic consistency
check.
