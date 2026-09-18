# AutoDRIVE exact-car minimum-time optimizer v1.3 — design contract

## Goal

Compute the minimum-lap-time periodic trajectory for the **current identified AutoDRIVE RoboRacer command-response model and current project safety corridor**. The optimizer must never silently substitute TUM/Pacejka/BachelorProject vehicle dynamics for the simulated car.

## Authoritative repository inputs

The optimizer reads its active vehicle values from the checked-out repository at runtime.

### Identified command-response dynamics

Source: `f1tenth_mpc/include/mpc_types.h`

Used directly:

- steering angle limit;
- steering-rate limit;
- identified yaw-response gain and time constant;
- identified longitudinal response coefficients;
- measured acceleration saturation;
- measured speed-dependent braking envelope;
- target-speed command ceiling;
- target-speed slew policy used by the current MPC.

### AutoDRIVE geometry and planning footprint

Source: `f1tenth_planning/config/autodrive_sim_vehicle.yaml`

Current values used by v1.3:

```text
physical body length        = 0.510 m
physical body width         = 0.273 m
rear overhang               = 0.080 m
planning footprint width    = 0.300 m
required wall clearance     = 0.150 m per side
```

The `0.300 m` planning footprint is intentionally wider than the physical `0.273 m` body because it is the safety footprint already used by the repository's existing `optimize_trajectory.py`.

### Validated current lateral operating envelope

Source: `f1tenth_control/config/path_tracking_autodrive.yaml`

Current value:

```text
max_lateral_accel = 6.50 m/s^2
```

This is the current repository's measured/runtime envelope and replaces the older 7.3 m/s² planning/bootstrap value. It remains an empirical operating limit that should later be refined from new high-speed closed-loop data, but v1.3 does not invent a different tire model.

### Explicitly not used by the v1.3 vehicle OCP

The minimum-time dynamics do **not** use:

- TUM double-track dynamics;
- Pacejka B/C/E coefficients;
- friction coefficient `mu`;
- BachelorProject cornering stiffness;
- old physical-car mass/inertia/tire force equations;
- TUM drive/brake actuator constants.

The existing repository's map/centerline preparation is reused because it describes track geometry, not because its TUM physical vehicle model is retained.

## Track coordinate

The repository produces a smooth closed reference centerline with center-to-wall widths. Let centerline arc length be `s`, curvature `kappa_ref(s)`, and lateral deviation `e_y` positive to the left.

The optimizer uses centerline arc length as the independent variable. This removes free final time and makes the periodic track bounds natural.

## State

```text
x = [e_y, e_psi, u, r, v_target, delta_command]
```

where:

- `e_y`: lateral offset from centerline;
- `e_psi`: heading relative to centerline tangent;
- `u`: body longitudinal speed;
- `r`: yaw rate;
- `v_target`: commanded speed-setpoint state;
- `delta_command`: steering-command state.

The accepted MPC does not contain a promoted predictive lateral-speed tire/slip model. The planning nominal therefore uses `v=0` rather than inventing one.

## Controls

```text
w = [q_delta, q_v]
```

with current MPC command limits loaded from the repository:

```text
|q_delta| <= 3.2 rad/s
-8 <= q_v <= +3 m/s^2
```

These are explicit command-policy limits. They can later be separated from a theoretical vehicle-only optimum, but v1.3 deliberately generates a line that the current controller command interface can request.

## Dynamics

Frenet progress:

```text
s_dot = u cos(e_psi) / (1 - kappa_ref e_y)
```

Lateral and heading motion:

```text
e_y_dot   = u sin(e_psi)
e_psi_dot = r - kappa_ref s_dot
```

Identified yaw response:

```text
r_ss  = K_yaw * u * tan(delta)
r_dot = (r_ss - r) / tau_yaw
```

Identified longitudinal response:

```text
a = b + c_u*u + k_target*(v_target-u) + c_q*q_v
u_dot = a
```

Command states:

```text
v_target_dot = q_v
delta_dot    = q_delta
```

All coefficients are read from the current MPC header rather than duplicated in the optimizer YAML.

## Spatial conversion

For any time derivative `x_dot`:

```text
dx/ds = x_dot / s_dot
```

and lap-time density is:

```text
dt/ds = 1 / s_dot
```

The NLP minimizes the periodic integral of `dt/ds`.

## Transcription and convergence strategy

The closed lap is discretized at fixed centerline-arc-length nodes. Periodic trapezoidal direct collocation enforces the spatial dynamics across every interval, including the seam.

Default continuation:

```text
0.25 m -> 0.18 m -> 0.14 m -> 0.11 m -> 0.08 m
```

V1.2 real-track data showed that repaired/raw continuation starts repeatedly consumed the 3000-iteration ceiling, while the existing production raceline warm start solved the same fine meshes in only tens of iterations. V1.3 therefore changes the policy:

- coarse level: try empirically successful starts first and stop after the first accepted optimum;
- refinement levels: try the production-raceline warm start first;
- stop after the first accepted refinement solution;
- fine-mesh IPOPT budget is intentionally bounded (default 450), not 3000;
- a failed start cannot erase the last converged checkpoint.

Observed validation on the uploaded real centerline with v1.3:

```text
mesh 0.25 m : 71 iterations
mesh 0.18 m : 61 iterations
mesh 0.14 m : 57 iterations
mesh 0.11 m : 100 iterations
mesh 0.08 m : 68 iterations
```

No 3000-iteration solve was needed.

## Wall safety — must match the existing repository semantics

The existing `optimize_trajectory.py` plans with:

```text
optimizer_width = 0.30 m
wall_clearance  = 0.15 m
```

Therefore, when aligned with the local track tangent, the optimized reference point must remain at least:

```text
0.30 / 2 + 0.15 = 0.30 m
```

from either wall.

V1.3 enforces **at least this much clearance** inside the NLP.

It additionally accounts for the actual `0.510 x 0.273 m` body when the vehicle is angled relative to the local track tangent. The lateral occupied half-extent is:

```text
max(
    0.5 * planning_footprint_width,
    0.5 * physical_width * |cos(e_psi)|
      + max(front_extent, rear_extent) * |sin(e_psi)|
)
```

and the `0.15 m` required wall clearance is added outside that footprint.

Thus the optimizer is never allowed to trade away the safety margin merely to reduce lap time.

After export, the runner also invokes the repository's existing `compute_wall_distances.py` on the actual map and hard-rejects the final trajectory if either raw center-to-wall minimum is below `0.30 m`. This is a second, independent map-space safety check using the same convention as the existing planning pipeline.

## Acceleration/braking

The identified longitudinal response is constrained by the accepted MPC envelope:

```text
a <= +6.0 m/s^2
a >= -(5.36267417 + 0.27655518*u)
```

The exact coefficients are parsed from `mpc_types.h`; the numbers above are shown only for explanation.

This prevents the optimizer from requesting the +8 to +9 m/s² accelerations present in older TUM-generated profiles when the current accepted AutoDRIVE model cannot reproduce them.

## Steering feasibility

Steering angle and steering rate are explicit optimization constraints. The line therefore cannot rely on geometric curvature transitions that the AutoDRIVE steering actuator cannot traverse at the optimized speed.

## Lateral operating limit

The accepted yaw-response model identifies how steering produces yaw, but by itself does not establish an ultimate tire/slip stability boundary. V1.3 therefore constrains:

```text
|u*r| <= ay_max(u)
```

with the current project runtime envelope sourced from `path_tracking_autodrive.yaml`.

For the current repository this is:

```text
ay_max = 6.50 m/s^2
```

No old TUM/Pacejka friction calculation is used to replace this value.

An optional combined `ax/ay` superellipse remains disabled until simulator data justify its shape.

## Export

V1.2's dense 0.02 m exporter was rejected because linear interpolation of `e_y` followed by another periodic cubic fit generated artificial curvature spikes far above the optimized dynamics.

V1.3 exports the **actual optimized node path** directly. At the final 0.08 m mesh this is roughly 700 points around the lap, already significantly denser than one point per 40 Hz vehicle movement at racing speed.

The controller CSV schema remains:

```text
s,x,y,psi,kappa,v,ax,d_left,d_right
```

`x/y/psi/speed/acceleration` come from the optimized node solution. The optimized dynamic curvature `r/u` is exported as the controller curvature reference, while an independent geometric curvature reconstruction is retained in the report as a validation diagnostic rather than being allowed to corrupt the path.

## Output and promotion checks

A final candidate is not promoted merely because IPOPT says `Solve_Succeeded`.

The report records:

- lap time;
- solver iterations/status;
- minimum wall safety slack;
- independent raw wall distances from the actual map;
- speed, yaw and steering extrema;
- steering-rate and target-rate utilization;
- longitudinal acceleration/braking utilization;
- lateral-envelope utilization;
- geometric-vs-dynamic curvature consistency.

Any violation of the required wall clearance is a hard failure.

## Tuning loop

1. Solve using the current repository's identified dynamics and 6.50 m/s² validated lateral envelope.
2. Verify actual-map wall clearance.
3. Inspect `report.json` and `solution_nodes.csv`.
4. Track at reduced speed with MPC.
5. Record actual `u`, `r`, cross-track error, steering, steering rate and command history.
6. Use new held-out simulator data to refine only the specific envelope that is demonstrably conservative or inaccurate.
7. Re-solve and compare actual closed-loop lap time.
8. Increase aggressiveness only when data support it.

The final objective is minimum **actual simulator lap time** for this AutoDRIVE car, not minimum time for an unrelated physical-car model.
