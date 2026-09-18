# V1.3 validation snapshot

This release was checked against the centerline produced by the user's first full V1.2 real-track run.

## Repository-derived active values

```text
physical body length             0.510 m
physical body width              0.273 m
rear overhang                    0.080 m
planning footprint width         0.300 m
required wall clearance          0.150 m
minimum aligned center-to-wall   0.300 m
max steering                     0.5235987756 rad
max steering rate                3.2 rad/s
yaw response tau                 0.087735 s
yaw response gain                3.011897 1/m
longitudinal bias               -0.37356440 m/s^2
longitudinal speed coefficient  -0.06389858 1/s
longitudinal target gain         9.11426915 1/s
longitudinal target-rate coeff  -0.06687204
longitudinal accel cap            6.0 m/s^2
brake envelope                    5.36267417 + 0.27655518*u m/s^2
max command speed                16.0 m/s
target-speed slew                +3 / -8 m/s^2
current lateral envelope          6.50 m/s^2
```

The optimizer reads these from the current repository; they are documented here only to show the reviewed state.

## Real-centerline continuation test

With the current 6.50 m/s² lateral envelope and the repository's 0.30 m planning footprint + 0.15 m wall-clearance rule:

```text
mesh       IPOPT iterations     predicted lap
0.25 m          71              10.053799 s
0.18 m          61              10.063899 s
0.14 m          57              10.064877 s
0.11 m         100              10.066432 s
0.08 m          68              10.068174 s
```

No 3000-iteration solve was needed.

## Export validation at 0.08 m

```text
optimizer/export nodes              705
optimized path length                52.5448 m
mean path sample spacing              0.0745 m
speed range                           ~1.96 to ~9.86 m/s
lateral acceleration maximum          ~6.50 m/s^2
approx. center-to-wall minimum        ~0.300 m before actual-map postcheck
```

The final runner additionally invokes the repository's existing `compute_wall_distances.py` against the user's map and rejects the final trajectory if either raw center-to-wall minimum is below 0.300 m.

## Important interpretation

An internal wall-safety slack near zero now means the trajectory is touching the **configured safety boundary**, not the physical wall. The safety boundary already includes the 0.300 m planning footprint and 0.150 m side clearance. When heading error increases the true rectangular body's lateral projection, the larger physical footprint is used.
