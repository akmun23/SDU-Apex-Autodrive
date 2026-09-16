# Exact Unity open-plane combined-slip collision audit

- Build: `exact_open_collision_audit_v1_20260916`
- Unity: `2022.3.52f1` Linux player
- Run mode: `-batchmode`; no `-nographics`
- Experiment: `combined_slip_matrix_v1`
- Duration: `70.027 s`
- Trace rate: `0.001 s` fixed-step, 70,028 wheel rows
- Collision rows: 290,569 read-only contact callbacks
- Maximum recorded body speed: 15.325 m/s
- Scene mode: open identification plane; no competition track walls
- Physics/controller/scene behavior: unchanged; diagnostic observations only

The collision trace shows chassis-floor contact callbacks during two turning
intervals. Full contact-point auditing found positive separation and zero
solver impulse for every chassis callback, so these are not measured body
forces. The callbacks remain useful for exact collision-state auditing but do
not invalidate the wheel force balance by themselves.

The lateral-wrench audit found 63,936 stable usable rows, of which 10,231
had chassis callbacks. The lower residual obtained after excluding them is a
data-regime effect, not evidence of a body-force mechanism; no body-force term
is promoted from this capture.
