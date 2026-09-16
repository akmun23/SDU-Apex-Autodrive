# Exact Unity open-plane combined-slip runtime-state capture

- Build: `exact_open_runtime_state_v2_20260916`
- Unity: `2022.3.52f1` Linux player
- Run mode: `-batchmode`; no `-nographics`
- Experiment: `combined_slip_matrix_v1`
- Duration: `70.027 s`
- Trace rate: `0.001 s` fixed-step, 70,028 wheel rows
- Trace columns: 154, including runtime inertia and sprung mass
- Collision rows: 290,569 read-only contact callbacks
- Maximum recorded body speed: 15.325 m/s
- Scene mode: open identification plane; no competition track walls
- Physics/controller/scene behavior: unchanged; diagnostic observations only

Runtime body-y inertia ranged from `0.09619075` initially to
`0.09557827 kg m^2`; each wheel's recorded sprung mass also varied over the
run. These values are required state, not fixed fit parameters.

Using the per-step inertia, the usable force audit contained 63,935 rows;
10,231 had chassis contact callbacks. The full contact audit found positive
separation and zero solver impulse for all of them, so no body-force channel
is identified. The lower residual obtained after excluding those rows is only
a data-regime comparison and is not promoted as a physical explanation.

Replacing static sprung loads with the recorded per-step sprung loads changed
the no-chassis-contact fitted holdout only to `1.477 N` force and `0.217 N m`
yaw, versus `1.477 N` and `0.217 N m` with static loads. The variation is
therefore a required exact input but not the missing explanatory mechanism.
