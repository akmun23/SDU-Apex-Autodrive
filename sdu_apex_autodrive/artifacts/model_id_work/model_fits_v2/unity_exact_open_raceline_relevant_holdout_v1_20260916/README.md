# Exact Unity raceline-relevant force holdout

- Unity: `2022.3.52f1` Linux player
- Run mode: `-batchmode`, normal graphics; `-nographics` was not used
- Experiment: `raceline_relevant_holdout_v1`
- Duration: `84.035 s`
- Rows: `84,036` at the Unity `0.001 s` fixed step
- Speed range: `0--15.36 m/s`; no command or model exceeds `16 m/s`
- Applied steering envelope: `-3.6--+3.6 deg`
- Simulator behavior/physics: unchanged

The bounded steering rows selected for force inversion cover
`5.94--15.31 m/s` after applying the offline `0.005 rad` minimum-steering
screen. This includes high-speed, small-steering raceline conditions and is
not an arbitrary high-speed/high-steering stress test.

The trace is split into `wheel_contact_trace_part01.csv` and
`wheel_contact_trace_part02.csv`, each below 100 MB. The force-response report
is `../unity_exact_open_raceline_axle_force_response_v1_20260916.json`.

The previous open-plane schedule identified front/rear proxy gains of
`0.6489/0.6869`. This fresh schedule identifies `0.7932/0.6781` on its
chronological training half, with holdout RMSE `0.234/0.183 N` using the
threshold-adjusted selection. The change is intentional evidence that one
constant gain would hide a state/regime dependency; neither pair is promoted
to the vehicle model.
