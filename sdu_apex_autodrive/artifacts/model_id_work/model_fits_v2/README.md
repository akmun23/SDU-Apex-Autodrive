# Model-fit artifacts

The active 2026-09-15/16 artifacts are the files whose names contain
`core_guard_075`, `ablation_A0_A5_first_order`, `ablation_A0_A6_first_order`,
`mpc_N30_discrete_stage_map`,
`n30_stage_map_validation`, `runtime_control_state_live`, or
`speed_command_pipeline_live`, plus the first-order candidate and
the first-order residual candidate, the 2026-09-16 residual diagnostic, plus
`steering_semantics_v2_train_validation_20260916.json` one level above this
directory. The earlier v1 steering report is training-only historical
evidence.

They are offline identification/validation outputs. They do not change the
Unity simulator or production MPC. The causally corrected raceline-filtered
candidate passes the provisional 0.75 s CORE numerical gates, but blind
validation, native parity, and live acceptance remain incomplete. No model is
currently approved for MPC migration.

Tire-peak bound experiments are documented in
`tire_peak_bound_diagnostic_20260916.md`; inflated peaks are diagnostic only
and are not approved parameters.

All other files in this directory are historical audit evidence from earlier
model experiments. They are intentionally retained for provenance but must
not be used as active candidate-selection inputs. In particular, reports from
2 s horizons or >16 m/s experiments are superseded for the current project
contract.

The direct Unity WheelCollider screen is
`direct_unity_wheel_collider_benchmark_075_ackermann_fix_20260916.json`.
It is offline-only and uses the Unity piecewise sideways curve and
Ackermann steering geometry directly. Its full holdout 0.75 s CORE p95 is
`0.0431 m` cross-track, `0.0528 rad` heading, `0.0100 m/s` lateral velocity,
and `0.1098 rad/s` yaw rate. It is a comparison model, not a promoted MPC
model. The companion
`unity_wheel_contact_trace_audit_20260916.json` identifies dynamic contact
load as the next explicit mechanism to investigate. The instantaneous
load-transfer screen is rejected for recursive instability.

The current force-mechanism audit is
`unity_exact_open_axle_force_response_v1_20260916.json` plus its repeat. It
uses wheel rotational balance and body-wrench inversion to recover front/rear
axle lateral response. Its gains are diagnostic Unity-solver observability
quantities only; they are not friction or cornering-stiffness parameters and
are not active MPC inputs.
