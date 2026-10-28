"""Check Phase-I reduction and exact dynamic re-entry without IPOPT."""
from __future__ import annotations

import sys
import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "SDU_Apex_IPOPT_PhysX_MinTime_v2"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from recover import load_modules, restore, LEGACY  # noqa: E402
from solver_overlay_v251 import (  # noqa: E402
    add_ipopt_cpu_time_limits,
    make_componentwise_time_trust_region, make_phase1_objective_only,
    make_phase1_source_equalities,
    prepare_working_solver_v251)
from solver_overlay_v251 import stop_source_phase1_when_feasible  # noqa: E402
from run_ipopt import (  # noqa: E402
    OPTIMIZER_TEMPORAL_SEMANTICS,
    annotate_spatial_temporal_semantics,
    load_optimizer_seed_checkpoint, load_reference_speed_seed,
    parse_trust_region_factors,
    replace_optimizer_seed_speed_profile,
    write_optimizer_seed_checkpoint,
)


def test_time_solver_controls_parse_as_explicit_continuation_settings():
    assert parse_trust_region_factors('1.5, 2, 4') == (1.5, 2.0, 4.0)
    for invalid in ('', '1', '2,2', '4,2', 'nan', 'inf'):
        try:
            parse_trust_region_factors(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f'accepted invalid trust radii: {invalid!r}')


def test_source_equality_time_continuation_uses_componentwise_trust_bounds():
    source = (
        "        opti.set_value(trust_region_bound, feasibility_reference * "
        "time_trust_region_factors[0])\n"
        "        opti.subject_to(feasibility_objective <= trust_region_bound)\n"
        "        constraint_labels.append('captured_motion_trust_region')\n"
        "        stage2_initial_diagnostics = _constraint_diagnostics("
        "opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), "
        "opti.value(opti.ubg, opti.initial()), constraint_labels)\n"
        "report = {'feasibility_trust_region': feasibility_trust_region, "
        "'ipopt_iteration_budgets': {}}"
    )
    transformed = make_componentwise_time_trust_region(source)
    assert 'time_reference_x = np.asarray(stage_values(' in transformed
    assert 'stage1_solution, stage1_stats, x)' in transformed
    assert 'time_trust_region_deviation = ca.vertcat(' in transformed
    assert 'x - ca.DM(time_reference_x)' in transformed
    assert 'x - ca.DM(seed_x_values)' not in transformed
    assert 'time_trust_region_component_limit = trust_component_radius(' in transformed
    assert "constraint_labels.extend(['captured_motion_trust_region'] * " in transformed
    assert 'opti.subject_to(feasibility_objective <= trust_region_bound)' not in transformed
    assert "'time_trust_region_geometry': os.environ.get('SDU_IPOPT_TRUST_GEOMETRY', 'node-mean')" in transformed


def test_ipopt_cpu_time_caps_are_optional_and_keep_iteration_limits():
    source = (
        "import math\n"
        "def solve():\n"
        "    opti.solver('ipopt', {'expand': False, "
        "'detect_simple_bounds': True, 'ipopt': {'max_iter': "
        "feasibility_max_iterations, 'hessian_approximation': "
        "'limited-memory', 'limited_memory_max_history': 20, 'tol': "
        "1e-06, 'acceptable_tol': 1e-05, 'print_level': "
        "int(print_level)}}, {})\n"
        "    if True:\n"
        "        time_ipopt_options = {'max_iter': "
        "time_optimization_max_iterations, 'tol': 1e-06, "
        "'acceptable_tol': 1e-05, 'print_level': int(print_level)}\n"
        "        if time_hessian_approximation == 'limited-memory':\n"
        "            pass\n")
    transformed = add_ipopt_cpu_time_limits(source)
    assert "import os\n" in transformed
    assert "phase1_ipopt_options['max_cpu_time'] = phase1_max_cpu_time_s" in transformed
    assert "'max_iter': feasibility_max_iterations" in transformed
    assert "float(os.environ.get('SDU_IPOPT_PHASE1_MAX_CPU_SECONDS', '0'))" in transformed
    assert "time_ipopt_options['max_cpu_time'] = time_max_cpu_time_s" in transformed
    assert "float(os.environ.get('SDU_IPOPT_TIME_MAX_CPU_SECONDS', '0'))" in transformed
    compile(transformed, "phase1_cpu_cap_fixture.py", "exec")


def test_reference_speed_seed_is_periodically_resampled_and_not_a_constraint(
        tmp_path):
    source = tmp_path / 'reference.csv'
    source.write_text(
        '# s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps,'
        'acceleration_mps2,d_left_m,d_right_m\n'
        '0,0,0,0,0,3,0,1,1\n'
        '1,1,0,0,0,5,0,1,1\n'
        '2,1,1,0,0,7,0,1,1\n'
        '3,0,1,0,0,4,0,1,1\n')
    track = SimpleNamespace(count=5, length=4.0,
                            s=np.asarray([0.0, 0.5, 1.0, 2.0, 3.0]))
    profile = load_reference_speed_seed(source, track)
    assert np.allclose(profile, [3.0, 4.0, 5.0, 7.0, 4.0])
    assert np.all(profile > 0.0)


def test_reference_speed_seed_accepts_standard_velocity_column(tmp_path):
    source = tmp_path / "baseline.csv"
    source.write_text(
        "# s_m,x_m,y_m,psi_rad,kappa_radpm,velocity_mps,acceleration_mps2,"
        "d_left_m,d_right_m\n"
        "0,0,0,0,0,4,0,1,1\n"
        "1,1,0,0,0,6,0,1,1\n"
        "2,1,1,0,0,8,0,1,1\n"
        "3,0,1,0,0,5,0,1,1\n")
    track = SimpleNamespace(count=5, length=4.0,
                            s=np.asarray([0.0, 0.5, 1.0, 2.0, 3.0]))
    profile = load_reference_speed_seed(source, track)
    assert np.allclose(profile, [4.0, 5.0, 6.0, 8.0, 5.0])


def test_hybrid_checkpoint_seed_replaces_only_forward_speed():
    seed = {
        'u_mps': np.array([3.0, 4.0, 5.0]),
        'v_mps': np.array([0.2, -0.1, 0.3]),
        'roll_rad': np.array([0.01, 0.02, -0.01]),
        'yaw_rate_radps': np.array([0.4, -0.5, 0.7]),
        'wheel_omega_radps': np.arange(12.0).reshape(4, 3),
        'steering_rate_radps': np.array([0.1, -0.2, 0.3]),
        'body_height_rate_mps': np.array([0.01, 0.02, -0.01]),
        'roll_rate_radps': np.array([0.03, -0.04, 0.05]),
        'pitch_rate_radps': np.array([-0.03, 0.02, 0.01]),
        'ax_mps2': np.array([1.0, -2.0, 3.0]),
        'ay_mps2': np.array([-1.0, 2.0, -3.0]),
        'throttle_norm': np.array([0.12, 0.17, 0.14]),
    }
    replacement = np.array([4.5, 6.0, 7.5])
    hybrid = replace_optimizer_seed_speed_profile(
        seed, replacement, minimum_mps=0.35, maximum_mps=16.0)
    assert np.array_equal(hybrid['u_mps'], replacement)
    assert np.array_equal(seed['u_mps'], [3.0, 4.0, 5.0])
    for field in seed.keys() - {'u_mps'}:
        assert np.array_equal(hybrid[field], seed[field])
        assert hybrid[field] is not seed[field]

    scaled = replace_optimizer_seed_speed_profile(
        seed, replacement, minimum_mps=0.35, maximum_mps=16.0,
        mode='time-scaled', steering_rate_limit_radps=0.4)
    ratio = replacement / seed['u_mps']
    assert np.allclose(scaled['v_mps'], seed['v_mps'] * ratio)
    assert np.allclose(scaled['wheel_omega_radps'],
                       seed['wheel_omega_radps'] * ratio[None, :])
    assert np.allclose(scaled['roll_rate_radps'],
                       seed['roll_rate_radps'] * ratio)
    assert np.max(np.abs(scaled['steering_rate_radps'])) <= 0.4
    assert np.allclose(scaled['ax_mps2'], seed['ax_mps2'] * ratio**2)
    assert np.array_equal(scaled['roll_rad'], seed['roll_rad'])
    assert np.array_equal(scaled['throttle_norm'], seed['throttle_norm'])

    for bad_speed in (np.array([1.0, 2.0]),
                      np.array([1.0, np.nan, 2.0]),
                      np.array([1.0, 2.0, 16.1])):
        try:
            replace_optimizer_seed_speed_profile(
                seed, bad_speed, minimum_mps=0.35, maximum_mps=16.0)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted an invalid hybrid speed profile')


def test_optimizer_timebase_is_spatial_not_runtime_command_rate():
    semantics = OPTIMIZER_TEMPORAL_SEMANTICS
    assert semantics['formulation'] == (
        'spatial OCP; minimize integrated lap time from ds/sdot')
    assert semantics['25ms_runtime_command_hold'] is False
    assert 'spatial nodes' in semantics['control_mesh']
    assert 'not used as the optimizer time grid' in (
        semantics['1ms_unity_fixed_step_role'])
    report = {'wheel_spin_discretization': {'reason': 'stale generated text'}}
    annotate_spatial_temporal_semantics(report, unity_source_fixed_step_s=0.001)
    wheel = report['wheel_spin_discretization']
    assert report['temporal_semantics'] == semantics
    assert wheel['unity_fixed_step_s'] == 0.001
    assert wheel['optimizer_time_grid_s'] is None
    assert wheel['branch_selection_per_spatial_segment'] is True
    assert wheel['within_segment_unity_branch_changes_reproduced'] is False
    assert 'not a Unity-per-tick replay' in wheel['reason']


def test_optimizer_checkpoint_round_trip_and_model_provenance(tmp_path):
    track = SimpleNamespace(
        count=3, length=9.0, s=np.array([0.0, 3.0, 6.0]))
    model = SimpleNamespace(metadata=lambda: {'vehicle': 'practice-v1'})
    map_path = tmp_path / 'map.pgm'
    centerline_path = tmp_path / 'centerline.csv'
    map_path.write_text('map fingerprint')
    centerline_path.write_text('line fingerprint')
    candidate = {
        'ey_m': np.zeros(3), 'epsi_rad': np.zeros(3),
        'speed_mps': np.ones(3), 'lateral_speed_mps': np.zeros(3),
        'yaw_rate_radps': np.zeros(3),
        'wheel_omega_radps': np.ones((4, 3)),
        'steering_rad': np.zeros(3), 'steering_rate_radps': np.zeros(3),
        'throttle_norm': np.full(3, 0.1), 'ax_mps2': np.zeros(3),
        'ay_mps2': np.zeros(3), 'normal_load_n': np.ones((4, 3)),
        'body_height_m': np.zeros(3), 'body_height_rate_mps': np.zeros(3),
        'roll_rad': np.zeros(3), 'roll_rate_radps': np.zeros(3),
        'pitch_rad': np.zeros(3), 'pitch_rate_radps': np.zeros(3),
    }
    checkpoint = tmp_path / 'seed.npz'
    write_optimizer_seed_checkpoint(
        candidate, checkpoint, track=track, map_path=map_path,
        centerline_path=centerline_path, model=model)
    loaded, station = load_optimizer_seed_checkpoint(
        checkpoint, track=track, map_path=map_path,
        centerline_path=centerline_path, model=model)
    assert np.array_equal(station, track.s)
    assert set(loaded) == {
        'ey_m', 'epsi_rad', 'u_mps', 'v_mps', 'yaw_rate_radps',
        'wheel_omega_radps', 'steering_rad', 'steering_rate_radps',
        'throttle_norm', 'ax_mps2', 'ay_mps2', 'normal_load_n',
        'body_height_m', 'body_height_rate_mps', 'roll_rad',
        'roll_rate_radps', 'pitch_rad', 'pitch_rate_radps'}
    assert np.array_equal(loaded['u_mps'], candidate['speed_mps'])
    assert np.array_equal(loaded['v_mps'], candidate['lateral_speed_mps'])
    mismatched_model = SimpleNamespace(
        metadata=lambda: {'vehicle': 'other-practice-v1'})
    try:
        load_optimizer_seed_checkpoint(
            checkpoint, track=track, map_path=map_path,
            centerline_path=centerline_path, model=mismatched_model)
    except ValueError as error:
        assert 'model_sha256 mismatch' in str(error)
    else:
        raise AssertionError('accepted checkpoint from a different vehicle model')


def test_optimizer_checkpoint_mesh_resampling_is_explicit_and_provenance_bound(
        tmp_path):
    source_track = SimpleNamespace(
        count=3, length=9.0, s=np.array([0.0, 3.0, 6.0]))
    target_track = SimpleNamespace(
        count=5, length=9.0, s=np.array([0.0, 1.8, 3.6, 5.4, 7.2]))
    model = SimpleNamespace(metadata=lambda: {'vehicle': 'practice-v1'})
    map_path = tmp_path / 'map.pgm'
    centerline_path = tmp_path / 'centerline.csv'
    map_path.write_text('map fingerprint')
    centerline_path.write_text('line fingerprint')
    candidate = {
        'ey_m': np.zeros(3), 'epsi_rad': np.zeros(3),
        'speed_mps': np.ones(3), 'lateral_speed_mps': np.zeros(3),
        'yaw_rate_radps': np.zeros(3),
        'wheel_omega_radps': np.ones((4, 3)),
        'steering_rad': np.zeros(3), 'steering_rate_radps': np.zeros(3),
        'throttle_norm': np.full(3, 0.1), 'ax_mps2': np.zeros(3),
        'ay_mps2': np.zeros(3), 'normal_load_n': np.ones((4, 3)),
        'body_height_m': np.zeros(3), 'body_height_rate_mps': np.zeros(3),
        'roll_rad': np.zeros(3), 'roll_rate_radps': np.zeros(3),
        'pitch_rad': np.zeros(3), 'pitch_rate_radps': np.zeros(3),
    }
    checkpoint = tmp_path / 'seed.npz'
    write_optimizer_seed_checkpoint(
        candidate, checkpoint, track=source_track, map_path=map_path,
        centerline_path=centerline_path, model=model)
    try:
        load_optimizer_seed_checkpoint(
            checkpoint, track=target_track, map_path=map_path,
            centerline_path=centerline_path, model=model)
    except ValueError as error:
        assert 'mesh does not match' in str(error)
    else:
        raise AssertionError('mesh resampling was allowed without opt-in')

    loaded, source_station = load_optimizer_seed_checkpoint(
        checkpoint, track=target_track, map_path=map_path,
        centerline_path=centerline_path, model=model,
        allow_mesh_resample=True)
    assert np.array_equal(source_station, source_track.s)
    assert loaded['u_mps'].shape == source_track.s.shape
    assert loaded['wheel_omega_radps'].shape == (4, source_track.count)


def test_objective_phase1_defers_dynamic_rows_and_keeps_exact_gate():
    optimizer, _, _, root, provenance = load_modules(
        ROOT, PACKAGE, allow_fetch=False,
        phase1_hessian="limited-memory", phase1_formulation="objective")
    source_path = Path(optimizer.__file__).resolve()
    assert root.resolve() in source_path.parents
    source = source_path.read_text(encoding="utf-8")
    compile(source, str(source_path), "exec")
    assert "    report = {" in source
    assert "    candidate = {" in source
    assert "report['constraint_diagnostics']" in source
    assert "'solver': solver_report" in source
    assert "optimized_brake_nodes" not in source
    assert "brake_mode_seed_source" not in source
    solve = next(node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "solve_bottom_up")
    time_loop = next(node for node in ast.walk(solve)
                     if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name)
                     and node.target.id == "trust_factor")
    assert not any(isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "solver"
                   for node in ast.walk(time_loop))
    assert "IPOPT time-continuation solver setup (one-time)" in source
    assert "IPOPT time-stage result:" in source
    assert "candidate['x_m']" in source
    assert "candidate['speed_mps']" in source
    assert 'initial_speed_seed_mps: np.ndarray | None=None' in source
    assert 'seed_u = np.clip(reference_speed, min_speed_mps, max_speed_mps)' in source
    assert 'reference speed seed must be positive, finite, shape (n,)' in source
    assert "'brake_mode':" not in source
    recovered = restore(ROOT, PACKAGE / "_restored_git_history", False)
    elastic_root, _ = prepare_working_solver_v251(
        recovered, PACKAGE, LEGACY, "limited-memory", "elastic")
    elastic_path = (elastic_root / LEGACY /
                    "f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py")
    elastic_source = elastic_path.read_text(encoding="utf-8")
    # Compare physical constraints after all formulation-independent geometry
    # and fixed-mesh substitutions, rather than re-transforming final source.

    def physical_constraint_calls(text):
        tree = ast.parse(text)
        return [ast.dump(node, include_attributes=False)
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "add_constraint"]

    assert physical_constraint_calls(source) == [
        row for row in physical_constraint_calls(elastic_source)
        if "captured_motion_trust_region" not in row]

    assert provenance["candidate"] == "v2.5.6-positive-drive-algebraic-source-state"
    assert provenance["phase1_formulation"] == "objective"
    assert provenance["phase1_dynamic_constraints_deferred_until_certified"] is True
    assert provenance["phase1_physics_equations_unchanged"] is True
    assert "_working_solver_v251_limited_memory_objective_only" in str(root)

    assert "elastic = opti.variable(residual.size1(), residual.size2())" not in source
    assert "opti.subject_to(residual - elastic == 0)" not in source
    assert "elastic_vars.append(elastic)" not in source
    assert "ca.vec(residual / scale)" in source
    assert "opti.bounded(-elastic_cap, residual, elastic_cap)" not in source
    assert "phase1_dynamic_labels.extend(labels)" in source
    assert "Phase-I residual gate passed; restoring exact vehicle equalities" in source
    assert "opti.subject_to(residual == 0)" in source
    assert "Phase-I objective-only NLP dimensions: variables={int(opti.nx)}, constraints={int(opti.ng)}, dynamic_residuals={dynamic_residuals}" in source
    assert "status={checkpoint_status}" in source
    assert "hard_feasible_trials={checkpoint_rejection_diagnostics[\"hard_feasible_trials\"]}" in source
    assert "min_hard_violation={checkpoint_rejection_diagnostics[\"min_hard_violation\"]}" in source
    assert "worst_hard_constraint={checkpoint_rejection_diagnostics[\"worst_hard_constraint\"]}" in source
    assert "feasible_no_descent" in source
    assert "hard_constraint_reject" in source
    assert "phase1_objective=objective" in source
    assert "reference_objective = None if reference_measure is None else reference_measure.get('phase1_objective')" in source
    assert "if np.isfinite(objective) and np.isfinite(merit) and np.isfinite(defect) and (reference_objective is None or objective < reference_objective - 1e-08)" in source
    assert "ipopt_trial_objective={checkpoint_rejection_diagnostics[\"ipopt_trial_objective\"]}" in source
    assert "ipopt_trial_hard_violation={checkpoint_rejection_diagnostics[\"ipopt_trial_hard_violation\"]}" in source
    assert "opti.minimize(elastic_merit)" in source
    assert "opti.minimize(elastic_merit + 0.0001 * feasibility_objective)" not in source
    assert "add_constraint(feasibility_objective <= trust_region_bound, 'captured_motion_trust_region')" not in source
    assert "opti.subject_to(feasibility_objective <= trust_region_bound)" in source
    assert 'constraint_labels.append(\'captured_motion_trust_region\')' in source

    for label in (
            "periodic_collocation_state_", "physx_implicit_wheel_update_",
            "longitudinal_force_balance_", "lateral_force_balance_",
            "unity_spring_damper_load_", "yaw_moment_balance_",
            "heave_force_balance_", "roll_moment_balance_",
            "pitch_moment_balance_"):
        assert label in source


def test_source_equalities_phase1_uses_exact_dynamics_and_seed_distance():
    optimizer, _, _, root, provenance = load_modules(
        ROOT, PACKAGE, allow_fetch=False,
        phase1_hessian="limited-memory", phase1_formulation="source-equalities")
    strict_seed_gate = optimizer.source_seed_is_strictly_feasible
    assert strict_seed_gate(True, 2.8e-8, 2.8e-8, 1.0e-3)
    assert not strict_seed_gate(False, 0.0, 0.0, 1.0e-3)
    assert not strict_seed_gate(True, 1.1e-3, 0.0, 1.0e-3)
    assert not strict_seed_gate(True, 0.0, 1.1e-3, 1.0e-3)
    assert not strict_seed_gate(True, float("nan"), 0.0, 1.0e-3)
    source_path = Path(optimizer.__file__).resolve()
    source = source_path.read_text(encoding="utf-8")
    compile(source, str(source_path), "exec")
    assert provenance["generated_solver_source_sha256"] == hashlib.sha256(
        source.encode("utf-8")).hexdigest()
    assert provenance["generated_solver_build_fingerprint"] in str(source_path)
    assert "    report = {" in source
    assert "    candidate = {" in source
    assert "report['constraint_diagnostics']" in source
    assert "'solver': solver_report" in source
    assert "optimized_brake_nodes" not in source
    assert "brake_mode_seed_source" not in source
    solve = next(node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "solve_bottom_up")
    time_loop = next(node for node in ast.walk(solve)
                     if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name)
                     and node.target.id == "trust_factor")
    assert not any(isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute)
                   and node.func.attr == "solver"
                   for node in ast.walk(time_loop))
    assert "IPOPT time-continuation solver setup (one-time)" in source
    assert "IPOPT time-stage result:" in source
    assert "candidate['x_m']" in source
    assert "candidate['speed_mps']" in source
    assert "'brake_mode':" not in source
    assert "opti.subject_to(residual == 0)" in source
    assert "phase1_equalities_in_nlp = True" in source
    assert "nonlocal phase1_equalities_in_nlp" in source
    assert "stage1_solution = None" in source
    assert "initial_is_feasible = source_seed_is_strictly_feasible(" in source
    assert "Phase-I seed reuse gate:" in source
    seed_audit_site = source.index(
        "Phase-I source-seed constraint audit begins")
    seed_gate_site = source.index(
        "initial_is_feasible = source_seed_is_strictly_feasible(")
    assert seed_audit_site < seed_gate_site
    assert "diagnostic_deferred_to_phase1_measure" not in source
    assert "_constraint_diagnostics(opti.value(opti.g, opti.initial())" in (
        source[:seed_gate_site])
    initial_measure_site = source.index("initial_measure = phase1_measure()")
    phase1_diagnostic_block = source[initial_measure_site:source.index(
        "Phase-I numeric seed diagnostics ready", initial_measure_site)]
    assert "if initial_measure.get('max_constraint_violation') is not None:" in phase1_diagnostic_block
    assert "initial_constraints['max_constraint_violation'] = initial_measure.get('max_constraint_violation')" in phase1_diagnostic_block
    assert "Phase-I skipped: source seed passes exact dynamics and constraints" in source
    assert "opti.minimize(feasibility_objective)" in source
    assert "opti.minimize(elastic_merit)" not in source
    assert "captured_motion_trust_region" not in source.split(
        "# First find a model-consistent lap", 1)[-1].split(
        "Phase-I source-equality NLP dimensions", 1)[0]
    assert provenance["phase1_source_dynamics_are_exact_equalities"] is True
    assert provenance["phase1_objective_is_normalized_seed_displacement"] is True
    assert provenance["time_trust_region_geometry"] == (
        "normalized_component_box_with_aggregate_l2_equivalent_radius")
    assert provenance["unity_source_fixed_step_s"] == 0.001
    assert provenance["unity_source_fixed_step_matches_production"] is True
    assert provenance["unity_source_fixed_step_is_optimizer_time_grid"] is False
    assert "fixed_step_s" not in provenance
    assert "_working_solver_v251_limited_memory_source_equalities" in str(root)
    assert source.count(
        "ay_residual_mps2 = acceleration[1] - source_acceleration[1]") == 1

    elastic_root, _ = prepare_working_solver_v251(
        restore(ROOT, PACKAGE / "_restored_git_history", False),
        PACKAGE, LEGACY, "limited-memory", "elastic")
    elastic_path = (elastic_root / LEGACY /
                    "f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py")
    generated = elastic_path.read_text(encoding="utf-8")
    # Both formulation routes must retain identical coordinate definitions.
    # Runtime caps and time-stage capture imports are installed separately.
    for name in ('_periodic_interpolants', '_path_geometry_at'):
        def definition(text):
            return next(node for node in ast.parse(text).body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(definition(source)) == ast.dump(definition(generated))


def test_source_phase1_accepts_feasibility_without_claiming_time_optimality():
    source = '''def options():
    phase1_ipopt_options = {}
    phase1_max_cpu_time_s = float('0')
    return phase1_ipopt_options
'''
    scope = {'NLP_FEASIBILITY_TOLERANCE': .001}
    exec(stop_source_phase1_when_feasible(source), scope)
    options = scope['options']()
    assert options['acceptable_constr_viol_tol'] == .0001
    # Verify option names and IPOPT's actual unscaled constraint termination,
    # not merely the generated source text. This is a tiny deterministic NLP.
    import casadi as ca
    x = ca.MX.sym('phase1_scalar')
    solver = ca.nlpsol('phase1_feasibility_options', 'ipopt',
        {'x':x, 'f':(x-100)**2, 'g':x*x-1},
        {'print_time':False, 'ipopt':{**options,'print_level':0,'max_iter':200}})
    result = solver(x0=2, lbg=0, ubg=0)
    assert abs(float(result['g'])) <= options['acceptable_constr_viol_tol']
    assert solver.stats()['return_status'] == 'Solved_To_Acceptable_Level'
    assert solver.stats()['iter_count'] < 10


def test_feasible_checkpoint_saved_before_time_solver_setup(tmp_path, monkeypatch):
    optimizer, *_ = load_modules(ROOT, PACKAGE,
                                  phase1_formulation='source-equalities')
    source = Path(optimizer.__file__).read_text()
    assert source.index("'phase1_feasible_raw_iterate.npz'") < source.index(
        "IPOPT time-continuation solver setup (one-time)")
    tree = ast.parse(source)
    capture = next(node for node in ast.walk(tree)
        if isinstance(node, ast.If) and any(
            isinstance(child, ast.Assign) and
            any(isinstance(target,ast.Name) and target.id=='capture_file'
                for target in child.targets) and
            'phase1_feasible_raw_iterate.npz' in ast.unparse(child)
            for child in node.body))
    values = {'x':np.arange(48.).reshape(16,3), 'qdelta':np.ones(3)*.2,
              'throttle':np.ones(3)*.15, 'phase_lengths':np.array([9.]),
              'lap_time':np.array(5.1)}
    def selected_values(solution, stats, variable):
        assert solution is None  # Selected certified snapshot, not raw result.
        assert stats['selected'] is True
        return values[variable]
    scope = {'np':np, 'os':os, 'Path':Path, 'json':json,
             'capture_dir':str(tmp_path),
             'stage_values':selected_values, 'stage1_solution':None,
             'stage1_stats':{'selected':True},
             'stage1_diagnostics':{'max_constraint_violation':2.e-5},
             'track':SimpleNamespace(s=np.array([0.,3.,6.]))}
    scope.update({key:key for key in values})
    monkeypatch.setenv('SDU_IPOPT_STAGE_CAPTURE_DIR',str(tmp_path))
    exec(compile(ast.Module(body=capture.body,type_ignores=[]),
                 '<phase1-checkpoint>', 'exec'), scope)
    with np.load(tmp_path/'phase1_feasible_raw_iterate.npz') as saved:
        np.testing.assert_array_equal(saved['state_x'],values['x'])
        assert float(saved['max_constraint_violation']) == 2.e-5
        assert float(saved['predicted_lap_time_s']) == 5.1
    assert not list(tmp_path.glob('*.tmp.npz'))
    # Derived loads/accelerations are not independent trust-box coordinates.
    assert 'time_reference_load' not in source
    assert 'time_reference_accel_xy' not in source
