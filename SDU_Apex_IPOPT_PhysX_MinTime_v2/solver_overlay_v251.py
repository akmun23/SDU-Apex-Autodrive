"""Source-equation corrections for a spatial minimum-time IPOPT model.

The OCP optimizes a path and speed profile over distance. It does not impose
the runtime MPC's 25 ms command hold or use a 1 kHz optimizer grid. Unity's
1 ms step appears only in source-derived discrete physics maps (notably wheel
spin), which are composed across each variable-duration spatial segment. The
spatial collocation still is not Unity's native scene/contact solver.
"""
from __future__ import annotations

import ast
import hashlib
import math
import re
import shutil
import textwrap
from pathlib import Path

from solver_overlay import (
    prepare_working_solver as prepare_v249,
)
from vehicle_mechanics_overlay_v250 import prepare_working_mechanics
from wheel_model_overlay_v251 import prepare_working_wheel_model
from path_frame import install_exact_path_frame, SOURCE_REAR_AXLE_FROM_COM_M


UNITY_SOURCE_FIXED_STEP_S = 0.001
UNITY_SOURCE_ANGULAR_DAMPING_PER_S = 0.1
UNITY_MPC_THROTTLE_MIN_DRIVE = 0.001


_STATE_MARKER = '''WHEEL_STATE_NAMES = (
    "wheel_omega_fl_radps", "wheel_omega_fr_radps",
    "wheel_omega_rl_radps", "wheel_omega_rr_radps",
)'''
_IMPORT_OLD = '''    physx_no_drive_wheel_speed_interval_casadi,
)'''
_IMPORT_NEW = '''    physx_no_drive_wheel_speed_interval_casadi,
    UNITY_PHYSICS_GRAVITY_MPS2,
)'''
_HELPERS = '''

UNITY_SOURCE_FIXED_STEP_S = 0.001
UNITY_SOURCE_ANGULAR_DAMPING_PER_S = 0.1
UNITY_MPC_THROTTLE_MIN_DRIVE = 0.001


def _unity_equivalent_damping(rate_per_s: float) -> float:
    """Equivalent rate of Unity's 1ms damping recurrence; not an OCP timestep."""
    return -math.log1p(-rate_per_s * UNITY_SOURCE_FIXED_STEP_S) / \\
        UNITY_SOURCE_FIXED_STEP_S


def _unity_euler_rates_casadi(roll, pitch, p, q, r):
    """Unity ZYX Euler rates from body angular velocity p/q/r."""
    sr, cr = ca.sin(roll), ca.cos(roll)
    sp, cp = ca.sin(pitch), ca.cos(pitch)
    roll_dot = p + (sp / cp) * (sr * q + cr * r)
    pitch_dot = cr * q - sr * r
    yaw_dot = (sr * q + cr * r) / cp
    return roll_dot, pitch_dot, yaw_dot


def _unity_body_vertical_speed_casadi(body_height_rate, u, v, roll, pitch):
    sr, cr = ca.sin(roll), ca.cos(roll)
    sp, cp = ca.sin(pitch), ca.cos(pitch)
    return (body_height_rate + sp * u - cp * sr * v) / (cp * cr)


def _unity_horizontal_velocity_casadi(u, v, body_vertical, roll, pitch):
    sr, cr = ca.sin(roll), ca.cos(roll)
    sp, cp = ca.sin(pitch), ca.cos(pitch)
    return (cp * u + sp * (sr * v + cr * body_vertical),
            cr * v - sr * body_vertical)


def _unity_gravity_body_casadi(roll, pitch):
    sr, cr = ca.sin(roll), ca.cos(roll)
    sp, cp = ca.sin(pitch), ca.cos(pitch)
    gravity = UNITY_PHYSICS_GRAVITY_MPS2
    return (gravity * sp, -gravity * cp * sr, -gravity * cp * cr)


def _unity_suspension_loads_numpy(model, height, height_rate, roll, p,
                                  pitch, q, r):
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    roll_dot = p + (sp / cp) * (sr * q + cr * r)
    pitch_dot = cr * q - sr * r
    return suspension_loads_numpy(
        model, height, height_rate, roll, roll_dot, pitch, pitch_dot)


def _unity_body_acceleration_numpy(model, fx, fy, u, v, yaw_rate,
                                   body_height_rate, roll, roll_rate,
                                   pitch, pitch_rate):
    """Source-equation body acceleration, evaluated independently for reports."""
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    w = (body_height_rate + sp * u - cp * sr * v) / (cp * cr)
    gravity_x = UNITY_PHYSICS_GRAVITY_MPS2 * sp
    gravity_y = -UNITY_PHYSICS_GRAVITY_MPS2 * cp * sr
    damping = _unity_equivalent_damping(model.drag_per_s)
    ax = (fx / model.mass_kg + gravity_x - damping * u
          - pitch_rate * w + yaw_rate * v)
    ay = (fy / model.mass_kg + gravity_y - damping * v
          + roll_rate * w - yaw_rate * u)
    return ax, ay
'''


def make_componentwise_time_trust_region(source: str) -> str:
    """Replace the fragile aggregate L2 cap with normalized component bounds.

    The aggregate squared-distance constraint was repeatedly violated by
    large factors during minimum-time IPOPT steps. Bounding each normalized
    decision/algebraic component prevents a single state from taking an
    arbitrarily large step. The box is centered on the *certified Phase-I
    trajectory*, not the original seed: Phase I may need to move a resampled
    seed substantially before time descent is allowed. Its radius is chosen
    from a per-node normalized budget, so refining the spatial mesh does not
    inadvertently freeze geometry. Old aggregate geometry is audit-only.
    """
    old = (
        "        opti.set_value(trust_region_bound, feasibility_reference * "
        "time_trust_region_factors[0])\n"
        "        opti.subject_to(feasibility_objective <= trust_region_bound)\n"
        "        constraint_labels.append('captured_motion_trust_region')\n"
        "        stage2_initial_diagnostics = _constraint_diagnostics("
        "opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), "
        "opti.value(opti.ubg, opti.initial()), constraint_labels)"
    )
    new = (
        "        opti.set_value(trust_region_bound, feasibility_reference * "
        "time_trust_region_factors[0])\n"
        "        time_reference_x = np.asarray(stage_values("
        "stage1_solution, stage1_stats, x), dtype=np.float64)\n"
        "        time_reference_qdelta = np.asarray(stage_values("
        "stage1_solution, stage1_stats, qdelta), dtype=np.float64)\n"
        "        time_reference_throttle = np.asarray(stage_values("
        "stage1_solution, stage1_stats, throttle), dtype=np.float64)\n"
        "        time_reference_phase_lengths = np.asarray(stage_values("
        "stage1_solution, stage1_stats, phase_lengths), dtype=np.float64)\n"
        "        time_reference_load = np.asarray(stage_values("
        "stage1_solution, stage1_stats, normal_load), dtype=np.float64)\n"
        "        time_reference_accel_xy = np.asarray(stage_values("
        "stage1_solution, stage1_stats, accel_xy), dtype=np.float64)\n"
        "        time_trust_region_deviation = ca.vertcat(\n"
        "            ca.vec((x - ca.DM(time_reference_x)) / state_scales),\n"
        "            (qdelta - ca.DM(time_reference_qdelta)) / 3.2,\n"
        "            throttle - ca.DM(time_reference_throttle),\n"
        "            (phase_lengths - ca.DM(time_reference_phase_lengths)) / "
        "ca.DM(np.maximum(time_reference_phase_lengths, 1.0)),\n"
        "            ca.vec((normal_load - ca.DM(time_reference_load)) / 10.0),\n"
        "            ca.vec((accel_xy - ca.DM(time_reference_accel_xy)) / 10.0))\n"
        "        time_trust_region_component_limit = trust_component_radius(\n"
        "            trust_region_bound, int(time_trust_region_deviation.numel()), n)\n"
        "        opti.subject_to(opti.bounded(-time_trust_region_component_limit, "
        "time_trust_region_deviation, time_trust_region_component_limit))\n"
        "        constraint_labels.extend(['captured_motion_trust_region'] * "
        "int(time_trust_region_deviation.numel()))\n"
        "        stage2_initial_diagnostics = _constraint_diagnostics("
        "opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), "
        "opti.value(opti.ubg, opti.initial()), constraint_labels)"
    )
    if source.count(old) != 1:
        raise RuntimeError(
            "Cannot safely replace the aggregate minimum-time trust region")
    source = source.replace(old, new, 1)
    old_report = "'feasibility_trust_region': feasibility_trust_region, 'ipopt_iteration_budgets':"
    new_report = (
        "'feasibility_trust_region': feasibility_trust_region, "
        "'time_trust_region_geometry': os.environ.get('SDU_IPOPT_TRUST_GEOMETRY', 'node-mean'), "
        "'ipopt_iteration_budgets':")
    if source.count(old_report) != 1:
        raise RuntimeError(
            "Cannot annotate the minimum-time trust-region formulation")
    return source.replace(old_report, new_report, 1)


def add_exploratory_time_stage_capture(source: str) -> str:
    """Optionally retain a failed time-stage iterate for Phase-I reseeding.

    This archive is deliberately separate from the source-feasible optimizer
    checkpoint. It is an exploratory NLP iterate, even when its lap-time
    objective is attractive. The normal candidate selection is unchanged.
    """
    import_anchor = "from pathlib import Path\n"
    if source.count(import_anchor) != 1:
        raise RuntimeError("Cannot add exploratory IPOPT capture import")
    source = source.replace(import_anchor, import_anchor + "import os\n", 1)
    stage_anchor = (
        "            print('IPOPT time-stage result: ' + "
        "json.dumps(stage_reports[-1], sort_keys=True, default=str), "
        "flush=True)\n")
    if source.count(stage_anchor) != 1:
        raise RuntimeError("Cannot locate IPOPT time-stage result capture point")
    capture = '''            capture_dir = os.environ.get('SDU_IPOPT_STAGE_CAPTURE_DIR')
            if capture_dir:
                capture_root = Path(capture_dir).resolve()
                capture_root.mkdir(parents=True, exist_ok=True)
                capture_file = capture_root / f'factor_{trust_factor:g}_raw_iterate.npz'
                capture_payload = {
                    'schema': np.asarray(1, dtype=np.int64),
                    'sample_s': np.asarray(track.s, dtype=np.float64),
                    'state_x': np.asarray(stage_values(
                        continuation_solution, continuation_stats, x),
                        dtype=np.float64),
                    'steering_rate_radps': np.asarray(stage_values(
                        continuation_solution, continuation_stats, qdelta),
                        dtype=np.float64).reshape(-1),
                    'throttle_norm': np.asarray(stage_values(
                        continuation_solution, continuation_stats, throttle),
                        dtype=np.float64).reshape(-1),
                    'phase_lengths_m': np.asarray(stage_values(
                        continuation_solution, continuation_stats, phase_lengths),
                        dtype=np.float64).reshape(-1),
                    'trust_factor': np.asarray(float(trust_factor)),
                    'predicted_lap_time_s': np.asarray(continuation_time),
                    'max_constraint_violation': np.asarray(
                        float(continuation_violation)
                        if continuation_violation is not None else np.nan),
                }
                if any(not np.isfinite(capture_payload[name]).all()
                       for name in ('sample_s', 'state_x',
                                    'steering_rate_radps', 'throttle_norm',
                                    'phase_lengths_m', 'trust_factor',
                                    'predicted_lap_time_s')):
                    raise ValueError('Exploratory IPOPT stage has nonfinite values')
                temporary_file = capture_file.with_suffix('.tmp.npz')
                np.savez_compressed(temporary_file, **capture_payload)
                temporary_file.replace(capture_file)
                capture_report = capture_file.with_suffix('.json')
                capture_report.write_text(json.dumps(
                    stage_reports[-1], indent=2, sort_keys=True, default=str)
                    + '\\n', encoding='utf-8')
                print(f'Exploratory time-stage iterate saved: {capture_file}',
                      flush=True)
'''
    source = source.replace(stage_anchor, stage_anchor + capture, 1)
    # Save the strictly checked Phase-I state BEFORE constructing/initializing
    # the expensive time-stage solver. A later interruption must not lose the
    # only feasible initializer. Reuse exactly the same raw-state schema; the
    # next process must still re-certify it against its own source fingerprint.
    phase_anchor = "        feasibility_reference = max(float(stage_values("
    if source.count(phase_anchor) != 1:
        raise RuntimeError('Cannot locate the certified Phase-I checkpoint boundary')
    phase_capture = capture.replace(
        "f'factor_{trust_factor:g}_raw_iterate.npz'", "'phase1_feasible_raw_iterate.npz'")
    phase_capture = phase_capture.replace('continuation_solution', 'stage1_solution')
    phase_capture = phase_capture.replace('continuation_stats', 'stage1_stats')
    phase_capture = phase_capture.replace('float(trust_factor)', '0.0')
    phase_capture = phase_capture.replace(
        'np.asarray(continuation_time)',
        'np.asarray(float(stage_values(stage1_solution, stage1_stats, lap_time)))')
    phase_capture = phase_capture.replace(
        'continuation_violation', "stage1_diagnostics.get('max_constraint_violation')")
    phase_capture = phase_capture.replace('stage_reports[-1]', 'stage1_diagnostics')
    phase_capture = phase_capture.replace('Exploratory time-stage iterate saved:',
                                          'Certified Phase-I initializer saved:')
    # Existing capture is nested in a trust loop (12 spaces); this boundary
    # is nested only in the strict-feasibility branch (8 spaces).
    phase_capture = textwrap.indent(textwrap.dedent(phase_capture), '        ')
    return source.replace(phase_anchor, phase_capture + phase_anchor, 1)


def compact_source_force_evaluation(source: str) -> str:
    """Reuse derivatives of unchanged scalar source equations at all nodes."""
    tree = ast.parse(source)
    solver = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'solve_bottom_up')
    calls = [node for node in ast.walk(solver)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == 'tire_forces_casadi']
    if len(calls) != 3:
        raise RuntimeError('Expected knot, midpoint and seed force evaluation sites')
    for node in calls:
        node.func.id = 'source_force_kernel'
    solver.body.insert(0, ast.parse(
        'source_force_kernel = make_force_kernel(model, tire_forces_casadi)').body[0])
    scale_index = next(i for i, node in enumerate(solver.body)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'state_scales'
                               for target in node.targets))
    solver.body.insert(scale_index+1, ast.parse(
        'scale_solver_coordinates(opti, x, state_scales, qdelta)').body[0])
    # The same time solver is installed once and reused across trust stages.
    # Only choose its barrier strategy; never relax physics or acceptance.
    class BarrierOptions(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if any(isinstance(target, ast.Name) and target.id == 'time_ipopt_options'
                   for target in node.targets):
                self.count += 1
                return [node, ast.parse(
                    'time_ipopt_options.update(time_barrier_options())').body[0]]
            return self.generic_visit(node)

    barrier = BarrierOptions()
    barrier.visit(solver)
    if barrier.count != 1:
        raise RuntimeError('Expected one time-stage barrier options assignment')
    # Exporting thousands of scalar Opti.value calls repeatedly reconstructs
    # the large primal assignment/evaluation graph. Batch and cache the exact
    # same expressions; retain every output channel and independent audit.
    value_index = next(i for i, node in enumerate(solver.body)
                       if isinstance(node, ast.FunctionDef) and node.name == 'value')
    solver.body.insert(value_index+1, ast.parse(
        "print('IPOPT exporting audited candidate: batched node evaluation', flush=True)").body[0])
    solver.body.insert(value_index+2, ast.parse(
        'node_values = NodeValueCache(value)').body[0])
    replacements = {}
    for name, loop in (
            ('time_density', 'expr'), ('step_nodes', 'expr'),
            ('midpoint_steering_nodes', 'item'), ('midpoint_qdelta_nodes', 'item'),
            ('midpoint_throttle_nodes', 'item'), ('body_path_heading_error_nodes', 'item')):
        replacements[f'[float(value({loop})) for {loop} in {name}]'] = (
            f'node_values({name}).reshape(-1)')
    for name in ('midpoint_state_nodes', 'midpoint_force_nodes',
                 'midpoint_acceleration_nodes'):
        replacements[f'[np.asarray(value(item), dtype=np.float64).reshape(-1) for item in {name}]'] = (
            f'node_values({name}).T')
    for name in ('midpoint_wheel_omega_nodes', 'midpoint_load_nodes',
                 'midpoint_slip_x_nodes', 'midpoint_slip_y_nodes',
                 'slip_x_nodes', 'slip_y_nodes'):
        replacements[f'[[float(value({name}[k][i])) for k in range(n)] for i in range(4)]'] = (
            f'node_values({name})')
    for axis in (0, 1):
        replacements[f'[float(value(force_nodes[k][{axis}])) for k in range(n)]'] = (
            f'node_values(force_nodes)[{axis}, :]')
    replacements['[[float(value(force_nodes[k][axis])) for k in range(n)] for axis in range(3)]'] = (
        'node_values(force_nodes)[:3, :]')

    class BatchExport(ast.NodeTransformer):
        count = 0

        def visit_ListComp(self, node):
            replacement = replacements.get(ast.unparse(node))
            if replacement is not None:
                self.count += 1
                return ast.copy_location(ast.parse(replacement, mode='eval').body, node)
            return self.generic_visit(node)

    batching = BatchExport()
    batching.visit(solver)
    if batching.count != 20:
        raise RuntimeError(f'Expected 20 batched export sites, found {batching.count}')
    # Preserve the exact collocation clock for independent command checks.
    # Re-deriving it from a rounded mesh CSV can select the other side of a
    # piecewise-linear coordinate knot and is not the same objective oracle.
    candidate_index = next(i for i, node in enumerate(solver.body)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
            and target.id == 'candidate' for target in node.targets))
    solver.body[candidate_index+1:candidate_index+1] = ast.parse(
        "candidate['node_time_density_s_per_m'] = node_values(time_density).reshape(-1)\n"
        "candidate['spatial_steps_m'] = node_values(step_nodes).reshape(-1)").body
    # Keep __future__ imports first.
    index = next(i for i, node in enumerate(tree.body)
                 if not (isinstance(node, ast.ImportFrom)
                         and node.module == '__future__'))
    tree.body.insert(index, ast.ImportFrom(
        module='source_equation_kernels',
        names=[ast.alias(name='make_force_kernel'),
               ast.alias(name='NodeValueCache'),
               ast.alias(name='scale_solver_coordinates'),
               ast.alias(name='trust_component_radius'),
               ast.alias(name='time_barrier_options')], level=0))
    return ast.unparse(ast.fix_missing_locations(tree))+'\n'


def add_time_stage_progress(source: str) -> str:
    """Persist sampled source-feasible incumbents without relaxing the NLP."""
    import_anchor = 'import os\n'
    if source.count(import_anchor) != 1:
        raise RuntimeError('Cannot add time-stage progress import')
    source = source.replace(import_anchor, import_anchor +
        'from ipopt_time_progress import TimeStageProgress, choose_sampled_incumbent, primal_assignments, prefer_terminal_restart\n'
        'from ipopt_nlp_audit import maybe_audit_checkpoint\n', 1)
    time_setup_anchor = "        print('IPOPT time-continuation solver setup (one-time)', flush=True)\n"
    callback_setup = '''        active_time_progress = [None]
        def time_progress_callback(iteration):
            if active_time_progress[0] is not None:
                active_time_progress[0].observe(iteration)
        # Register once: replacing Opti callbacks inside the trust loop can
        # invalidate/rebuild the solver and its expensive compiled oracles.
        opti.callback(time_progress_callback)
'''
    if source.count(time_setup_anchor) != 1:
        raise RuntimeError('Cannot register the one-time progress dispatcher')
    source = source.replace(time_setup_anchor, callback_setup+time_setup_anchor, 1)
    solve_anchor = '''            try:
                continuation_solution = opti.solve_limited()
'''
    setup = '''            maybe_audit_checkpoint(opti, constraint_labels,
                lap_time=lap_time, factor=trust_factor,
                time_density=ca.vertcat(*time_density),
                spatial_steps=ca.vertcat(*step_nodes))
            checkpoint_initial = opti.initial()
            checkpoint_parameters = np.asarray(opti.value(opti.p, checkpoint_initial))
            checkpoint_lower = np.asarray(opti.value(opti.lbg, checkpoint_initial))
            checkpoint_upper = np.asarray(opti.value(opti.ubg, checkpoint_initial))
            checkpoint_evaluate = ca.Function(f'time_checkpoint_{id(opti)}',
                [opti.x, opti.p], [opti.g, lap_time, x, qdelta, throttle, phase_lengths])
            def time_snapshot():
                flat = stage_values(None, {}, opti.x).reshape(-1)
                g, sampled_time, sampled_x, sampled_qdelta, sampled_q, sampled_lengths = checkpoint_evaluate(
                    flat, checkpoint_parameters)
                diagnostic = _constraint_diagnostics(
                    np.asarray(g), checkpoint_lower, checkpoint_upper, constraint_labels)
                return {
                    'sample_s': np.asarray(track.s, dtype=np.float64),
                    'state_x': np.asarray(sampled_x),
                    'steering_rate_radps': np.asarray(sampled_qdelta).reshape(-1),
                    'throttle_norm': np.asarray(sampled_q).reshape(-1),
                    'phase_lengths_m': np.asarray(sampled_lengths).reshape(-1),
                    'decision_x': flat,
                    'trust_geometry': np.asarray(os.environ.get(
                        'SDU_IPOPT_TRUST_GEOMETRY', 'node-mean')),
                    'constraint_duals': stage_values(None, {}, opti.lam_g).reshape(-1),
                    'predicted_lap_time_s': float(sampled_time),
                    'max_constraint_violation': diagnostic.get('max_constraint_violation', np.nan),
                }
            time_progress = TimeStageProgress(
                os.environ.get('SDU_IPOPT_STAGE_CAPTURE_DIR'), trust_factor,
                NLP_FEASIBILITY_TOLERANCE, time_snapshot,
                stride=int(os.environ.get('SDU_IPOPT_TIME_CHECKPOINT_STRIDE', '25')),
                interval_s=float(os.environ.get('SDU_IPOPT_TIME_CHECKPOINT_SECONDS', '120')))
            active_time_progress[0] = time_progress
'''
    if source.count(solve_anchor) != 1:
        raise RuntimeError('Cannot locate time-stage solve for progress capture')
    source = source.replace(solve_anchor, setup+solve_anchor, 1)
    reuse_anchor = '    phase1_elastic_reports = []\n'
    if source.count(reuse_anchor) != 1:
        raise RuntimeError('Cannot guard no-solve audit Phase-I seed reuse')
    source = source.replace(reuse_anchor,
        "    if os.environ.get('SDU_IPOPT_AUDIT_CHECKPOINT') and not initial_is_feasible:\n"
        "        raise ValueError('NLP audit requires an already source-feasible original seed; no solve allowed')\n"
        + reuse_anchor, 1)
    stats_anchor = '            continuation_stats = opti.stats()\n'
    stats_with_failure_context = '''            try:
                continuation_stats = opti.stats()
            except RuntimeError as stats_error:
                raise RuntimeError('IPOPT time-stage initialization failed: '
                    + str(continuation_error)) from stats_error
            terminal_solution = continuation_solution
            terminal_stats = continuation_stats
            time_progress.finish()
'''
    if source.count(stats_anchor) != 1:
        raise RuntimeError('Cannot preserve time-stage initialization error context')
    source = source.replace(stats_anchor, stats_with_failure_context, 1)
    violation_anchor = "            continuation_violation = continuation_diagnostics.get('max_constraint_violation')\n"
    selection = '''            terminal_time = continuation_time
            terminal_violation = continuation_diagnostics.get('max_constraint_violation')
            terminal_status = continuation_stats.get('return_status')
            sampled_incumbent_selected = choose_sampled_incumbent(
                time_progress.best, terminal_time, terminal_violation,
                NLP_FEASIBILITY_TOLERANCE)
            if sampled_incumbent_selected:
                # Preserve the exact sampled primal. Do not mutate the solver
                # or label this earlier point as IPOPT-converged.
                continuation_solution = None
                continuation_stats = {
                    **continuation_stats, 'success': False,
                    'return_status': 'Sampled_Feasible_Incumbent',
                    '_initial_snapshot': primal_assignments(opti, time_progress.best['decision_x'])}
                continuation_diagnostics = stage_constraint_diagnostics(
                    continuation_solution, continuation_stats)
                continuation_time = float(stage_values(
                    continuation_solution, continuation_stats, lap_time))
            terminal_restart_selected = prefer_terminal_restart(
                sampled_incumbent_selected, terminal_time, terminal_violation,
                continuation_time, NEAR_FEASIBLE_WARM_START_MAX_VIOLATION)
'''
    if source.count(violation_anchor) != 1:
        raise RuntimeError('Cannot locate time-stage incumbent selection')
    source = source.replace(violation_anchor, selection+violation_anchor, 1)
    report_anchor = "            print('IPOPT time-stage result: ' + json.dumps(stage_reports[-1], sort_keys=True, default=str), flush=True)\n"
    report = '''            stage_reports[-1]['time_checkpointing'] = {
                'sample_count': time_progress.sample_count,
                'callback_wall_s': time_progress.callback_wall_s,
                'callback_errors': time_progress.errors,
                'sampled_incumbent_selected': sampled_incumbent_selected,
                'selected_sample_iteration': (int(time_progress.best['iteration'])
                    if sampled_incumbent_selected else None),
                'terminal_return_status': terminal_status,
                'terminal_restart_selected': terminal_restart_selected,
                'terminal_objective_lap_time_s': terminal_time,
                'terminal_max_constraint_violation': terminal_violation}
            stage_reports[-1]['evaluation_timing'] = {
                key: value for key, value in continuation_stats.items()
                if key.startswith(('n_call_', 't_wall_', 't_proc_'))
                and isinstance(value, (int, float)) and math.isfinite(value)}
'''
    if source.count(report_anchor) != 1:
        raise RuntimeError('Cannot locate time-stage progress report')
    source = source.replace(report_anchor, report+report_anchor, 1)
    restart_anchor = '''            if continuation_restartable:
                restart_solution = continuation_solution
                restart_stats = continuation_stats
'''
    restart = '''            if terminal_restart_selected:
                restart_solution, restart_stats = terminal_solution, terminal_stats
            elif continuation_restartable:
                restart_solution = continuation_solution
                restart_stats = continuation_stats
'''
    if source.count(restart_anchor) != 1:
        raise RuntimeError('Cannot separate feasible incumbent from near-feasible restart')
    source = source.replace(restart_anchor, restart, 1)
    continuation_field = "'continued_after_failed_time_stage': retry_at_larger_radius"
    if source.count(continuation_field) != 1:
        raise RuntimeError('Cannot clarify time-stage continuation status')
    return source.replace(continuation_field,
        "'advance_to_next_trust_stage': (trust_factor < trust_region_factors[-1] "
        "and (continuation_feasible or retry_at_larger_radius))", 1)


def stop_source_phase1_when_feasible(source: str) -> str:
    """Phase I seeks feasibility, not stationarity of a seed-distance cost.

    IPOPT's acceptable constraint check is unscaled and 10x tighter than the
    subsequent independent source gate. Large dual/complementarity tolerances
    apply ONLY to Phase I. Minimum-time optimization keeps its original KKT
    convergence criteria and every physical constraint remains in the NLP.
    """
    anchor = '    phase1_max_cpu_time_s = float('
    if source.count(anchor) != 1:
        raise RuntimeError('Cannot locate source Phase-I IPOPT options')
    options = '''    phase1_ipopt_options.update({
        'acceptable_iter': 1,
        'acceptable_tol': 1.e10,
        'acceptable_constr_viol_tol': NLP_FEASIBILITY_TOLERANCE * .1,
        'acceptable_dual_inf_tol': 1.e10,
        'acceptable_compl_inf_tol': 1.e10,
        'acceptable_obj_change_tol': 1.e10,
    })
'''
    return source.replace(anchor, options + anchor, 1)

_PATH_OLD = '''        sdot_unclipped = ((u[k] * ca.cos(epsi[k]) - v[k] * ca.sin(epsi[k]))
                          / ca.fmax(den, 0.20))'''
_PATH_NEW = '''        body_vertical = _unity_body_vertical_speed_casadi(
            body_height_rate[k], u[k], v[k], roll[k], pitch[k])
        horizontal_u, horizontal_v = _unity_horizontal_velocity_casadi(
            u[k], v[k], body_vertical, roll[k], pitch[k])
        path_tangent_speed = (horizontal_u * ca.cos(epsi[k])
                              - horizontal_v * ca.sin(epsi[k]))
        path_lateral_speed = (horizontal_u * ca.sin(epsi[k])
                              + horizontal_v * ca.cos(epsi[k]))
        sdot_unclipped = path_tangent_speed / ca.fmax(den, 0.20)'''

_TIRE_CALL_OLD = '''        fx, fy, mz, fx_w, _, slip_x, slip_y = tire_forces_casadi(
            model, u[k], v[k], yaw_rate[k], steering[k],
            [wheel_omega[i][k] for i in range(4)], fz,
            roll_rate[k], pitch_rate[k])'''
_TIRE_CALL_NEW = '''        (fx, fy, fz_body, mx, my, mz, fx_w, _, slip_x,
         slip_y) = tire_forces_casadi(
            model, u[k], v[k], yaw_rate[k], steering[k],
            [wheel_omega[i][k] for i in range(4)], fz,
            body_height[k], body_height_rate[k], roll[k], roll_rate[k],
            pitch[k], pitch_rate[k])'''

_FORCE_BALANCE_OLD = '''        add_constraint(ax - (fx / model.mass_kg
                             - model.drag_per_s * u[k]) == 0,
                       f"longitudinal_force_balance_k{k}")
        add_constraint(ay - (fy / model.mass_kg
                             - model.drag_per_s * v[k]) == 0,
                       f"lateral_force_balance_k{k}")'''
_FORCE_BALANCE_NEW = '''        linear_damping = _unity_equivalent_damping(model.drag_per_s)
        angular_damping = _unity_equivalent_damping(
            UNITY_SOURCE_ANGULAR_DAMPING_PER_S)
        body_vertical = _unity_body_vertical_speed_casadi(
            body_height_rate[k], u[k], v[k], roll[k], pitch[k])
        gravity_body = _unity_gravity_body_casadi(roll[k], pitch[k])
        # Newton in the rotating body frame: v_dot = F/m + g - omega x v,
        # with the same per-step Rigidbody linear damping as the 1 ms plant.
        u_dot = (fx / model.mass_kg + gravity_body[0]
                 - linear_damping * u[k]
                 - pitch_rate[k] * body_vertical
                 + yaw_rate[k] * v[k])
        v_dot = (fy / model.mass_kg + gravity_body[1]
                 - linear_damping * v[k]
                 + roll_rate[k] * body_vertical
                 - yaw_rate[k] * u[k])
        add_constraint(ax - u_dot == 0,
                       f"longitudinal_force_balance_k{k}")
        add_constraint(ay - v_dot == 0,
                       f"lateral_force_balance_k{k}")'''

_SUSPENSION_CALL_OLD = '''        predicted_load = suspension_loads_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_rate[k], pitch[k], pitch_rate[k])
        suspension_travel, _ = suspension_travel_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_rate[k], pitch[k], pitch_rate[k])'''
_SUSPENSION_CALL_NEW = '''        roll_euler_rate, pitch_euler_rate, _ = _unity_euler_rates_casadi(
            roll[k], pitch[k], roll_rate[k], pitch_rate[k], yaw_rate[k])
        predicted_load = suspension_loads_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_euler_rate, pitch[k], pitch_euler_rate)
        suspension_travel, _ = suspension_travel_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_euler_rate, pitch[k], pitch_euler_rate)'''

_DYNAMIC_OLD = '''        longitudinal_dot = ax + yaw_rate[k] * v[k]
        lateral_dot = ay - yaw_rate[k] * u[k]
        yaw_dot = mz / model.yaw_inertia_kgm2 - 0.1 * yaw_rate[k]'''
_DYNAMIC_NEW = '''        longitudinal_dot = ax
        lateral_dot = ay
        roll_euler_rate, pitch_euler_rate, yaw_euler_rate = (
            _unity_euler_rates_casadi(
                roll[k], pitch[k], roll_rate[k], pitch_rate[k],
                yaw_rate[k]))
        yaw_acceleration = (
            mz / model.yaw_inertia_kgm2
            - angular_damping * yaw_rate[k])'''
_DYNAMIC_PREVIOUS = '''        longitudinal_dot = ax
        lateral_dot = ay
        roll_euler_rate, pitch_euler_rate, yaw_dot = (
            _unity_euler_rates_casadi(
                roll[k], pitch[k], roll_rate[k], pitch_rate[k],
                yaw_rate[k]))'''

_VERTICAL_OLD = '''        heave_acceleration = (
            sum(fz) / model.mass_kg - 9.80665
            - model.drag_per_s * body_height_rate[k])
        roll_moment = (
            sum(-WHEEL_Y_M[i] * fz[i] for i in range(4))
            - model.body_com_height_m * fy)
        pitch_moment = (
            sum(-WHEEL_X_M[i] * fz[i] for i in range(4))
            - model.body_com_height_m * fx)
        roll_acceleration = (roll_moment / model.inertia_roll_kgm2
                             - 0.1 * roll_rate[k])
        pitch_acceleration = (pitch_moment / model.inertia_pitch_kgm2
                              - 0.1 * pitch_rate[k])'''
_VERTICAL_NEW = '''        roll_s, roll_c = ca.sin(roll[k]), ca.cos(roll[k])
        pitch_s, pitch_c = ca.sin(pitch[k]), ca.cos(pitch[k])
        world_vertical_force_acceleration = (
            -pitch_s * fx / model.mass_kg
            + pitch_c * (roll_s * fy / model.mass_kg
                         + roll_c * fz_body / model.mass_kg))
        heave_acceleration = (
            world_vertical_force_acceleration
            - UNITY_PHYSICS_GRAVITY_MPS2
            - linear_damping * body_height_rate[k])
        roll_acceleration = (
            mx / model.inertia_roll_kgm2
            - angular_damping * roll_rate[k])
        pitch_acceleration = (
            my / model.inertia_pitch_kgm2
            - angular_damping * pitch_rate[k])'''

_PATH_RHS_OLD = '''            (u[k] * ca.sin(epsi[k]) + v[k] * ca.cos(epsi[k])) / sdot,
            yaw_rate[k] / sdot - curvature,'''
_PATH_RHS_NEW = '''            path_lateral_speed / sdot,
            yaw_euler_rate / sdot - curvature,'''
_YAW_RATE_RHS_OLD = '''            yaw_dot / sdot,
            qdelta[k] / sdot,'''
_YAW_RATE_RHS_NEW = '''            yaw_acceleration / sdot,
            qdelta[k] / sdot,'''
_ATTITUDE_RHS_OLD = '''            roll_rate[k] / sdot,
            roll_acceleration / sdot,
            pitch_rate[k] / sdot,
            pitch_acceleration / sdot))'''
_ATTITUDE_RHS_NEW = '''            roll_euler_rate / sdot,
            roll_acceleration / sdot,
            pitch_euler_rate / sdot,
            pitch_acceleration / sdot))'''

_PATH_NODE_CONSTRAINT_OLD = '''            add_constraint(opti.bounded(
                -max_abs_lateral_speed_mps, v[k], max_abs_lateral_speed_mps),
                f"lateral_velocity_trackability_k{k}")'''
_PATH_NODE_CONSTRAINT_NEW = '''            add_constraint(opti.bounded(
                -max_abs_lateral_speed_mps, path_lateral_speed,
                max_abs_lateral_speed_mps),
                f"lateral_velocity_trackability_k{k}")'''
_PATH_NODE_LIST_OLD = '''    body_path_heading_error_nodes = []'''
_PATH_NODE_LIST_NEW = '''    body_path_heading_error_nodes = []
    path_lateral_speed_nodes = []'''
_PATH_NODE_APPEND_OLD = '''        time_density.append(1.0 / sdot)
        slip_x_nodes.append(slip_x)'''
_PATH_NODE_APPEND_NEW = '''        time_density.append(1.0 / sdot)
        path_lateral_speed_nodes.append(path_lateral_speed)
        slip_x_nodes.append(slip_x)'''
_PATH_HEADING_OLD = '''        lateral_path_rate = (
            u[k] * ca.sin(epsi[k]) + v[k] * ca.cos(epsi[k])) / sdot
        path_heading_offset = ca.atan2(lateral_path_rate, den)'''
_PATH_HEADING_NEW = '''        lateral_path_rate = path_lateral_speed / sdot
        path_heading_offset = ca.atan2(lateral_path_rate, den)'''
_PATH_MIDPOINT_OLD = '''                -max_abs_lateral_speed_mps,
                0.5 * (v[k] + v[kp]),
                max_abs_lateral_speed_mps),'''
_PATH_MIDPOINT_NEW = '''                -max_abs_lateral_speed_mps,
                0.5 * (path_lateral_speed_nodes[k]
                       + path_lateral_speed_nodes[kp]),
                max_abs_lateral_speed_mps),'''
_FORCE_NODE_APPEND_OLD = '''        force_nodes.append((fx, fy, mz))'''
_FORCE_NODE_APPEND_NEW = '''        force_nodes.append((fx, fy, mz, u_dot, v_dot))'''
_FORCE_RESIDUAL_OLD = '''    ax_residual_mps2 = acceleration[0] - (
        force_x_values / model.mass_kg - model.drag_per_s * states[2])
    ay_residual_mps2 = acceleration[1] - (
        force_y_values / model.mass_kg - model.drag_per_s * states[3])'''
_FORCE_RESIDUAL_NEW = '''    source_acceleration = np.asarray([
        _unity_body_acceleration_numpy(
            model, force_x_values[k], force_y_values[k], states[2, k],
            states[3, k], states[4, k], states[11, k], states[12, k],
            states[13, k], states[14, k], states[15, k])
        for k in range(n)], dtype=np.float64).T
    ax_residual_mps2 = acceleration[0] - source_acceleration[0]
    ay_residual_mps2 = acceleration[1] - source_acceleration[1]'''

_RACELINE_WRITE_OLD = '''    raceline = output_dir / "unity_wheel_hybrid_raceline.csv"
    with raceline.open("w", encoding="utf-8") as stream:
        stream.write("# x_m,y_m,s_ref_m,speed_mps,steering_rad,throttle_norm,"
                     "path_curvature_per_m,ey_m,epsi_rad\\n")
        for k in range(n):
            stream.write(
                f"{candidate['x_m'][k]:.9f},{candidate['y_m'][k]:.9f},"
                f"{candidate['s_ref_m'][k]:.9f},"
                f"{candidate['speed_mps'][k]:.9f},"
                f"{candidate['steering_rad'][k]:.9f},"
                f"{candidate['throttle_norm'][k]:.9f},"
                f"{candidate['path_curvature_per_m'][k]:.9f},"
                f"{candidate['ey_m'][k]:.9f},"
                f"{candidate['epsi_rad'][k]:.9f}\\n")'''
_RACELINE_WRITE_NEW = '''    raceline = output_dir / "unity_wheel_target_raceline.csv"
    x_values = np.asarray(candidate["x_m"], dtype=np.float64)
    y_values = np.asarray(candidate["y_m"], dtype=np.float64)
    dx = np.roll(x_values, -1) - np.roll(x_values, 1)
    dy = np.roll(y_values, -1) - np.roll(y_values, 1)
    heading = np.unwrap(np.arctan2(dy, dx))
    with raceline.open("w", encoding="utf-8") as stream:
        stream.write("# s_m,x_m,y_m,psi_rad,kappa_radpm,body_forward_speed_mps\\n")
        for k in range(n):
            stream.write(
                f"{candidate['route_arc_m'][k]:.9f},"
                f"{candidate['x_m'][k]:.9f},{candidate['y_m'][k]:.9f},"
                f"{heading[k]:.9f},"
                f"{candidate['path_curvature_per_m'][k]:.9f},"
                f"{candidate['speed_mps'][k]:.9f}\\n")'''
_RACELINE_WRITE_PREVIOUS = _RACELINE_WRITE_NEW.replace(
    "body_forward_speed_mps", "velocity_mps")
_STATE_FILE_OLD = '    state_path = output_dir / "unity_wheel_hybrid_states.csv"'
_STATE_FILE_NEW = ('    state_path = output_dir / '
                   '"unity_wheel_feasibility_witness_states.csv"')
_STATE_COLUMNS_OLD = '''        "lateral_speed_mps", "yaw_rate_radps", "steering_rad",
        "body_height_m", "body_height_rate_mps", "roll_rad",
        "roll_rate_radps", "pitch_rad", "pitch_rate_radps",
        "steering_rate_radps", "throttle_norm", "ax_mps2", "ay_mps2",'''
_STATE_COLUMNS_NEW = '''        "lateral_speed_mps", "yaw_rate_radps", "witness_steering_rad",
        "body_height_m", "body_height_rate_mps", "roll_rad",
        "roll_rate_radps", "pitch_rad", "pitch_rate_radps",
        "witness_steering_rate_radps", "witness_throttle_norm",
        "ax_mps2", "ay_mps2",'''
_REPORT_OUTPUT_OLD = '''        "trajectory_csv": str(raceline),
        "state_csv": str(state_path),
    })'''
_REPORT_OUTPUT_NEW = '''        "trajectory_csv": str(raceline),
        "trajectory_semantics": (
            "target geometry, body-forward speed and curvature; no steering/throttle commands"),
        "feasibility_witness_csv": str(state_path),
        "witness_controls_are_runtime_commands": False,
    })'''
_REPORT_OUTPUT_PREVIOUS = _REPORT_OUTPUT_NEW.replace(
    "target geometry, body-forward speed and curvature",
    "target geometry, speed and curvature")
_WALL_CLEARANCE_OLD = "*, wall_clearance_m: float = 0.15,"
_WALL_CLEARANCE_NEW = "*, wall_clearance_m: float = 0.05,"

_SOURCE_INTERVAL_MIDPOINT_BLOCK = '''    # Re-evaluate the source-derived tire/suspension/body equations at each
    # spatial interval midpoint. Node-only force and wheel constraints can
    # admit an interpolated target whose local witness is infeasible between
    # knots. The midpoint wheel state is advanced with the same source wheel
    # update over the first half-interval; this is denser collocation, not a
    # claim of native Unity/PhysX substep parity.
    midpoint_state_nodes = []
    midpoint_steering_nodes = []
    midpoint_qdelta_nodes = []
    midpoint_wheel_omega_nodes = []
    midpoint_load_nodes = []
    midpoint_throttle_nodes = []
    midpoint_slip_x_nodes = []
    midpoint_slip_y_nodes = []
    midpoint_force_nodes = []
    midpoint_acceleration_nodes = []
    for k in range(n):
        kp = (k + 1) % n
        midpoint_x = 0.5 * (x[:, k] + x[:, kp])
        midpoint_u = midpoint_x[2]
        midpoint_v = midpoint_x[3]
        midpoint_yaw_rate = midpoint_x[4]
        midpoint_steering = 0.5 * (steering[k] + steering[kp])
        midpoint_qdelta = 0.5 * (qdelta[k] + qdelta[kp])
        midpoint_station_end = station_nodes[kp]
        if kp == 0:
            midpoint_station_end += track.length
        midpoint_station = 0.5 * (
            station_nodes[k] + midpoint_station_end)
        midpoint_curvature = interp["kappa"](midpoint_station)
        midpoint_denominator = (
            1.0 - midpoint_curvature * midpoint_x[0])
        add_constraint(midpoint_denominator >= 0.25,
                       f"periodic_collocation_state_midpoint_frenet_den_k{k}")
        midpoint_body_vertical = _unity_body_vertical_speed_casadi(
            midpoint_x[11], midpoint_u, midpoint_v,
            midpoint_x[12], midpoint_x[14])
        midpoint_horizontal_u, midpoint_horizontal_v = (
            _unity_horizontal_velocity_casadi(
                midpoint_u, midpoint_v, midpoint_body_vertical,
                midpoint_x[12], midpoint_x[14]))
        midpoint_tangent_speed = (
            midpoint_horizontal_u * ca.cos(midpoint_x[1])
            - midpoint_horizontal_v * ca.sin(midpoint_x[1]))
        midpoint_sdot = midpoint_tangent_speed / ca.fmax(
            midpoint_denominator, 0.20)
        add_constraint(midpoint_sdot >= 0.30,
                       f"periodic_collocation_state_midpoint_progress_k{k}")
        segment_time = 0.5 * step_nodes[k] * (
            time_density[k] + time_density[kp])
        half_time = 0.5 * segment_time
        midpoint_tire_torque = [0.5 * (
            wheel_tire_torque_nodes[k][i]
            + wheel_tire_torque_nodes[kp][i]) for i in range(4)]
        midpoint_wheel_omega = []
        for i in range(4):
            if brake_modes_array[k]:
                omega_mid = source_wheel_update(
                    wheel_omega[i][k], 0.0, model.motor_torque_nm,
                    midpoint_tire_torque[i], half_time, x[2, k])
            else:
                if brake_modes_array[kp]:
                    mean_throttle_first_half = throttle[k]
                else:
                    mean_throttle_first_half = (
                        0.75 * throttle[k] + 0.25 * throttle[kp])
                omega_mid = source_wheel_update(
                    wheel_omega[i][k],
                    model.motor_torque_nm * mean_throttle_first_half / 4.0,
                    0.0, midpoint_tire_torque[i], half_time, x[2, k])
            midpoint_wheel_omega.append(omega_mid)
            add_constraint(omega_mid >= 0.0,
                           f"midpoint_forward_wheel_rotation_w{i}_k{k}")
        # Endpoint steering, rate, and throttle bounds also bound their
        # arithmetic midpoint by convexity; avoid duplicate inequalities.
        if brake_modes_array[k] == brake_modes_array[kp]:
            midpoint_throttle = 0.5 * (throttle[k] + throttle[kp])
            if brake_modes_array[k]:
                add_constraint(midpoint_throttle == 0.0,
                               f"midpoint_unity_exact_zero_cawb_brake_k{k}")
        elif brake_modes_array[k]:
            midpoint_throttle = 0.0
        else:
            midpoint_throttle = throttle[k]
        mid_roll_rate, mid_pitch_rate, _ = _unity_euler_rates_casadi(
            midpoint_x[12], midpoint_x[14], midpoint_x[13],
            midpoint_x[15], midpoint_yaw_rate)
        mid_suspension_load = suspension_loads_casadi(
            model, midpoint_x[10], midpoint_x[11], midpoint_x[12],
            mid_roll_rate, midpoint_x[14], mid_pitch_rate)
        mid_suspension_travel, _ = suspension_travel_casadi(
            model, midpoint_x[10], midpoint_x[11], midpoint_x[12],
            mid_roll_rate, midpoint_x[14], mid_pitch_rate)
        # Midpoint normal loads are deterministic source-model quantities, not
        # independent collocation states. Reusing the average endpoint load
        # and constraining it to this nonlinear evaluation added four extra
        # equalities per interval without adding a midpoint state equation.
        midpoint_load = list(mid_suspension_load)
        for i in range(4):
            add_constraint(midpoint_load[i] >= minimum_wheel_normal_load_n,
                           f"midpoint_normal_load_w{i}_k{k}")
            add_constraint(opti.bounded(
                0.0, mid_suspension_travel[i], model.suspension_distance_m),
                f"midpoint_unity_suspension_travel_w{i}_k{k}")
        (mid_fx, mid_fy, mid_fz_body, mid_mx, mid_my, mid_mz, _, _,
         mid_slip_x, mid_slip_y) = (
            tire_forces_casadi(
                model, midpoint_u, midpoint_v, midpoint_yaw_rate,
                midpoint_steering, midpoint_wheel_omega, midpoint_load,
                midpoint_x[10], midpoint_x[11], midpoint_x[12],
                midpoint_x[13], midpoint_x[14], midpoint_x[15]))
        mid_gravity = _unity_gravity_body_casadi(
            midpoint_x[12], midpoint_x[14])
        mid_linear_damping = _unity_equivalent_damping(model.drag_per_s)
        mid_angular_damping = _unity_equivalent_damping(
            UNITY_SOURCE_ANGULAR_DAMPING_PER_S)
        mid_ax = (mid_fx / model.mass_kg + mid_gravity[0]
                  - mid_linear_damping * midpoint_u
                  - midpoint_x[15] * midpoint_body_vertical
                  + midpoint_yaw_rate * midpoint_v)
        mid_ay = (mid_fy / model.mass_kg + mid_gravity[1]
                  - mid_linear_damping * midpoint_v
                  + midpoint_x[13] * midpoint_body_vertical
                  - midpoint_yaw_rate * midpoint_u)
        mid_world_vertical_force_acceleration = (
            -ca.sin(midpoint_x[14]) * mid_fx / model.mass_kg
            + ca.cos(midpoint_x[14]) * (
                ca.sin(midpoint_x[12]) * mid_fy / model.mass_kg
                + ca.cos(midpoint_x[12]) * mid_fz_body / model.mass_kg))
        mid_heave_acceleration = (
            mid_world_vertical_force_acceleration
            - UNITY_PHYSICS_GRAVITY_MPS2
            - mid_linear_damping * midpoint_x[11])
        mid_roll_acceleration = (
            mid_mx / model.inertia_roll_kgm2
            - mid_angular_damping * midpoint_x[13])
        mid_pitch_acceleration = (
            mid_my / model.inertia_pitch_kgm2
            - mid_angular_damping * midpoint_x[15])
        midpoint_yaw_acceleration = (
            mid_mz / model.yaw_inertia_kgm2
            - mid_angular_damping * midpoint_yaw_rate)
        midpoint_state_nodes.append(midpoint_x)
        midpoint_steering_nodes.append(midpoint_steering)
        midpoint_qdelta_nodes.append(midpoint_qdelta)
        midpoint_wheel_omega_nodes.append(tuple(midpoint_wheel_omega))
        midpoint_load_nodes.append(tuple(midpoint_load))
        midpoint_throttle_nodes.append(midpoint_throttle)
        midpoint_slip_x_nodes.append(mid_slip_x)
        midpoint_slip_y_nodes.append(mid_slip_y)
        midpoint_force_nodes.append(ca.vertcat(
            mid_fx, mid_fy, mid_fz_body, mid_mx, mid_my, mid_mz))
        midpoint_acceleration_nodes.append(ca.vertcat(
            mid_ax, mid_ay, midpoint_yaw_acceleration,
            mid_heave_acceleration, mid_roll_acceleration,
            mid_pitch_acceleration))

'''

_SOURCE_INTERVAL_MIDPOINT_MARKER = '''    for k in range(n):
        kp = (k + 1) % n
        xk = ca.vertcat(*[x[index, k] for index in dynamic_indices])'''

_DYNAMIC_PREFIXES_OLD = '''            'longitudinal_force_balance_', 'lateral_force_balance_',
            'unity_spring_damper_load_',
        )'''
_DYNAMIC_PREFIXES_NEW = '''            'longitudinal_force_balance_', 'lateral_force_balance_',
            'unity_spring_damper_load_', 'yaw_moment_balance_',
            'heave_force_balance_', 'roll_moment_balance_',
            'pitch_moment_balance_',
        )'''
_INITIAL_DYNAMIC_PREFIXES_OLD = '''        'longitudinal_force_balance_', 'lateral_force_balance_',
        'unity_spring_damper_load_'))]'''
_INITIAL_DYNAMIC_PREFIXES_NEW = '''        'longitudinal_force_balance_', 'lateral_force_balance_',
        'unity_spring_damper_load_', 'yaw_moment_balance_',
        'heave_force_balance_', 'roll_moment_balance_',
        'pitch_moment_balance_'))]'''
_DEFECT_DYNAMIC_PREFIXES_OLD = '''                          'lateral_force_balance_',
                          'unity_spring_damper_load_'))]'''
_DEFECT_DYNAMIC_PREFIXES_NEW = '''                          'lateral_force_balance_',
                          'unity_spring_damper_load_',
                          'yaw_moment_balance_',
                          'heave_force_balance_',
                          'roll_moment_balance_',
                          'pitch_moment_balance_'))]'''
_YAW_DEFECT_GROUP_OLD = '''            elif label.startswith('lateral_force_balance_'):
                group = 'lateral_force_balance'
            elif label.startswith('unity_spring_damper_load_'):'''
_YAW_DEFECT_GROUP_NEW = '''            elif label.startswith('lateral_force_balance_'):
                group = 'lateral_force_balance'
            elif label.startswith('yaw_moment_balance_'):
                group = 'yaw_moment_balance'
            elif label.startswith('heave_force_balance_'):
                group = 'heave_force_balance'
            elif label.startswith('roll_moment_balance_'):
                group = 'roll_moment_balance'
            elif label.startswith('pitch_moment_balance_'):
                group = 'pitch_moment_balance'
            elif label.startswith('unity_spring_damper_load_'):'''
_YAW_BREAKDOWN_GROUP_OLD = '''                elif label.startswith('lateral_force_balance_'):
                    group = 'lateral_force_balance'
                else:'''
_YAW_BREAKDOWN_GROUP_NEW = '''                elif label.startswith('lateral_force_balance_'):
                    group = 'lateral_force_balance'
                elif label.startswith('yaw_moment_balance_'):
                    group = 'yaw_moment_balance'
                elif label.startswith('heave_force_balance_'):
                    group = 'heave_force_balance'
                elif label.startswith('roll_moment_balance_'):
                    group = 'roll_moment_balance'
                elif label.startswith('pitch_moment_balance_'):
                    group = 'pitch_moment_balance'
                else:'''
_MIDPOINT_EXTRACTION_MARKER = '''    ay_residual_mps2 = acceleration[1] - source_acceleration[1]
    report["constraint_diagnostics"] = {'''
_MIDPOINT_EXTRACTION_BLOCK = '''    midpoint_states = np.column_stack([
        np.asarray(value(item), dtype=np.float64).reshape(-1)
        for item in midpoint_state_nodes])
    midpoint_wheel_omega_values = np.asarray([
        [float(value(midpoint_wheel_omega_nodes[k][i]))
         for k in range(n)] for i in range(4)], dtype=np.float64)
    midpoint_load_values = np.asarray([
        [float(value(midpoint_load_nodes[k][i]))
         for k in range(n)] for i in range(4)], dtype=np.float64)
    midpoint_slip_x_values = np.asarray([
        [float(value(midpoint_slip_x_nodes[k][i]))
         for k in range(n)] for i in range(4)], dtype=np.float64)
    midpoint_slip_y_values = np.asarray([
        [float(value(midpoint_slip_y_nodes[k][i]))
         for k in range(n)] for i in range(4)], dtype=np.float64)
    midpoint_force_values = np.asarray([
        np.asarray(value(item), dtype=np.float64).reshape(-1)
        for item in midpoint_force_nodes], dtype=np.float64).T
    midpoint_acceleration_values = np.asarray([
        np.asarray(value(item), dtype=np.float64).reshape(-1)
        for item in midpoint_acceleration_nodes], dtype=np.float64).T
    midpoint_steering_values = np.asarray(
        [float(value(item)) for item in midpoint_steering_nodes])
    midpoint_qdelta_values = np.asarray(
        [float(value(item)) for item in midpoint_qdelta_nodes])
    midpoint_throttle_values = np.asarray(
        [float(value(item)) for item in midpoint_throttle_nodes])
    report["constraint_diagnostics"] = {'''
_MIDPOINT_REPORT_MARKER = '''        "peak_speed_mps": float(np.max(states[2])),'''
_MIDPOINT_REPORT_BLOCK = '''        "midpoint_max_abs_longitudinal_slip_by_wheel": np.max(
            np.abs(midpoint_slip_x_values), axis=1).tolist(),
        "midpoint_max_abs_lateral_slip_by_wheel": np.max(
            np.abs(midpoint_slip_y_values), axis=1).tolist(),
        "midpoint_source_dynamic_balance_axes": [
            "body_u", "body_v", "yaw", "world_vertical", "roll", "pitch"],
        "source_dynamic_constraint_sampling": (
            "six body force/moment balance equalities at OCP nodes; source "
            "tire, suspension, and acceleration equations are also evaluated "
            "at segment midpoints for feasibility inequalities and witness "
            "diagnostics, not extra midpoint collocation equalities"),
'''
_MIDPOINT_CANDIDATE_MARKER = '''        "force_summary": np.asarray([[float(value(force_nodes[k][axis]))'''
_MIDPOINT_CANDIDATE_BLOCK = '''        "midpoint_s_ref_m": midpoint_s,
        "midpoint_state": midpoint_states,
        "midpoint_wheel_omega_radps": midpoint_wheel_omega_values,
        "midpoint_steering_rad": midpoint_steering_values,
        "midpoint_steering_rate_radps": midpoint_qdelta_values,
        "midpoint_throttle_norm": midpoint_throttle_values,
        "midpoint_normal_load_n": midpoint_load_values,
        "midpoint_slip_x": midpoint_slip_x_values,
        "midpoint_slip_y": midpoint_slip_y_values,
        "midpoint_force_summary": midpoint_force_values,
        "midpoint_acceleration_summary": midpoint_acceleration_values,
'''


def _upgrade_previous_v251(source: str) -> str:
    """Upgrade only the exact earlier v251 candidate written in this workspace."""
    if ("CasADi/IPOPT positive-drive spatial direct collocation" in source
            and ("UNITY_MPC_THROTTLE_MIN_DRIVE = 0.01" in source
                 or "UNITY_MPC_THROTTLE_MIN_DRIVE = 0.001" in source)
            and "phase_count = 1" in source
            and "phase_modes" not in source
            and "path_lateral_speed_nodes = []" in source
            and "yaw_acceleration =" in source
            and "midpoint_force_nodes" in source
            and not re.search(
                r"brake_mode|brake_torque|_brake_phases|CAWB|zero.throttle|"
                r"brake_modes_array|mode_transition_nodes|\bbrakes?\b|"
                r"\bbraking\b", source, re.IGNORECASE)):
        # Exact locally generated positive-drive cache from the prior v2.5.4
        # run. It is checked here, then regenerated from recovered source;
        # the cached file does not become authoritative.
        return source
    if ("phase_modes = [False]" in source
            and "phase_count = 1" in source
            and "UNITY_MPC_THROTTLE_MIN_DRIVE" in source
            and "path_lateral_speed_nodes = []" in source
            and "yaw_acceleration =" in source
            and "phase-switched spatial direct collocation" in source
            and "'brake' if mode else 'drive'" in source):
        # Exact transient drive-only generated cache from the interrupted
        # positive-throttle conversion. It is regenerated from recovered
        # source below; do not treat the cached body as authoritative.
        return source
    if ("yaw_acceleration = (" in source
            and "path_lateral_speed_nodes = []" in source
            and source.count(_RACELINE_WRITE_NEW) == 1
            and source.count(_STATE_FILE_NEW) == 1
            and source.count(_STATE_COLUMNS_NEW) == 1
            and source.count(_REPORT_OUTPUT_NEW) == 1):
        return source
    if ("yaw_acceleration = (" in source
            and "path_lateral_speed_nodes = []" in source
            and source.count(_RACELINE_WRITE_PREVIOUS) == 1
            and source.count(_STATE_FILE_NEW) == 1
            and source.count(_STATE_COLUMNS_NEW) == 1
            and source.count(_REPORT_OUTPUT_PREVIOUS) == 1):
        source = source.replace(
            _RACELINE_WRITE_PREVIOUS, _RACELINE_WRITE_NEW, 1)
        source = source.replace(
            _REPORT_OUTPUT_PREVIOUS, _REPORT_OUTPUT_NEW, 1)
        return source
    if ("yaw_acceleration = (" in source
            and "path_lateral_speed_nodes = []" in source
            and source.count(_RACELINE_WRITE_OLD) == 1
            and source.count(_STATE_FILE_OLD) == 1
            and source.count(_STATE_COLUMNS_OLD) == 1
            and source.count(_REPORT_OUTPUT_OLD) == 1):
        for old, new in (
                (_RACELINE_WRITE_OLD, _RACELINE_WRITE_NEW),
                (_STATE_FILE_OLD, _STATE_FILE_NEW),
                (_STATE_COLUMNS_OLD, _STATE_COLUMNS_NEW),
                (_REPORT_OUTPUT_OLD, _REPORT_OUTPUT_NEW)):
            source = source.replace(old, new, 1)
        return source
    if ("yaw_dot = (" not in source
            or "yaw_acceleration = (" in source
            or "path_lateral_speed_nodes = []" in source):
        raise RuntimeError("Unrecognized cached IPOPT candidate; refusing rewrite")
    replacements = (
        (_DYNAMIC_PREVIOUS, _DYNAMIC_NEW),
        ('''            yaw_dot / sdot - curvature,''',
         '''            yaw_euler_rate / sdot - curvature,'''),
        ('''            yaw_dot / sdot,
            qdelta[k] / sdot,''',
         _YAW_RATE_RHS_NEW),
        (_PATH_NODE_CONSTRAINT_OLD, _PATH_NODE_CONSTRAINT_NEW),
        (_PATH_NODE_LIST_OLD, _PATH_NODE_LIST_NEW),
        (_PATH_NODE_APPEND_OLD, _PATH_NODE_APPEND_NEW),
        (_PATH_HEADING_OLD, _PATH_HEADING_NEW),
        (_PATH_MIDPOINT_OLD, _PATH_MIDPOINT_NEW),
        (_FORCE_NODE_APPEND_OLD, _FORCE_NODE_APPEND_NEW),
        (_FORCE_RESIDUAL_OLD, _FORCE_RESIDUAL_NEW),
        (_RACELINE_WRITE_OLD, _RACELINE_WRITE_NEW),
        (_STATE_FILE_OLD, _STATE_FILE_NEW),
        (_STATE_COLUMNS_OLD, _STATE_COLUMNS_NEW),
        (_REPORT_OUTPUT_OLD, _REPORT_OUTPUT_NEW),
    )
    for old, new in replacements:
        if source.count(old) != 1:
            raise RuntimeError("Cached v251 source differs at a parity patch site")
        source = source.replace(old, new, 1)
    return source

_SEED_MAP_OLD = '''    seed_force_state = ca.MX.sym(f"seed_force_state_{id(opti)}", 12, n)
    seed_fx, seed_fy, _, seed_fx_w, _, _, _ = tire_forces_casadi(
        model, seed_force_state[0, :], seed_force_state[1, :],
        seed_force_state[2, :], seed_force_state[3, :],
        [seed_force_state[4 + i, :] for i in range(4)],
        [seed_force_state[8 + i, :] for i in range(4)])
    seed_force_map = ca.Function(
        f"seed_force_map_{id(opti)}", [seed_force_state],
        [ca.vertcat(seed_fx, seed_fy, *seed_fx_w)])'''
_SEED_MAP_NEW = '''    seed_force_state = ca.MX.sym(f"seed_force_state_{id(opti)}", 18, n)
    (seed_fx, seed_fy, seed_fz, seed_mx, seed_my, seed_mz,
     seed_fx_w, _, _, _) = tire_forces_casadi(
        model, seed_force_state[0, :], seed_force_state[1, :],
        seed_force_state[2, :], seed_force_state[3, :],
        [seed_force_state[4 + i, :] for i in range(4)],
        [seed_force_state[8 + i, :] for i in range(4)],
        seed_force_state[12, :], seed_force_state[13, :],
        seed_force_state[14, :], seed_force_state[15, :],
        seed_force_state[16, :], seed_force_state[17, :])
    seed_body_vertical = _unity_body_vertical_speed_casadi(
        seed_force_state[13, :], seed_force_state[0, :],
        seed_force_state[1, :], seed_force_state[14, :],
        seed_force_state[16, :])
    seed_gravity_body = _unity_gravity_body_casadi(
        seed_force_state[14, :], seed_force_state[16, :])
    seed_linear_damping = _unity_equivalent_damping(model.drag_per_s)
    seed_ax_model = (seed_fx / model.mass_kg + seed_gravity_body[0]
               - seed_linear_damping * seed_force_state[0, :]
               - seed_force_state[17, :] * seed_body_vertical
               + seed_force_state[2, :] * seed_force_state[1, :])
    seed_ay_model = (seed_fy / model.mass_kg + seed_gravity_body[1]
               - seed_linear_damping * seed_force_state[1, :]
               + seed_force_state[15, :] * seed_body_vertical
               - seed_force_state[2, :] * seed_force_state[0, :])
    seed_force_map = ca.Function(
        f"seed_force_map_{id(opti)}", [seed_force_state],
        [ca.vertcat(seed_fx, seed_fy, seed_fz, seed_mx, seed_my, seed_mz,
                    *seed_fx_w, seed_ax_model, seed_ay_model)])'''

_SEED_STACK_OLD = '''        force_state = np.vstack((seed_u, seed_v, seed_yaw, seed_steering,
                                 seed_omega, seed_load))'''
_SEED_STACK_NEW = '''        force_state = np.vstack((seed_u, seed_v, seed_yaw, seed_steering,
                                 seed_omega, seed_load, seed_body_height,
                                 seed_body_height_rate, seed_roll,
                                 seed_roll_rate, seed_pitch,
                                 seed_pitch_rate))'''
_SYNTHETIC_SPEED_SEED_OLD = '''        seed_u = np.clip(np.minimum(
            min(5.5, max_speed_mps - 0.2),
            np.sqrt(seed_lateral_accel * 0.72 /
                    np.maximum(np.abs(kappa_seed), 1.0e-5))),
            min_speed_mps, max_speed_mps)'''
_SYNTHETIC_SPEED_SEED_NEW = '''        # This is only a source-curve-based initial point, not a lateral-
        # acceleration constraint. Do not retain the former fitted 0.72
        # derate or 5.5 m/s ceiling as an implicit racing-speed prior.
        source_curve_speed = np.sqrt(
            seed_lateral_accel / np.maximum(np.abs(kappa_seed), 1.0e-5))
        if initial_speed_seed_mps is None:
            seed_u = np.clip(source_curve_speed, min_speed_mps, max_speed_mps)
        else:
            reference_speed = np.asarray(initial_speed_seed_mps,
                                         dtype=np.float64)
            if (reference_speed.shape != (n,)
                    or not np.isfinite(reference_speed).all()
                    or np.any(reference_speed <= 0.0)):
                raise ValueError(
                    'reference speed seed must be positive, finite, shape (n,)')
            seed_u = np.clip(reference_speed, min_speed_mps, max_speed_mps)'''
_SPEED_SEED_SIGNATURE_OLD = '''                    initial_state_seed_is_model_consistent: bool = False,
                    minimum_trust_region_reference: float = 1.0,'''
_SPEED_SEED_SIGNATURE_NEW = '''                    initial_state_seed_is_model_consistent: bool = False,
                    initial_speed_seed_mps: np.ndarray | None = None,
                    minimum_trust_region_reference: float = 1.0,'''
_SYNTHETIC_SLIP_SEED_OLD = '''        seed_v = seed_u * (0.15532 * kappa_seed
                           - 0.025 * np.sign(kappa_seed))
        course_speed = np.hypot(seed_u, seed_v)
        seed_yaw = course_speed * kappa_seed
        seed_epsi = -np.arctan2(seed_v, seed_u)'''
_SYNTHETIC_SLIP_SEED_NEW = '''        # Start with a geometric zero-slip guess. The old empirical lateral-
        # speed relation was not a Unity/PhysX source equation; solved lateral
        # motion and heading error come from the OCP tire/collocation model.
        seed_v = np.zeros(n)
        course_speed = seed_u.copy()
        seed_yaw = seed_u * kappa_seed
        seed_epsi = np.zeros(n)'''
_SEED_ACCEL_OLD = '''        ax_values = (force_values[0] / model.mass_kg
                     - model.drag_per_s * seed_u)
        ay_values = (force_values[1] / model.mass_kg
                     - model.drag_per_s * seed_v)'''
_SEED_ACCEL_NEW = '''        ax_values = np.asarray(force_values[-2], dtype=np.float64).reshape(-1)
        ay_values = np.asarray(force_values[-1], dtype=np.float64).reshape(-1)'''
_SEED_TORQUE_OLD = '''            model.wheel_radius_m * force_values[2:6]
            + model.wheel_damping_nms * seed_omega)'''
_SEED_TORQUE_NEW = '''            model.wheel_radius_m * force_values[6:10]
            + model.wheel_damping_nms * seed_omega)'''
_SEED_THROTTLE_CLIP_OLD = '''        seed_throttle = np.clip(
            4.0 * model.wheel_damping_nms * np.mean(seed_omega, axis=0)
            / model.motor_torque_nm, 0.015, 0.95)'''
_SEED_THROTTLE_CLIP_NEW = '''        seed_throttle = np.clip(
            4.0 * model.wheel_damping_nms * np.mean(seed_omega, axis=0)
            / model.motor_torque_nm, UNITY_MPC_THROTTLE_MIN_DRIVE, 0.95)'''
_SEED_THROTTLE_INITIAL_FLOOR_OLD = '''            0.015, seed_throttle[~brake_modes_array])'''
_SEED_THROTTLE_INITIAL_FLOOR_NEW = '''            UNITY_MPC_THROTTLE_MIN_DRIVE,
            seed_throttle[~brake_modes_array])'''
_SEED_THROTTLE_NODE_BOUND_OLD = '''opti.bounded(0.015, throttle[k], 1.0)'''
_SEED_THROTTLE_NODE_BOUND_NEW = '''opti.bounded(UNITY_MPC_THROTTLE_MIN_DRIVE,
                                           throttle[k], 1.0)'''

_PERIODIC_SEED_START = "\ndef _periodic_suspension_seed("
_PERIODIC_SEED_END = "\ndef _constraint_diagnostics("
_SEED_CONSISTENCY_START = (
    "    initialization_consistency_iterations = 0\n"
    "    for iteration in range(0 if initial_state_seed_is_model_consistent else 5):")
_SEED_CONSISTENCY_END = (
    "\n    # First find a model-consistent lap near the measured motion.")
_SOURCE_FORCE_SEED_BLOCK = '''    initialization_consistency_iterations = 0
    if not initial_state_seed_is_model_consistent:
        # Evaluate the source-derived tire/body equations once at the seeded
        # suspension state. This initializes acceleration and drive torque;
        # it does not integrate a separate approximate suspension lap. The
        # optimizer still solves the full heave/roll/pitch dynamics jointly.
        force_state = np.vstack((seed_u, seed_v, seed_yaw, seed_steering,
                                 seed_omega, seed_load, seed_body_height,
                                 seed_body_height_rate, seed_roll,
                                 seed_roll_rate, seed_pitch,
                                 seed_pitch_rate))
        force_values = np.asarray(seed_force_map(force_state),
                                  dtype=np.float64)
        seed_ax = np.asarray(force_values[-2], dtype=np.float64).reshape(-1)
        seed_ay = np.asarray(force_values[-1], dtype=np.float64).reshape(-1)
        required_per_wheel = (
            model.wheel_radius_m * force_values[6:10]
            + model.wheel_damping_nms * seed_omega)
        seed_throttle = np.clip(
            4.0 * np.mean(required_per_wheel, axis=0)
            / model.motor_torque_nm, 0.015, 1.0)
        seed_throttle[brake_modes_array] = 0.0
        opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))
        opti.set_initial(throttle, seed_throttle)
        initialization_consistency_iterations = 1
        suspension_seed_summary = {
            "method": "source force evaluation at seeded suspension state",
            "outer_force_load_iterations": 1,
            "periodic_laps": 0,
            "fixed_point_residual_scaled": None,
        }
'''
_SEED_LOADS_OLD = '''        suspension_loads_numpy(
            model, seed_body_height[k], seed_body_height_rate[k],
            seed_roll[k], seed_roll_rate[k], seed_pitch[k],
            seed_pitch_rate[k])'''
_SEED_LOADS_NEW = '''        _unity_suspension_loads_numpy(
            model, seed_body_height[k], seed_body_height_rate[k],
            seed_roll[k], seed_roll_rate[k], seed_pitch[k],
            seed_pitch_rate[k], seed_yaw[k])'''


def _drive_only_optimizer_source(source: str) -> str:
    """Remove the optimizer's discrete exact-zero-throttle mode entirely.

    The native vehicle model still describes Unity's real input semantics for
    odometry and replay. This generated IPOPT transcription, however, has only
    positive drive torque as an actuator choice, matching the production MPC
    envelope. Old seed brake masks must not create hidden zero-throttle arcs.
    """
    brake_phase_start = source.find("def _brake_phases(")
    periodic_start = source.find("def _periodic_interpolants(", brake_phase_start)
    if brake_phase_start < 0 or periodic_start < 0:
        raise RuntimeError("Cannot locate legacy discrete-mode helper to remove")
    source = source[:brake_phase_start] + source[periodic_start:]

    source, signature_count = re.subn(
        r"(?m)^\s*brake_mode_seed: np\.ndarray \| None = None,\n"
        r"\s*brake_mode_seed_source: str = \"drive-mode initialization\",\n",
        "", source, count=1)
    if signature_count != 1:
        raise RuntimeError("Cannot remove the optimizer's legacy mode arguments")

    old_doc = '''    """Jointly optimize a free path, motion states, throttle, and brake phases.

    Unity changes discontinuously between CAWB full braking at zero throttle
    and CAWD positive drive. Instead of 257 independent integer decisions,
    measured training-only mode data initializes the count/order of contiguous
    phases. IPOPT optimizes their lengths and all path/state/control variables;
    captured phase locations are not constraints. The vehicle force model is
    the Unity-source-calibrated model described in ``UnityWheelModel``.
    """'''
    new_doc = '''    """Optimize a free path using positive throttle and physical steering only.

    Every spatial node and interval midpoint is constrained to the production
    MPC drive-throttle envelope. The NLP has no full-brake mode or zero-throttle
    decision. Vehicle force equations remain source-derived.
    """'''
    if source.count(old_doc) != 1:
        raise RuntimeError("Cannot replace legacy hybrid-control solver description")
    source = source.replace(old_doc, new_doc, 1)

    setup_start = source.find("    if brake_mode_seed is None:")
    setup_end = source.find(
        "    node_counts = np.asarray(phase_nodes, dtype=np.int64)", setup_start)
    if setup_start < 0 or setup_end < 0:
        raise RuntimeError("Cannot locate legacy mode/phase setup")
    drive_setup = '''    if not math.isfinite(minimum_trust_region_reference) or minimum_trust_region_reference <= 0.0:
        raise ValueError("minimum trust-region reference must be positive")
    if (throttle_fall_rate_limit_norm_per_s is not None
            and (not math.isfinite(throttle_fall_rate_limit_norm_per_s)
                 or throttle_fall_rate_limit_norm_per_s <= 0.0)):
        raise ValueError("throttle fall-rate limit must be finite and positive")
    phase_nodes = [n]
    phase_seed_lengths = np.asarray([track.length], dtype=np.float64)
    phase_count = 1
    if phase_lengths_initial is None:
        phase_initial_lengths = phase_seed_lengths.copy()
    else:
        phase_initial_lengths = np.asarray(phase_lengths_initial, dtype=np.float64)
        if (phase_initial_lengths.shape != (1,)
                or not np.isfinite(phase_initial_lengths).all()
                or np.any(phase_initial_lengths <= 0.0)
                or not np.isclose(np.sum(phase_initial_lengths), track.length,
                                  rtol=0.0, atol=1.0e-5)):
            raise ValueError("invalid drive-only phase length")
'''
    source = source[:setup_start] + drive_setup + source[setup_end:]
    source = source.replace(
        "for phase, (mode, count) in enumerate(zip(phase_modes, node_counts)):",
        "for phase, count in enumerate(node_counts):")
    source = source.replace(
        '"CasADi/IPOPT phase-switched spatial direct collocation"',
        '"CasADi/IPOPT positive-drive spatial direct collocation"')
    source = source.replace(
        '"modes": ["brake" if mode else "drive" for mode in phase_modes]',
        '"modes": ["positive_drive"]')

    wheel_update_start = source.find("    def source_wheel_update(")
    wheel_update_end = source.find(
        "    # Re-evaluate the source-derived tire/suspension/body equations",
        wheel_update_start)
    if wheel_update_start < 0 or wheel_update_end < 0:
        raise RuntimeError("Cannot locate legacy wheel/brake update helper")
    drive_wheel_update = '''    def source_wheel_update(omega, drive_torque, tire_torque,
                            duration_s, forward_speed):
        def advance(substeps):
            # duration_s is the variable travel time across a spatial OCP
            # segment. Compose Unity's scalar wheel recurrence over that
            # duration with the collocated torques held constant; this is not
            # a 25 ms command hold or an optimizer integration grid. It does
            # not resolve changing tire forces inside this spatial segment.
            internal_step_s = UNITY_SOURCE_FIXED_STEP_S / substeps
            damping = model.wheel_damping_nms
            inertia = model.wheel_inertia_kgm2
            damping_step = damping * internal_step_s / inertia
            step_ratio = 1.0 / (1.0 + damping_step)
            step_count = duration_s / internal_step_s
            decay = ca.exp(step_count * ca.log(step_ratio))
            total_torque = drive_torque + tire_torque
            equilibrium = total_torque / damping
            undamped = omega + duration_s * total_torque / inertia
            return ca.if_else(
                damping > 0.0,
                decay * omega + (1.0 - decay) * equilibrium,
                undamped)

        low_speed = advance(UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS)
        high_speed = advance(UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS)
        return ca.if_else(
            ca.fabs(forward_speed) < UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS,
            low_speed, high_speed)

'''
    source = source[:wheel_update_start] + drive_wheel_update + source[wheel_update_end:]

    seed_start = source.find(
        "    # For a fresh guess, initialize locked brake/transition wheel states at")
    seed_end = source.find(
        '        seed_ey[0] = gate_state_anchor["ey_m"]', seed_start)
    if seed_start < 0 or seed_end < 0:
        raise RuntimeError("Cannot remove legacy brake-seed initialization")
    seed_setup = '''    if not initial_state_seed_is_model_consistent:
        seed_throttle = np.clip(
            seed_throttle, UNITY_MPC_THROTTLE_MIN_DRIVE, 1.0)
    if gate_state_anchor is not None:
'''
    source = source[:seed_start] + seed_setup + source[seed_end:]
    # The OCP's positive-drive wheel update has no brake-torque argument. Also
    # remove the generic helper import so the generated optimizer cannot call
    # the source's zero-throttle wheel-lock branch.
    source, import_count = source.replace(
        "    physx_no_drive_wheel_speed_interval_casadi,\n", ""), source.count(
            "    physx_no_drive_wheel_speed_interval_casadi,\n")
    if import_count != 1:
        raise RuntimeError("Unexpected generic wheel-update import count")

    class DriveOnly(ast.NodeTransformer):
        @staticmethod
        def _zero_mode_value(node):
            if isinstance(node, ast.Constant) and isinstance(node.value, bool):
                return node.value
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
                if node.value.id in {"brake_modes_array", "mode_transition_nodes"}:
                    return False
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                value = DriveOnly._zero_mode_value(node.operand)
                return None if value is None else not value
            if isinstance(node, ast.BoolOp):
                values = [DriveOnly._zero_mode_value(value) for value in node.values]
                if any(value is None for value in values):
                    return None
                return (all(values) if isinstance(node.op, ast.And)
                        else any(values))
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left = DriveOnly._zero_mode_value(node.left)
                right = DriveOnly._zero_mode_value(node.comparators[0])
                if left is None or right is None:
                    return None
                if isinstance(node.ops[0], ast.Eq):
                    return left == right
                if isinstance(node.ops[0], ast.NotEq):
                    return left != right
            return None

        @staticmethod
        def _mentions_removed_mode(node):
            return any(isinstance(item, ast.Name) and item.id in {
                "brake_modes", "brake_modes_array", "mode_transition_nodes",
                "brake_mode_seed", "brake_mode_seed_source"}
                for item in ast.walk(node))

        def visit_FunctionDef(self, node):
            if node.name == "_brake_phases":
                return None
            if node.name == "solve_bottom_up":
                args = node.args
                keep = [i for i, arg in enumerate(args.args)
                        if arg.arg not in {"brake_mode_seed", "brake_mode_seed_source"}]
                args.args = [args.args[i] for i in keep]
                if args.defaults:
                    removed = len(keep) < len(args.args) + 2
                    # The textual signature patch above removes both optional
                    # arguments, so defaults already align; no slicing needed.
                    del removed
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                        node.body[0].value, ast.Constant) and isinstance(
                        node.body[0].value.value, str):
                    node.body[0].value.value = (
                        "Optimize a free path with positive throttle and "
                        "physical steering only; every command obeys the "
                        "production MPC throttle lower bound.")
            return self.generic_visit(node)

        def visit_If(self, node):
            node.test = self.visit(node.test)
            value = self._zero_mode_value(node.test)
            if value is not None:
                chosen = node.body if value else node.orelse
                result = []
                for statement in chosen:
                    transformed = self.visit(statement)
                    if transformed is None:
                        continue
                    result.extend(transformed if isinstance(transformed, list)
                                  else [transformed])
                return result
            return self.generic_visit(node)

        def visit_Subscript(self, node):
            if (isinstance(node.value, ast.Name)
                    and node.value.id in {"brake_modes_array", "mode_transition_nodes"}):
                return ast.copy_location(ast.Constant(False), node)
            return self.generic_visit(node)

        def visit_UnaryOp(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.op, ast.Not)
                    and isinstance(node.operand, ast.Constant)
                    and isinstance(node.operand.value, bool)):
                return ast.copy_location(ast.Constant(not node.operand.value), node)
            return node

        def visit_BoolOp(self, node):
            node.values = [self.visit(value) for value in node.values]
            if isinstance(node.op, ast.And):
                if any(isinstance(value, ast.Constant)
                       and value.value is False for value in node.values):
                    return ast.copy_location(ast.Constant(False), node)
                node.values = [value for value in node.values if not (
                    isinstance(value, ast.Constant) and value.value is True)]
            elif isinstance(node.op, ast.Or):
                if any(isinstance(value, ast.Constant)
                       and value.value is True for value in node.values):
                    return ast.copy_location(ast.Constant(True), node)
                node.values = [value for value in node.values if not (
                    isinstance(value, ast.Constant) and value.value is False)]
            if not node.values:
                return ast.copy_location(ast.Constant(
                    isinstance(node.op, ast.And)), node)
            if len(node.values) == 1:
                return node.values[0]
            return node

        def visit_Compare(self, node):
            node = self.generic_visit(node)
            value = self._zero_mode_value(node)
            if value is not None:
                return ast.copy_location(ast.Constant(value), node)
            return node

        def visit_Assign(self, node):
            if self._mentions_removed_mode(node.targets[0]):
                return None
            # The historical solver report is a dictionary with a few
            # brake-mode-only entries.  Filtering the whole assignment when
            # one value mentions brake_modes_array also deletes the report
            # used by the post-solve diagnostics, causing a NameError after
            # IPOPT has finished.  Let visit_Dict remove only those retired
            # fields; retain the common report and feasibility result.
            if (isinstance(node.value, ast.Dict)
                    and any(isinstance(target, ast.Name)
                            and target.id in {"report", "candidate"}
                            for target in node.targets)):
                node.value = self.visit(node.value)
                return node
            if self._mentions_removed_mode(node):
                # Drop mode-mask initialization and any masked zero-throttle
                # assignments. Positive seed throttle is floored separately.
                return None
            return self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if self._mentions_removed_mode(node):
                return None
            return self.generic_visit(node)

        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and isinstance(node.value.func.value, ast.Name)
                    and node.value.func.value.id == "brake_modes"):
                return None
            if self._mentions_removed_mode(node):
                return None
            return self.generic_visit(node)

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (isinstance(node.func, ast.Name)
                    and node.func.id == "source_wheel_update"
                    and len(node.args) == 6):
                node.args.pop(2)  # fixed brake-torque argument is eliminated
            return node

        def visit_Dict(self, node):
            kept_keys, kept_values = [], []
            for key, value in zip(node.keys, node.values):
                key_text = key.value if isinstance(key, ast.Constant) and isinstance(
                    key.value, str) else ""
                value_text = (value.value if isinstance(value, ast.Constant)
                              and isinstance(value.value, str) else "")
                if re.search(r"brake|cawb|zero.throttle", key_text + " " + value_text,
                             re.IGNORECASE):
                    continue
                if key_text == "controls":
                    value = ast.List(elts=[
                        ast.Constant("physical steering rate"),
                        ast.Constant("positive normalized motor torque")],
                        ctx=ast.Load())
                kept_keys.append(self.visit(key) if key is not None else None)
                kept_values.append(self.visit(value))
            node.keys, node.values = kept_keys, kept_values
            return node

        def visit_List(self, node):
            values = []
            for value in node.elts:
                if (isinstance(value, ast.Constant) and isinstance(value.value, str)
                        and re.search(r"brake|cawb|zero.throttle", value.value,
                                      re.IGNORECASE)):
                    continue
                transformed = self.visit(value)
                if transformed is not None:
                    values.append(transformed)
            node.elts = values
            return node

    tree = DriveOnly().visit(ast.parse(source))
    ast.fix_missing_locations(tree)
    result = ast.unparse(tree) + "\n"
    # Initial elastic values are independent residual blocks in the same NLP.
    # Evaluate their concatenation once instead of asking Opti to build/evaluate
    # a separate numeric expression for every block.  This avoids repeated
    # traversal of the large 474-node graph before IPOPT is even entered.
    result, elastic_eval_sites = re.subn(
        r"(?m)^    elastic_initial_values = \{\}\n"
        r"    for elastic, residual in elastic_pairs:\n"
        r"        values = np\.asarray\(opti\.value\(residual, opti\.initial\(\)\), dtype=np\.float64\)\n"
        r"        if not np\.isfinite\(values\)\.all\(\):\n"
        r"            raise RuntimeError\('Non-finite original vehicle dynamics in synthetic seed'\)\n"
        r"        opti\.set_initial\(elastic, values\)\n"
        r"        elastic_initial_values\[elastic\] = values\.copy\(\)",
        "    elastic_initial_values = {}\n"
        "    elastic_residual_vector = ca.vertcat(*[ca.vec(residual) "
        "for _, residual in elastic_pairs])\n"
        "    print('Phase-I elastic seed evaluation begins', flush=True)\n"
        "    elastic_seed_vector = np.asarray(opti.value(\n"
        "        elastic_residual_vector, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "    print('Phase-I elastic seed evaluation complete', flush=True)\n"
        "    elastic_offset = 0\n"
        "    for elastic, residual in elastic_pairs:\n"
        "        elastic_count = int(elastic.numel())\n"
        "        values = elastic_seed_vector[elastic_offset:elastic_offset + elastic_count].reshape((int(elastic.size1()), int(elastic.size2())), order='F')\n"
        "        elastic_offset += elastic_count\n"
        "        if not np.isfinite(values).all():\n"
        "            raise RuntimeError('Non-finite original vehicle dynamics in synthetic seed')\n"
        "        opti.set_initial(elastic, values)\n"
        "        elastic_initial_values[elastic] = values.copy()\n"
        "    if elastic_offset != elastic_seed_vector.size:\n"
        "        raise RuntimeError('Elastic residual seed dimension mismatch')",
        result, count=1)
    if elastic_eval_sites != 1:
        raise RuntimeError(
            "Expected to combine initial elastic residual evaluation; "
            f"found {elastic_eval_sites}")
    # The source-feasible-seed gate runs before the Phase-I evaluator is
    # constructed, so it must have a real full-NLP constraint audit here.
    # Deferring this value leaves a `None` placeholder at the gate and forces
    # an already-feasible source seed through hundreds of pointless Phase-I
    # iterations.
    result, initial_diag_sites = re.subn(
        r"(?m)^    try:\n"
        r"        initial_constraints = _constraint_diagnostics\(opti\.value\(opti\.g, opti\.initial\(\)\), opti\.value\(opti\.lbg, opti\.initial\(\)\), opti\.value\(opti\.ubg, opti\.initial\(\)\), constraint_labels\)\n"
        r"    except \(RuntimeError, ValueError\) as error:\n"
        r"        initial_constraints = \{'max_constraint_violation': None, 'worst_constraints': \[\], 'diagnostic_error': str\(error\)\}",
        "    if diagnose_initial_only:\n"
        "        print('Initial-only constraint evaluation begins', flush=True)\n"
        "        try:\n"
        "            initial_constraints = _constraint_diagnostics(opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), opti.value(opti.ubg, opti.initial()), constraint_labels)\n"
        "        except (RuntimeError, ValueError) as error:\n"
        "            initial_constraints = {'max_constraint_violation': None, 'worst_constraints': [], 'diagnostic_error': str(error)}\n"
        "        print('Initial-only constraint evaluation complete', flush=True)\n"
        "    else:\n"
        "        print('Phase-I source-seed constraint audit begins', flush=True)\n"
        "        try:\n"
        "            initial_constraints = _constraint_diagnostics(opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), opti.value(opti.ubg, opti.initial()), constraint_labels)\n"
        "        except (RuntimeError, ValueError) as error:\n"
        "            initial_constraints = {'max_constraint_violation': None, 'worst_constraints': [], 'diagnostic_error': str(error)}\n"
        "        print('Phase-I source-seed constraint audit complete', flush=True)",
        result, count=1)
    if initial_diag_sites != 1:
        raise RuntimeError(
            "Expected to establish the Phase-I seed full-NLP constraint audit; "
            f"found {initial_diag_sites}")
    # The restored spatial OCP builds a large symbolic graph before it reaches
    # IPOPT.  Emit a sparse node counter from each full-mesh construction loop
    # so a long setup phase is visibly progressing in persistent/background
    # runs.  This is logging only and leaves the equations and iteration path
    # unchanged.
    result, node_progress_markers = re.subn(
        r"(?m)^([ ]*)for k in range\(n\):$",
        r"\1for k in range(n):\n"
        r"\1    if k % 25 == 0:\n"
        r"\1        print(f'Symbolic mesh pass node {k}/{n}', flush=True)",
        result)
    if node_progress_markers < 3:
        raise RuntimeError(
            "Expected sparse progress markers in the full-mesh OCP loops; "
            f"found {node_progress_markers}")
    # Keep CasADi's MX graph intact for this dense, high-resolution OCP.
    # Expanding all 474 nodes to SX was observed to consume many minutes
    # before the first numeric seed diagnostic or IPOPT iteration. This is a
    # representation/setup choice only; it does not alter the NLP equations,
    # bounds, objective, or throttle envelope.
    result, expansion_sites = re.subn(
        r"opti\.solver\('ipopt', \{'expand': True,",
        "opti.solver('ipopt', {'expand': False,",
        result)
    if expansion_sites != 2:
        raise RuntimeError(
            "Expected to disable eager expansion in both IPOPT stages; "
            f"found {expansion_sites} solver sites")
    result, solver_markers = re.subn(
        r"(?m)^([ ]*)opti\.solver\('ipopt',",
        r"\1print('IPOPT stage solver-interface setup begins', flush=True)\n"
        r"\1opti.solver('ipopt',",
        result)
    if solver_markers != 2:
        raise RuntimeError(
            "Expected progress markers at both IPOPT solver setup sites")
    result, phase1_setup_markers = re.subn(
        r"(?m)^(    )started = time\.perf_counter\(\)$",
        r"\1print('Phase-I solver interface ready; evaluating initial seed', flush=True)\n"
        r"\1started = time.perf_counter()",
        result, count=1)
    if phase1_setup_markers != 1:
        raise RuntimeError("Cannot mark completion of Phase-I IPOPT setup")
    result, numeric_function_markers = re.subn(
        r"(?m)^(        )(numerical_phase1 = ca\.Function\(.+\))$",
        r"\1print('Phase-I numeric function construction begins', flush=True)\n"
        r"\1\2\n"
        r"\1print('Phase-I numeric function construction complete', flush=True)",
        result, count=1)
    if numeric_function_markers != 1:
        raise RuntimeError("Cannot mark Phase-I numeric function construction")
    result, seed_measure_markers = re.subn(
        r"(?m)^(        )initial_measure = phase1_measure\(\)$",
        r"\1print('Phase-I initial numeric constraint evaluation begins', flush=True)\n"
        r"\1initial_measure = phase1_measure()\n"
        r"\1print('Phase-I initial numeric constraint evaluation complete', flush=True)\n"
        r"\1if initial_measure.get('max_constraint_violation') is not None:\n"
        r"\1    initial_constraints['max_constraint_violation'] = initial_measure.get('max_constraint_violation')\n"
        r"\1    initial_constraints['worst_constraints'] = initial_measure.get('worst_constraints', [])\n"
        r"\1initial_constraints['phase1_measure_error'] = initial_measure.get('error')",
        result, count=1)
    if seed_measure_markers != 1:
        raise RuntimeError("Cannot mark Phase-I initial constraint evaluation")
    result, bounds_cache_sites = re.subn(
        r"(?m)^        def phase1_measure\(debug=False\):$",
        "        phase1_bounds_cache = {}\n\n"
        "        def phase1_measure(debug=False):",
        result, count=1)
    if bounds_cache_sites != 1:
        raise RuntimeError("Cannot establish Phase-I numeric-bound cache")
    result, bounds_capture_sites = re.subn(
        r"(?m)^(                )g, low, high, raw, energy = numeric_evaluation\(primal_vector\(values\)\)$",
        r"\1g, low, high, raw, energy = numeric_evaluation(primal_vector(values))\n"
        r"\1phase1_bounds_cache['low'] = np.asarray(low, dtype=np.float64).reshape(-1).copy()\n"
        r"\1phase1_bounds_cache['high'] = np.asarray(high, dtype=np.float64).reshape(-1).copy()",
        result, count=1)
    if bounds_capture_sites != 1:
        raise RuntimeError("Cannot capture evaluated Phase-I constraint bounds")
    result, initial_bounds_markers = re.subn(
        r"(?m)^(        )_, low, high, _, _ = numeric_evaluation\(primal_vector\(initial_snapshot\)\)$",
        r"\1print('Phase-I hard-bound preparation begins', flush=True)\n"
        r"\1if 'low' not in phase1_bounds_cache or 'high' not in phase1_bounds_cache:\n"
        r"\1    raise RuntimeError('Phase-I initial measure did not cache constraint bounds')\n"
        r"\1low = phase1_bounds_cache['low']\n"
        r"\1high = phase1_bounds_cache['high']\n"
        r"\1print('Phase-I hard-bound preparation complete', flush=True)",
        result, count=1)
    if initial_bounds_markers != 1:
        raise RuntimeError("Cannot mark Phase-I hard-bound preparation")
    result, solve_call_markers = re.subn(
        r"(?m)^(            )result = opti\.solve_limited\(\)$",
        r"\1print('Phase-I entering CasADi/IPOPT solve call', flush=True)\n"
        r"\1result = opti.solve_limited()\n"
        r"\1print('Phase-I returned from CasADi/IPOPT solve call', flush=True)",
        result, count=1)
    if solve_call_markers != 1:
        raise RuntimeError("Cannot mark the actual Phase-I solver call")
    result, iteration_markers = re.subn(
        r"(?m)^([ ]*)if int\(iteration\) % checkpoint_stride:",
        r"\1print(f'Phase-I IPOPT iteration={int(iteration)}', flush=True)\n"
        r"\1if int(iteration) % checkpoint_stride:",
        result, count=1)
    if iteration_markers != 1:
        raise RuntimeError("Cannot mark per-iteration Phase-I progress")
    forbidden = re.compile(
        r"brake_mode|brake_torque|_brake_phases|CAWB|zero.throttle|"
        r"brake_modes_array|mode_transition_nodes|\bbrakes?\b|\bbraking\b",
        re.IGNORECASE)
    match = forbidden.search(result)
    if match:
        context = result[max(0, match.start() - 80):match.end() + 120]
        raise RuntimeError("Drive-only optimizer retained forbidden input behavior: "
                           + context.replace("\n", " "))
    return result
def transform_solver(source: str, mode: str = "limited-memory") -> str:
    expected = (
        _IMPORT_NEW, _HELPERS, _PATH_NEW, _TIRE_CALL_NEW,
        _FORCE_BALANCE_NEW, _SUSPENSION_CALL_NEW, _DYNAMIC_NEW,
        _VERTICAL_NEW, _PATH_RHS_NEW, _ATTITUDE_RHS_NEW,
        _SEED_MAP_NEW, _SEED_STACK_NEW, _SYNTHETIC_SPEED_SEED_NEW,
        _SYNTHETIC_SLIP_SEED_NEW, _SEED_ACCEL_NEW,
        _SEED_TORQUE_NEW, _SEED_THROTTLE_CLIP_NEW,
        _SEED_THROTTLE_INITIAL_FLOOR_NEW, _SEED_THROTTLE_NODE_BOUND_NEW,
        _SOURCE_FORCE_SEED_BLOCK,
        _SEED_LOADS_NEW, _PATH_NODE_CONSTRAINT_NEW,
        _PATH_NODE_LIST_NEW, _PATH_NODE_APPEND_NEW, _PATH_HEADING_NEW,
        _PATH_MIDPOINT_NEW, _FORCE_NODE_APPEND_NEW, _FORCE_RESIDUAL_NEW,
        _YAW_RATE_RHS_NEW, _RACELINE_WRITE_NEW, _STATE_FILE_NEW,
        _STATE_COLUMNS_NEW, _REPORT_OUTPUT_NEW,
        _WALL_CLEARANCE_NEW,
        "source-derived tire/suspension/body equations at each",
        "midpoint_load = list(mid_suspension_load)",
        "midpoint_source_dynamic_balance_axes",
        _DYNAMIC_PREFIXES_NEW, _INITIAL_DYNAMIC_PREFIXES_NEW,
        _DEFECT_DYNAMIC_PREFIXES_NEW, _YAW_DEFECT_GROUP_NEW,
        _YAW_BREAKDOWN_GROUP_NEW, _MIDPOINT_EXTRACTION_BLOCK,
        _MIDPOINT_REPORT_BLOCK, _MIDPOINT_CANDIDATE_BLOCK)
    expected = (*expected, _SPEED_SEED_SIGNATURE_NEW)
    if (all(token in source for token in expected)
            and "def _periodic_suspension_seed(" not in source):
        return source
    patched = source
    replacements = (
        (_IMPORT_OLD, _IMPORT_NEW),
        (_STATE_MARKER, _STATE_MARKER + _HELPERS),
        (_PATH_OLD, _PATH_NEW),
        (_TIRE_CALL_OLD, _TIRE_CALL_NEW),
        (_FORCE_BALANCE_OLD, _FORCE_BALANCE_NEW),
        (_SUSPENSION_CALL_OLD, _SUSPENSION_CALL_NEW),
        (_DYNAMIC_OLD, _DYNAMIC_NEW),
        (_VERTICAL_OLD, _VERTICAL_NEW),
        (_PATH_RHS_OLD, _PATH_RHS_NEW),
        (_YAW_RATE_RHS_OLD, _YAW_RATE_RHS_NEW),
        (_ATTITUDE_RHS_OLD, _ATTITUDE_RHS_NEW),
        (_PATH_NODE_CONSTRAINT_OLD, _PATH_NODE_CONSTRAINT_NEW),
        (_PATH_NODE_LIST_OLD, _PATH_NODE_LIST_NEW),
        (_PATH_NODE_APPEND_OLD, _PATH_NODE_APPEND_NEW),
        (_PATH_HEADING_OLD, _PATH_HEADING_NEW),
        (_PATH_MIDPOINT_OLD, _PATH_MIDPOINT_NEW),
        (_FORCE_NODE_APPEND_OLD, _FORCE_NODE_APPEND_NEW),
        (_FORCE_RESIDUAL_OLD, _FORCE_RESIDUAL_NEW),
        (_RACELINE_WRITE_OLD, _RACELINE_WRITE_NEW),
        (_STATE_FILE_OLD, _STATE_FILE_NEW),
        (_STATE_COLUMNS_OLD, _STATE_COLUMNS_NEW),
        (_REPORT_OUTPUT_OLD, _REPORT_OUTPUT_NEW),
        (_WALL_CLEARANCE_OLD, _WALL_CLEARANCE_NEW),
        (_SEED_MAP_OLD, _SEED_MAP_NEW),
        (_SEED_STACK_OLD, _SEED_STACK_NEW),
        (_SYNTHETIC_SPEED_SEED_OLD, _SYNTHETIC_SPEED_SEED_NEW),
        (_SYNTHETIC_SLIP_SEED_OLD, _SYNTHETIC_SLIP_SEED_NEW),
        (_SEED_ACCEL_OLD, _SEED_ACCEL_NEW),
        (_SEED_TORQUE_OLD, _SEED_TORQUE_NEW),
        (_SEED_THROTTLE_CLIP_OLD, _SEED_THROTTLE_CLIP_NEW),
        (_SEED_THROTTLE_INITIAL_FLOOR_OLD,
         _SEED_THROTTLE_INITIAL_FLOOR_NEW),
        (_SEED_THROTTLE_NODE_BOUND_OLD, _SEED_THROTTLE_NODE_BOUND_NEW),
    )
    for old, new in replacements:
        if patched.count(old) != 1:
            raise RuntimeError(
                "Unrecognized source OCP at v2.5.1 parity site: "
                + old.splitlines()[0].strip())
        patched = patched.replace(old, new, 1)
    if patched.count(_SPEED_SEED_SIGNATURE_OLD) != 1:
        raise RuntimeError('Cannot add the optional reference-speed seed input')
    patched = patched.replace(
        _SPEED_SEED_SIGNATURE_OLD, _SPEED_SEED_SIGNATURE_NEW, 1)
    for old, new, expected_count in (
            (_DYNAMIC_PREFIXES_OLD, _DYNAMIC_PREFIXES_NEW, 2),
            (_INITIAL_DYNAMIC_PREFIXES_OLD, _INITIAL_DYNAMIC_PREFIXES_NEW, 1),
            (_DEFECT_DYNAMIC_PREFIXES_OLD, _DEFECT_DYNAMIC_PREFIXES_NEW, 1),
            (_YAW_DEFECT_GROUP_OLD, _YAW_DEFECT_GROUP_NEW, 1),
            (_YAW_BREAKDOWN_GROUP_OLD, _YAW_BREAKDOWN_GROUP_NEW, 1)):
        if patched.count(old) != expected_count:
            raise RuntimeError(
                "Unrecognized source OCP at midpoint residual diagnostic site")
        patched = patched.replace(old, new)
    for marker, block in (
            (_MIDPOINT_EXTRACTION_MARKER, _MIDPOINT_EXTRACTION_BLOCK),
            (_MIDPOINT_REPORT_MARKER, _MIDPOINT_REPORT_BLOCK
             + _MIDPOINT_REPORT_MARKER),
            (_MIDPOINT_CANDIDATE_MARKER, _MIDPOINT_CANDIDATE_BLOCK
             + _MIDPOINT_CANDIDATE_MARKER)):
        if patched.count(marker) != 1:
            raise RuntimeError(
                "Unrecognized source OCP at midpoint witness output site")
        patched = patched.replace(marker, block, 1)
    if patched.count(_SEED_LOADS_OLD) != 1:
        raise RuntimeError("Unrecognized source suspension seed load site")
    patched = patched.replace(_SEED_LOADS_OLD, _SEED_LOADS_NEW, 1)
    if (patched.count(_PERIODIC_SEED_START) != 1
            or patched.count(_PERIODIC_SEED_END) != 1):
        raise RuntimeError("Unrecognized periodic suspension seed helper")
    helper_start = patched.index(_PERIODIC_SEED_START)
    helper_end = patched.index(_PERIODIC_SEED_END, helper_start)
    patched = patched[:helper_start] + patched[helper_end:]
    if (patched.count(_SEED_CONSISTENCY_START) != 1
            or patched.count(_SEED_CONSISTENCY_END) != 1):
        raise RuntimeError("Unrecognized suspension-seed consistency block")
    seed_start = patched.index(_SEED_CONSISTENCY_START)
    seed_end = patched.index(_SEED_CONSISTENCY_END, seed_start)
    patched = (patched[:seed_start] + _SOURCE_FORCE_SEED_BLOCK
               + patched[seed_end:])
    if patched.count(_SOURCE_INTERVAL_MIDPOINT_MARKER) != 1:
        raise RuntimeError(
            "Unrecognized source OCP at interval-midpoint feasibility site")
    patched = patched.replace(
        _SOURCE_INTERVAL_MIDPOINT_MARKER,
        _SOURCE_INTERVAL_MIDPOINT_BLOCK + _SOURCE_INTERVAL_MIDPOINT_MARKER,
        1)
    # The report consumes both body-axis force-balance residuals.  Some
    # locally cached v251 sources had the lateral definition dropped while
    # retaining its report field, causing a NameError after a complete solve.
    ay_residual = (
        "    ay_residual_mps2 = acceleration[1] - source_acceleration[1]")
    if ay_residual not in patched:
        ax_residual = (
            "    ax_residual_mps2 = acceleration[0] - source_acceleration[0]")
        if patched.count(ax_residual) != 1:
            raise RuntimeError(
                "Cannot restore the missing lateral force-balance diagnostic")
        patched = patched.replace(
            ax_residual, ax_residual + "\n" + ay_residual, 1)
    if patched.count(ay_residual) != 1:
        raise RuntimeError("Lateral force-balance diagnostic is ambiguous")
    return _drive_only_optimizer_source(patched)


def add_ipopt_cpu_time_limits(source: str) -> str:
    """Allow bounded Phase-I/time IPOPT calls without changing either NLP.

    IPOPT then returns a time-limit status through the existing diagnostic and
    checkpoint path instead of leaving an unbounded feasibility solve running
    for hours. A zero/unset environment value preserves the historical limit.
    """
    import_site = "import math\n"
    if source.count(import_site) != 1:
        raise RuntimeError("Cannot add Phase-I time-limit environment support")
    if "import os\n" not in source:
        source = source.replace(import_site, import_site + "import os\n", 1)

    solver_matches = list(re.finditer(
        r"(?m)^    opti\.solver\('ipopt', \{'expand': False, "
        r"'detect_simple_bounds': True, 'ipopt': "
        r"(?P<options>\{'max_iter': feasibility_max_iterations, [^\n]+\})\}, \{\}\)$",
        source))
    if len(solver_matches) != 1:
        raise RuntimeError(
            "Cannot locate the unique Phase-I IPOPT options for a finite time cap")
    solver_site = solver_matches[0].group(0)
    original_options = solver_matches[0].group('options')
    solver_replacement = (
        "    phase1_ipopt_options = " + original_options + "\n"
        "    phase1_max_cpu_time_s = float(os.environ.get("
        "'SDU_IPOPT_PHASE1_MAX_CPU_SECONDS', '0'))\n"
        "    if not math.isfinite(phase1_max_cpu_time_s) or phase1_max_cpu_time_s < 0.0:\n"
        "        raise ValueError('Phase-I CPU time limit must be finite and non-negative')\n"
        "    if phase1_max_cpu_time_s > 0.0:\n"
        "        phase1_ipopt_options['max_cpu_time'] = phase1_max_cpu_time_s\n"
        "        print(f'Phase-I IPOPT CPU-time limit: {phase1_max_cpu_time_s:.1f}s', flush=True)\n"
        "    opti.solver('ipopt', {'expand': False, 'detect_simple_bounds': True, "
        "'ipopt': phase1_ipopt_options}, {})")
    source = source.replace(solver_site, solver_replacement, 1)
    time_solver_site = (
        "        time_ipopt_options = {'max_iter': time_optimization_max_iterations, "
        "'tol': 1e-06, 'acceptable_tol': 1e-05, "
        "'print_level': int(print_level)}")
    time_solver_replacement = (
        time_solver_site + "\n"
        "        time_max_cpu_time_s = float(os.environ.get("
        "'SDU_IPOPT_TIME_MAX_CPU_SECONDS', '0'))\n"
        "        if not math.isfinite(time_max_cpu_time_s) or time_max_cpu_time_s < 0.0:\n"
        "            raise ValueError('Time-stage CPU time limit must be finite and non-negative')\n"
        "        if time_max_cpu_time_s > 0.0:\n"
        "            time_ipopt_options['max_cpu_time'] = time_max_cpu_time_s\n"
        "            print(f'IPOPT time-stage CPU-time limit: {time_max_cpu_time_s:.1f}s per continuation stage', flush=True)")
    if source.count(time_solver_site) != 1:
        raise RuntimeError(
            "Cannot locate the unique consolidated time-stage IPOPT options")
    source = source.replace(time_solver_site, time_solver_replacement, 1)
    wall_anchor = "        if time_hessian_approximation == 'limited-memory':"
    wall_options = (
        "        time_max_wall_time_s = float(os.environ.get('SDU_IPOPT_TIME_MAX_WALL_SECONDS', '0'))\n"
        "        if not math.isfinite(time_max_wall_time_s) or time_max_wall_time_s < 0.0:\n"
        "            raise ValueError('Time-stage wall limit must be finite and non-negative')\n"
        "        if time_max_wall_time_s > 0.0:\n"
        "            time_ipopt_options['max_wall_time'] = time_max_wall_time_s\n")
    if source.count(wall_anchor) != 1:
        raise RuntimeError('Cannot install time-stage wall limit')
    source = source.replace(wall_anchor, wall_options + wall_anchor, 1)
    ast.parse(source)
    return source


def eliminate_algebraic_vehicle_states(source: str) -> str:
    """Inline node loads/accelerations that are exact functions of plant state.

    Normal load is computed by the source suspension equations, then body
    acceleration by the source tire/body equations. Keeping either as an
    independent NLP coordinate only to add an equality back to that formula
    enlarges the KKT system without adding vehicle dynamics.
    """
    replacements = (
        (
            "    normal_load = opti.variable(4, n)\n"
            "    accel_xy = opti.variable(2, n)",
            "    normal_load_nodes = []\n"
            "    accel_xy_nodes = []",
        ),
        (
            "    opti.set_initial(normal_load, seed_load)\n"
            "    opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))\n",
            "",
        ),
        (
            "        opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))\n",
            "",
        ),
        (
            "        fz = [normal_load[i, k] for i in range(4)]",
            "        roll_euler_rate, pitch_euler_rate, _ = (\n"
            "            _unity_euler_rates_casadi(\n"
            "                roll[k], pitch[k], roll_rate[k], pitch_rate[k],\n"
            "                yaw_rate[k]))\n"
            "        predicted_load = suspension_loads_casadi(\n"
            "            model, body_height[k], body_height_rate[k], roll[k],\n"
            "            roll_euler_rate, pitch[k], pitch_euler_rate)\n"
            "        normal_load_nodes.append(ca.vertcat(*predicted_load))\n"
            "        suspension_travel, _ = suspension_travel_casadi(\n"
            "            model, body_height[k], body_height_rate[k], roll[k],\n"
            "            roll_euler_rate, pitch[k], pitch_euler_rate)\n"
            "        fz = list(predicted_load)",
        ),
        (
            "            add_constraint(normal_load[i, k] >= minimum_wheel_normal_load_n, f'normal_load_w{i}')",
            "            add_constraint(predicted_load[i] >= minimum_wheel_normal_load_n, f'normal_load_w{i}')",
        ),
        (
            "        ax, ay = (accel_xy[0, k], accel_xy[1, k])\n",
            "",
        ),
        (
            "        v_dot = fy / model.mass_kg + gravity_body[1] - linear_damping * v[k] + roll_rate[k] * body_vertical - yaw_rate[k] * u[k]\n",
            "        v_dot = fy / model.mass_kg + gravity_body[1] - linear_damping * v[k] + roll_rate[k] * body_vertical - yaw_rate[k] * u[k]\n"
            "        ax, ay = u_dot, v_dot\n"
            "        accel_xy_nodes.append(ca.vertcat(ax, ay))\n",
        ),
        (
            "        add_constraint(ax - u_dot == 0, f'longitudinal_force_balance_k{k}')\n"
            "        add_constraint(ay - v_dot == 0, f'lateral_force_balance_k{k}')\n",
            "",
        ),
        (
            "        roll_euler_rate, pitch_euler_rate, _ = _unity_euler_rates_casadi(roll[k], pitch[k], roll_rate[k], pitch_rate[k], yaw_rate[k])\n"
            "        predicted_load = suspension_loads_casadi(model, body_height[k], body_height_rate[k], roll[k], roll_euler_rate, pitch[k], pitch_euler_rate)\n"
            "        suspension_travel, _ = suspension_travel_casadi(model, body_height[k], body_height_rate[k], roll[k], roll_euler_rate, pitch[k], pitch_euler_rate)",
            "        suspension_travel, _ = suspension_travel_casadi(model, body_height[k], body_height_rate[k], roll[k], roll_euler_rate, pitch[k], pitch_euler_rate)",
        ),
        (
            "            add_constraint((normal_load[i, k] - predicted_load[i]) / 10.0 == 0, f'unity_spring_damper_load_w{i}')\n",
            "",
        ),
        (
            "    lap_time = 0.0\n    dynamic_indices =",
            "    normal_load = ca.horzcat(*normal_load_nodes)\n"
            "    accel_xy = ca.horzcat(*accel_xy_nodes)\n"
            "    lap_time = 0.0\n    dynamic_indices =",
        ),
        (
            "        physical_vars = (x, qdelta, throttle, phase_lengths, normal_load, accel_xy, *elastic_vars)",
            "        physical_vars = (x, qdelta, throttle, phase_lengths, *elastic_vars)",
        ),
        (
            ", phase_lengths: seed_matrix(phase_lengths, phase_initial_lengths), normal_load: seed_matrix(normal_load, seed_load), accel_xy: seed_matrix(accel_xy, np.vstack((seed_ax, seed_ay)))",
            ", phase_lengths: seed_matrix(phase_lengths, phase_initial_lengths)",
        ),
    )
    transformed = source
    for old, new in replacements:
        count = transformed.count(old)
        if count != 1:
            raise RuntimeError(
                "Cannot algebraically condense v2.5.1 vehicle states; "
                f"expected one source anchor, found {count}: "
                + old.splitlines()[0].strip())
        transformed = transformed.replace(old, new, 1)

    repeated_time_solver_setup = (
        "            opti.minimize(lap_time)\n"
        "            time_ipopt_options = {'max_iter': time_optimization_max_iterations, 'tol': 1e-06, 'acceptable_tol': 1e-05, 'print_level': int(print_level)}\n"
        "            if time_hessian_approximation == 'limited-memory':\n"
        "                time_ipopt_options.update({'hessian_approximation': 'limited-memory', 'limited_memory_max_history': 20})\n"
        "            print('IPOPT stage solver-interface setup begins', flush=True)\n"
        "            opti.solver('ipopt', {'expand': False, 'ipopt': time_ipopt_options}, {})\n")
    if transformed.count(repeated_time_solver_setup) != 1:
        raise RuntimeError(
            "Cannot consolidate the repeated IPOPT time-stage setup")
    transformed = transformed.replace(repeated_time_solver_setup, "", 1)
    time_loop_anchor = (
        "        trust_region_factors = time_trust_region_factors\n"
        "        for trust_factor in trust_region_factors:\n")
    time_solver_once = (
        "        trust_region_factors = time_trust_region_factors\n"
        "        opti.minimize(lap_time)\n"
        "        time_ipopt_options = {'max_iter': time_optimization_max_iterations, 'tol': 1e-06, 'acceptable_tol': 1e-05, 'print_level': int(print_level)}\n"
        "        if time_hessian_approximation == 'limited-memory':\n"
        "            time_ipopt_options.update({'hessian_approximation': 'limited-memory', 'limited_memory_max_history': 20})\n"
        "        print('IPOPT time-continuation solver setup (one-time)', flush=True)\n"
        "        opti.solver('ipopt', {'expand': False, 'ipopt': time_ipopt_options}, {})\n"
        "        for trust_factor in trust_region_factors:\n"
        "            print(f'IPOPT time stage trust_factor={trust_factor:g}', flush=True)\n")
    if transformed.count(time_loop_anchor) != 1:
        raise RuntimeError("Cannot locate IPOPT time-continuation loop")
    transformed = transformed.replace(time_loop_anchor, time_solver_once, 1)
    stage_diagnostic_anchor = "            if continuation_restartable:\n"
    if transformed.count(stage_diagnostic_anchor) != 1:
        raise RuntimeError("Cannot add minimum-time stage diagnostics")
    transformed = transformed.replace(
        stage_diagnostic_anchor,
        "            print('IPOPT time-stage result: ' + json.dumps(stage_reports[-1], sort_keys=True, default=str), flush=True)\n"
        + stage_diagnostic_anchor,
        1)

    warm_start_loop = (
        "        for variable in (x, qdelta, throttle, phase_lengths, "
        "normal_load, accel_xy):")
    if transformed.count(warm_start_loop) != 2:
        raise RuntimeError("Cannot condense all Phase-I/time-stage warm starts")
    transformed = transformed.replace(
        warm_start_loop,
        "        for variable in (x, qdelta, throttle, phase_lengths):")

    restart_loads = "                loads = np.asarray(winning_snapshot[normal_load], dtype=np.float64)\n"
    restart_accel = "                accel = np.asarray(winning_snapshot[accel_xy], dtype=np.float64)\n"
    if transformed.count(restart_loads) != 1 or transformed.count(restart_accel) != 1:
        raise RuntimeError("Cannot evaluate condensed vehicle states for Phase-I restart")
    transformed = transformed.replace(
        restart_loads + restart_accel,
        "                algebraic_state_values = ca.Function(\n"
        "                    'phase1_condensed_vehicle_state',\n"
        "                    [opti.x, opti.p], [normal_load, accel_xy])\n"
        "                restart_load_values, restart_accel_values = algebraic_state_values(\n"
        "                    primal_vector(winning_snapshot),\n"
        "                    np.asarray(opti.value(opti.p), dtype=np.float64).reshape(-1))\n"
        "                loads = np.asarray(restart_load_values, dtype=np.float64)\n"
        "                accel = np.asarray(restart_accel_values, dtype=np.float64)\n",
        1)
    diagnose_marker = (
        "    if diagnose_initial_only:\n"
        "        print('Initial-only constraint evaluation begins', flush=True)")
    if transformed.count(diagnose_marker) != 1:
        raise RuntimeError("Cannot report condensed OCP dimensions in diagnostic mode")
    transformed = transformed.replace(
        diagnose_marker,
        "    if diagnose_initial_only:\n"
        "        dynamic_residuals = sum(int(residual.numel()) for _, residual in elastic_pairs)\n"
        "        print(f'Initial-only NLP dimensions: variables={int(opti.nx)}, constraints={int(opti.ng)}, dynamic_residuals={dynamic_residuals}', flush=True)\n"
        "        print('Initial-only constraint evaluation begins', flush=True)",
        1)
    ast.parse(transformed)
    if "normal_load = opti.variable(4, n)" in transformed:
        raise RuntimeError("Condensed OCP retained independent normal-load variables")
    if "accel_xy = opti.variable(2, n)" in transformed:
        raise RuntimeError("Condensed OCP retained independent acceleration variables")
    if "unity_spring_damper_load_w{i}" in transformed:
        raise RuntimeError("Condensed OCP retained redundant node-load equations")
    if "longitudinal_force_balance_k{k}" in transformed or "lateral_force_balance_k{k}" in transformed:
        raise RuntimeError("Condensed OCP retained redundant node-acceleration equations")
    if "normal_load = ca.horzcat(*normal_load_nodes)" not in transformed:
        raise RuntimeError("Condensed normal-load expressions are not assembled")
    if "accel_xy = ca.horzcat(*accel_xy_nodes)" not in transformed:
        raise RuntimeError("Condensed acceleration expressions are not assembled")
    return transformed


def eliminate_phase1_slack_variables(source: str) -> str:
    """Algebraically substitute each residual for its auxiliary slack variable.

    The original Phase-I system has r-z=0, |z|<=cap and minimizes ||S z||^2.
    This form substitutes z=r exactly, preserving the same feasible set,
    objective and strict cap-zero check while reducing decision variables.
    """
    replacements = (
        (
            """            elastic = opti.variable(residual.size1(), residual.size2())
            elastic_pairs.append((elastic, residual))
            elastic_vars.append(elastic)
            scale = 10.0 if any((name.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for name in labels)) else 1.0
            elastic_scaled.append(ca.vec(elastic / scale))
            opti.subject_to(residual - elastic == 0)
            constraint_labels.extend(labels)
            opti.subject_to(opti.bounded(-elastic_cap, elastic, elastic_cap))
            constraint_labels.extend(['elastic_bound_' + label for label in labels])""",
            """            elastic = None
            elastic_pairs.append((elastic, residual))
            scale = 10.0 if any((name.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for name in labels)) else 1.0
            elastic_scaled.append(ca.vec(residual / scale))
            constraint_labels.extend(labels)
            opti.subject_to(opti.bounded(-elastic_cap, residual, elastic_cap))
            constraint_labels.extend(['elastic_bound_' + label for label in labels])""",
        ),
        (
            """    elastic_initial_values = {}
    elastic_residual_vector = ca.vertcat(*[ca.vec(residual) for _, residual in elastic_pairs])
    print('Phase-I elastic seed evaluation begins', flush=True)
    elastic_seed_vector = np.asarray(opti.value(
        elastic_residual_vector, opti.initial()), dtype=np.float64).reshape(-1)
    print('Phase-I elastic seed evaluation complete', flush=True)
    elastic_offset = 0
    for elastic, residual in elastic_pairs:
        elastic_count = int(elastic.numel())
        values = elastic_seed_vector[elastic_offset:elastic_offset + elastic_count].reshape((int(elastic.size1()), int(elastic.size2())), order='F')
        elastic_offset += elastic_count
        if not np.isfinite(values).all():
            raise RuntimeError('Non-finite original vehicle dynamics in synthetic seed')
        opti.set_initial(elastic, values)
        elastic_initial_values[elastic] = values.copy()
    if elastic_offset != elastic_seed_vector.size:
        raise RuntimeError('Elastic residual seed dimension mismatch')""",
            """    elastic_initial_values = {}
    elastic_residual_vector = ca.vertcat(*[ca.vec(residual) for _, residual in elastic_pairs])
    print('Phase-I direct-residual seed evaluation begins', flush=True)
    elastic_seed_vector = np.asarray(opti.value(
        elastic_residual_vector, opti.initial()), dtype=np.float64).reshape(-1)
    print('Phase-I direct-residual seed evaluation complete', flush=True)
    if not np.isfinite(elastic_seed_vector).all():
        raise RuntimeError('Non-finite original vehicle dynamics in synthetic seed')
    elastic_offset = 0
    for elastic_index, (_, residual) in enumerate(elastic_pairs):
        residual_count = int(residual.numel())
        values = elastic_seed_vector[elastic_offset:elastic_offset + residual_count].reshape((int(residual.size1()), int(residual.size2())), order='F')
        elastic_offset += residual_count
        elastic_initial_values[elastic_index] = values.copy()
    if elastic_offset != elastic_seed_vector.size:
        raise RuntimeError('Direct residual seed dimension mismatch')""",
        ),
        (
            """    for elastic in elastic_vars:
        values = np.asarray(elastic_initial_values[elastic], dtype=np.float64).reshape(-1, order='F')
        labels = initial_dynamic_labels[label_offset:label_offset + values.size]
        label_offset += values.size""",
            """    for elastic_index, (_, _) in enumerate(elastic_pairs):
        values = np.asarray(elastic_initial_values[elastic_index], dtype=np.float64).reshape(-1, order='F')
        labels = initial_dynamic_labels[label_offset:label_offset + values.size]
        label_offset += values.size""",
        ),
        (
            """                actual_elastic = np.concatenate([np.asarray(values[elastic]).reshape(-1, order='F') for elastic in elastic_vars])
                defect = float(np.max(np.abs(raw)))
                elast_max = float(np.max(np.abs(actual_elastic)))
                if not (np.isfinite(raw).all() and np.isfinite(actual_elastic).all() and np.isfinite(energy)):
                    diag['max_constraint_violation'] = None
                diag.update(raw_max_dynamic_defect=defect, max_elastic_slack=elast_max, elastic_merit=energy, raw_dynamic_breakdown=defect_breakdown(raw))""",
            """                defect = float(np.max(np.abs(raw)))
                elast_max = defect
                if not (np.isfinite(raw).all() and np.isfinite(energy)):
                    diag['max_constraint_violation'] = None
                diag.update(raw_max_dynamic_defect=defect, max_elastic_slack=elast_max, elastic_merit=energy, raw_dynamic_breakdown=defect_breakdown(raw))""",
        ),
        (
            """            for elastic, residual in elastic_pairs:
                size = int(elastic.numel())
                raw = all_raw[raw_offset:raw_offset + size].reshape((int(elastic.size1()), int(elastic.size2())), order='F')
                raw_offset += size
                if float(np.max(np.abs(raw))) > elastic_limit:
                    return (None, None)
                labels = dynamic_labels[label_offset:label_offset + size]
                label_offset += size
                norm = 10.0 if any((label.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for label in labels)) else 1.0
                energy += float(np.sum((raw / norm) ** 2))
                max_defect = max(max_defect, float(np.max(np.abs(raw))))
                values[elastic] = raw.copy()""",
            """            for _, residual in elastic_pairs:
                size = int(residual.numel())
                raw = all_raw[raw_offset:raw_offset + size].reshape((int(residual.size1()), int(residual.size2())), order='F')
                raw_offset += size
                if float(np.max(np.abs(raw))) > elastic_limit:
                    return (None, None)
                labels = dynamic_labels[label_offset:label_offset + size]
                label_offset += size
                norm = 10.0 if any((label.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for label in labels)) else 1.0
                energy += float(np.sum((raw / norm) ** 2))
                max_defect = max(max_defect, float(np.max(np.abs(raw))))""",
        ),
    )
    transformed = source
    for old, new in replacements:
        count = transformed.count(old)
        if count != 1:
            raise RuntimeError(
                "Cannot construct algebraically reduced v2.5.1 Phase-I; "
                f"expected one occurrence of anchor, found {count}: "
                + old.splitlines()[0].strip())
        transformed = transformed.replace(old, new, 1)
    solve_marker = "            print('Phase-I entering CasADi/IPOPT solve call', flush=True)"
    if transformed.count(solve_marker) != 1:
        raise RuntimeError("Cannot instrument reduced Phase-I NLP dimensions")
    transformed = transformed.replace(
        solve_marker,
        "            dynamic_residuals = sum(int(residual.numel()) for _, residual in elastic_pairs)\n"
        "            print(f'Phase-I reduced NLP dimensions: variables={int(opti.nx)}, constraints={int(opti.ng)}, dynamic_residuals={dynamic_residuals}', flush=True)\n"
        + solve_marker,
        1)
    ast.parse(transformed)
    if "elastic = opti.variable(residual.size1(), residual.size2())" in transformed:
        raise RuntimeError("Reduced Phase-I retained residual slack variables")
    if "opti.subject_to(residual - elastic == 0)" in transformed:
        raise RuntimeError("Reduced Phase-I retained slack equality constraints")
    return transformed


def make_phase1_objective_only(source: str) -> str:
    """Use residuals as the Phase-I objective; restore exact equalities after."""
    transformed = eliminate_phase1_slack_variables(source)

    def replace_once(old: str, new: str, label: str) -> None:
        nonlocal transformed
        count = transformed.count(old)
        if count != 1:
            raise RuntimeError(
                f"Cannot build objective-only Phase-I ({label}); found {count} anchors")
        transformed = transformed.replace(old, new, 1)

    replace_once(
        """    elastic_pairs = []
    elastic_label_by_variable = {}
    elastic_vars = []
    elastic_scaled = []
    elastic_cap = opti.parameter()
    opti.set_value(elastic_cap, 100000.0)""",
        """    elastic_pairs = []
    phase1_dynamic_labels = []
    elastic_label_by_variable = {}
    elastic_vars = []
    elastic_scaled = []
    phase1_residual_checkpoint_limit = 100000.0""",
        "residual registry")
    replace_once(
        """            elastic_pairs.append((elastic, residual))
            scale = 10.0 if any((name.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for name in labels)) else 1.0
            elastic_scaled.append(ca.vec(residual / scale))
            constraint_labels.extend(labels)
            opti.subject_to(opti.bounded(-elastic_cap, residual, elastic_cap))
            constraint_labels.extend(['elastic_bound_' + label for label in labels])""",
        """            elastic_pairs.append((elastic, residual))
            phase1_dynamic_labels.extend(labels)
            scale = 10.0 if any((name.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for name in labels)) else 1.0
            elastic_scaled.append(ca.vec(residual / scale))""",
        "dynamic residual constraints")
    replace_once("    opti.set_value(elastic_cap, 100000.0)\n", "",
                 "post-objective cap")
    replace_once(
        "    add_constraint(feasibility_objective <= trust_region_bound, 'captured_motion_trust_region')\n",
        "",
        "defer seed-centered trust region until time optimization")
    replace_once(
        "    opti.minimize(elastic_merit + 0.0001 * feasibility_objective)",
        "    opti.minimize(elastic_merit)",
        "make Phase-I objective pure dynamics residual")
    replace_once(
        "    initial_dynamic_labels = [label for label in constraint_labels if label.startswith(('periodic_collocation_state_', 'physx_implicit_wheel_update_', 'longitudinal_force_balance_', 'lateral_force_balance_', 'unity_spring_damper_load_', 'yaw_moment_balance_', 'heave_force_balance_', 'roll_moment_balance_', 'pitch_moment_balance_'))]",
        "    initial_dynamic_labels = phase1_dynamic_labels.copy()",
        "initial residual labels")
    replace_once(
        "    initial_raw_values = [item['value'] for samples in initial_raw_samples.values() for item in samples]",
        "    for group, samples in initial_raw_samples.items():\n"
        "        group_values = np.asarray([item['value'] for item in samples], dtype=np.float64)\n"
        "        worst_index = int(np.argmax(np.abs(group_values)))\n"
        "        initial_raw_dynamic_defects[group]['worst_signed_value'] = float(group_values[worst_index])\n"
        "        initial_raw_dynamic_defects[group]['worst_neighborhood'] = [\n"
        "            {'sample_index': index, 'label': samples[index]['label'],\n"
        "             'signed_value': float(group_values[index])}\n"
        "            for index in range(max(0, worst_index - 2),\n"
        "                               min(len(samples), worst_index + 3))]\n"
        "    initial_raw_values = [item['value'] for samples in initial_raw_samples.values() for item in samples]",
        "raw residual neighborhood diagnostics")
    replace_once(
        "            labels = [name for name in constraint_labels if name.startswith(('periodic_collocation_state_', 'physx_implicit_wheel_update_', 'longitudinal_force_balance_', 'lateral_force_balance_', 'unity_spring_damper_load_', 'yaw_moment_balance_', 'heave_force_balance_', 'roll_moment_balance_', 'pitch_moment_balance_'))]",
        "            labels = phase1_dynamic_labels",
        "residual breakdown labels")
    replace_once(
        """                elast_max = defect
                if not (np.isfinite(raw).all() and np.isfinite(energy)):
                    diag['max_constraint_violation'] = None
                diag.update(raw_max_dynamic_defect=defect, max_elastic_slack=elast_max, elastic_merit=energy, raw_dynamic_breakdown=defect_breakdown(raw))""",
        """                if not (np.isfinite(raw).all() and np.isfinite(energy)):
                    diag['max_constraint_violation'] = None
                diag.update(raw_max_dynamic_defect=defect, elastic_merit=energy, raw_dynamic_breakdown=defect_breakdown(raw))""",
        "residual diagnostics")
    replace_once(
        "            return {'max_constraint_violation': None, 'raw_max_dynamic_defect': None, 'max_elastic_slack': None, 'elastic_merit': None, 'worst_constraints': [], 'error': str(exc)}",
        "            return {'max_constraint_violation': None, 'raw_max_dynamic_defect': None, 'elastic_merit': None, 'worst_constraints': [], 'error': str(exc)}",
        "residual diagnostic error")
    replace_once(
        "'seed_trust_region_bound': float(opti.value(trust_region_bound))",
        "'phase1_seed_trust_region_active': False",
        "report Phase-I trust-region semantics")
    replace_once(
        "        stage2_initial_diagnostics = _constraint_diagnostics(opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), opti.value(opti.ubg, opti.initial()), constraint_labels)",
        "        feasibility_reference = max(float(stage_values(stage1_solution, stage1_stats, feasibility_objective)), minimum_trust_region_reference)\n        opti.set_value(trust_region_bound, feasibility_reference * time_trust_region_factors[0])\n        opti.subject_to(feasibility_objective <= trust_region_bound)\n        constraint_labels.append('captured_motion_trust_region')\n        stage2_initial_diagnostics = _constraint_diagnostics(opti.value(opti.g, opti.initial()), opti.value(opti.lbg, opti.initial()), opti.value(opti.ubg, opti.initial()), constraint_labels)",
        "restore seed trust region only for minimum-time phase")
    replace_once(
        "        feasibility_reference = max(float(stage_values(stage1_solution, stage1_stats, feasibility_objective)), minimum_trust_region_reference)\n        stage_reports[0]['feasibility_objective_value'] = feasibility_reference",
        "        stage_reports[0]['feasibility_objective_value'] = feasibility_reference",
        "reuse Phase-I reference for time trust region")
    replace_once(
        "        dynamic_labels = [name for name in constraint_labels if name.startswith(elastic_physical_prefixes)]",
        "        dynamic_labels = phase1_dynamic_labels",
        "checkpoint residual labels")
    replace_once(
        "        numerical_phase1 = ca.Function('phase1_vectorized_evaluation', [opti.x, opti.p], [opti.g, opti.lbg, opti.ubg, ca.vertcat(*[ca.vec(residual) for _, residual in elastic_pairs]), elastic_merit])",
        "        numerical_phase1 = ca.Function('phase1_vectorized_evaluation', [opti.x, opti.p], [opti.g, opti.lbg, opti.ubg, ca.vertcat(*[ca.vec(residual) for _, residual in elastic_pairs]), elastic_merit, opti.f])",
        "capture exact Phase-I objective")
    replace_once(
        "            g, low, high, raw, energy = numerical_phase1(primal, params)\n            return (np.asarray(g, dtype=np.float64).reshape(-1), np.asarray(low, dtype=np.float64).reshape(-1), np.asarray(high, dtype=np.float64).reshape(-1), np.asarray(raw, dtype=np.float64).reshape(-1), float(energy))",
        "            g, low, high, raw, energy, objective = numerical_phase1(primal, params)\n            return (np.asarray(g, dtype=np.float64).reshape(-1), np.asarray(low, dtype=np.float64).reshape(-1), np.asarray(high, dtype=np.float64).reshape(-1), np.asarray(raw, dtype=np.float64).reshape(-1), float(energy), float(objective))",
        "evaluate exact Phase-I objective")
    replace_once(
        "                g, low, high, raw, energy = numeric_evaluation(primal_vector(values))",
        "                g, low, high, raw, energy, objective = numeric_evaluation(primal_vector(values))",
        "measure exact Phase-I objective")
    replace_once(
        "                diag.update(raw_max_dynamic_defect=defect, elastic_merit=energy, raw_dynamic_breakdown=defect_breakdown(raw))",
        "                diag.update(raw_max_dynamic_defect=defect, elastic_merit=energy, phase1_objective=objective, raw_dynamic_breakdown=defect_breakdown(raw))",
        "record exact Phase-I objective")
    replace_once(
        "            elastic_limit = float(opti.value(elastic_cap))",
        "            elastic_limit = phase1_residual_checkpoint_limit",
        "checkpoint residual cap")
    replace_once(
        "            g, _, _, all_raw, _ = numeric_evaluation(primal_vector(values))",
        "            g, _, _, all_raw, _, objective = numeric_evaluation(primal_vector(values))",
        "evaluate exact checkpoint objective")
    replace_once(
        "        def projected_trial(values=None):",
        "        checkpoint_rejection_diagnostics = {'hard_feasible_trials': 0, 'finite_trials': 0, 'min_hard_violation': None, 'worst_hard_constraint': None, 'min_residual_violation': None, 'ipopt_trial_objective': None, 'ipopt_trial_max_defect': None, 'ipopt_trial_hard_violation': None, 'ipopt_trial_worst_constraint': None}\n\n        def projected_trial(values=None):",
        "checkpoint rejection diagnostics registry")
    replace_once(
        "        def checked_checkpoint(trial, reference, reference_measure):\n            for alpha in checkpoint_alphas:",
        "        def checked_checkpoint(trial, reference, reference_measure):\n            checkpoint_rejection_diagnostics.update(hard_feasible_trials=0, finite_trials=0, min_hard_violation=None, worst_hard_constraint=None, min_residual_violation=None, ipopt_trial_objective=None, ipopt_trial_max_defect=None, ipopt_trial_hard_violation=None, ipopt_trial_worst_constraint=None)\n            for alpha in checkpoint_alphas:",
        "reset checkpoint diagnostics")
    replace_once(
        "            g, _, _, all_raw, _, objective = numeric_evaluation(primal_vector(values))\n            if not np.isfinite(g).all() or not np.isfinite(all_raw).all():",
        "            g, _, _, all_raw, _, objective = numeric_evaluation(primal_vector(values))\n            if checkpoint_rejection_diagnostics['finite_trials'] == 0:\n                checkpoint_rejection_diagnostics['ipopt_trial_objective'] = objective\n                checkpoint_rejection_diagnostics['ipopt_trial_max_defect'] = float(np.max(np.abs(all_raw)))\n            if not np.isfinite(g).all() or not np.isfinite(all_raw).all():",
        "record unprojected IPOPT objective and defect")
    replace_once(
        "            hard_viol = max(float(np.max(low[hard_indices] - g[hard_indices])), float(np.max(g[hard_indices] - high[hard_indices])), 0.0)\n            if hard_viol > NLP_FEASIBILITY_TOLERANCE:\n                return (None, None)",
        "            hard_low_viol = np.maximum(low[hard_indices] - g[hard_indices], 0.0)\n            hard_high_viol = np.maximum(g[hard_indices] - high[hard_indices], 0.0)\n            hard_violations = np.maximum(hard_low_viol, hard_high_viol)\n            hard_viol = float(np.max(hard_violations)) if hard_violations.size else 0.0\n            hard_names = np.asarray(constraint_labels, dtype=object)[hard_indices]\n            hard_worst_constraint = str(hard_names[int(np.argmax(hard_violations))]) if hard_violations.size and hard_viol > 0.0 else None\n            checkpoint_rejection_diagnostics['finite_trials'] += 1\n            previous_min = checkpoint_rejection_diagnostics['min_hard_violation']\n            if previous_min is None or hard_viol < previous_min:\n                checkpoint_rejection_diagnostics['min_hard_violation'] = hard_viol\n                checkpoint_rejection_diagnostics['worst_hard_constraint'] = hard_worst_constraint\n            if hard_viol > NLP_FEASIBILITY_TOLERANCE:\n                return (None, {'hard_constraint_violation': hard_viol, 'worst_hard_constraint': hard_worst_constraint})\n            checkpoint_rejection_diagnostics['hard_feasible_trials'] += 1",
        "checkpoint hard-constraint diagnosis")
    replace_once(
        "            checkpoint_rejection_diagnostics['finite_trials'] += 1",
        "            if checkpoint_rejection_diagnostics['finite_trials'] == 0:\n                checkpoint_rejection_diagnostics['ipopt_trial_hard_violation'] = hard_viol\n                checkpoint_rejection_diagnostics['ipopt_trial_worst_constraint'] = hard_worst_constraint\n            checkpoint_rejection_diagnostics['finite_trials'] += 1",
        "record first IPOPT hard-row violation")
    replace_once(
        "                if float(np.max(np.abs(raw))) > elastic_limit:\n                    return (None, None)",
        "                residual_violation = float(np.max(np.abs(raw)))\n                previous_residual_violation = checkpoint_rejection_diagnostics['min_residual_violation']\n                if previous_residual_violation is None or residual_violation < previous_residual_violation:\n                    checkpoint_rejection_diagnostics['min_residual_violation'] = residual_violation\n                if residual_violation > elastic_limit:\n                    return (None, {'hard_constraint_violation': hard_viol, 'worst_hard_constraint': hard_worst_constraint, 'residual_violation': residual_violation})",
        "checkpoint residual-bound diagnosis")
    replace_once(
        "            return (values, {'elastic_merit': energy, 'raw_max_dynamic_defect': max_defect, 'max_elastic_slack': max_defect, 'max_constraint_violation': hard_viol})",
        "            return (values, {'elastic_merit': energy, 'phase1_objective': objective, 'raw_max_dynamic_defect': max_defect, 'max_elastic_slack': max_defect, 'max_constraint_violation': hard_viol, 'hard_constraint_violation': hard_viol, 'worst_hard_constraint': hard_worst_constraint})",
        "checkpoint diagnostics")
    replace_once(
        "                reference_merit = None if reference_measure is None else reference_measure.get('elastic_merit')\n                reference_defect = None if reference_measure is None else reference_measure.get('raw_max_dynamic_defect')\n                improves_worst_defect = reference_defect is None or defect < reference_defect - 1e-08\n                holds_worst_defect_and_improves_merit = reference_defect is not None and defect <= reference_defect + 1e-08 and (reference_merit is None or merit < reference_merit - 1e-08)\n                if np.isfinite(merit) and np.isfinite(defect) and (improves_worst_defect or holds_worst_defect_and_improves_merit):\n                    return (candidate, diag, alpha)",
        "                reference_objective = None if reference_measure is None else reference_measure.get('phase1_objective')\n                objective = diag['phase1_objective']\n                if np.isfinite(objective) and np.isfinite(merit) and np.isfinite(defect) and (reference_objective is None or objective < reference_objective - 1e-08):\n                    return (candidate, diag, alpha)",
        "checkpoint selection matches NLP objective")
    replace_once(
        "                    checkpoint_status = 'hard_reject_or_no_descent'",
        "                    checkpoint_status = ('hard_constraint_reject' if checkpoint_rejection_diagnostics['hard_feasible_trials'] == 0 else 'feasible_no_descent')",
        "distinguish checkpoint rejection causes")
    replace_once(
        "                print(f'Phase-I iter={int(iteration):4d} status={checkpoint_status} alpha={checkpoint_alpha} candidate_merit={(checkpoint_measure or {}).get('elastic_merit')} best_merit={progress_measure.get('elastic_merit')} best_max_defect={progress_measure.get('raw_max_dynamic_defect')} accepted={accepted_ipopt_iters} callback_errors={callback_errors}', flush=True)",
        "                print(f'Phase-I iter={int(iteration):4d} status={checkpoint_status} alpha={checkpoint_alpha} candidate_objective={(checkpoint_measure or {}).get('phase1_objective')} best_objective={progress_measure.get('phase1_objective')} residual_sum_squares={progress_measure.get('elastic_merit')} max_defect={progress_measure.get('raw_max_dynamic_defect')} accepted={accepted_ipopt_iters} callback_errors={callback_errors} ipopt_trial_objective={checkpoint_rejection_diagnostics[\"ipopt_trial_objective\"]} ipopt_trial_defect={checkpoint_rejection_diagnostics[\"ipopt_trial_max_defect\"]} ipopt_trial_hard_violation={checkpoint_rejection_diagnostics[\"ipopt_trial_hard_violation\"]} ipopt_trial_worst_constraint={checkpoint_rejection_diagnostics[\"ipopt_trial_worst_constraint\"]} hard_feasible_trials={checkpoint_rejection_diagnostics[\"hard_feasible_trials\"]} min_hard_violation={checkpoint_rejection_diagnostics[\"min_hard_violation\"]} worst_hard_constraint={checkpoint_rejection_diagnostics[\"worst_hard_constraint\"]}', flush=True)",
        "checkpoint rejection progress output")
    replace_once(
        """        opti.set_value(elastic_cap, 0.0)
        strict_diag = phase1_measure()
        strict_ok = eligible(strict_diag) and strict_diag['raw_max_dynamic_defect'] <= NLP_FEASIBILITY_TOLERANCE and (strict_diag['max_elastic_slack'] <= NLP_FEASIBILITY_TOLERANCE)""",
        """        strict_diag = phase1_measure()
        strict_ok = (eligible(strict_diag) and strict_diag['raw_max_dynamic_defect'] is not None and strict_diag['raw_max_dynamic_defect'] <= NLP_FEASIBILITY_TOLERANCE)
        if strict_ok:
            print('Phase-I residual gate passed; restoring exact vehicle equalities', flush=True)
            for _, residual in elastic_pairs:
                opti.subject_to(residual == 0)
            constraint_labels.extend(phase1_dynamic_labels)
            strict_g = np.asarray(opti.value(opti.g, opti.initial()), dtype=np.float64).reshape(-1)
            strict_low = np.asarray(opti.value(opti.lbg, opti.initial()), dtype=np.float64).reshape(-1)
            strict_high = np.asarray(opti.value(opti.ubg, opti.initial()), dtype=np.float64).reshape(-1)
            exact_diag = _constraint_diagnostics(strict_g, strict_low, strict_high, constraint_labels)
            strict_diag.update(exact_diag)
            strict_ok = (exact_diag.get('max_constraint_violation') is not None and exact_diag['max_constraint_violation'] <= NLP_FEASIBILITY_TOLERANCE)""",
        "strict feasibility gate")
    solve_marker = "Phase-I reduced NLP dimensions"
    if solve_marker not in transformed:
        raise RuntimeError("Cannot add objective-only dimensions marker")
    transformed = transformed.replace(
        solve_marker, "Phase-I objective-only NLP dimensions", 1)
    ast.parse(transformed)
    if "opti.bounded(-elastic_cap, residual, elastic_cap)" in transformed:
        raise RuntimeError("Objective-only Phase-I retained residual bound rows")
    if "opti.set_value(elastic_cap, 0.0)" in transformed:
        raise RuntimeError("Objective-only Phase-I retained stale cap-zero logic")
    if transformed.count("opti.subject_to(residual == 0)") != 1:
        raise RuntimeError("Exact physical equalities are not restored after Phase I")
    return transformed


def make_phase1_source_equalities(source: str) -> str:
    """Restore source dynamics as equalities while minimizing seed deviation.

    The prior objective-only formulation minimized the squared dynamic
    residuals.  This formulation instead asks IPOPT for the closest
    hard-feasible trajectory satisfying the original source-derived dynamics
    exactly.  It removes neither physics nor actuator/corridor constraints.
    """
    transformed = make_phase1_objective_only(source)

    def replace_once(old: str, new: str, label: str) -> None:
        nonlocal transformed
        count = transformed.count(old)
        if count != 1:
            raise RuntimeError(
                f"Cannot build source-equality Phase-I ({label}); found {count} anchors")
        transformed = transformed.replace(old, new, 1)

    replace_once(
        "    def add_constraint(expression, label: str | list[str]) -> None:\n"
        "        count = int(expression.numel())",
        "    def add_constraint(expression, label: str | list[str]) -> None:\n"
        "        nonlocal phase1_equalities_in_nlp\n"
        "        count = int(expression.numel())",
        "propagate exact-equality state to Phase-I acceptance closure")
    replace_once(
        "    phase1_dynamic_labels = []\n"
        "    elastic_label_by_variable = {}",
        "    phase1_dynamic_labels = []\n"
        "    phase1_equalities_in_nlp = False\n"
        "    elastic_label_by_variable = {}",
        "record exact-equality formulation")
    replace_once(
        "            phase1_dynamic_labels.extend(labels)\n"
        "            scale = 10.0 if any((name.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for name in labels)) else 1.0\n"
        "            elastic_scaled.append(ca.vec(residual / scale))",
        "            phase1_dynamic_labels.extend(labels)\n"
        "            constraint_labels.extend(labels)\n"
        "            opti.subject_to(residual == 0)\n"
        "            phase1_equalities_in_nlp = True\n"
        "            scale = 10.0 if any((name.startswith(('longitudinal_force_balance_', 'lateral_force_balance_')) for name in labels)) else 1.0\n"
        "            elastic_scaled.append(ca.vec(residual / scale))",
        "put original dynamics directly in NLP")
    replace_once(
        "    opti.minimize(elastic_merit)",
        "    opti.minimize(feasibility_objective)",
        "minimize normalized seed displacement under exact dynamics")
    replace_once(
        "def should_retain_diagnostic_iterate(",
        "def source_seed_is_strictly_feasible(\n"
        "        source_equalities_active: bool,\n"
        "        raw_dynamic_max_defect: float | None,\n"
        "        max_constraint_violation: float | None,\n"
        "        tolerance: float) -> bool:\n"
        "    # Require an exact-source audit, not only a small elastic residual.\n"
        "    values = (raw_dynamic_max_defect, max_constraint_violation, tolerance)\n"
        "    return bool(\n"
        "        source_equalities_active\n"
        "        and all(value is not None and math.isfinite(float(value))\n"
        "                for value in values)\n"
        "        and float(tolerance) > 0.0\n"
        "        and float(raw_dynamic_max_defect) <= float(tolerance)\n"
        "        and float(max_constraint_violation) <= float(tolerance))\n"
        "\n\n"
        "def should_retain_diagnostic_iterate(",
        "add exact source-feasible seed predicate")
    replace_once(
        "    initial_is_feasible = False\n",
        "    initial_is_feasible = source_seed_is_strictly_feasible(\n"
        "        source_equalities_active=phase1_equalities_in_nlp,\n"
        "        raw_dynamic_max_defect=initial_raw_max_defect,\n"
        "        max_constraint_violation=initial_constraints.get(\n"
        "            'max_constraint_violation'),\n"
        "        tolerance=NLP_FEASIBILITY_TOLERANCE)\n"
        "    print('Phase-I seed reuse gate: source_equalities_active={} raw_dynamic_max_defect={} max_constraint_violation={} phase1_measure_error={} diagnostic_error={} tolerance={} reuse={}'.format(phase1_equalities_in_nlp, initial_raw_max_defect, initial_constraints.get('max_constraint_violation'), initial_constraints.get('phase1_measure_error'), initial_constraints.get('diagnostic_error'), NLP_FEASIBILITY_TOLERANCE, initial_is_feasible), flush=True)\n"
        "    phase1_elastic_reports = []\n"
        "    strict_ok = initial_is_feasible\n",
        "reuse only a fully source-feasible seed")
    replace_once(
        "    if initial_is_feasible:\n"
        "        stage1_solution = None\n",
        "    if initial_is_feasible:\n"
        "        # Source dynamics and every original constraint already passed\n"
        "        # the exact seed audit; do not optimize distance to this seed.\n"
        "        seed_max_violation = initial_constraints['max_constraint_violation']\n"
        "        print(f'Phase-I skipped: source seed passes exact dynamics and constraints (max defect={initial_raw_max_defect:.9g}, max violation={seed_max_violation:.9g})', flush=True)\n"
        "        phase1_elastic_reports.append({\n"
        "            'name': 'source-feasible seed reuse',\n"
        "            'initial_raw_max_defect': initial_raw_max_defect,\n"
        "            'best_raw_max_defect': initial_raw_max_defect,\n"
        "            'strict_max_constraint_violation': initial_constraints[\n"
        "                'max_constraint_violation'],\n"
        "            'final_exact_physics_feasible': True,\n"
        "            'accepted_iterate': 'verified_initial_source_seed',\n"
        "            'ipopt_iterations': 0,\n"
        "        })\n"
        "        stage1_solution = None\n",
        "record exact feasible seed reuse")
    replace_once(
        "        phase1_elastic_reports = []\n"
        "        stage1_solution = None\n",
        "        stage1_solution = None\n",
        "initialize Phase-I report list for both seed reuse and IPOPT paths")
    replace_once(
        "                if np.isfinite(objective) and np.isfinite(merit) and np.isfinite(defect) and (reference_objective is None or objective < reference_objective - 1e-08):\n"
        "                    return (candidate, diag, alpha)",
        "                source_equalities_satisfied = (not phase1_equalities_in_nlp or defect <= NLP_FEASIBILITY_TOLERANCE)\n"
        "                if (source_equalities_satisfied and np.isfinite(objective) and np.isfinite(merit) and np.isfinite(defect) and (reference_objective is None or objective < reference_objective - 1e-08)):\n"
        "                    return (candidate, diag, alpha)",
        "checkpoint must satisfy exact source dynamics before seed-distance ranking")
    replace_once(
        "                    checkpoint_status = ('hard_constraint_reject' if checkpoint_rejection_diagnostics['hard_feasible_trials'] == 0 else 'feasible_no_descent')",
        "                    checkpoint_status = ('hard_constraint_reject' if checkpoint_rejection_diagnostics['hard_feasible_trials'] == 0 else 'source_equality_reject' if phase1_equalities_in_nlp and checkpoint_rejection_diagnostics['min_residual_violation'] is not None and checkpoint_rejection_diagnostics['min_residual_violation'] > NLP_FEASIBILITY_TOLERANCE else 'feasible_no_descent')",
        "explain equality-gate checkpoint rejection")
    replace_once(
        "hard_feasible_trials={checkpoint_rejection_diagnostics[\"hard_feasible_trials\"]} min_hard_violation={checkpoint_rejection_diagnostics[\"min_hard_violation\"]}",
        "hard_feasible_trials={checkpoint_rejection_diagnostics[\"hard_feasible_trials\"]} min_hard_violation={checkpoint_rejection_diagnostics[\"min_hard_violation\"]} min_residual_violation={checkpoint_rejection_diagnostics[\"min_residual_violation\"]}",
        "report exact-equality residual gate")
    replace_once(
        "        if strict_ok:\n"
        "            print('Phase-I residual gate passed; restoring exact vehicle equalities', flush=True)\n"
        "            for _, residual in elastic_pairs:\n"
        "                opti.subject_to(residual == 0)\n"
        "            constraint_labels.extend(phase1_dynamic_labels)\n"
        "            strict_g = np.asarray(opti.value(opti.g, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "            strict_low = np.asarray(opti.value(opti.lbg, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "            strict_high = np.asarray(opti.value(opti.ubg, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "            exact_diag = _constraint_diagnostics(strict_g, strict_low, strict_high, constraint_labels)\n"
        "            strict_diag.update(exact_diag)\n"
        "            strict_ok = (exact_diag.get('max_constraint_violation') is not None and exact_diag['max_constraint_violation'] <= NLP_FEASIBILITY_TOLERANCE)",
        "        if strict_ok and not phase1_equalities_in_nlp:\n"
        "            print('Phase-I residual gate passed; restoring exact vehicle equalities', flush=True)\n"
        "            for _, residual in elastic_pairs:\n"
        "                opti.subject_to(residual == 0)\n"
        "            constraint_labels.extend(phase1_dynamic_labels)\n"
        "            strict_g = np.asarray(opti.value(opti.g, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "            strict_low = np.asarray(opti.value(opti.lbg, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "            strict_high = np.asarray(opti.value(opti.ubg, opti.initial()), dtype=np.float64).reshape(-1)\n"
        "            exact_diag = _constraint_diagnostics(strict_g, strict_low, strict_high, constraint_labels)\n"
        "            strict_diag.update(exact_diag)\n"
        "            strict_ok = (exact_diag.get('max_constraint_violation') is not None and exact_diag['max_constraint_violation'] <= NLP_FEASIBILITY_TOLERANCE)",
        "avoid duplicating already-active source equalities")
    replace_once(
        "        stage1_stats = {'success': bool(strict_ok), 'return_status': 'Elastic_Exact_Dynamics_Feasible' if strict_ok else stats.get('return_status', 'Unknown'), 'iter_count': stats.get('iter_count', 0), '_initial_snapshot': opti.initial() if strict_ok else None}\n"
        "        if not strict_ok:",
        "        stage1_stats = {'success': bool(strict_ok), 'return_status': 'Elastic_Exact_Dynamics_Feasible' if strict_ok else stats.get('return_status', 'Unknown'), 'iter_count': stats.get('iter_count', 0), '_initial_snapshot': opti.initial() if strict_ok else None}\n"
        "        if strict_ok:\n"
        "            # Phase-I's IPOPT result may be a worse raw iterate than the\n"
        "            # checkpoint selected by the exact-dynamics gate.  Force\n"
        "            # stage_values() to consume the saved certified checkpoint.\n"
        "            stage1_solution = None\n"
        "        if not strict_ok:",
        "start minimum-time stage from the retained certified Phase-I snapshot")
    transformed = transformed.replace(
        "Phase-I objective-only NLP dimensions",
        "Phase-I source-equality NLP dimensions", 1)
    ast.parse(transformed)
    # One equality is active during Phase-I; the second is guarded by
    # `not phase1_equalities_in_nlp` for legacy deferred formulations.
    if transformed.count("opti.subject_to(residual == 0)") != 2:
        raise RuntimeError("Source dynamics are not represented by one exact equality block")
    if "opti.minimize(feasibility_objective)" not in transformed:
        raise RuntimeError("Source-equality Phase-I does not minimize seed displacement")
    if "source_equalities_satisfied = (not phase1_equalities_in_nlp or defect <= NLP_FEASIBILITY_TOLERANCE)" not in transformed:
        raise RuntimeError("Source-equality checkpoints can retain a dynamically infeasible seed")
    if "initial_is_feasible = source_seed_is_strictly_feasible(" not in transformed:
        raise RuntimeError("Source-feasible seed reuse lacks the strict feasibility gate")
    if "phase1_equalities_in_nlp = True" not in transformed:
        raise RuntimeError("Source-equality Phase-I marker is missing")
    if "stage1_solution = None" not in transformed:
        raise RuntimeError("Minimum-time stage can bypass the certified Phase-I snapshot")
    return transformed


def prepare_working_solver_v251(recovered_root: Path, package_root: Path,
                                legacy_name: str,
                                mode: str = "limited-memory",
                                phase1_formulation: str = "elastic"):
    """Generate an isolated source-parity candidate without touching v249."""
    package_root = Path(package_root).resolve()
    recovered_root = Path(recovered_root).resolve()
    repo_root = package_root.parent
    from source_vehicle import _cpp_number
    mpc_types = (repo_root / "f1tenth_mpc/include/mpc_types.h").read_text(
        encoding="utf-8")
    throttle_min_match = re.search(
        r"(?m)^\s*#define\s+MPC_THROTTLE_MIN_DRIVE\s+"
        r"([0-9]+(?:\.[0-9]*)?(?:[eE][+-]?[0-9]+)?)f\s*$",
        mpc_types)
    if throttle_min_match is None:
        raise RuntimeError(
            "Cannot resolve MPC's admissible minimum drive throttle")
    mpc_throttle_min_drive = float(throttle_min_match.group(1))
    if not math.isclose(mpc_throttle_min_drive,
                        UNITY_MPC_THROTTLE_MIN_DRIVE,
                        rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            "IPOPT drive-throttle envelope is stale relative to MPC_THROTTLE_MIN_DRIVE")
    plant_source = (repo_root / "f1tenth_mpc/src/unity_source_plant.cpp").read_text(
        encoding="utf-8")
    time_manager = (repo_root /
                    "external/autodrive-unity-source/ProjectSettings/TimeManager.asset")
    time_text = time_manager.read_text(encoding="utf-8")
    fixed_steps = re.findall(r"(?m)^\s*Fixed Timestep:\s*([0-9.eE+-]+)\s*$",
                             time_text)
    plant_step = _cpp_number(plant_source, "kDt")
    rear_axle_offset = _cpp_number(plant_source, "kComForwardFromRearAxleM")
    if not math.isclose(rear_axle_offset, SOURCE_REAR_AXLE_FROM_COM_M,
                        rel_tol=0.0, abs_tol=1.e-12):
        raise RuntimeError('IPOPT rear-axle reference differs from production plant')
    source_angular_damping = _cpp_number(
        plant_source, "kAngularDampingPerS")
    if (len(fixed_steps) != 1
            or not math.isclose(float(fixed_steps[0]), plant_step,
                                rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(plant_step, UNITY_SOURCE_FIXED_STEP_S,
                                rel_tol=0.0, abs_tol=1.0e-12)
            or not math.isclose(source_angular_damping,
                                UNITY_SOURCE_ANGULAR_DAMPING_PER_S,
                                rel_tol=0.0, abs_tol=1.0e-12)):
        raise RuntimeError(
            "IPOPT source-step model is stale: Unity Fixed Timestep, "
            "production kDt, angular damping, and overlay constants must match")
    v249_root, base_provenance = prepare_v249(
        recovered_root,
        package_root / ("_working_solver_v249_" + mode.replace("-", "_")),
        legacy_name, mode)
    if phase1_formulation not in ("elastic", "direct", "objective", "source-equalities"):
        raise ValueError(
            "v2.5.1 Phase-I formulation must be elastic, direct, objective, or source-equalities")
    formulation_suffix = ("" if phase1_formulation == "elastic" else
                          "_direct_residual" if phase1_formulation == "direct" else
                          "_objective_only" if phase1_formulation == "objective" else
                          "_source_equalities")
    base_solver_path = (v249_root / legacy_name /
                        "f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py")
    patched = eliminate_algebraic_vehicle_states(
        transform_solver(base_solver_path.read_text(encoding="utf-8"), mode))
    if phase1_formulation == "direct":
        patched = eliminate_phase1_slack_variables(patched)
    elif phase1_formulation == "objective":
        patched = make_phase1_objective_only(patched)
    elif phase1_formulation == "source-equalities":
        patched = make_phase1_source_equalities(patched)
        patched = make_componentwise_time_trust_region(patched)
        patched = add_exploratory_time_stage_capture(patched)
        patched = add_time_stage_progress(patched)
    patched = add_ipopt_cpu_time_limits(patched)
    if phase1_formulation == 'source-equalities':
        patched = stop_source_phase1_when_feasible(patched)
    patched = install_exact_path_frame(patched)
    if phase1_formulation == 'source-equalities':
        patched = compact_source_force_evaluation(patched)
    # Never mutate a generated solver module after another optimizer process
    # may have imported it. Concurrent campaigns used to share one path, so a
    # later source edit could leave a long-running process silently executing
    # a stale Phase-I gate. Content-address the complete generated module and
    # the transformers that create its vehicle equations.
    source_digest = hashlib.sha256()
    for source_path in (
            Path(__file__),
            Path(__file__).with_name("vehicle_mechanics_overlay_v250.py"),
            Path(__file__).with_name("wheel_model_overlay_v251.py"),
            Path(__file__).with_name("path_frame.py"),
            Path(__file__).with_name("source_equation_kernels.py"),
            Path(__file__).with_name("ipopt_time_progress.py"),
            base_solver_path):
        source_digest.update(source_path.name.encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(source_path.read_bytes())
        source_digest.update(b"\0")
    source_digest.update(mode.encode("utf-8"))
    source_digest.update(b"\0" + phase1_formulation.encode("utf-8"))
    generated_source_sha256 = hashlib.sha256(patched.encode("utf-8")).hexdigest()
    build_fingerprint = source_digest.hexdigest()[:16]
    target_root = package_root / (
        "_working_solver_v251_" + mode.replace("-", "_") + formulation_suffix
        + "_src_" + build_fingerprint)
    if not target_root.exists():
        shutil.copytree(v249_root, target_root)
        (target_root / ".generated_solver_fingerprint").write_text(
            build_fingerprint + "\n", encoding="utf-8")
    mechanics = prepare_working_mechanics(recovered_root, target_root)
    wheel = prepare_working_wheel_model(
        recovered_root, target_root, legacy_name)
    wheel_path = (target_root / legacy_name /
                  "f1tenth_planning/autodrive_mintime/unity_wheel_model.py")
    wheel_source = wheel_path.read_text(encoding="utf-8")
    if "physx_no_drive_wheel_speed_interval_casadi" in wheel_source:
        raise RuntimeError("Active IPOPT wheel model still exposes zero-throttle lock")
    solver_path = (target_root / legacy_name /
                   "f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py")
    current_on_disk = solver_path.read_text(encoding="utf-8")
    current = current_on_disk
    baseline = base_solver_path.read_text(encoding="utf-8")
    if current not in (baseline, patched):
        current = _upgrade_previous_v251(current)
    # Compare against the actual bytes on disk. `current` may have been
    # upgraded in memory to `patched`; comparing those two would falsely skip
    # persisting the upgrade and leave stale generated solver code active.
    if current_on_disk != patched:
        solver_path.write_text(patched, encoding="utf-8")
    elif hashlib.sha256(current_on_disk.encode("utf-8")).hexdigest() != generated_source_sha256:
        raise RuntimeError("Content-addressed IPOPT workspace contains stale solver source")
    return target_root, {
        "candidate": "v2.5.6-positive-drive-algebraic-source-state",
        "algebraic_load_and_acceleration_states_eliminated": True,
        "algebraic_elimination_is_source_equation_substitution": True,
        "v249_baseline_preserved": True,
        "base_solver": base_provenance,
        "mechanics": mechanics,
        "wheel_model": wheel,
        "spatial_collocation_is_not_native_1khz_map": True,
        "unity_source_fixed_step_s": plant_step,
        "unity_source_fixed_step_matches_production": True,
        "unity_source_fixed_step_is_optimizer_time_grid": False,
        "mpc_throttle_min_drive": mpc_throttle_min_drive,
        "optimizer_input_is_positive_throttle_only": True,
        "optimizer_minimum_throttle_matches_mpc": True,
        "ipopt_symbolic_graph_expansion": False,
        "phase1_formulation": phase1_formulation,
        "generated_solver_source_sha256": generated_source_sha256,
        "generated_solver_build_fingerprint": build_fingerprint,
        "phase1_seed_trust_region_active": phase1_formulation != "objective",
        "phase1_candidate_selection": (
            "lowest exact NLP objective among hard-feasible projected iterates; "
            "maximum dynamics residual remains a strict final feasibility gate"),
        "phase1_slack_variables_eliminated": phase1_formulation == "direct",
        "phase1_dynamic_constraints_deferred_until_certified": phase1_formulation == "objective",
        "phase1_source_dynamics_are_exact_equalities": phase1_formulation == "source-equalities",
        "phase1_objective_is_normalized_seed_displacement": phase1_formulation == "source-equalities",
        "time_trust_region_geometry": (
            "normalized_component_box_with_aggregate_l2_equivalent_radius"
            if phase1_formulation == "source-equalities" else
            "aggregate_squared_l2"),
        "phase1_physics_equations_unchanged": True,
        "ipopt_progress_markers": True,
        "source_equation_corrections": [
            "PhysX normalized tire-load filter before tire callback",
            "full 3D tire and suspension force/moment at configured point",
            "body-frame gravity, angular-velocity cross-velocity, and tilt",
            "ZYX Euler rates from body p/q/r for yaw and tilt kinematics",
            "horizontal-projected velocity for Frenet path progress",
            "exact source 1ms damping represented by equivalent continuous rate",
            "source sprung mass, wheel and suspension constants",
            "source-derived tire, suspension, wheel-speed, force and yaw checks at every OCP interval midpoint",
        ],
        "initialization": (
            "single source-equation tire/body force evaluation at seeded "
            "suspension state; no recursive periodic suspension seed rollout"),
        "raceline_semantics": (
            "optimized geometry/speed/curvature target; internal steering and "
            "throttle are feasibility witnesses, not runtime commands"),
        "remaining_known_transcription_limits": [
            "spatial collocation is not a complete discrete 1ms Unity scene simulation",
            "runtime 25ms command holds are intentionally omitted; spatial controls are theoretical feasibility witnesses, not an online command schedule",
            "collocated wheel torque is held constant and averaged across a variable-length spatial interval",
            "native PhysX contact/constraint solver is not replicated by IPOPT",
            "initial body heave/roll/pitch values are an NLP seed; their dynamic evolution remains an optimizer state",
        ],
        # Resume certification must include imported frame equations too.
        # The generated Python file alone can stay byte-identical while a
        # helper changes its geometry/velocity semantics.
        "optimizer_source_sha256": source_digest.hexdigest(),
        "optimizer_source_path": str(solver_path),
        "path_frame": "exact_linear_xy_and_unwrapped_heading_as_production_mpc",
        "spatial_phase_length_is_optimization_variable": False,
        "path_position_origin": "rear_axle_as_Unity_GPS_and_production_MPC",
    }
