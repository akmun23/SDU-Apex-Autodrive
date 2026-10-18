"""Numerical overlay for the recovered Unity-wheel IPOPT minimum-time OCP.

Keep historical source immutable. The working copy adds *temporary* dynamic
constraint defect continuation to Phase I. Every dynamic equality is enforced
exactly (up to the existing 1e-3 residual acceptance) before minimum-time
optimization and native PhysX replay. Path, footprint, tire-slip bounds and
other inequalities are NEVER relaxed.

The old historical feasibility reference-distance cap came from a specific
measured Unity capture; it is disabled only during Phase I because this
package synthesizes a fresh initial guess instead of using that capture.
Phase-II uses the historical objective/trust progression unmodified.
Unlike the captured-trajectory historical solver, Phase I minimizes the
normalized *dynamic defects*, not distance from the synthetic seed.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import shutil

HESSIAN_SITE = '"max_iter": feasibility_max_iterations,\n'
HESSIAN_ADDITION = ('"max_iter": feasibility_max_iterations,\n'
                    '            "hessian_approximation": "limited-memory",\n'
                    '            "limited_memory_max_history": 20,\n')

OPTIMIZER_DECLARATION = '    opti = ca.Opti()\n'
CONSTRAINT_FN = '''    def add_constraint(expression, label: str | list[str]) -> None:
        count = int(expression.numel())
        opti.subject_to(expression)
        if isinstance(label, str):
            constraint_labels.extend([label] * count)
        elif len(label) == count:
            constraint_labels.extend(label)
        else:
            raise ValueError("constraint labels do not match expression size")
'''
CONSTRAINT_REPLACEMENT = '''    def add_constraint(expression, label: str | list[str]) -> None:
        count = int(expression.numel())
        labels = ([label] * count if isinstance(label, str) else list(label))
        if len(labels) != count:
            raise ValueError("constraint labels do not match expression size")
        # Only the DYNAMICS equalities can have a temporary homotopy defect.
        # Never relax lane boundaries, physical footprint, tire-slip, normal
        # load limits, steering, actuator limits, or lap periodic length.
        dynamics_prefixes = (
            'periodic_collocation_state_', 'physx_implicit_wheel_update_',
            'longitudinal_force_balance_', 'lateral_force_balance_',
            'unity_spring_damper_load_',
        )
        if (expression.is_op(ca.OP_EQ) and
                all(any(name.startswith(prefix) for prefix in dynamics_prefixes)
                    for name in labels)):
            residual = expression.dep(0) - expression.dep(1)
            # The historical residuals are scaled for spatial collocation;
            # force balances are still in m/s^2, so scale those for the
            # Phase-I least-squares merit, not for physical feasibility.
            is_force_balance = any(name.startswith((
                'longitudinal_force_balance_', 'lateral_force_balance_'))
                for name in labels)
            phase1_defect_residuals.append(
                (residual / 10.0) if is_force_balance else residual)
            opti.subject_to(opti.bounded(
                -phase1_defect_tolerance, residual, phase1_defect_tolerance))
        else:
            opti.subject_to(expression)
        constraint_labels.extend(labels)
'''

TRUST_SITE = '    opti.set_value(trust_region_bound, feasibility_trust_region)\n'
TRUST_REPLACEMENT = '''    # Original distance-to-captured-run bound is not meaningful here: the
    # seed is synthetic, not the measured capture that justified radius 25.
    # Disable it during Phase I. The Phase-II continuation resets this bound
    # using the actually feasible incumbent and the original trust factors.
    opti.set_value(trust_region_bound, 1.0e12)
'''

INIT_SITE = '''    opti.minimize(feasibility_objective)
    try:
        initial_constraints'''
INIT_REPLACEMENT = '''    if not phase1_defect_residuals:
        raise RuntimeError('No physical dynamics equality defects collected')
    # A synthetic guess is not measured ground truth. Minimizing squared
    # dynamic defects supplies a direction toward the REAL hard equations.
    phase1_defect_energy = ca.sumsqr(ca.vertcat(*phase1_defect_residuals))
    opti.minimize(phase1_defect_energy + 1.0e-3 * feasibility_objective)
    # Initial Phase-I defects are allowed only as an initialization homotopy.
    # At the final stage this parameter becomes EXACTLY zero.
    phase1_defect_schedule = (0.5, 0.35, 0.20, 0.10, 0.05, 0.02, 0.005, 0.0)
    opti.set_value(phase1_defect_tolerance, phase1_defect_schedule[0])
    try:
        initial_constraints'''

FEASIBILITY_SITE = '''    initial_is_feasible = (
        initial_constraints.get("max_constraint_violation") is not None
        and initial_constraints["max_constraint_violation"]
        <= NLP_FEASIBILITY_TOLERANCE)'''
FEASIBILITY_REPLACEMENT = '''    # A point satisfying a temporary relaxed bound is NOT a feasible lap.
    # Always complete the final zero-defect continuation stage first.
    initial_is_feasible = False'''

SOLVE_SITE = '''    else:
        try:
            stage1_solution = opti.solve_limited()
        except RuntimeError as error:
            stage1_solution = None
            solver_error = str(error)
        stage1_stats = opti.stats()

    def stage_values'''
SOLVE_REPLACEMENT = '''    else:
        # IPOPT may traverse feasible iterates and finish at an infeasible
        # point (iteration-limit / restoration). Save the BEST feasible
        # intermediate iterate, not merely the final returned state.
        # Also avoid requiring an abrupt 0.5 -> 0.35 reduction when the
        # actual seed still needs a defect tolerance of 0.4665.
        phase1_target_schedule = (0.5, 0.35, 0.20, 0.10, 0.05, 0.02, 0.005, 0.0)
        phase1_dynamics_prefixes = (
            'periodic_collocation_state_', 'physx_implicit_wheel_update_',
            'longitudinal_force_balance_', 'lateral_force_balance_',
            'unity_spring_damper_load_',
        )
        dynamic_indices = np.asarray([
            i for i, name in enumerate(constraint_labels)
            if any(name.startswith(prefix) for prefix in phase1_dynamics_prefixes)
        ], dtype=int)
        if not len(dynamic_indices):
            raise RuntimeError('Cannot identify any vehicle dynamics constraints')
        stage1_solution = None
        stage1_stats = {}
        phase1_homotopy_reports = []
        phase1_reached_strict = False
        phase1_decision_vars = (x, qdelta, throttle, phase_lengths,
                                normal_load, accel_xy)

        def current_snapshot():
            return {var: np.asarray(opti.value(var, opti.initial()),
                                    dtype=np.float64).copy()
                    for var in phase1_decision_vars}

        def assign_snapshot(snapshot):
            for var, val in snapshot.items():
                opti.set_initial(var, val)

        def phase1_diagnostics(debug=False):
            try:
                getter = opti.debug.value if debug else (
                    lambda expression: opti.value(expression, opti.initial()))
                g = np.asarray(getter(opti.g), dtype=np.float64).reshape(-1)
                # Use live parameter values, never stale OptiSol bounds.
                l = np.asarray(opti.value(opti.lbg, opti.initial()),
                               dtype=np.float64).reshape(-1)
                h = np.asarray(opti.value(opti.ubg, opti.initial()),
                               dtype=np.float64).reshape(-1)
                result = _constraint_diagnostics(g, l, h, constraint_labels)
                energy = float(getter(phase1_defect_energy))
                result['normalized_physics_defect_energy'] = energy
                result['unrelaxed_max_dynamics_defect'] = float(
                    np.max(np.abs(g[dynamic_indices])))
                if not np.isfinite(energy) or not np.isfinite(
                        result['unrelaxed_max_dynamics_defect']):
                    result['max_constraint_violation'] = None
                return result
            except (RuntimeError, ValueError, OverflowError) as exc:
                return {'max_constraint_violation': None,
                        'normalized_physics_defect_energy': None,
                        'unrelaxed_max_dynamics_defect': None,
                        'worst_constraints': [], 'diagnostic_error': str(exc)}

        def is_valid(diag):
            v = diag.get('max_constraint_violation')
            return v is not None and np.isfinite(v) and v <= NLP_FEASIBILITY_TOLERANCE

        # Keep the total Phase-I IPOPT budget bounded, despite additional
        # adaptive homotopy stages. A failed step is not silently accepted.
        total_iterations_used = 0
        max_phase1_stages = 28
        current_tolerance = None
        current_target_index = 0
        rejected_stage_count = 0
        stalled_stage_count = 0
        retry_tolerance = None
        for stage_index in range(max_phase1_stages):
            if retry_tolerance is not None:
                defect_tol = retry_tolerance
                retry_tolerance = None
            elif current_tolerance is None:
                defect_tol = phase1_target_schedule[0]
            else:
                prev_diag = phase1_diagnostics()
                required = prev_diag.get('unrelaxed_max_dynamics_defect')
                if required is None:
                    break
                next_target = phase1_target_schedule[current_target_index]
                if current_tolerance <= next_target + 1.e-9:
                    current_target_index += 1
                    if current_target_index >= len(phase1_target_schedule):
                        break
                    next_target = phase1_target_schedule[current_target_index]
                # If the next target would violate the starting point, make
                # a smaller step, just above its measured current defect.
                # This is a numerical homotopy parameter, NEVER accepted
                # as vehicle physics at the end of Phase I.
                defect_tol = max(next_target, required + 0.002)
                if defect_tol >= current_tolerance - 0.001:
                    defect_tol = max(next_target, current_tolerance - 0.004)
                if defect_tol >= current_tolerance - 1.e-9:
                    print('Phase-I continuation stalled: no smaller feasible '
                          'defect target; cannot certify a zero-defect lap.',
                          flush=True)
                    break
            opti.set_value(phase1_defect_tolerance, defect_tol)
            before = phase1_diagnostics()
            seed_is_feasible = is_valid(before)
            start_energy = before.get('normalized_physics_defect_energy')
            initial_snapshot = current_snapshot()
            winner_snapshot = initial_snapshot if seed_is_feasible else None
            winner_energy = start_energy if seed_is_feasible else float('inf')
            winner_source = 'start' if seed_is_feasible else 'none'
            checked_intermediate = 0
            candidate_diag = None
            stage_error = None
            iterations_left = feasibility_max_iterations - total_iterations_used
            if iterations_left <= 0:
                print('Phase-I iteration budget exhausted before reaching '
                      'exact dynamics. No raceline exported.', flush=True)
                break
            budget = min(iterations_left, max(30, int(
                np.ceil(feasibility_max_iterations / 12))))

            # IPOPT iterate callbacks expose the actual primal iterate. This
            # catches a useful feasible point even if IPOPT later diverges.
            bound_l = np.asarray(opti.value(opti.lbg, opti.initial()),
                                 dtype=np.float64).reshape(-1)
            bound_h = np.asarray(opti.value(opti.ubg, opti.initial()),
                                 dtype=np.float64).reshape(-1)
            callback_errors = 0

            def keep_feasible_iterate(it):
                nonlocal winner_snapshot, winner_energy, winner_source
                nonlocal checked_intermediate, callback_errors
                try:
                    g = np.asarray(opti.debug.value(opti.g),
                                   dtype=np.float64).reshape(-1)
                    if not np.isfinite(g).all() or len(g) != len(bound_l):
                        return
                    violation = max(float(np.max(bound_l - g)),
                                    float(np.max(g - bound_h)), 0.0)
                    if violation > NLP_FEASIBILITY_TOLERANCE:
                        return
                    energy = float(opti.debug.value(phase1_defect_energy))
                    if (not np.isfinite(energy) or
                            energy >= winner_energy - 1.e-9):
                        return
                    snapshot = {var: np.asarray(opti.debug.value(var),
                                                dtype=np.float64).copy()
                                for var in phase1_decision_vars}
                    if not all(np.isfinite(v).all() for v in snapshot.values()):
                        return
                    winner_snapshot = snapshot
                    winner_energy = energy
                    winner_source = 'ipopt_iteration'
                    checked_intermediate += 1
                except (RuntimeError, ValueError, OverflowError):
                    callback_errors += 1

            opti.callback(keep_feasible_iterate)
            options = {'max_iter': budget, 'tol': 1.e-6,
                       'acceptable_tol': 1.e-5,
                       'print_level': int(print_level)}
            if phase1_limited_memory:
                options.update({'hessian_approximation': 'limited-memory',
                                'limited_memory_max_history': 20})
            opti.solver('ipopt', {'expand': True, 'ipopt': options}, {})
            try:
                opti.solve_limited()
            except RuntimeError as exc:
                stage_error = str(exc)
                solver_error = stage_error
            solver_stats = opti.stats()
            total_iterations_used += int(solver_stats.get('iter_count', budget))
            raw_candidate = None
            try:
                raw_candidate = {
                    var: np.asarray(opti.debug.value(var),
                                    dtype=np.float64).copy()
                    for var in phase1_decision_vars}
                if not all(np.isfinite(v).all() for v in raw_candidate.values()):
                    raw_candidate = None
                if raw_candidate is not None:
                    assign_snapshot(raw_candidate)
                    candidate_diag = phase1_diagnostics()
                    if is_valid(candidate_diag):
                        keep_energy = candidate_diag['normalized_physics_defect_energy']
                        if keep_energy < winner_energy - 1.e-9:
                            winner_snapshot = raw_candidate
                            winner_energy = keep_energy
                            winner_source = 'ipopt_final'
            except (RuntimeError, ValueError, OverflowError) as exc:
                stage_error = str(exc)

            # Safeguarded backtracking along the numerical solver step can
            # recover a feasible, lower-defect intermediate. This is NOT a
            # substitute for the strict final dynamics-constrained solve.
            if raw_candidate is not None:
                for alpha in (0.5, 0.25, 0.125, 0.0625, 0.03125,
                              0.015625, 0.0078125):
                    blended = {var: (initial_snapshot[var] + alpha *
                                     (raw_candidate[var] - initial_snapshot[var]))
                               for var in phase1_decision_vars}
                    assign_snapshot(blended)
                    diag = phase1_diagnostics()
                    energy = diag.get('normalized_physics_defect_energy')
                    if (is_valid(diag) and energy is not None and
                            energy < winner_energy - 1.e-9):
                        winner_snapshot = blended
                        winner_energy = energy
                        winner_source = 'backtracked_ipopt_step'
            if winner_snapshot is None:
                assign_snapshot(initial_snapshot)
                accepted = False
                after = phase1_diagnostics()
            else:
                assign_snapshot(winner_snapshot)
                after = phase1_diagnostics()
                accepted = is_valid(after)
            accepted_iterate = winner_source
            # Keep final stage status strictly about feasibility, not about
            # reaching Ipopt's KKT tolerance on the relaxed subproblem.
            stage1_stats = {
                'success': bool(accepted),
                'return_status': ('Feasible_Defect_Homotopy_Incumbent' if accepted
                                  else solver_stats.get('return_status', 'Unknown')),
                'iter_count': int(solver_stats.get('iter_count', 0)),
                '_initial_snapshot': opti.initial(),
            }
            stage1_solution = None
            phase1_homotopy_reports.append({
                'dynamic_equality_tolerance': float(defect_tol),
                'iterations_budget': int(budget),
                'iterations_used_total': total_iterations_used,
                'status': solver_stats.get('return_status'),
                'initial_max_constraint_violation': before.get(
                    'max_constraint_violation'),
                'max_constraint_violation': after.get('max_constraint_violation'),
                'unrelaxed_max_dynamics_defect': after.get(
                    'unrelaxed_max_dynamics_defect'),
                'initial_normalized_physics_defect_energy': start_energy,
                'normalized_physics_defect_energy': after.get(
                    'normalized_physics_defect_energy'),
                'solver_candidate_max_constraint_violation': (
                    candidate_diag.get('max_constraint_violation')
                    if candidate_diag else None),
                'accepted_iterate': accepted_iterate,
                'feasible_iteration_improvements': checked_intermediate,
                'callback_errors': callback_errors,
                'worst_constraints': after.get('worst_constraints', [])[:12],
                'accepted_stage': bool(accepted),
                'exception': stage_error,
            })
            print(f'Phase-I dynamics merit {stage_index+1}: '
                  f'tolerance={defect_tol:.5f}, '
                  f'violation={after.get("max_constraint_violation")}, '
                  f'raw_defect={after.get("unrelaxed_max_dynamics_defect")}, '
                  f'energy={after.get("normalized_physics_defect_energy")}, '
                  f'accepted={accepted}, iterate={accepted_iterate}', flush=True)
            if not accepted:
                rejected_stage_count += 1
                # The previous iterate remains feasible at the PREVIOUS
                # tolerance, so retry with a smaller contraction once.
                assign_snapshot(initial_snapshot)
                if (current_tolerance is None or rejected_stage_count >= 2
                        or current_tolerance - defect_tol < 0.004):
                    print('Phase-I stopped: could not find a feasible smaller '
                          'dynamics tolerance. No minimum-time solve.', flush=True)
                    break
                # Reset to the last accepted tolerance and try a smaller step.
                # Do NOT change current_tolerance: it denotes only an
                # already-accepted relaxed physics constraint set.
                retry_tolerance = 0.5 * (current_tolerance + defect_tol)
                opti.set_value(phase1_defect_tolerance, current_tolerance)
                continue
            rejected_stage_count = 0
            current_tolerance = defect_tol
            if (start_energy is not None and winner_energy is not None and
                    winner_energy >= start_energy - 1.e-6):
                stalled_stage_count += 1
            else:
                stalled_stage_count = 0
            if stalled_stage_count >= 5 and defect_tol > 0.0:
                print('Phase-I stalled: dynamics defect energy is not '
                      'decreasing. Investigate the physics/initial seed.',
                      flush=True)
                break
            if defect_tol == 0.0:
                phase1_reached_strict = True
                break
        # Final audit MUST use exact dynamics, regardless of relaxed stages.
        opti.callback(lambda iteration: None)
        opti.set_value(phase1_defect_tolerance, 0.0)
        if not phase1_reached_strict:
            stage1_stats['_phase1_failed_before_strict'] = True

    def stage_values'''

STRICT_BOUND_DIAGNOSTICS_SITE = '    def stage_constraint_diagnostics(solution, stats):\n        try:\n            values = stage_values(solution, stats, opti.g).reshape(-1)\n            lower = stage_values(solution, stats, opti.lbg).reshape(-1)\n            upper = stage_values(solution, stats, opti.ubg).reshape(-1)'
STRICT_BOUND_DIAGNOSTICS_REPLACEMENT = '    def stage_constraint_diagnostics(solution, stats):\n        try:\n            values = stage_values(solution, stats, opti.g).reshape(-1)\n            # A previous OptiSol can retain relaxed parameter values. Never\n            # use that snapshot for deciding strict physical feasibility.\n            lower = np.asarray(opti.value(opti.lbg, opti.initial()),\n                               dtype=np.float64).reshape(-1)\n            upper = np.asarray(opti.value(opti.ubg, opti.initial()),\n                               dtype=np.float64).reshape(-1)'

REPORT_SITE = '''    stage_reports.append({
        "name": "captured-motion feasibility",
        **_solver_status(stage1_stats),
        "max_constraint_violation": stage1_diagnostics.get(
            "max_constraint_violation"),
    })\n'''
REPORT_REPLACEMENT = REPORT_SITE + '    stage_reports.extend(phase1_homotopy_reports)\n'

STRICT_GATE_SITE = '    if (stage1_diagnostics.get("max_constraint_violation") is not None\n            and stage1_diagnostics["max_constraint_violation"]\n            <= NLP_FEASIBILITY_TOLERANCE):'
STRICT_GATE_REPLACEMENT = '    if (phase1_reached_strict\n            and stage1_diagnostics.get("max_constraint_violation") is not None\n            and stage1_diagnostics["max_constraint_violation"]\n            <= NLP_FEASIBILITY_TOLERANCE):'

# After insertion, the pre-existing Ipopt solver block is still defined and
# used in the historical stage-I site. The continuation explicitly replaces
# its budget/derivative options once per homotopy stage.


def _replace_once(source: str, before: str, after: str, label: str) -> str:
    matches = source.count(before)
    if matches != 1:
        raise RuntimeError(f'Historical source mismatch at {label}: expected one match, got {matches}')
    return source.replace(before, after, 1)


def transform_solver(source: str, mode: str) -> str:
    """Apply a strictly checked transformation; NEVER touch the legacy file."""
    if mode not in ('limited-memory', 'exact'):
        raise ValueError('Phase-I Hessian must be limited-memory or exact')
    original = source
    if mode == 'limited-memory':
        source = _replace_once(source, HESSIAN_SITE, HESSIAN_ADDITION, 'hessian')
    source = _replace_once(source, OPTIMIZER_DECLARATION,
        OPTIMIZER_DECLARATION +
        '    phase1_defect_residuals = []\n'
        '    phase1_defect_tolerance = opti.parameter()\n'
        '    opti.set_value(phase1_defect_tolerance, 0.0)\n', 'Opti construction')
    source = _replace_once(source, CONSTRAINT_FN, CONSTRAINT_REPLACEMENT, 'add_constraint')
    source = _replace_once(source, TRUST_SITE, TRUST_REPLACEMENT, 'Phase-I captured motion bound')
    source = _replace_once(source, INIT_SITE, INIT_REPLACEMENT, 'initial diagnostics')
    source = _replace_once(source, FEASIBILITY_SITE, FEASIBILITY_REPLACEMENT, 'strict feasibility test')
    source = _replace_once(source, SOLVE_SITE,
        SOLVE_REPLACEMENT.replace('if phase1_limited_memory:',
                                  'if '+str(mode == 'limited-memory')+':'),
        'phase1 solve')
    source = _replace_once(source, STRICT_GATE_SITE,
                           STRICT_GATE_REPLACEMENT, 'strict Phase-I gate')
    source = _replace_once(source, STRICT_BOUND_DIAGNOSTICS_SITE,
        STRICT_BOUND_DIAGNOSTICS_REPLACEMENT, 'strict bound auditing')
    source = _replace_once(source, REPORT_SITE, REPORT_REPLACEMENT, 'phase-I reports')
    if source == original:
        raise AssertionError('Numerical overlay made no change')
    return source


def prepare_working_solver(recovered_root: Path, working_root: Path,
                           legacy_name: str, mode: str = 'limited-memory') -> tuple[Path, dict]:
    """Upgrade an old VERIFIED working copy in place. Keep user outputs intact."""
    source_root, target_root = Path(recovered_root).resolve(), Path(working_root).resolve()
    original = (source_root / legacy_name /
                'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py')
    if not original.is_file():
        raise FileNotFoundError(original)
    source_bytes = original.read_bytes()
    raw = source_bytes.decode('utf-8')
    patched = transform_solver(raw, mode)
    dest = (target_root / legacy_name /
            'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py')
    if source_root == target_root:
        raise ValueError('Cannot alter original recovered IPOPT solver')
    if not target_root.exists():
        shutil.copytree(source_root, target_root)
    elif dest.is_file():
        existing = dest.read_text(encoding='utf-8')
        previous_overlay = (raw.replace(HESSIAN_SITE, HESSIAN_ADDITION, 1)
                            if mode == 'limited-memory' else raw)
        # Accept the verified v2.2.3 and v2.2.4 overlays, in addition to the earlier
        # verified v2.2.x Hessian-only overlay. No arbitrary cache edits.
        from solver_overlay_v223 import transform_solver as v223_transform
        previous_v223 = v223_transform(raw, mode)
        from solver_overlay_v224 import transform_solver as v224_transform
        previous_v224 = v224_transform(raw, mode)
        from solver_overlay_v225 import transform_solver as v225_transform
        previous_v225 = v225_transform(raw, mode)
        if existing not in (previous_overlay, previous_v223,
                            previous_v224, previous_v225, patched):
            raise RuntimeError(
                'Working IPOPT solver has unknown edits; refusing to overwrite. '
                f'Inspect {dest} before continuing.')
    else:
        raise FileNotFoundError('Incomplete working IPOPT cache: ' + str(dest))
    if dest.read_text(encoding='utf-8') != patched:
        dest.write_text(patched, encoding='utf-8')
    return target_root, {
        'phase1_hessian': mode,
        'phase1_method': 'v2.2.6 defect-merit plus safeguarded feasible iterates, adaptive defect homotopy, mandatory zero tolerance',
        'dynamics_relaxation_schedule': [0.5, 0.35, 0.20, 0.10, 0.05, 0.02, 0.005, 0.0],
        'adaptive_intermediate_tolerances': True,
        'original_solver_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'working_solver_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'source_untouched': original.read_bytes() == source_bytes,
        'working_solver_path': str(dest),
        'patch_method': 'Phase-I feasible-iterate checkpointing and adaptive weighted dynamics merit homotopy; zero-defect final audit required',
    }
