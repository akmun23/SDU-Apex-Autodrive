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
        # Stage I is a constrained residual MINIMIZATION, not a minimization
        # of distance from the initial synthetic trajectory. A feasible
        # relaxed guess is worth refining, so that the next, tighter stage
        # starts near the true zero-defect vehicle dynamics manifold.
        phase1_defect_schedule = (0.5, 0.35, 0.20, 0.10, 0.05, 0.02, 0.005, 0.0)
        stage1_solution = None
        stage1_stats = {}
        phase1_homotopy_reports = []
        phase1_reached_strict = False
        phase1_decision_vars = (x, qdelta, throttle, phase_lengths,
                                normal_load, accel_xy)

        def diagnose_current_phase1():
            try:
                values = np.asarray(opti.value(
                    opti.g, opti.initial()), dtype=np.float64).reshape(-1)
                lower = np.asarray(opti.value(
                    opti.lbg, opti.initial()), dtype=np.float64).reshape(-1)
                upper = np.asarray(opti.value(
                    opti.ubg, opti.initial()), dtype=np.float64).reshape(-1)
                result = _constraint_diagnostics(
                    values, lower, upper, constraint_labels)
                energy = float(opti.value(phase1_defect_energy, opti.initial()))
                result['normalized_physics_defect_energy'] = energy
                if not np.isfinite(energy):
                    result['max_constraint_violation'] = None
                return result
            except (RuntimeError, ValueError, OverflowError) as exc:
                return {'max_constraint_violation': None,
                        'normalized_physics_defect_energy': None,
                        'worst_constraints': [], 'diagnostic_error': str(exc)}

        def phase1_feasible(diag):
            v = diag.get('max_constraint_violation')
            return (v is not None and np.isfinite(v)
                    and v <= NLP_FEASIBILITY_TOLERANCE)

        for stage_index, defect_tol in enumerate(phase1_defect_schedule):
            opti.set_value(phase1_defect_tolerance, defect_tol)
            before = diagnose_current_phase1()
            start_energy = before.get('normalized_physics_defect_energy')
            seed_is_feasible = phase1_feasible(before)
            start_variables = {
                var: np.asarray(opti.value(var, opti.initial()), dtype=np.float64)
                for var in phase1_decision_vars
            }
            budget = max(80, int(round(
                feasibility_max_iterations / len(phase1_defect_schedule))))
            stage_error = None
            candidate_diag = None
            accepted_iterate = 'start'
            # A relaxed feasibility seed can have large physical defects.
            # Unlike v2.2.4, actively minimize the dynamics defects even
            # when the current relaxed inequalities are already feasible.
            need_solve = (not seed_is_feasible or start_energy is None
                          or start_energy > 1.0e-8)
            if need_solve:
                options = {'max_iter': budget,
                           'tol': 1.0e-6, 'acceptable_tol': 1.0e-5,
                           'print_level': int(print_level)}
                if phase1_limited_memory:
                    options.update({
                        'hessian_approximation': 'limited-memory',
                        'limited_memory_max_history': 20})
                opti.solver('ipopt', {'expand': True, 'ipopt': options}, {})
                try:
                    candidate_sol = opti.solve_limited()
                except RuntimeError as exc:
                    candidate_sol = None
                    stage_error = str(exc)
                    solver_error = stage_error
                solver_stats = opti.stats()
                candidate_vals = {}
                try:
                    for var in phase1_decision_vars:
                        value = np.asarray(
                            (candidate_sol.value(var) if candidate_sol is not None
                             else opti.debug.value(var)), dtype=np.float64)
                        if not np.isfinite(value).all():
                            raise ValueError('nonfinite IPOPT candidate state')
                        candidate_vals[var] = value
                    for var, value in candidate_vals.items():
                        opti.set_initial(var, value)
                    candidate_diag = diagnose_current_phase1()
                except (RuntimeError, ValueError, OverflowError) as exc:
                    stage_error = str(exc)
                    candidate_diag = {'max_constraint_violation': None,
                                      'normalized_physics_defect_energy': None,
                                      'worst_constraints': [],
                                      'diagnostic_error': str(exc)}
                candidate_energy = candidate_diag.get(
                    'normalized_physics_defect_energy')
                improved_energy = (
                    candidate_energy is not None
                    and np.isfinite(candidate_energy)
                    and (start_energy is None
                         or candidate_energy < start_energy - 1.0e-9))
                if (phase1_feasible(candidate_diag)
                        and (not seed_is_feasible or improved_energy)):
                    accepted_iterate = 'ipopt_candidate'
                    stage1_stats = solver_stats
                else:
                    # Preserve the most recently feasible relaxed point.
                    # A failed restoration or an iteration-limit exit must
                    # not overwrite it with an infeasible or worse guess.
                    for var, value in start_variables.items():
                        opti.set_initial(var, value)
                    stage1_stats = {
                        'success': bool(seed_is_feasible),
                        'return_status': (
                            'Preserved_Initial_Feasible_Incumbent' if seed_is_feasible
                            else solver_stats.get('return_status', 'Unknown')),
                        'iter_count': int(solver_stats.get('iter_count', 0)),
                    }
            else:
                stage1_stats = {
                    'success': True,
                    'return_status': 'Near_Zero_Physics_Defect_Seed',
                    'iter_count': 0,
                }
            # Always use the CURRENT parameters and chosen seed to compute
            # the result. Old OptiSol objects snapshot relaxed bounds.
            after = diagnose_current_phase1()
            accepted_stage = phase1_feasible(after)
            stage1_stats['_initial_snapshot'] = opti.initial()
            stage1_solution = None
            phase1_homotopy_reports.append({
                'dynamic_equality_tolerance': float(defect_tol),
                'iterations_budget': budget if need_solve else 0,
                'status': stage1_stats.get('return_status'),
                'initial_max_constraint_violation': before.get('max_constraint_violation'),
                'max_constraint_violation': after.get('max_constraint_violation'),
                'initial_normalized_physics_defect_energy': start_energy,
                'normalized_physics_defect_energy': after.get(
                    'normalized_physics_defect_energy'),
                'solver_candidate_max_constraint_violation': (
                    candidate_diag.get('max_constraint_violation')
                    if candidate_diag else None),
                'accepted_iterate': accepted_iterate,
                'worst_constraints': after.get('worst_constraints', [])[:12],
                'accepted_stage': bool(accepted_stage),
                'exception': stage_error,
            })
            print(f'Phase-I dynamics merit {stage_index+1}/'
                  f'{len(phase1_defect_schedule)}: tolerance={defect_tol:.4g}, '
                  f'violation={after.get("max_constraint_violation")}, '
                  f'energy={after.get("normalized_physics_defect_energy")}, '
                  f'accepted={accepted_stage}, iterate={accepted_iterate}, '
                  f'status={stage1_stats.get("return_status")}', flush=True)
            if not accepted_stage:
                print('Phase-I stopping: could not satisfy the current '
                      'relaxed physics constraints; a tighter or minimum-time '
                      'problem would be invalid.', flush=True)
                break
            if defect_tol == 0.0:
                phase1_reached_strict = True
        # Final audit against the ORIGINAL zero-defect constraints.
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
        if existing not in (previous_overlay, previous_v223,
                            previous_v224, patched):
            raise RuntimeError(
                'Working IPOPT solver has unknown edits; refusing to overwrite. '
                f'Inspect {dest} before continuing.')
    else:
        raise FileNotFoundError('Incomplete working IPOPT cache: ' + str(dest))
    if dest.read_text(encoding='utf-8') != patched:
        dest.write_text(patched, encoding='utf-8')
    return target_root, {
        'phase1_hessian': mode,
        'phase1_method': 'v2.2.5 defect-merit minimization and stage-gated homotopy, mandatory zero tolerance',
        'dynamics_relaxation_schedule': [0.5, 0.35, 0.20, 0.10, 0.05, 0.02, 0.005, 0.0],
        'original_solver_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'working_solver_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'source_untouched': original.read_bytes() == source_bytes,
        'working_solver_path': str(dest),
        'patch_method': 'Phase-I weighted dynamics residual least-squares homotopy; original constraints required at zero-defect final stage and Phase II',
    }
