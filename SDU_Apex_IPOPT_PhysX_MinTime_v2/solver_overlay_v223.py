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
INIT_REPLACEMENT = '''    opti.minimize(feasibility_objective)
    # Initial Phase-I defects are allowed only as an initialization homotopy.
    # At the final stage this parameter becomes EXACTLY zero.
    phase1_defect_schedule = (0.5, 0.2, 0.07, 0.015, 0.0)
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
        # A high-dimensional tire/suspension OCP with periodic dynamics is
        # difficult to initialize from centerline geometry alone. Solve a
        # sequence of NLPs, decreasing only the allowed dynamics equality
        # residual. Phase-II can run *only* after the final zero-defect stage.
        fractions = (0.10, 0.14, 0.18, 0.22, 0.36)
        stage1_solution = None
        stage1_stats = {}
        phase1_homotopy_reports = []
        for stage_index, (defect_tol, fraction) in enumerate(zip(
                phase1_defect_schedule, fractions)):
            opti.set_value(phase1_defect_tolerance, defect_tol)
            stage_budget = max(20, int(round(feasibility_max_iterations * fraction)))
            stage_ipopt_options = {
                "max_iter": stage_budget,
                "tol": 1.0e-6,
                "acceptable_tol": 1.0e-5,
                "print_level": int(print_level),
            }
            if phase1_limited_memory:
                stage_ipopt_options.update({
                    "hessian_approximation": "limited-memory",
                    "limited_memory_max_history": 20,
                })
            opti.solver("ipopt", {"expand": True,
                                  "ipopt": stage_ipopt_options}, {})
            try:
                stage1_solution = opti.solve_limited()
                stage_error = None
            except RuntimeError as error:
                stage1_solution = None
                stage_error = str(error)
                solver_error = stage_error
            stage1_stats = opti.stats()
            # IPOPT solve_limited() can return a point even when it hits its
            # iteration budget. Transfer only finite numerical iterates.
            warm_start_ok = True
            for variable in (x, qdelta, throttle, phase_lengths,
                             normal_load, accel_xy):
                try:
                    values = np.asarray(
                        (stage1_solution.value(variable)
                         if stage1_solution is not None
                         else opti.debug.value(variable)), dtype=np.float64)
                    if not np.isfinite(values).all():
                        warm_start_ok = False
                        break
                    opti.set_initial(variable, values)
                except (RuntimeError, ValueError):
                    warm_start_ok = False
                    break
            try:
                g = np.asarray(opti.debug.value(opti.g), dtype=np.float64).reshape(-1)
                lbg = np.asarray(opti.debug.value(opti.lbg), dtype=np.float64).reshape(-1)
                ubg = np.asarray(opti.debug.value(opti.ubg), dtype=np.float64).reshape(-1)
                diagnostics = _constraint_diagnostics(g, lbg, ubg, constraint_labels)
                maximum = diagnostics.get("max_constraint_violation")
            except (RuntimeError, ValueError):
                maximum = None
            phase1_homotopy_reports.append({
                "dynamic_equality_tolerance": float(defect_tol),
                "iterations_budget": stage_budget,
                "status": stage1_stats.get("return_status"),
                "max_constraint_violation": maximum,
                "finite_warm_start": bool(warm_start_ok),
                "exception": stage_error,
            })
            print(f"Phase-I defect continuation {stage_index+1}/"
                  f"{len(phase1_defect_schedule)}: "
                  f"dynamics tolerance={defect_tol:.4g}, "
                  f"max violation={maximum}, "
                  f"status={stage1_stats.get('return_status')}", flush=True)
            if not warm_start_ok:
                # Still force zero-defect diagnosis below, never report a
                # relaxed/invalid point as a strict feasible incumbent.
                break
        opti.set_value(phase1_defect_tolerance, 0.0)

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
        if existing not in (previous_overlay, patched):
            raise RuntimeError(
                'Working IPOPT solver has unknown edits; refusing to overwrite. '
                f'Inspect {dest} before continuing.')
    else:
        raise FileNotFoundError('Incomplete working IPOPT cache: ' + str(dest))
    if dest.read_text(encoding='utf-8') != patched:
        dest.write_text(patched, encoding='utf-8')
    return target_root, {
        'phase1_hessian': mode,
        'phase1_method': 'dynamic equality defect homotopy, final zero tolerance',
        'dynamics_relaxation_schedule': [0.5, 0.2, 0.07, 0.015, 0.0],
        'original_solver_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'working_solver_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'source_untouched': original.read_bytes() == source_bytes,
        'working_solver_path': str(dest),
        'patch_method': 'Phase-I numerical initializer only; exact original constraints restored at zero-defect final stage and Phase II',
    }
