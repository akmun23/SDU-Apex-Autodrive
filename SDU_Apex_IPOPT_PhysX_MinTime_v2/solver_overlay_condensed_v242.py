"""Alternative v2.4.2 Phase-I NLP: direct residual least-squares, no slacks.

This is a DIAGNOSTIC solver only. It can produce a physically infeasible
hard-constraint-safe seed; it can NEVER export a lap or enter Phase II.
The same original dynamic residual expressions are evaluated at every iterate.
Historical git source and the v2.4.1 elastic cache remain unchanged.
"""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path

from solver_overlay import (transform_solver_v240, CONSTRAINT_ELASTIC,
    INIT_ELASTIC, SOLVE_ELASTIC, _PHASE1_FAILURE_MARKER,
    _PHASE1_RESTART_EXPORT, _PHASE1_REPORT_MARKER, _PHASE1_REPORT_EXPORT)
from wheel_model_overlay_v241 import prepare_working_wheel_model

CONSTRAINT_CONDENSED = '''    def add_constraint(expression, label: str | list[str]) -> None:
        count = int(expression.numel())
        labels = ([label] * count if isinstance(label, str) else list(label))
        if len(labels) != count:
            raise ValueError("constraint labels do not match expression size")
        physical_dynamic_labels = (
            'periodic_collocation_state_', 'physx_implicit_wheel_update_',
            'longitudinal_force_balance_', 'lateral_force_balance_',
            'unity_spring_damper_load_',
        )
        if (expression.is_op(ca.OP_EQ) and all(any(
                name.startswith(prefix) for prefix in physical_dynamic_labels)
                for name in labels)):
            # Condense F(z)-e=0: eliminate ALL slack variables and their
            # equalities, keep the exact F(z) in least-squares residual vector.
            residual = expression.dep(0) - expression.dep(1)
            elastic_pairs.append((None, residual))
            dynamic_labels.extend(labels)
            scale = (10.0 if any(name.startswith((
                'longitudinal_force_balance_', 'lateral_force_balance_'))
                for name in labels) else 1.0)
            elastic_scaled.append(ca.vec(residual / scale))
        else:
            # Wall/footprint/steering/speed/throttle/load/trust bounds stay HARD.
            opti.subject_to(expression)
            constraint_labels.extend(labels)
'''

INIT_CONDENSED = '''    if not elastic_pairs:
        raise RuntimeError('Condensed Phase I has no original dynamics')
    elastic_merit = ca.sumsqr(ca.vertcat(*elastic_scaled))
    # Stabilize otherwise weakly identified states near the measured seed.
    opti.minimize(elastic_merit + 1.0e-2 * feasibility_objective)
    opti.set_value(elastic_cap, 0.0)
    try:
        initial_constraints'''

SOLVE_CONDENSED = '''    else:
        # Phase-I ONLY: 0 original dynamics equalities are imposed here.
        # Any candidate must subsequently pass the FULL original constrained
        # solver before it can be used as a race lap. This solve never enters
        # the minimum-time stage or exports a native-validated result.
        phase1_elastic_reports = []
        physical_vars = (x, qdelta, throttle, phase_lengths, normal_load,
                         accel_xy)
        seed_arrays = (seed_x_values, seed_qdelta, seed_throttle,
                       phase_initial_lengths, seed_load,
                       np.vstack((seed_ax, seed_ay)))
        last_snapshot = {}
        dimensions = []
        for var, values in zip(physical_vars, seed_arrays):
            arr = np.asarray(values, dtype=float)
            shape = (int(var.size1()), int(var.size2()))
            if arr.size != int(var.numel()) or not np.isfinite(arr).all():
                raise RuntimeError('Invalid condensed Phase-I seed')
            last_snapshot[var] = arr.reshape(shape, order='F').copy()
            dimensions.append(shape)
        total_vars = sum(int(var.numel()) for var in physical_vars)
        if total_vars != int(opti.nx):
            raise RuntimeError('Condensation unexpectedly left slack variables')

        def flatten(snapshot):
            return np.concatenate([snapshot[var].reshape(-1, order='F')
                                   for var in physical_vars])

        def unflatten(vec):
            vec = np.asarray(vec, dtype=float).reshape(-1)
            if vec.size != total_vars or not np.isfinite(vec).all():
                raise RuntimeError('Invalid condensed IPOPT primal')
            result, offset = {}, 0
            for var, shape in zip(physical_vars, dimensions):
                count = int(var.numel())
                result[var] = vec[offset:offset+count].reshape(
                    shape, order='F').copy()
                offset += count
            return result

        numeric = ca.Function('condensed_phase1_evaluator', [opti.x, opti.p],
            [opti.g, opti.lbg, opti.ubg,
             ca.vertcat(*[ca.vec(expr) for _, expr in elastic_pairs]),
             elastic_merit])
        parameters = np.asarray(opti.value(opti.p), dtype=float).reshape(-1)

        def measure(snapshot):
            g, lo, hi, raw, merit = numeric(flatten(snapshot), parameters)
            g = np.asarray(g, dtype=float).reshape(-1)
            lo = np.asarray(lo, dtype=float).reshape(-1)
            hi = np.asarray(hi, dtype=float).reshape(-1)
            raw = np.asarray(raw, dtype=float).reshape(-1)
            merit = float(merit)
            if (not np.isfinite(g).all() or not np.isfinite(raw).all()
                    or not np.isfinite(merit)):
                return None
            hard = _constraint_diagnostics(g, lo, hi, constraint_labels)
            if hard['max_constraint_violation'] > NLP_FEASIBILITY_TOLERANCE:
                return None
            groups = {}
            for label, value in zip(dynamic_labels, raw):
                group = ('periodic_collocation_state_' +
                         label.split('periodic_collocation_state_', 1)[1]
                         if label.startswith('periodic_collocation_state_')
                         else ('physx_implicit_wheel_update' if
                          label.startswith('physx_implicit_wheel_update_')
                          else label.rsplit('_k', 1)[0]
                          if label.startswith(('longitudinal_force_balance_',
                                                'lateral_force_balance_'))
                          else 'unity_spring_damper_load'))
                groups.setdefault(group, []).append(float(value))
            breakdown = [dict(group=k, count=len(v),
                  max_abs=float(np.max(np.abs(v))),
                  rms=float(np.sqrt(np.mean(np.square(v)))))
                  for k,v in groups.items()]
            breakdown.sort(key=lambda row: -row['max_abs'])
            return dict(max_constraint_violation=hard['max_constraint_violation'],
                raw_max_dynamic_defect=float(np.max(np.abs(raw))),
                elastic_merit=merit,
                raw_dynamic_breakdown=breakdown,
                hard_worst_constraints=hard['worst_constraints'])

        initial_snapshot = {var:val.copy() for var,val in last_snapshot.items()}
        initial_measure = measure(initial_snapshot)
        if initial_measure is None:
            raise RuntimeError('Condensed Phase-I native seed violates hard limits')
        winning_snapshot, winning_measure = initial_snapshot, initial_measure
        winning_origin = 'initial_native_seed'
        accepted_ipopt_iters, accepted_backtracks, callback_errors = 0, 0, 0
        # IPOPT's filter can visit infeasible intermediates. A bounded
        # backtrack from the current hard-feasible incumbent prevents them
        # from contaminating the saved checkpoint.
        alphas = (1.0, .5, .25, .125, .0625, .03125, .015625,
                  .0078125, .00390625, .001953125, .0009765625)

        def accept_trial(trial, origin):
            nonlocal winning_snapshot, winning_measure, winning_origin
            nonlocal accepted_ipopt_iters, accepted_backtracks
            for alpha in alphas:
                proposed = {var:(winning_snapshot[var] + alpha *
                         (trial[var] - winning_snapshot[var]))
                         for var in physical_vars}
                diag = measure(proposed)
                if diag is None:
                    continue
                # Prefer reducing MAX original dynamics residual. Squared
                # merit can improve while one physically worst node worsens.
                improved_max = (diag['raw_max_dynamic_defect'] <
                    winning_measure['raw_max_dynamic_defect'] - 1e-8)
                improved_merit = (diag['elastic_merit'] <
                    winning_measure['elastic_merit'] - 1e-8)
                if improved_max and (improved_merit or
                    diag['elastic_merit'] <= 1.10 * winning_measure['elastic_merit']):
                    winning_snapshot, winning_measure = proposed, diag
                    winning_origin = origin
                    accepted_ipopt_iters += 1
                    accepted_backtracks += int(alpha < 1.0)
                    return

        def record(iteration):
            nonlocal callback_errors
            try:
                accept_trial(unflatten(opti.debug.value(opti.x)),
                             'condensed_ipopt_iterate')
            except (RuntimeError, ValueError, OverflowError):
                callback_errors += 1

        print('== Condensed Phase-I: original residual least squares, no '
              'elastic equalities; strict certification deferred ==', flush=True)
        print('Condensed NLP size: variables=', int(opti.nx),
              'hard constraints=', int(opti.ng),
              'original residual components=', sum(int(expr.numel())
                   for _,expr in elastic_pairs), flush=True)
        opti.callback(record)
        try:
            result = opti.solve_limited()
            final_error = None
        except RuntimeError as error:
            result = None
            final_error = str(error)
        stats = opti.stats()
        try:
            accept_trial(unflatten(opti.debug.value(opti.x)),
                         'condensed_ipopt_final')
        except (RuntimeError, ValueError, OverflowError):
            callback_errors += 1
        opti.callback(lambda it: None)
        for var, val in winning_snapshot.items():
            opti.set_initial(var, val)
        strict_diag = dict(winning_measure)
        strict_diag['max_constraint_violation'] = max(
            strict_diag['max_constraint_violation'],
            strict_diag['raw_max_dynamic_defect'])
        strict_diag['worst_constraints'] = [dict(
            label='original_'+row['group'],
            violation=row['max_abs']) for row in
            strict_diag['raw_dynamic_breakdown'][:12]]
        phase1_elastic_reports.append(dict(
            name='condensed original-residual least squares (diagnostic)',
            initial_raw_max_defect=initial_measure['raw_max_dynamic_defect'],
            initial_elastic_merit=initial_measure['elastic_merit'],
            best_raw_max_defect=winning_measure['raw_max_dynamic_defect'],
            best_elastic_merit=winning_measure['elastic_merit'],
            strict_max_constraint_violation=strict_diag['max_constraint_violation'],
            final_exact_physics_feasible=False,
            accepted_iterate=winning_origin,
            feasible_iteration_improvements=accepted_ipopt_iters,
            accepted_hard_feasible_backtracks=accepted_backtracks,
            callback_errors=callback_errors,
            ipopt_status=stats.get('return_status'),
            ipopt_iterations=stats.get('iter_count'),
            solver_exception=final_error,
            initial_dynamic_defect_breakdown=initial_measure['raw_dynamic_breakdown'],
            selected_dynamic_defect_breakdown=winning_measure['raw_dynamic_breakdown'],
            worst_constraints=strict_diag['worst_constraints']))
        # Export a hard-feasible warm start in exactly the v2.4.0 schema,
        # while explicitly REFUSING to interpret it as physical feasibility.
        sx = np.asarray(winning_snapshot[x], dtype=float)
        loads = np.asarray(winning_snapshot[normal_load], dtype=float)
        accel = np.asarray(winning_snapshot[accel_xy], dtype=float)
        lengths = np.asarray(winning_snapshot[phase_lengths], dtype=float).reshape(-1)
        beginnings = np.r_[0.0, np.cumsum(lengths[:-1])]
        stations = np.concatenate([beginnings[i] + lengths[i] *
            np.arange(int(count)) / int(count)
            for i, count in enumerate(node_counts)])
        if (sx.shape != (16,n) or loads.shape != (4,n) or accel.shape != (2,n)
                or stations.shape != (n,) or np.any(np.diff(stations) <= 0)
                or abs(stations[0]) > 1e-8 or stations[-1] >= track.length):
            raise RuntimeError('Invalid condensed checkpoint dimensions/mesh')
        names = ('ey_m','epsi_rad','u_mps','v_mps','yaw_rate_radps',
                 'steering_rad','body_height_m','body_height_rate_mps',
                 'roll_rad','roll_rate_radps','pitch_rad','pitch_rate_radps')
        ixs = (0,1,2,3,4,9,10,11,12,13,14,15)
        seed = {key:sx[idx].tolist() for key,idx in zip(names,ixs)}
        seed.update(wheel_omega_radps=sx[5:9,:].tolist(),
                    normal_load_n=loads.tolist(),
                    ax_mps2=accel[0].tolist(),ay_mps2=accel[1].tolist(),
                    steering_rate_radps=np.asarray(
                        winning_snapshot[qdelta]).reshape(-1).tolist(),
                    throttle_norm=np.asarray(
                        winning_snapshot[throttle]).reshape(-1).tolist())
        checkpoint = dict(seed=seed,station_nodes_m=stations.tolist(),
            phase_lengths_m=lengths.tolist(),
            raw_max_dynamic_defect=winning_measure['raw_max_dynamic_defect'],
            elastic_merit=winning_measure['elastic_merit'],
            hard_constraints_satisfied=True,
            strict_original_dynamics_feasible=False,origin=winning_origin)
        print('Condensed Phase-I diagnostics: seed_max=',
              initial_measure['raw_max_dynamic_defect'], 'best_max=',
              winning_measure['raw_max_dynamic_defect'],
              'hard_feasible=', True, 'exact_dynamics_certified=', False,
              flush=True)
        return {}, {
            'optimizer':'CasADi/IPOPT reduced original-residual Phase-I',
            'track':track.to_dict(),
            'unity_vehicle_model':model.metadata(),
            'elapsed_s':time.perf_counter()-started,
            'solver':dict(success=False,
                return_status=stats.get('return_status','Unknown'),
                ipopt_converged=False,iterations=stats.get('iter_count',0),
                exception=final_error, feasible_incumbent=False),
            'solve_stages':phase1_elastic_reports,
            'selected_stage':'condensed original-residual least squares',
            'nlp_diagnostics':dict(**strict_diag,initial_guess=initial_constraints,
                feasible_incumbent=False,solver_error=final_error),
            'phase1_strict_feasible':False,
            'condensed_candidate_raw_below_tolerance':bool(
                winning_measure['raw_max_dynamic_defect'] <= NLP_FEASIBILITY_TOLERANCE),
            'phase1_exact_zero_slack_audited':False,
            'phase1_restart_checkpoint':checkpoint,
            'brake_mode_seed_source':brake_mode_seed_source,
        }

    def stage_values'''


def transform_solver_condensed(raw: str, hessian: str = 'limited-memory') -> str:
    """Transform ONLY byte-exact known upstream sites; fail closed."""
    generated = transform_solver_v240(raw, hessian)
    original_solve = SOLVE_ELASTIC.replace(
        _PHASE1_FAILURE_MARKER, _PHASE1_RESTART_EXPORT + '\n', 1).replace(
        _PHASE1_REPORT_MARKER, _PHASE1_REPORT_EXPORT, 1)
    replacements = [
        (CONSTRAINT_ELASTIC, CONSTRAINT_CONDENSED, 'constraint-condenser'),
        (INIT_ELASTIC, INIT_CONDENSED, 'least-squares initializer'),
        (original_solve, SOLVE_CONDENSED, 'Phase-I solve'),
        ('    elastic_scaled = []\n',
         '    elastic_scaled = []\n    dynamic_labels = []\n', 'dynamic labels'),
        ('    opti.set_value(trust_region_bound, 1.0e12)\n',
         '    opti.set_value(trust_region_bound, 4.0 * feasibility_trust_region)\n',
         'finite initialization trust'),
    ]
    for old,new,name in replacements:
        count=generated.count(old)
        if count!=1:raise RuntimeError(f'Unknown {name} site (count={count})')
        generated=generated.replace(old,new,1)
    # Original physical equations remain in the dynamic least-squares vector.
    assert 'ca.sumsqr(ca.vertcat(*elastic_scaled))' in generated
    assert 'opti.subject_to(residual - elastic == 0)' not in generated
    return generated


def prepare_working_solver_condensed(recovered_root: Path, target_root: Path,
                                     legacy_name: str, hessian: str):
    source_root=Path(recovered_root).resolve()
    target_root=Path(target_root).resolve()
    original=source_root/legacy_name/'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py'
    if source_root==target_root:raise ValueError('Historical source is read-only')
    raw=original.read_text(encoding='utf-8')
    generated=transform_solver_condensed(raw,hessian)
    target=target_root/legacy_name/'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py'
    if not target_root.exists():
        shutil.copytree(source_root,target_root)
    elif not target.is_file():
        raise RuntimeError('Incomplete condensed model cache')
    elif target.read_text(encoding='utf-8')!=generated:
        raise RuntimeError('Unknown edits in condensed model cache; refuse overwrite')
    provenance=prepare_working_wheel_model(source_root,target_root,legacy_name)
    if target.read_text(encoding='utf-8')!=generated:
        target.write_text(generated,encoding='utf-8')
    return target_root, dict(
        phase1_method='v2.4.2 condensed exact-original-residual least squares',
        phase1_hessian=hessian,
        condensed_phase1_diagnostic_only=True,
        exact_zero_dynamics_accepted_from_condensed=False,
        physical_dynamics_changed=False,
        physical_equations_stored_as_objective=True,
        original_solver_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        working_solver_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        source_untouched=original.read_text(encoding='utf-8')==raw,
        working_solver_path=str(target), **provenance)
