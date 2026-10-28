"""v2.2.7 elastic-dynamics feasibility overlay for historical Unity-wheel IPOPT.

The source model, tire laws, steering/braking and track safety remain intact.
Only the numeric form of Phase I is altered: a dynamic equality F(z)=0 is
represented by F(z)-e=0 with an elastic defect variable e. During Phase I,
minimize the weighted squared defects. Phase II is strictly gated by e=0 and
an explicit audit of the ORIGINAL dynamic residuals.

The historical source in _restored_git_history is never edited.
"""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path

from solver_numerics import guard_solver

from solver_overlay_v226 import (
    HESSIAN_SITE, HESSIAN_ADDITION, OPTIMIZER_DECLARATION, CONSTRAINT_FN,
    TRUST_SITE, INIT_SITE, FEASIBILITY_SITE, SOLVE_SITE,
    STRICT_BOUND_DIAGNOSTICS_SITE, REPORT_SITE, STRICT_GATE_SITE,
    _replace_once,
)

CONSTRAINT_ELASTIC = '''    def add_constraint(expression, label: str | list[str]) -> None:
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
            # Introducing independent elastic variables makes the synthetic
            # guess an EXACT feasible point for the modified dynamics while
            # allowing the nonlinear solver to reduce their magnitude.
            residual = expression.dep(0) - expression.dep(1)
            elastic = opti.variable(residual.size1(), residual.size2())
            elastic_pairs.append((elastic, residual))
            elastic_vars.append(elastic)
            scale = (10.0 if any(name.startswith((
                'longitudinal_force_balance_', 'lateral_force_balance_'))
                for name in labels) else 1.0)
            elastic_scaled.append(ca.vec(elastic / scale))
            # Record each equality at the original insertion site, preserving
            # the ordering expected by the existing diagnostics.
            opti.subject_to(residual - elastic == 0)
            constraint_labels.extend(labels)
            # This is a dynamic-only cap: 1e5 during Phase I, EXACTLY 0
            # during Phase II. Never a relaxation of lane/footprint limits.
            opti.subject_to(opti.bounded(-elastic_cap, elastic, elastic_cap))
            constraint_labels.extend(['elastic_bound_' + label for label in labels])
        else:
            opti.subject_to(expression)
            constraint_labels.extend(labels)
'''

INIT_ELASTIC = "    if not elastic_pairs:\n        raise RuntimeError('No dynamic equalities were selected for elastic initialization')\n    elastic_initial_values = {}\n    for elastic, residual in elastic_pairs:\n        values = np.asarray(opti.value(residual, opti.initial()), dtype=np.float64)\n        if not np.isfinite(values).all():\n            raise RuntimeError('Non-finite original vehicle dynamics in synthetic seed')\n        opti.set_initial(elastic, values)\n        elastic_initial_values[elastic] = values.copy()\n    # Initial point is now algebraically compatible with the relaxed\n    # dynamic equations. All safety and actuator constraints remain hard.\n    elastic_merit = ca.sumsqr(ca.vertcat(*elastic_scaled))\n    # Small regularizer prevents unlimited movement of unconstrained\n    # variables without pinning the solution to inaccurate seed dynamics.\n    opti.minimize(elastic_merit + 1.0e-4 * feasibility_objective)\n    opti.set_value(elastic_cap, 1.0e5)\n    try:\n        initial_constraints"

SOLVE_ELASTIC_V233 = '    else:\n        # Solve ONE well-defined elastic feasibility problem instead of\n        # tightening tolerances around a synthetic iterate that never moves.\n        phase1_elastic_reports = []\n        stage1_solution = None\n        stage1_stats = {}\n        physical_vars = (x, qdelta, throttle, phase_lengths, normal_load,\n                         accel_xy, *elastic_vars)\n        # CasADi Opti.initial() reconstructs substitutions across the full\n        # 6,000+ variable graph. Calling it for each state, wheel, or defect\n        # cost minutes in the measured user traceback. Cache the numeric\n        # initialization and compile one vectorized equality evaluator.\n        def seed_matrix(var, values):\n            arr = np.asarray(values, dtype=np.float64)\n            if arr.size != int(var.numel()) or not np.isfinite(arr).all():\n                raise RuntimeError(\'Invalid numeric Phase-I warm-start variable\')\n            return arr.reshape((int(var.size1()), int(var.size2())),\n                               order=\'F\').copy()\n\n        last_snapshot = {\n            x: seed_matrix(x, seed_x_values),\n            qdelta: seed_matrix(qdelta, seed_qdelta),\n            throttle: seed_matrix(throttle, seed_throttle),\n            phase_lengths: seed_matrix(phase_lengths, phase_initial_lengths),\n            normal_load: seed_matrix(normal_load, seed_load),\n            accel_xy: seed_matrix(accel_xy, np.vstack((seed_ax, seed_ay))),\n        }\n        for elastic in elastic_vars:\n            last_snapshot[elastic] = seed_matrix(elastic,\n                                                  elastic_initial_values[elastic])\n        var_slices = {}\n        offset = 0\n        for var in physical_vars:\n            count = int(var.numel())\n            var_slices[var] = (offset, offset + count,\n                               int(var.size1()), int(var.size2()))\n            offset += count\n        if offset != int(opti.nx):\n            raise RuntimeError(\'Phase-I cache does not cover all Opti variables\')\n\n        def primal_vector(snapshot):\n            return np.concatenate([\n                snapshot[var].reshape(-1, order=\'F\') for var in physical_vars])\n\n        def snapshot_from_vector(values):\n            flat = np.asarray(values, dtype=np.float64).reshape(-1)\n            if flat.size != offset:\n                raise RuntimeError(\'Unexpected IPOPT primal vector length\')\n            return {var: flat[lo:hi].reshape((rows, cols), order=\'F\').copy()\n                    for var, (lo, hi, rows, cols) in var_slices.items()}\n\n        # Evaluate original g, original bounds and original dynamic residuals\n        # in one CasADi function. Bounds still come from the CURRENT parameters\n        # (the slack cap switches to strict zero before final acceptance).\n        numerical_phase1 = ca.Function(\n            \'phase1_vectorized_evaluation\', [opti.x, opti.p],\n            [opti.g, opti.lbg, opti.ubg,\n             ca.vertcat(*[ca.vec(residual) for _, residual in elastic_pairs]),\n             elastic_merit])\n        raw_shapes = [(var, int(var.numel())) for var in elastic_vars]\n\n        def numeric_evaluation(primal):\n            params = np.asarray(opti.value(opti.p),\n                                dtype=np.float64).reshape(-1)\n            g, low, high, raw, energy = numerical_phase1(primal, params)\n            return (np.asarray(g, dtype=np.float64).reshape(-1),\n                    np.asarray(low, dtype=np.float64).reshape(-1),\n                    np.asarray(high, dtype=np.float64).reshape(-1),\n                    np.asarray(raw, dtype=np.float64).reshape(-1),\n                    float(energy))\n\n        def snapshot_current(debug=False):\n            if debug:\n                # One Opti call per IPOPT callback, not one per variable.\n                return snapshot_from_vector(opti.debug.value(opti.x))\n            return {var: value.copy() for var, value in last_snapshot.items()}\n\n        def set_snapshot(snapshot):\n            for var, val in snapshot.items():\n                opti.set_initial(var, val)\n            last_snapshot.clear()\n            last_snapshot.update({var: np.asarray(val).copy()\n                                  for var, val in snapshot.items()})\n\n        def phase1_measure(debug=False):\n            try:\n                values = snapshot_current(debug=debug)\n                g, low, high, raw, energy = numeric_evaluation(\n                    primal_vector(values))\n                diag = _constraint_diagnostics(g, low, high, constraint_labels)\n                actual_elastic = np.concatenate([\n                    np.asarray(values[elastic]).reshape(-1, order=\'F\')\n                    for elastic in elastic_vars])\n                defect = float(np.max(np.abs(raw)))\n                elast_max = float(np.max(np.abs(actual_elastic)))\n                if not (np.isfinite(raw).all() and np.isfinite(actual_elastic).all()\n                        and np.isfinite(energy)):\n                    diag[\'max_constraint_violation\'] = None\n                diag.update(raw_max_dynamic_defect=defect,\n                            max_elastic_slack=elast_max,\n                            elastic_merit=energy)\n                return diag\n            except (RuntimeError, ValueError, OverflowError) as exc:\n                return {\'max_constraint_violation\': None,\n                        \'raw_max_dynamic_defect\': None,\n                        \'max_elastic_slack\': None,\n                        \'elastic_merit\': None,\n                        \'worst_constraints\': [], \'error\': str(exc)}\n\n        def eligible(diag):\n            violation = diag.get(\'max_constraint_violation\')\n            energy = diag.get(\'elastic_merit\')\n            return (violation is not None and energy is not None and\n                    np.isfinite(energy) and np.isfinite(violation) and\n                    violation <= NLP_FEASIBILITY_TOLERANCE)\n\n        initial_snapshot = snapshot_current()\n        initial_measure = phase1_measure()\n        print(\'Phase-I numeric seed diagnostics ready; starting IPOPT\', flush=True)\n        winning_snapshot = initial_snapshot if eligible(initial_measure) else None\n        winning_measure = initial_measure if eligible(initial_measure) else None\n        winning_origin = \'initial_elastic_guess\' if winning_snapshot else \'none\'\n        accepted_ipopt_iters = 0\n        callback_errors = 0\n        # Cache constraints from the same numerical evaluator, with exact\n        # current parameter values and without Opti.initial() substitutions.\n        _, low, high, _, _ = numeric_evaluation(primal_vector(initial_snapshot))\n        # Project each trial\'s ELASTIC VARIABLES onto its actual dynamic\n        # residuals before acceptance. Elastic e is a numerical artifact:\n        # an IPOPT trial with valid safety/actuator constraints and a better\n        # physical residual must not be discarded just because e differs\n        # slightly from F(z). No physical constraint is relaxed here.\n        elastic_physical_prefixes = (\n            \'periodic_collocation_state_\', \'physx_implicit_wheel_update_\',\n            \'longitudinal_force_balance_\', \'lateral_force_balance_\',\n            \'unity_spring_damper_load_\',\n        )\n        hard_indices = np.array([not (name.startswith(elastic_physical_prefixes)\n                           or name.startswith(\'elastic_bound_\'))\n                           for name in constraint_labels], dtype=bool)\n        dynamic_labels = [name for name in constraint_labels\n                          if name.startswith(elastic_physical_prefixes)]\n        def projected_trial():\n            # Evaluate the actual IPOPT primal once, then project only the\n            # numerical slack variables. Never relax physical hard bounds.\n            values = snapshot_current(debug=True)\n            g, _, _, all_raw, _ = numeric_evaluation(primal_vector(values))\n            if not np.isfinite(g).all() or not np.isfinite(all_raw).all():\n                return None, None\n            hard_viol = max(float(np.max(low[hard_indices]-g[hard_indices])),\n                            float(np.max(g[hard_indices]-high[hard_indices])), 0.)\n            if hard_viol > NLP_FEASIBILITY_TOLERANCE:\n                return None, None\n            energy = 0.0\n            max_defect = 0.0\n            label_offset = 0\n            raw_offset = 0\n            elastic_limit = float(opti.value(elastic_cap))\n            for elastic, residual in elastic_pairs:\n                size = int(elastic.numel())\n                raw = all_raw[raw_offset:raw_offset + size].reshape(\n                    (int(elastic.size1()), int(elastic.size2())), order=\'F\')\n                raw_offset += size\n                if float(np.max(np.abs(raw))) > elastic_limit:\n                    return None, None\n                labels = dynamic_labels[label_offset:label_offset + size]\n                label_offset += size\n                norm = 10.0 if any(label.startswith((\n                    \'longitudinal_force_balance_\', \'lateral_force_balance_\'))\n                    for label in labels) else 1.0\n                energy += float(np.sum((raw / norm) ** 2))\n                max_defect = max(max_defect, float(np.max(np.abs(raw))))\n                values[elastic] = raw.copy()\n            return values, {\'elastic_merit\': energy,\n                            \'raw_max_dynamic_defect\': max_defect,\n                            \'max_elastic_slack\': max_defect,\n                            \'max_constraint_violation\': hard_viol}\n\n        # An IPOPT trial is accepted only if *all nonelastic hard constraints*\n        # are satisfied and the projected dynamic slack improves the merit.\n        def save_feasible_iterate(iteration):\n            nonlocal winning_snapshot, winning_measure, winning_origin\n            nonlocal accepted_ipopt_iters, callback_errors\n            try:\n                candidate, diag = projected_trial()\n                if candidate is None:\n                    return\n                if (winning_measure is not None and diag[\'elastic_merit\'] >=\n                        winning_measure[\'elastic_merit\'] - 1.0e-8):\n                    return\n                winning_snapshot, winning_measure = candidate, diag\n                winning_origin = \'ipopt_projected_iteration\'\n                accepted_ipopt_iters += 1\n            except (RuntimeError, ValueError, OverflowError, KeyError):\n                callback_errors += 1\n\n        opti.callback(save_feasible_iterate)\n        try:\n            result = opti.solve_limited()\n            final_error = None\n        except RuntimeError as exc:\n            result = None\n            final_error = str(exc)\n            solver_error = final_error\n        stats = opti.stats()\n        try:\n            final_candidate, final_measure = projected_trial()\n            if (final_candidate is not None and\n                    (winning_measure is None or final_measure[\'elastic_merit\'] <\n                     winning_measure[\'elastic_merit\'] - 1.0e-8)):\n                winning_snapshot, winning_measure = final_candidate, final_measure\n                winning_origin = \'ipopt_projected_final\'\n        except (RuntimeError, ValueError, OverflowError):\n            final_measure = {\'max_constraint_violation\': None}\n        if winning_snapshot is None:\n            set_snapshot(initial_snapshot)\n        else:\n            set_snapshot(winning_snapshot)\n        # CRITICAL: force zero elastic slack bounds BEFORE any acceptance,\n        # before checking the legacy NLP and before switching to lap time.\n        opti.callback(lambda it: None)\n        opti.set_value(elastic_cap, 0.0)\n        strict_diag = phase1_measure()\n        strict_ok = (eligible(strict_diag) and\n                     strict_diag[\'raw_max_dynamic_defect\']\n                     <= NLP_FEASIBILITY_TOLERANCE and\n                     strict_diag[\'max_elastic_slack\']\n                     <= NLP_FEASIBILITY_TOLERANCE)\n        phase1_elastic_reports.append({\n            \'name\': \'elastic model-consistent initialization\',\n            \'initial_raw_max_defect\': initial_measure.get(\'raw_max_dynamic_defect\'),\n            \'initial_elastic_merit\': initial_measure.get(\'elastic_merit\'),\n            \'best_raw_max_defect\': (winning_measure or {}).get(\'raw_max_dynamic_defect\'),\n            \'best_elastic_merit\': (winning_measure or {}).get(\'elastic_merit\'),\n            \'strict_max_constraint_violation\': strict_diag.get(\'max_constraint_violation\'),\n            \'final_exact_physics_feasible\': bool(strict_ok),\n            \'accepted_iterate\': winning_origin,\n            \'feasible_iteration_improvements\': accepted_ipopt_iters,\n            \'callback_errors\': callback_errors,\n            \'ipopt_status\': stats.get(\'return_status\'),\n            \'ipopt_iterations\': stats.get(\'iter_count\'),\n            \'solver_exception\': final_error,\n            \'worst_constraints\': strict_diag.get(\'worst_constraints\', [])[:12],\n        })\n        print(\'Phase-I elastic dynamics: \'\n              f\'seed_defect={initial_measure.get("raw_max_dynamic_defect")}, \'\n              f\'best_defect={(winning_measure or {}).get("raw_max_dynamic_defect")}, \'\n              f\'strict_violation={strict_diag.get("max_constraint_violation")}, \'\n              f\'accepted={strict_ok}, source={winning_origin}\', flush=True)\n        stage1_stats = {\n            \'success\': bool(strict_ok),\n            \'return_status\': (\'Elastic_Exact_Dynamics_Feasible\' if strict_ok\n                              else stats.get(\'return_status\', \'Unknown\')),\n            \'iter_count\': stats.get(\'iter_count\', 0),\n            \'_initial_snapshot\': opti.initial() if strict_ok else None,\n        }\n        if not strict_ok:\n            stage1_stats[\'_phase1_elastic_strict_failed\'] = True\n\n    def stage_values'
# v2.3.4: preserve strict Phase-I diagnostics without querying an unsolved Opti stack.
SOLVE_ELASTIC = SOLVE_ELASTIC_V233.replace("        if not strict_ok:\n            stage1_stats['_phase1_elastic_strict_failed'] = True\n\n    def stage_values", "        if not strict_ok:\n            stage1_stats['_phase1_elastic_strict_failed'] = True\n            # The historical reporting path expects a solved Opti stack.\n            # An elastic Phase-I failure is *not* an Opti solution and must\n            # not fall through to Opti.value() or Opti.debug.value().\n            # We have already audited the same candidate against the strict\n            # zero-slack bounds using the cached, vectorized evaluator.\n            return {}, {\n                'optimizer': 'CasADi/IPOPT phase-switched spatial direct collocation',\n                'track': track.to_dict(),\n                'unity_vehicle_model': model.metadata(),\n                'elapsed_s': time.perf_counter() - started,\n                'solver': {\n                    'success': False,\n                    'return_status': stats.get('return_status', 'Unknown'),\n                    'ipopt_converged': False,\n                    'iterations': stats.get('iter_count', 0),\n                    'exception': final_error,\n                    'feasible_incumbent': False,\n                },\n                'solve_stages': phase1_elastic_reports,\n                'selected_stage': 'elastic model-consistent initialization',\n                'nlp_diagnostics': {\n                    **strict_diag,\n                    'initial_guess': initial_constraints,\n                    'feasible_incumbent': False,\n                    'solver_error': final_error,\n                },\n                'phase1_exact_zero_slack_audited': True,\n                'phase1_strict_feasible': False,\n                'brake_mode_seed_source': brake_mode_seed_source,\n            }\n\n    def stage_values", 1)



# Preserve the byte-exact v2.3.4 generated solver for fail-closed cache upgrades.
SOLVE_ELASTIC_V234 = SOLVE_ELASTIC

# v2.3.5 Phase-I checkpointing: an IPOPT full step can have valid physical
# directions but violate nonlinear track/actuator bounds before line search.
# Evaluate shorter steps FROM THE LAST ACCEPTED hard-feasible candidate,
# project ONLY the numerical elastic variables, and accept a point only when
# every hard constraint still passes and the physical-defect merit decreases.
# IPOPT's own iterate and optimization constraints are not changed.
_old_projected_trial = '''        def projected_trial():
            # Evaluate the actual IPOPT primal once, then project only the
            # numerical slack variables. Never relax physical hard bounds.
            values = snapshot_current(debug=True)'''
_new_projected_trial = '''        def projected_trial(values=None):
            # Evaluate the actual IPOPT primal once, then project only the
            # numerical slack variables. Never relax physical hard bounds.
            if values is None:
                values = snapshot_current(debug=True)'''
assert SOLVE_ELASTIC.count(_old_projected_trial)==1
SOLVE_ELASTIC = SOLVE_ELASTIC.replace(_old_projected_trial, _new_projected_trial, 1)

_old_callback = '''        # An IPOPT trial is accepted only if *all nonelastic hard constraints*
        # are satisfied and the projected dynamic slack improves the merit.
        def save_feasible_iterate(iteration):
            nonlocal winning_snapshot, winning_measure, winning_origin
            nonlocal accepted_ipopt_iters, callback_errors
            try:
                candidate, diag = projected_trial()
                if candidate is None:
                    return
                if (winning_measure is not None and diag['elastic_merit'] >=
                        winning_measure['elastic_merit'] - 1.0e-8):
                    return
                winning_snapshot, winning_measure = candidate, diag
                winning_origin = 'ipopt_projected_iteration'
                accepted_ipopt_iters += 1
            except (RuntimeError, ValueError, OverflowError, KeyError):
                callback_errors += 1'''
_new_callback = '''        # A full IPOPT iterate may be outside safety bounds. Never accept it
        # directly in that case, but examine shorter physical-variable steps
        # from the last verified hard-feasible checkpoint. Slack is explicitly
        # reprojected onto the original dynamics after each trial step. Retain
        # checkpoints by the strict gate's max-defect metric first; scaled
        # sum-of-squares is only a tie-breaker at equal worst defect.
        checkpoint_alphas = (1., .5, .25, .125, .0625, .03125,
                             .015625, .0078125, .00390625,
                             .001953125, .0009765625, .00048828125)
        accepted_backtracks = 0
        def checked_checkpoint(trial, reference, reference_measure):
            # Side-effect-free w.r.t. IPOPT; used only for recording candidate
            # states and potential subsequent starts, never to certify physics.
            for alpha in checkpoint_alphas:
                if alpha == 1.:
                    proposed = trial
                else:
                    proposed = {var: (reference[var] +
                        alpha * (trial[var] - reference[var]))
                        for var in physical_vars}
                candidate, diag = projected_trial(proposed)
                if candidate is None:
                    continue
                merit = diag['elastic_merit']
                defect = diag['raw_max_dynamic_defect']
                reference_merit = (None if reference_measure is None else
                                   reference_measure.get('elastic_merit'))
                reference_defect = (None if reference_measure is None else
                                    reference_measure.get('raw_max_dynamic_defect'))
                improves_worst_defect = (reference_defect is None or
                    defect < reference_defect - 1.0e-8)
                holds_worst_defect_and_improves_merit = (
                    reference_defect is not None and
                    defect <= reference_defect + 1.0e-8 and
                    (reference_merit is None or
                     merit < reference_merit - 1.0e-8))
                if (np.isfinite(merit) and np.isfinite(defect) and
                    (improves_worst_defect or
                     holds_worst_defect_and_improves_merit)):
                    return candidate, diag, alpha
            return None, None, None

        def save_feasible_iterate(iteration):
            nonlocal winning_snapshot, winning_measure, winning_origin
            nonlocal accepted_ipopt_iters, callback_errors, accepted_backtracks
            checkpoint_status = 'no_improvement'
            checkpoint_alpha = None
            checkpoint_measure = None
            try:
                trial = snapshot_current(debug=True)
                reference = (winning_snapshot if winning_snapshot is not None
                             else initial_snapshot)
                base = winning_measure
                candidate, diag, alpha = checked_checkpoint(trial, reference, base)
                if candidate is None:
                    checkpoint_status = 'hard_reject_or_no_descent'
                    return
                winning_snapshot, winning_measure = candidate, diag
                winning_origin = ('ipopt_projected_iteration' if alpha == 1.
                                  else 'ipopt_hard_feasible_backtrack')
                accepted_ipopt_iters += 1
                checkpoint_status = 'accepted'
                checkpoint_alpha = alpha
                checkpoint_measure = diag
                if alpha != 1.:
                    accepted_backtracks += 1
            except (RuntimeError, ValueError, OverflowError, KeyError):
                callback_errors += 1
                checkpoint_status = 'callback_error' '''
assert SOLVE_ELASTIC.count(_old_callback)==1
SOLVE_ELASTIC = SOLVE_ELASTIC.replace(_old_callback, _new_callback, 1)
_old_final = '''        try:
            final_candidate, final_measure = projected_trial()
            if (final_candidate is not None and
                    (winning_measure is None or final_measure['elastic_merit'] <
                     winning_measure['elastic_merit'] - 1.0e-8)):
                winning_snapshot, winning_measure = final_candidate, final_measure
                winning_origin = 'ipopt_projected_final'
        except (RuntimeError, ValueError, OverflowError):
            final_measure = {'max_constraint_violation': None}'''
_new_final = '''        try:
            final_trial = snapshot_current(debug=True)
            final_ref = (winning_snapshot if winning_snapshot is not None
                         else initial_snapshot)
            final_base = winning_measure
            final_candidate, final_measure, final_alpha = checked_checkpoint(
                final_trial, final_ref, final_base)
            if final_candidate is not None:
                winning_snapshot, winning_measure = final_candidate, final_measure
                winning_origin = ('ipopt_projected_final' if final_alpha == 1.
                                  else 'ipopt_hard_feasible_final_backtrack')
                if final_alpha != 1.:
                    accepted_backtracks += 1
        except (RuntimeError, ValueError, OverflowError):
            final_measure = {'max_constraint_violation': None}'''
assert SOLVE_ELASTIC.count(_old_final)==1
SOLVE_ELASTIC = SOLVE_ELASTIC.replace(_old_final, _new_final, 1)
_old_metric="            'feasible_iteration_improvements': accepted_ipopt_iters,\n"
assert SOLVE_ELASTIC.count(_old_metric)==1
SOLVE_ELASTIC = SOLVE_ELASTIC.replace(_old_metric,
    _old_metric+"            'accepted_hard_feasible_backtracks': accepted_backtracks,\n",1)


# Keep the exact generated v2.3.6 string for fail-closed cache checks.
SOLVE_ELASTIC_V236 = SOLVE_ELASTIC
# Add raw-physical dynamic residual group diagnostics, based on the same CasADi
# numerical evaluator and original g label order; no new physics model.
needle="""        def phase1_measure(debug=False):
"""
insert="""        def defect_breakdown(raw):
            # Dynamic labels are placed before their corresponding slack caps.
            labels = [name for name in constraint_labels if
                      name.startswith(('periodic_collocation_state_',
                          'physx_implicit_wheel_update_',
                          'longitudinal_force_balance_',
                          'lateral_force_balance_',
                          'unity_spring_damper_load_'))]
            if len(labels) != len(raw):
                # Unit-test surrogate programs may not use vehicle labels.
                # Never fabricate wheel/state attributions in that case.
                return [{'group': 'unlabeled_residual',
                         'count': int(len(raw)),
                         'max_abs': float(np.max(np.abs(raw))),
                         'rms': float(np.sqrt(np.mean(raw * raw))),
                         'worst_label': None, 'worst_signed_value': None}]
            buckets = {}
            for label, val in zip(labels, raw):
                if label.startswith('periodic_collocation_state_'):
                    group = label
                elif label.startswith('physx_implicit_wheel_update_'):
                    group = 'physx_implicit_wheel_update'
                elif label.startswith('longitudinal_force_balance_'):
                    group = 'longitudinal_force_balance'
                elif label.startswith('lateral_force_balance_'):
                    group = 'lateral_force_balance'
                else:
                    group = 'unity_spring_damper_load'
                bucket = buckets.setdefault(group, [])
                bucket.append((label, float(val)))
            breakdown = []
            for group, entries in buckets.items():
                worst_label, worst_value = max(entries,
                                                key=lambda item: abs(item[1]))
                vec = np.fromiter((val for _, val in entries), dtype=float)
                breakdown.append({
                    'group': group, 'count': len(entries),
                    'max_abs': abs(worst_value),
                    'rms': float(np.sqrt(np.mean(vec * vec))),
                    'worst_label': worst_label,
                    'worst_sample_index': int(np.argmax(np.abs(vec))),
                    'worst_signed_value': worst_value,
                })
            return sorted(breakdown, key=lambda item: -item['max_abs'])

"""
assert SOLVE_ELASTIC.count(needle)==1
SOLVE_ELASTIC=SOLVE_ELASTIC.replace(needle,insert+needle,1)
needle="                diag.update(raw_max_dynamic_defect=defect,\n                            max_elastic_slack=elast_max,\n                            elastic_merit=energy)"
replacement="""                diag.update(raw_max_dynamic_defect=defect,
                            max_elastic_slack=elast_max,
                            elastic_merit=energy,
                            raw_dynamic_breakdown=defect_breakdown(raw))"""
assert SOLVE_ELASTIC.count(needle)==1
SOLVE_ELASTIC=SOLVE_ELASTIC.replace(needle,replacement,1)
# For accepted checkpoint, obtain full raw breakdown once, not per backtrack.
needle="""        if winning_snapshot is None:
            set_snapshot(initial_snapshot)
        else:
            set_snapshot(winning_snapshot)
        # CRITICAL"""
repl="""        if winning_snapshot is None:
            set_snapshot(initial_snapshot)
        else:
            set_snapshot(winning_snapshot)
        selected_pre_strict_measure = phase1_measure()
        # CRITICAL"""
assert SOLVE_ELASTIC.count(needle)==1
SOLVE_ELASTIC=SOLVE_ELASTIC.replace(needle,repl,1)
report_seed = "        phase1_elastic_reports.append({\n            'name': 'elastic model-consistent initialization',\n"
report_seed_with_radius = (report_seed +
    "            'seed_trust_region_bound': float(opti.value(trust_region_bound)),\n")
assert SOLVE_ELASTIC.count(report_seed) == 1
SOLVE_ELASTIC = SOLVE_ELASTIC.replace(report_seed, report_seed_with_radius, 1)
_stride_callback = _new_callback.replace(
    "        def save_feasible_iterate(iteration):",
    "        checkpoint_stride = 10\n"
    "        checkpoint_callbacks_skipped = 0\n"
    "        checkpoint_callback_count = 0\n"
    "        checkpoint_callback_wall_s = 0.0\n"
    "        def save_feasible_iterate(iteration):", 1)
_stride_callback = _stride_callback.replace(
    "            nonlocal accepted_ipopt_iters, callback_errors, accepted_backtracks\n"
    "            checkpoint_status = 'no_improvement'\n"
    "            checkpoint_alpha = None\n"
    "            checkpoint_measure = None\n"
    "            try:\n                trial = snapshot_current(debug=True)",
    "            nonlocal accepted_ipopt_iters, callback_errors, accepted_backtracks\n"
    "            nonlocal checkpoint_callbacks_skipped, checkpoint_callback_count\n"
    "            nonlocal checkpoint_callback_wall_s\n"
    "            if int(iteration) % checkpoint_stride:\n"
    "                checkpoint_callbacks_skipped += 1\n"
    "                return\n"
    "            checkpoint_status = 'no_improvement'\n"
    "            checkpoint_alpha = None\n"
    "            checkpoint_measure = None\n"
    "            callback_started = time.perf_counter()\n"
    "            checkpoint_callback_count += 1\n"
    "            try:\n                trial = snapshot_current(debug=True)", 1)
_stride_callback = _stride_callback.replace(
    "            except (RuntimeError, ValueError, OverflowError, KeyError):\n"
    "                callback_errors += 1",
    "            except (RuntimeError, ValueError, OverflowError, KeyError):\n"
    "                callback_errors += 1\n"
    "            finally:\n"
    "                checkpoint_callback_wall_s += (\n"
    "                    time.perf_counter() - callback_started)\n"
    "                progress_measure = (checkpoint_measure or winning_measure or {})\n"
    "                print(\n"
    "                    f'Phase-I iter={int(iteration):4d} '\n"
    "                    f'status={checkpoint_status} '\n"
    "                    f'alpha={checkpoint_alpha} '\n"
    "                    f'candidate_merit={(checkpoint_measure or {}).get(\"elastic_merit\")} '\n"
    "                    f'best_merit={progress_measure.get(\"elastic_merit\")} '\n"
    "                    f'best_max_defect={progress_measure.get(\"raw_max_dynamic_defect\")} '\n"
    "                    f'accepted={accepted_ipopt_iters} '\n"
    "                    f'callback_errors={callback_errors}', flush=True)", 1)
assert _stride_callback != _new_callback
assert SOLVE_ELASTIC.count(_new_callback) == 1
SOLVE_ELASTIC = SOLVE_ELASTIC.replace(_new_callback, _stride_callback, 1)

needle="            'accepted_hard_feasible_backtracks': accepted_backtracks,\n"
repl="""            'accepted_hard_feasible_backtracks': accepted_backtracks,
            'phase1_checkpoint_stride': checkpoint_stride,
            'phase1_checkpoint_callback_count': checkpoint_callback_count,
            'phase1_checkpoint_callbacks_skipped': checkpoint_callbacks_skipped,
            'phase1_checkpoint_callback_wall_s': checkpoint_callback_wall_s,
            'initial_dynamic_defect_breakdown': initial_measure.get('raw_dynamic_breakdown', []),
            'selected_dynamic_defect_breakdown': selected_pre_strict_measure.get('raw_dynamic_breakdown', []),
"""
assert SOLVE_ELASTIC.count(needle)==1
SOLVE_ELASTIC=SOLVE_ELASTIC.replace(needle,repl,1)

from solver_overlay_v229_compat import SOLVE_ELASTIC_V229

TRUST_ELASTIC_V236 = '''    # The synthetic starting profile is not measured capture data.
    # Disable the capture-specific distance cap during feasibility; the
    # original minimum-time continuation restores its own trust factors.
    opti.set_value(trust_region_bound, 1.0e12)
'''

TRUST_ELASTIC = '''    # Phase I starts from measured native motion. Keep model-consistency
    # search inside the configured normalized neighborhood; only after exact
    # dynamics feasibility does the time-optimal continuation widen its own
    # trust region.
    opti.set_value(trust_region_bound, feasibility_trust_region)
'''

STRICT_DIAGNOSTICS_ELASTIC = '''    def stage_constraint_diagnostics(solution, stats):
        try:
            values = stage_values(solution, stats, opti.g).reshape(-1)
            # ALWAYS read the strict zero-slack bounds from the live model.
            lower = np.asarray(opti.value(opti.lbg, opti.initial()),
                               dtype=np.float64).reshape(-1)
            upper = np.asarray(opti.value(opti.ubg, opti.initial()),
                               dtype=np.float64).reshape(-1)'''


def transform_solver(source: str, mode: str, *, _numeric_guards: bool = True, _project_elastic_trials: bool = True, _fix_wheel_substep_speed: bool = True) -> str:
    if mode not in ('limited-memory', 'exact'):
        raise ValueError('Phase-I Hessian must be limited-memory or exact')
    if mode == 'limited-memory':
        source = _replace_once(source, HESSIAN_SITE, HESSIAN_ADDITION,
                               'Phase-I Hessian')
    source = _replace_once(source, OPTIMIZER_DECLARATION,
        OPTIMIZER_DECLARATION +
        '    elastic_pairs = []\n'
        '    elastic_label_by_variable = {}\n'
        '    elastic_vars = []\n'
        '    elastic_scaled = []\n'
        '    elastic_cap = opti.parameter()\n'
        '    opti.set_value(elastic_cap, 1.0e5)\n', 'Opti construction')
    source = _replace_once(source, CONSTRAINT_FN, CONSTRAINT_ELASTIC,
                           'dynamic constraints')
    source = _replace_once(source, TRUST_SITE, TRUST_ELASTIC,
                           'synthetic seed captured-motion cap')
    source = _replace_once(source, INIT_SITE, INIT_ELASTIC,
                           'initialize elastic defects')
    source = _replace_once(source, FEASIBILITY_SITE,
                           '    initial_is_feasible = False',
                           'strict model feasibility guard')
    source = _replace_once(source, SOLVE_SITE,
                           SOLVE_ELASTIC if _project_elastic_trials else SOLVE_ELASTIC_V229,
                           'elastic solve with strict gate')
    source = _replace_once(source, STRICT_BOUND_DIAGNOSTICS_SITE,
                           STRICT_DIAGNOSTICS_ELASTIC,
                           'strict bound diagnostics')
    source = _replace_once(source, REPORT_SITE,
                           REPORT_SITE + '    stage_reports.extend(phase1_elastic_reports)\n',
                           'elastic reports')
    source = _replace_once(source, STRICT_GATE_SITE,
        '    if (strict_ok\n'
        '            and stage1_diagnostics.get("max_constraint_violation") is not None\n'
        '            and stage1_diagnostics["max_constraint_violation"]\n'
        '            <= NLP_FEASIBILITY_TOLERANCE):', 'minimum-time strict gate')
    return guard_solver(source, fix_wheel_substep_speed=_fix_wheel_substep_speed) if _numeric_guards else source


def _remove_v231_only_declaration(code: str) -> str:
    """Restore byte-exact older cache output without relaxing cache validation.

    v2.3.1 added this unused symbol while incorporating its new projected-trial
    handler. The legacy handlers never included it. Its presence meant the
    existing v2.3.0 generated file was incorrectly classified as user-edited.
    """
    declaration = '    elastic_label_by_variable = {}\n'
    if code.count(declaration) != 1:
        raise RuntimeError('Cannot reconstruct historical overlay: unexpected declaration count')
    return code.replace(declaration, '', 1)


def _old_projected_seed_init(source: str) -> str:
    """Restore historical elastic seed creation for verified old cache hashes."""
    from solver_overlay_v232_compat import INIT_ELASTIC_V232
    if source.count(INIT_ELASTIC) != 1:
        raise RuntimeError('Unexpected elastic seed site in historical overlay')
    return source.replace(INIT_ELASTIC, INIT_ELASTIC_V232, 1)


def transform_solver_228(source: str, mode: str) -> str:
    """Byte-exact previous v2.2.8 overlay for verified cache upgrades."""
    return _remove_v231_only_declaration(_old_projected_seed_init(transform_solver(
        source, mode, _numeric_guards=False, _project_elastic_trials=False)))


def transform_solver_v229(source: str, mode: str) -> str:
    """Byte-exact v2.2.9/v2.3.0 generated solver for safe in-place upgrade."""
    return _remove_v231_only_declaration(_old_projected_seed_init(transform_solver(
        source, mode, _project_elastic_trials=False, _fix_wheel_substep_speed=False)))


def transform_solver_v234(source: str, mode: str) -> str:
    """Byte-exact v2.3.4 generated solver for safe cache upgrade."""
    latest = transform_solver(source, mode, _fix_wheel_substep_speed=False)
    if latest.count(SOLVE_ELASTIC) != 1:
        raise RuntimeError('Cannot reproduce v2.3.4 optimizer overlay')
    return latest.replace(SOLVE_ELASTIC, SOLVE_ELASTIC_V234, 1)


def transform_solver_v233(source: str, mode: str) -> str:
    """Reconstruct the exact v2.3.3 generated solver before reporting fix."""
    latest = transform_solver(source, mode, _fix_wheel_substep_speed=False)
    if latest.count(SOLVE_ELASTIC) != 1:
        raise RuntimeError('Cannot reproduce v2.3.3 optimizer overlay')
    return latest.replace(SOLVE_ELASTIC, SOLVE_ELASTIC_V233, 1)


def transform_solver_v232(source: str, mode: str) -> str:
    """Reconstruct byte-exact v2.3.2 working solver for safe cache upgrade."""
    from solver_overlay_v232_compat import INIT_ELASTIC_V232, SOLVE_ELASTIC_V232
    latest = transform_solver(source, mode, _fix_wheel_substep_speed=False)
    if latest.count(INIT_ELASTIC) != 1 or latest.count(SOLVE_ELASTIC) != 1:
        raise RuntimeError('Cannot reproduce v2.3.2 optimizer overlay')
    return latest.replace(INIT_ELASTIC, INIT_ELASTIC_V232, 1).replace(
        SOLVE_ELASTIC, SOLVE_ELASTIC_V232, 1)


def transform_solver_v236(source: str, mode: str) -> str:
    """Byte-exact v2.3.6 cache, before max-defect-monotone checkpoint fix."""
    latest = transform_solver(source, mode)
    if latest.count(SOLVE_ELASTIC) != 1:
        raise RuntimeError('Cannot reconstruct v2.3.6 generated solver')
    latest = latest.replace(SOLVE_ELASTIC, SOLVE_ELASTIC_V236, 1)
    return latest.replace(TRUST_ELASTIC, TRUST_ELASTIC_V236, 1)


def transform_solver_v235(source: str, mode: str) -> str:
    """Byte-exact v2.3.5 working solver, before forward-speed index repair."""
    return transform_solver(source, mode, _fix_wheel_substep_speed=False)


_CONTACT_SOLVER_CALL_OLD = '''        fx, fy, mz, fx_w, _, slip_x, slip_y = tire_forces_casadi(
            model, u[k], v[k], yaw_rate[k], steering[k],
            [wheel_omega[i][k] for i in range(4)], fz)'''
_CONTACT_SOLVER_CALL_NEW = '''        fx, fy, mz, fx_w, _, slip_x, slip_y = tire_forces_casadi(
            model, u[k], v[k], yaw_rate[k], steering[k],
            [wheel_omega[i][k] for i in range(4)], fz,
            roll_rate[k], pitch_rate[k])'''

def transform_solver_v2316(source: str, mode: str) -> str:
    """Pass OCP roll/pitch rate states to native tire contact velocity."""
    patched=transform_solver(source,mode)
    if '    def source_wheel_update(' not in source:
        return patched  # Isolated historical toy fixture has no real OCP.
    count=patched.count(_CONTACT_SOLVER_CALL_OLD)
    if count != 1:
        raise RuntimeError('Unexpected historical OCP tire contact site: '+str(count))
    return patched.replace(_CONTACT_SOLVER_CALL_OLD,_CONTACT_SOLVER_CALL_NEW,1)



# v2.4.0: preserve the best *nonzero-defect* Phase-I diagnostic state for
# controlled solver continuation. This is never a physically feasible result.
_PHASE1_FAILURE_MARKER = """        if not strict_ok:
            stage1_stats['_phase1_elastic_strict_failed'] = True
"""
_PHASE1_RESTART_EXPORT = """        if not strict_ok:
            stage1_stats['_phase1_elastic_strict_failed'] = True
            # A selected checkpoint has satisfied ALL hard safety/actuator
            # constraints; it is NOT certified against original dynamics.
            # Export its entire state and optimized spatial mesh, so another
            # Phase-I solve can start HERE rather than at the original native
            # trajectory. Never export arbitrary/off-bounds IPOPT trial data.
            phase1_restart_checkpoint = None
            if (winning_snapshot is not None and winning_measure is not None
                    and eligible(winning_measure)):
                sx = np.asarray(winning_snapshot[x], dtype=np.float64)
                loads = np.asarray(winning_snapshot[normal_load], dtype=np.float64)
                accel = np.asarray(winning_snapshot[accel_xy], dtype=np.float64)
                lengths = np.asarray(winning_snapshot[phase_lengths],
                                     dtype=np.float64).reshape(-1)
                beginnings = np.r_[0.0, np.cumsum(lengths[:-1])]
                stations = np.concatenate([
                    beginnings[i] + lengths[i] * np.arange(int(count)) / int(count)
                    for i, count in enumerate(node_counts)])
                if (sx.shape != (16, n) or loads.shape != (4, n)
                        or accel.shape != (2, n) or stations.shape != (n,)
                        or not np.isfinite(stations).all()
                        or not np.all(np.diff(stations) > 0.0)
                        or stations[0] < -1.0e-8 or stations[-1] >= track.length):
                    raise RuntimeError('Invalid safe Phase-I restart mesh/state')
                names = ('ey_m', 'epsi_rad', 'u_mps', 'v_mps',
                         'yaw_rate_radps', 'steering_rad', 'body_height_m',
                         'body_height_rate_mps', 'roll_rad', 'roll_rate_radps',
                         'pitch_rad', 'pitch_rate_radps')
                ixs = (0, 1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 15)
                seed = {key: sx[idx].tolist() for key, idx in zip(names, ixs)}
                seed.update({
                    'wheel_omega_radps': sx[5:9, :].tolist(),
                    'normal_load_n': loads.tolist(),
                    'ax_mps2': accel[0].tolist(),
                    'ay_mps2': accel[1].tolist(),
                    'steering_rate_radps': np.asarray(
                        winning_snapshot[qdelta], dtype=np.float64).reshape(-1).tolist(),
                    'throttle_norm': np.asarray(
                        winning_snapshot[throttle], dtype=np.float64).reshape(-1).tolist(),
                })
                phase1_restart_checkpoint = {
                    'seed': seed,
                    'station_nodes_m': stations.tolist(),
                    'phase_lengths_m': lengths.tolist(),
                    'raw_max_dynamic_defect': winning_measure['raw_max_dynamic_defect'],
                    'elastic_merit': winning_measure['elastic_merit'],
                    'hard_constraints_satisfied': True,
                    'strict_original_dynamics_feasible': False,
                    'origin': winning_origin,
                }
"""
_PHASE1_REPORT_MARKER = """                'brake_mode_seed_source': brake_mode_seed_source,
            }
"""
_PHASE1_REPORT_EXPORT = """                'brake_mode_seed_source': brake_mode_seed_source,
                'phase1_restart_checkpoint': phase1_restart_checkpoint,
            }
"""

def transform_solver_v240(source: str, mode: str) -> str:
    """Consolidated Phase-I continuation for the v2.3.16 physics model."""
    patched = transform_solver_v2316(source, mode)
    # Exact historical tests may use stripped-down solver fixtures.
    if '    def source_wheel_update(' not in source:
        return patched
    if patched.count(_PHASE1_FAILURE_MARKER) != 1:
        raise RuntimeError('Unknown Phase-I failure site: refuse unsafe rewrite')
    patched = patched.replace(_PHASE1_FAILURE_MARKER, _PHASE1_RESTART_EXPORT + '\n', 1)
    if patched.count(_PHASE1_REPORT_MARKER) != 1:
        raise RuntimeError('Unknown Phase-I failure report site')
    return patched.replace(_PHASE1_REPORT_MARKER, _PHASE1_REPORT_EXPORT, 1)


def transform_solver_phase1_raw_diagnostics(
        source: str, mode: str, *, include_initial_raw_diagnostics: bool = True) -> str:
    """Add raw Phase-I dynamics diagnostics without changing vehicle physics."""
    patched = transform_solver_v240(source, mode)
    if '    def source_wheel_update(' not in source:
        return patched  # stripped unit-test fixture, not the active OCP
    if not include_initial_raw_diagnostics:
        return patched
    diagnostic_site = '    if diagnose_initial_only:\n        return {}, {'
    diagnostic = '''    # The relaxed elastic equalities make the NLP seed constraint-feasible,
    # which can hide large defects in the original vehicle equations. Report
    # those source-model residuals explicitly without another OCP evaluation.
    initial_dynamic_labels = [label for label in constraint_labels if label.startswith((
        'periodic_collocation_state_', 'physx_implicit_wheel_update_',
        'longitudinal_force_balance_', 'lateral_force_balance_',
        'unity_spring_damper_load_'))]
    initial_raw_samples = {}
    label_offset = 0
    for elastic in elastic_vars:
        values = np.asarray(elastic_initial_values[elastic], dtype=np.float64).reshape(-1, order='F')
        labels = initial_dynamic_labels[label_offset:label_offset + values.size]
        label_offset += values.size
        if len(labels) != values.size:
            raise RuntimeError('Initial raw-dynamics labels do not match elastic residuals')
        for label, value in zip(labels, values):
            if label.startswith('periodic_collocation_state_'):
                group = 'periodic_collocation_state:' + label.removeprefix('periodic_collocation_state_')
            elif label.startswith('longitudinal_force_balance_'):
                group = 'longitudinal_force_balance'
            elif label.startswith('lateral_force_balance_'):
                group = 'lateral_force_balance'
            elif label.startswith('unity_spring_damper_load_'):
                group = 'unity_spring_damper_load'
            else:
                group = 'physx_implicit_wheel_update'
            initial_raw_samples.setdefault(group, []).append({
                'label': label,
                'value': float(value),
            })
    initial_raw_dynamic_defects = {
        group: {
            'count': len(samples),
            'max_abs': float(np.max(np.abs([item['value'] for item in samples]))),
            'rmse': float(np.sqrt(np.mean(np.square(
                [item['value'] for item in samples])))),
            'p95_abs': float(np.percentile(np.abs(
                [item['value'] for item in samples]), 95.0)),
            'worst_sample_index': int(np.argmax(np.abs(
                [item['value'] for item in samples]))),
            'worst_constraint_label': max(
                samples, key=lambda item: abs(item['value']))['label'],
        }
        for group, samples in initial_raw_samples.items()
    }
    initial_raw_values = [item['value']
                          for samples in initial_raw_samples.values()
                          for item in samples]
    initial_raw_max_defect = (float(np.max(np.abs(initial_raw_values)))
                              if initial_raw_values else None)
'''
    if patched.count(diagnostic_site) != 1:
        if 'diagnose_initial_only' not in patched:
            return patched  # unit-test fixture omits the initial-only path
        raise RuntimeError('Unknown initial-only diagnostics insertion site')
    patched = patched.replace(diagnostic_site, diagnostic + diagnostic_site, 1)
    report_site = '            "nlp_diagnostics": {"initial_guess": initial_constraints},'
    report = ('            "nlp_diagnostics": {"initial_guess": initial_constraints,\n'
              '                                "initial_raw_dynamic_max_defect": initial_raw_max_defect,\n'
              '                                "initial_raw_dynamic_defects": initial_raw_dynamic_defects},')
    if patched.count(report_site) != 1:
        raise RuntimeError('Unknown initial-only report insertion site')
    return patched.replace(report_site, report, 1)


def _previous_raw_diagnostic_cache(patched: str) -> str:
    """Return the exact pre-label cache variant for safe generated-cache upgrade."""
    if "initial_raw_samples.setdefault(group, []).append({" not in patched:
        return patched  # Minimal test fixture has no raw-diagnostic block.
    replacements = (
        ("""            initial_raw_samples.setdefault(group, []).append({
                'label': label,
                'value': float(value),
            })""",
         """            initial_raw_samples.setdefault(group, []).append(float(value))"""),
        ("""            'max_abs': float(np.max(np.abs([item['value'] for item in samples]))),
            'rmse': float(np.sqrt(np.mean(np.square(
                [item['value'] for item in samples])))),
            'p95_abs': float(np.percentile(np.abs(
                [item['value'] for item in samples]), 95.0)),
            'worst_sample_index': int(np.argmax(np.abs(
                [item['value'] for item in samples]))),
            'worst_constraint_label': max(
                samples, key=lambda item: abs(item['value']))['label'],""",
         """            'max_abs': float(np.max(np.abs(samples))),
            'rmse': float(np.sqrt(np.mean(np.square(samples)))),
            'p95_abs': float(np.percentile(np.abs(samples), 95.0)),
            'worst_sample_index': int(np.argmax(np.abs(samples))),"""),
        ("""    initial_raw_values = [item['value']
                          for samples in initial_raw_samples.values()
                          for item in samples]""",
         """    initial_raw_values = [value for samples in initial_raw_samples.values()
                          for value in samples]"""),
    )
    result = patched
    for current, previous in replacements:
        if result.count(current) != 1:
            raise RuntimeError('Cannot recognize prior raw-diagnostic cache form')
        result = result.replace(current, previous, 1)
    return result


def prepare_working_physx_substep_source(recovered_root: Path,
                                         working_root: Path) -> dict:
    """Pin the generated wheel helper to the verified practice-player setup."""
    rel = Path('tools/vehicle_dynamics_learning/physx_no_drive_source.py')
    historical = Path(recovered_root).resolve() / rel
    generated = Path(working_root).resolve() / rel
    if not historical.is_file() or not generated.is_file():
        return {'physx_substep_source_available': False}
    original = historical.read_text(encoding='utf-8')
    old_block = ('UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 5.0\n'
                 'UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5\n'
                 'UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1')
    new_block = ('# Pinned practice UnityPlayer.so Build ID\n'
                 '# 79f401cda3570218568e40b5331a0e68aa86f11c calls\n'
                 '# setSubStepCount(5.0f, 5, 1). PhysX selects five substeps\n'
                 '# when abs(body velocity projected onto vehicle forward) < 5 m/s.\n'
                 'UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 5.0\n'
                 'UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5\n'
                 'UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1')
    old_bad_block = ('# Unity VehicleCollider initialization calls\n'
                     '# setSubStepCount(128.0f, 5, 1); keep this distinct from the\n'
                     '# unrelated 5 m/s wheel-angle integration threshold.\n'
                     'UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 128.0\n'
                     'UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5\n'
                     'UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1')
    if original.count(old_block) != 1:
        if new_block not in original:
            raise RuntimeError('Unknown recovered PhysX wheel-substep constants')
        patched = original
    else:
        patched = original.replace(old_block, new_block, 1)
    current = generated.read_text(encoding='utf-8')
    prior_bad_cache = (original.replace(old_block, old_bad_block, 1)
                       if original.count(old_block) == 1 else None)
    if current not in (original, patched, prior_bad_cache):
        raise RuntimeError('Unknown generated PhysX substep helper; refusing overwrite: '
                           + str(generated))
    if current != patched:
        generated.write_text(patched, encoding='utf-8')
    return {
        'physx_substep_source_available': True,
        'physx_substep_source_path': str(generated),
        'physx_substep_source_sha256': hashlib.sha256(
            patched.encode('utf-8')).hexdigest(),
        'physx_substep_threshold_mps': 5.0,
        'physx_substep_counts': [5, 1],
        'physx_substep_call': 'setSubStepCount(5.0f, 5, 1)',
        'unity_player_build_id': '79f401cda3570218568e40b5331a0e68aa86f11c',
        'recovered_source_untouched': historical.read_text(
            encoding='utf-8') == original,
    }

def prepare_working_solver(recovered_root: Path, working_root: Path,
                           legacy_name: str,
                           mode: str = 'limited-memory') -> tuple[Path, dict]:
    source_root = Path(recovered_root).resolve()
    target_root = Path(working_root).resolve()
    original = (source_root / legacy_name /
                'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py')
    if not original.is_file():
        raise FileNotFoundError(original)
    source_bytes = original.read_bytes()
    raw = source_bytes.decode('utf-8')
    patched = transform_solver_phase1_raw_diagnostics(raw, mode)
    if '    def source_wheel_update(' in raw and (
            patched.count(_CONTACT_SOLVER_CALL_NEW) != 1 or
            _CONTACT_SOLVER_CALL_OLD in patched):
        raise RuntimeError('Generated OCP does not include full roll/pitch contact velocity')
    dest = (target_root / legacy_name /
            'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py')
    if source_root == target_root:
        raise ValueError('Refusing to patch the historical source directory')
    if not target_root.exists():
        shutil.copytree(source_root, target_root)
    elif dest.is_file():
        old_hessian = (raw.replace(HESSIAN_SITE, HESSIAN_ADDITION, 1)
                       if mode == 'limited-memory' else raw)
        from solver_overlay_v223 import transform_solver as v223
        from solver_overlay_v224 import transform_solver as v224
        from solver_overlay_v225 import transform_solver as v225
        from solver_overlay_v226 import transform_solver as v226
        # Permit upgrading the exact known v2.2.8 overlay, but refuse
        # unexpected edits to the solver cache.
        old_elastic = transform_solver_228(raw, mode)
        known = (old_hessian, v223(raw, mode), v224(raw, mode),
                 v225(raw, mode), v226(raw, mode), old_elastic,
                 transform_solver_v229(raw, mode),
                 transform_solver_v232(raw, mode),
                 transform_solver_v233(raw, mode),
                 transform_solver_v234(raw, mode),
                 transform_solver_v235(raw, mode),
                 transform_solver_v236(raw, mode),
                 transform_solver(raw, mode), transform_solver_v2316(raw, mode),
                 transform_solver_v240(raw, mode),
                 transform_solver_v240(raw,
                     'exact' if mode == 'limited-memory' else 'limited-memory'),
                 transform_solver_phase1_raw_diagnostics(raw, mode,
                     include_initial_raw_diagnostics=False),
                 transform_solver_phase1_raw_diagnostics(raw,
                     'exact' if mode == 'limited-memory' else 'limited-memory'),
                 transform_solver_v2316(raw,
                     'exact' if mode == 'limited-memory' else 'limited-memory'),
                 # Exact previously generated cache with the now-corrected
                 # Phase-I diagnostic overlay but the erroneous 128 m/s text.
                 transform_solver_phase1_raw_diagnostics(raw, mode).replace(
                     '5 m/s', '128 m/s'),
                 _previous_raw_diagnostic_cache(patched),
                 patched)
        if dest.read_text(encoding='utf-8') not in known:
            raise RuntimeError('Working solver has unknown local edits; refusing to overwrite '
                               + str(dest))
    else:
        raise FileNotFoundError('Incomplete solver cache: ' + str(dest))
    # Upgrade the source-matched wheel callback only in the GENERATED working
    # tree. Unknown local edits to either cached file fail before overwrite.
    from wheel_model_overlay_v241 import prepare_working_wheel_model
    wheel_provenance = prepare_working_wheel_model(source_root, target_root,
                                                  legacy_name)
    substep_provenance = prepare_working_physx_substep_source(
        source_root, target_root)
    if dest.read_text(encoding='utf-8') != patched:
        dest.write_text(patched, encoding='utf-8')
    return target_root, {
        'phase1_hessian': mode,
        'phase1_method': 'v2.4.9 measured-seed trust region + flushed max-defect-first checkpoints; verified 3-axis contact velocity and wheel-substep corrections',
        'numerical_guards_equal_on_physical_domain': True,
        'phase1_detect_simple_bounds': True,
        'physical_dynamics_changed': True,
        'physical_dynamics_change': '3-axis contact-point velocity passes roll_rate and pitch_rate into the tire model; contact-arm correction; prior wheel-state index u=x[2,k] retained; PhysX NoDrive substep branch uses the pinned practice UnityPlayer setSubStepCount(5.0f,5,1); Phase-I max-defect-first checkpoint selection',
        'generated_ocp_full_contact_velocity_verified': (
            patched.count(_CONTACT_SOLVER_CALL_NEW) == 1 and
            _CONTACT_SOLVER_CALL_OLD not in patched),
        'wheel_substep_selection': 'Pinned practice UnityPlayer setSubStepCount(5.0f, 5, 1): abs(body velocity projected onto vehicle forward) <5m/s -> 5 substeps; >=5m/s -> 1',
        'phase1_checkpoint_continuation': True,
        'phase1_checkpoint_stride': 10,
        'phase1_seed_trust_region': 'configured feasibility_trust_region (default 200 normalized squared-distance units)',
        'elastic_slack_bound_phase1': 1.0e5,
        'elastic_slack_bound_final': 0.0,
        'original_solver_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'working_solver_sha256': hashlib.sha256(patched.encode()).hexdigest(),
        'source_untouched': original.read_bytes() == source_bytes,
        'working_solver_path': str(dest),
        **wheel_provenance,
        **substep_provenance,
    }
