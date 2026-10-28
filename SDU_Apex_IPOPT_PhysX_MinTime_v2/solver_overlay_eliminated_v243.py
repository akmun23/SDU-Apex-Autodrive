"""v2.4.3: reduced-algebraic diagnostic Phase I.

Source-preserving transform of the v2.4.2 direct-residual formulation:
wheel loads and body accelerations are **expressions**, not free NLP vars.
Their original equalities are satisfied by construction; surviving dynamic
residuals are evaluated unrelaxed. Numeric report never certifies a lap.
"""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from solver_overlay_condensed_v242 import (transform_solver_condensed,
    SOLVE_CONDENSED, INIT_CONDENSED)
from wheel_model_overlay_v241 import prepare_working_wheel_model

_STATE_SITE = '''    body_height, body_height_rate, roll, roll_rate, pitch, pitch_rate = (
        x[i, :] for i in range(10, 16))'''
_EXPRESSIONS = '''

    # Algebraic variable elimination (not a physical-model approximation).
    # The exact same spring and tire functions used in the former equalities
    # now DEFINE loads and accelerations, leaving no independent load/accel DOFs.
    normal_load = ca.horzcat(*[
        ca.vertcat(*suspension_loads_casadi(
            model, body_height[k], body_height_rate[k], roll[k],
            roll_rate[k], pitch[k], pitch_rate[k])) for k in range(n)])
    _derived_accel_columns = []
    for k in range(n):
        _fz = [normal_load[i, k] for i in range(4)]
        _fx, _fy, _, _, _, _, _ = tire_forces_casadi(
            model, x[2, k], x[3, k], x[4, k], x[9, k],
            [x[5 + i, k] for i in range(4)], _fz,
            roll_rate[k], pitch_rate[k])
        _derived_accel_columns.append(ca.vertcat(
            _fx / model.mass_kg - model.drag_per_s * x[2, k],
            _fy / model.mass_kg - model.drag_per_s * x[3, k]))
    accel_xy = ca.horzcat(*_derived_accel_columns)
'''
_ACCEL_SITE = '''        add_constraint(ax - (fx / model.mass_kg
                             - model.drag_per_s * u[k]) == 0,
                       f"longitudinal_force_balance_k{k}")
        add_constraint(ay - (fy / model.mass_kg
                             - model.drag_per_s * v[k]) == 0,
                       f"lateral_force_balance_k{k}")'''
_LOAD_SITE = '''            add_constraint((normal_load[i, k] - predicted_load[i]) / 10.0 == 0,
                           f"unity_spring_damper_load_w{i}")'''


def _one(source: str, old: str, new: str, name: str, count: int = 1) -> str:
    actual = source.count(old)
    if actual != count:
        raise RuntimeError(f'Unexpected {name}: {actual} occurrences (need {count})')
    return source.replace(old, new)


def transform_solver_eliminated(raw: str, hessian: str = 'limited-memory') -> str:
    code = transform_solver_condensed(raw, hessian)
    # Avoid the common v2.4.2 failure where load residuals (already /10 N)
    # contributed 70% of the objective while the larger force errors were
    # discounted another factor 100. Here surviving residuals use THEIR
    # physical-equation scalings, no additional arbitrary /10 for accelerations.
    scale_site = '''            scale = (10.0 if any(name.startswith((
                'longitudinal_force_balance_', 'lateral_force_balance_'))
                for name in labels) else 1.0)
            elastic_scaled.append(ca.vec(residual / scale))'''
    code = _one(code, scale_site, '            elastic_scaled.append(ca.vec(residual))',
                'force objective reweighting')
    code = _one(code, '    normal_load = opti.variable(4, n)\n',
                '    normal_load = None  # derived below\n', 'free normal-load declaration')
    code = _one(code, '    accel_xy = opti.variable(2, n)\n',
                '    accel_xy = None  # derived below\n', 'free acceleration declaration')
    code = _one(code, _STATE_SITE, _STATE_SITE + _EXPRESSIONS,
                'body-state algebraic construction')
    code = _one(code, _ACCEL_SITE,
                '        # Fx/m, Fy/m force balance is an algebraic identity.\n'
                '        # Original residual cannot be relaxed or mismatched.',
                'redundant accelerations equalities')
    code = _one(code, _LOAD_SITE,
                '            # Spring load equality is an algebraic identity.',
                'redundant suspension equality')
    code = _one(code, '    opti.set_initial(normal_load, seed_load)\n',
                '    # normal_load is derived from pose/rates, not initialized separately.\n',
                'top-level normal-load setter', count=2)
    code = _one(code, '    opti.set_initial(accel_xy, np.vstack((seed_ax, seed_ay)))\n',
                '    # accel_xy is derived from exact tire forces, not separately initialized.\n',
                'top-level acceleration setter', count=2)
    # Fix fresh-seed projection: v2.3.16 added roll/pitch contact velocities
    # to the OCP but the initializer continued evaluating them at zero rates.
    code = _one(code, '    seed_force_state = ca.MX.sym(f"seed_force_state_{id(opti)}", 12, n)',
                '    seed_force_state = ca.MX.sym(f"seed_force_state_{id(opti)}", 14, n)',
                'seed force input dimensions')
    code = _one(code, '''        [seed_force_state[8 + i, :] for i in range(4)])''',
                '''        [seed_force_state[8 + i, :] for i in range(4)],
        seed_force_state[12, :], seed_force_state[13, :])''',
                'projected contact velocities')
    code = _one(code, '''                                 seed_omega, seed_load))''',
                '''                                 seed_omega, seed_load,
                                 seed_roll_rate, seed_pitch_rate))''',
                'seed force vector')
    # Replace only the diagnostic-phase solver block; never enable Phase II.
    solve = SOLVE_CONDENSED
    solve = _one(solve, '''        physical_vars = (x, qdelta, throttle, phase_lengths, normal_load,
                         accel_xy)''',
                  '''        physical_vars = (x, qdelta, throttle, phase_lengths)''',
                  'optimizer physical-variable list')
    solve = _one(solve, '''        seed_arrays = (seed_x_values, seed_qdelta, seed_throttle,
                       phase_initial_lengths, seed_load,
                       np.vstack((seed_ax, seed_ay)))''',
                  '''        seed_arrays = (seed_x_values, seed_qdelta, seed_throttle,
                       phase_initial_lengths)''', 'optimizer seed arrays')
    solve = _one(solve, '''        loads = np.asarray(winning_snapshot[normal_load], dtype=float)
        accel = np.asarray(winning_snapshot[accel_xy], dtype=float)''',
                  '''        # Compute eliminated observables directly at the winning NLP primal.
        # They are not optional free variables or fitted measurements.
        algebraic_output = ca.Function('algebraic_checkpoint_map',
            [opti.x, opti.p], [normal_load, accel_xy])
        loads_eval, accel_eval = algebraic_output(
            flatten(winning_snapshot), parameters)
        loads = np.asarray(loads_eval, dtype=float)
        accel = np.asarray(accel_eval, dtype=float)''',
                  'checkpoint derived outputs')
    code = _one(code, SOLVE_CONDENSED, solve,
                'diagnostic phase solver substitution')
    assert 'normal_load = opti.variable(4, n)' not in code
    assert 'accel_xy = opti.variable(2, n)' not in code
    assert 'opti.subject_to(residual - elastic == 0)' not in code
    assert "'phase1_exact_zero_slack_audited':False" in code
    return code


def prepare_working_solver_eliminated(recovered_root: Path, target_root: Path,
                                      legacy_name: str, hessian: str):
    recovered_root = Path(recovered_root).resolve()
    target_root = Path(target_root).resolve()
    if recovered_root == target_root:
        raise ValueError('Historical source is read only')
    original = recovered_root / legacy_name / 'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py'
    raw = original.read_text(encoding='utf-8')
    generated = transform_solver_eliminated(raw, hessian)
    target = target_root / legacy_name / 'f1tenth_planning/autodrive_mintime/optimizer_unity_wheels.py'
    if not target_root.exists():
        shutil.copytree(recovered_root, target_root)
    elif not target.is_file():
        raise RuntimeError('Incomplete eliminated cache')
    elif target.read_text(encoding='utf-8') != generated:
        raise RuntimeError('Unknown edits to eliminated cache; refusing overwrite')
    provenance = prepare_working_wheel_model(recovered_root, target_root, legacy_name)
    if target.read_text(encoding='utf-8') != generated:
        target.write_text(generated, encoding='utf-8')
    return target_root, dict(
        phase1_method='v2.4.3 algebraically eliminated original-residual least squares',
        phase1_hessian=hessian,
        algebraic_load_and_acceleration_eliminated=True,
        removed_algebraic_variables_per_node=6,
        eliminated_dynamic_equalities_satisfied_by_definition=True,
        retained_dynamic_equalities_minimized_not_enforced=True,
        diagnostic_only=True,
        physical_dynamics_changed=False,
        original_solver_sha256=hashlib.sha256(raw.encode()).hexdigest(),
        working_solver_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        source_untouched=original.read_text(encoding='utf-8') == raw,
        working_solver_path=str(target), **provenance)
