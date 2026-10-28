"""Evaluate saved IPOPT primals/duals against the rebuilt, unchanged NLP.

No solve, bound relaxation, fitted vehicle constants or candidate promotion.
Stationarity is reported in physical decision coordinates, not mislabeled as
IPOPT's internally scaled dual infeasibility. Equality duals are unrestricted;
inequality duals use CasADi's signed lower/upper-bound convention.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

import casadi as ca
import numpy as np


class NlpAuditComplete(Exception):
    """Intentional no-solve termination after a saved-checkpoint audit."""


def summarize_rows(values, lower, upper, duals, labels, *, active_tol=1.e-5):
    arrays = [np.asarray(v, dtype=float).reshape(-1)
              for v in (values, lower, upper, duals)]
    g, lo, hi, lam = arrays
    if (not len(g) or any(a.size != len(g) for a in arrays)
            or len(labels) != len(g) or not np.isfinite(g).all()
            or not np.isfinite(lam).all() or np.isnan(lo).any()
            or np.isnan(hi).any() or np.any(lo > hi) or active_tol <= 0):
        raise ValueError('invalid constraint values/bounds/duals/labels')
    equality = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    violation = np.maximum(np.maximum(lo-g, g-hi), 0.)
    distance = np.minimum(g-lo, hi-g)
    # A negative lambda belongs to a finite lower bound, positive to upper.
    lower_dual, upper_dual = np.maximum(-lam, 0.), np.maximum(lam, 0.)
    complementarity = np.zeros(len(g))
    lower_mask, upper_mask = np.isfinite(lo) & ~equality, np.isfinite(hi) & ~equality
    complementarity[lower_mask] += lower_dual[lower_mask]*np.abs(g-lo)[lower_mask]
    complementarity[upper_mask] += upper_dual[upper_mask]*np.abs(hi-g)[upper_mask]
    wrong_sign = (~equality & ((~np.isfinite(lo) & (lam < -1.e-8)) |
                              (~np.isfinite(hi) & (lam > 1.e-8))))
    groups = {}
    for i, label in enumerate(labels):
        name = re.sub(r'_(?:k|mid)?\d+(?=_|$)', '', str(label))
        group = groups.setdefault(name, dict(rows=0, equalities=0, active=0,
            max_violation=0., dual_l1=0., max_abs_dual=0.,
            max_complementarity=0., wrong_sign_duals=0))
        group['rows'] += 1
        group['equalities'] += int(equality[i])
        group['active'] += int(not equality[i] and distance[i] <= active_tol)
        group['max_violation'] = max(group['max_violation'], float(violation[i]))
        group['dual_l1'] += float(abs(lam[i]))
        group['max_abs_dual'] = max(group['max_abs_dual'], float(abs(lam[i])))
        group['max_complementarity'] = max(group['max_complementarity'],
                                             float(complementarity[i]))
        group['wrong_sign_duals'] += int(wrong_sign[i])
    return dict(max_constraint_violation=float(np.max(violation)),
                max_complementarity=float(np.max(complementarity)),
                wrong_sign_duals=int(np.count_nonzero(wrong_sign)),
                constraint_groups=groups)


def maybe_audit_checkpoint(opti, labels, *, lap_time, factor,
                           time_density=None, spatial_steps=None):
    """Opt-in CLI hook, invoked before the first time-stage solve only."""
    source = os.environ.get('SDU_IPOPT_AUDIT_CHECKPOINT')
    if not source:
        return
    source = Path(source)
    output = Path(os.environ['SDU_IPOPT_AUDIT_OUTPUT'])
    if output.exists() or output.with_suffix('.npz').exists():
        raise ValueError('refusing to overwrite an NLP audit')
    with np.load(source, allow_pickle=False) as archive:
        x = np.asarray(archive['decision_x'], dtype=float).reshape(-1)
        lam = np.asarray(archive['constraint_duals'], dtype=float).reshape(-1)
        saved_lap = float(archive['predicted_lap_time_s'])
        saved_violation = float(archive['max_constraint_violation'])
        saved_factor = float(archive['trust_factor'])
    if (x.size != opti.nx or lam.size != opti.ng
            or not np.isfinite(x).all() or not np.isfinite(lam).all()
            or saved_factor != factor):
        raise ValueError('audit checkpoint does not match NLP dimensions/trust factor')
    print('IPOPT audit: rebuilding exact objective/constraint derivatives; no solve', flush=True)
    parameter = np.asarray(opti.value(opti.p, opti.initial()))
    lower = np.asarray(opti.value(opti.lbg, opti.initial())).reshape(-1)
    upper = np.asarray(opti.value(opti.ubg, opti.initial())).reshape(-1)
    lam_symbol = ca.MX.sym('audit_lambda', opti.ng)
    grad_f = ca.gradient(opti.f, opti.x)
    grad_l = grad_f + ca.jtimes(opti.g, opti.x, lam_symbol, True)
    extras = [] if time_density is None else [time_density, spatial_steps]
    oracle = ca.Function('checkpoint_kkt', [opti.x, opti.p, lam_symbol],
                         [opti.g, lap_time, opti.f, grad_f, grad_l, *extras])
    evaluated = oracle(x, parameter, lam)
    g, actual_lap, objective, grad_f, grad_l = evaluated[:5]
    g, grad_f, grad_l = (np.asarray(v).reshape(-1) for v in (g, grad_f, grad_l))
    if not all(np.isfinite(v).all() for v in (g, grad_f, grad_l)):
        raise ValueError('nonfinite audit oracle values')
    report = summarize_rows(g, lower, upper, lam, labels)
    # Fail closed: a different center, parameter set or model is not an audit
    # of this saved solve merely because the vector lengths happen to match.
    if (abs(float(actual_lap)-saved_lap) > 1.e-9 or
            abs(report['max_constraint_violation']-saved_violation) > 1.e-7):
        raise ValueError('rebuilt checkpoint time/residual differ from saved NLP')
    report.update(schema=1, checkpoint=str(source.resolve()),
        checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        trust_factor=factor, decisions=opti.nx, constraints=opti.ng,
        predicted_lap_time_s=float(actual_lap), objective=float(objective),
        stationarity_physical_coordinates_inf=float(np.max(np.abs(grad_l))),
        objective_gradient_physical_coordinates_inf=float(np.max(np.abs(grad_f))),
        stationarity_is_ipopt_scaled_dual_infeasibility=False,
        ipopt_invoked=False, unity_validated=False,
        limitations=['Duals belong to the saved iterate, not a KKT certificate.',
                     'Constraint activity alone does not establish a physical time floor.'])
    trust = np.array([name == 'captured_motion_trust_region' for name in labels])
    if np.any(trust):
        report['trust_box'] = dict(rows=int(np.count_nonzero(trust)),
            normalized_radius=float(np.max(np.abs(upper[trust]))),
            largest_normalized_displacement=float(np.max(np.abs(g[trust]))))
    output.parent.mkdir(parents=True, exist_ok=True)
    extra_payload = {} if not extras else dict(
        node_time_density_s_per_m=np.asarray(evaluated[5]).reshape(-1),
        spatial_steps_m=np.asarray(evaluated[6]).reshape(-1))
    np.savez_compressed(output.with_suffix('.npz'), decision_x=x,
        constraint_values=g, lower=lower, upper=upper, duals=lam,
        labels=np.asarray(labels, dtype=str), objective_gradient=grad_f,
        lagrangian_gradient=grad_l, parameters=parameter, **extra_payload)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'IPOPT audit complete: {output}', flush=True)
    raise NlpAuditComplete(str(output))
