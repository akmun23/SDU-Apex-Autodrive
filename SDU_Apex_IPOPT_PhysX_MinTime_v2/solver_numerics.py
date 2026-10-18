"""Numerical safety overlay for the unchanged *feasible* physical OCP.

These substitutions apply to the historical full solver after the elastic
constraint overlay. The spatial guards are identical to the source equations
on the physical admissible domain den>=0.25 and sdot>=0.30. They merely keep
infeasible IPOPT trial evaluations finite instead of dividing by zero.
"""
from __future__ import annotations

SPATIAL_SOURCE = '''        den = 1.0 - curvature * ey[k]
        sdot = (u[k] * ca.cos(epsi[k]) - v[k] * ca.sin(epsi[k])) / den
        add_constraint(den >= 0.25, "frenet_denominator")
        add_constraint(sdot >= 0.30, "path_speed")'''

SPATIAL_GUARDED = '''        den = 1.0 - curvature * ey[k]
        # Numeric guard *only outside* the feasible den >= 0.25 region.
        # Keeps an off-domain line-search trial from dividing by zero.
        sdot_unclipped = ((u[k] * ca.cos(epsi[k]) - v[k] * ca.sin(epsi[k]))
                          / ca.fmax(den, 0.20))
        add_constraint(den >= 0.25, "frenet_denominator")
        add_constraint(sdot_unclipped >= 0.30, "path_speed")
        # Every valid lap has sdot >= 0.30, so this guard has NO effect on
        # feasible states/dynamics or the minimum-time objective.
        sdot = ca.fmax(sdot_unclipped, 0.20)'''

# Use exact quote matching. The historical source is pinned by commit.
IPOPT_PHASE1_SOURCE = '''    opti.solver("ipopt", {
        "expand": True,
        "ipopt": {'''
IPOPT_PHASE1_BOUNDED = '''    opti.solver("ipopt", {
        "expand": True,
        # Extract simple physical/slack inequalities as true variable bounds.
        # Previously IPOPT reported ZERO bounded variables despite thousands
        # of steering, wheel, throttle and elastic bounds in opti.g.
        "detect_simple_bounds": True,
        "ipopt": {'''


def _unique_replace(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f'Expected exactly one {label} site in recovered solver; found {count}')
    return source.replace(old, new, 1)


def guard_solver(source: str, *, fix_wheel_substep_speed: bool = True) -> str:
    """Apply numeric changes to the *real* historical solver only.

    Unit-test mock historical files do not include source spatial dynamics;
    retaining their behavior lets the original overlay regression suite work.
    Any real spatial solver must match the pinned source exactly, fail closed.
    """
    if '    spatial_rhs = []' not in source:
        return source
    source = _unique_replace(source, SPATIAL_SOURCE, SPATIAL_GUARDED,
                             'Frenet positive-speed division')
    source = _unique_replace(source, IPOPT_PHASE1_SOURCE, IPOPT_PHASE1_BOUNDED,
                             'phase-I bound detection')
    if fix_wheel_substep_speed and '    def source_wheel_update(' in source:
        # x = (ey, epsi, u, v, r, wheel_omega[4], steering, body...).
        # Selecting PhysX wheel substep mode from x[0] uses lateral offset,
        # whereas native PhysX switches on actual longitudinal speed x[2].
        source = _unique_replace(
            source, '        interval_start_forward_speed = x[0, k]',
            '        interval_start_forward_speed = x[2, k]',
            'PhysX wheel-substep actual-forward-speed state')
    return source
