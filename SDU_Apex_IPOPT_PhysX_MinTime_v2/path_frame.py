"""Exact coordinates of the piecewise linear path used by production MPC.

For P=C(s)+ey*N(theta(s)), the reference chord need not be tangent to
theta(s), and theta'(s) need not equal an independently stored curvature.
These are coordinate derivatives, not a change to the vehicle dynamics.
"""
from __future__ import annotations

import ast
import textwrap
import numpy as np

SOURCE_REAR_AXLE_FROM_COM_M = 0.15532


def _grid_and_heading(track):
    grid = np.r_[np.asarray(track.s, dtype=float), float(track.length)]
    psi = np.unwrap(np.asarray(track.psi, dtype=float))
    if (psi.shape != (track.count,) or grid.size != track.count + 1
            or not np.isfinite(grid).all() or not np.isfinite(psi).all()
            or grid[0] != 0.0 or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("invalid periodic path frame")
    closing = psi[-1] + np.arctan2(
        np.sin(psi[0] - psi[-1]), np.cos(psi[0] - psi[-1]))
    return grid, np.r_[psi, closing]


def add_symbolic_path_frame(track, interpolants):
    """Use the same linear XY and unwrapped heading interpolation as MPC."""
    import casadi as ca

    grid, headings = _grid_and_heading(track)
    suffix = str(id(track))
    station = ca.MX.sym("frame_station")
    heading = ca.interpolant(
        f"frame_heading_{suffix}", "linear", [grid], headings)(station)
    c, s = ca.cos(heading), ca.sin(heading)
    dx = ca.jacobian(interpolants["x"](station), station)
    dy = ca.jacobian(interpolants["y"](station), station)
    expressions = {
        "cos_psi": c, "sin_psi": s,
        "frame_tangent_metric": c * dx + s * dy,
        "frame_normal_drift": -s * dx + c * dy,
        "frame_heading_rate": ca.jacobian(heading, station),
    }
    interpolants.update({
        key: ca.Function(f"frame_{key}_{suffix}", [station], [value])
        for key, value in expressions.items()
    })
    return interpolants


def path_frame_numpy(track, station):
    grid, headings = _grid_and_heading(track)
    query = np.mod(np.asarray(station, dtype=float), track.length)
    psi = np.interp(query, grid, headings)
    xy = [np.interp(query, grid, np.r_[values, values[0]])
          for values in (track.x, track.y)]
    return xy[0], xy[1], psi


def path_progress_numpy(track, station, ey, epsi, u, v, *,
                        roll=0.0, pitch=0.0, height_rate=0.0,
                        yaw_rate=0.0, pitch_rate=0.0):
    """Numeric counterpart of the OCP's exact horizontal progress equation."""
    grid, headings = _grid_and_heading(track)
    query = np.mod(np.asarray(station, dtype=float), track.length)
    index = np.searchsorted(grid, query, side='right') - 1
    ds = np.diff(grid)[index]
    _, _, heading = path_frame_numpy(track, station)
    dx = np.diff(np.r_[track.x, track.x[0]])[index] / ds
    dy = np.diff(np.r_[track.y, track.y[0]])[index] / ds
    dh = np.diff(headings)[index] / ds
    metric = np.cos(heading)*dx + np.sin(heading)*dy - dh*ey
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    w = (height_rate + sp*u - cp*sr*v) / (cp*cr)
    # World position is at the rear axle (Unity GPS VehicleTransform), while
    # the force model carries COM velocity. omega cross (-L,0,0) transports it.
    v = v - yaw_rate*SOURCE_REAR_AXLE_FROM_COM_M
    w = w + pitch_rate*SOURCE_REAR_AXLE_FROM_COM_M
    horizontal_u = cp*u + sp*(sr*v + cr*w)
    horizontal_v = cr*v - sr*w
    return (horizontal_u*np.cos(epsi) - horizontal_v*np.sin(epsi)) / metric


def install_exact_path_frame(source: str) -> str:
    """Correct every node, midpoint and export in the generated spatial OCP."""
    substitutions = [
        ("import math\n", "import math\nfrom path_frame import "
         "add_symbolic_path_frame, path_frame_numpy\n"),
        ('''    return {
        key: ca.interpolant(
            f"free_path_{key}_{suffix}", "linear", [grid],
            np.r_[np.asarray(values, dtype=np.float64), float(values[0])])
        for key, values in periodic.items()
    }''', '''    interpolants = {
        key: ca.interpolant(
            f"free_path_{key}_{suffix}", "linear", [grid],
            np.r_[np.asarray(values, dtype=np.float64), float(values[0])])
        for key, values in periodic.items()
    }
    return add_symbolic_path_frame(track, interpolants)'''),
        ('''    px = periodic(track.x) - periodic(np.sin(track.psi)) * ey
    py = periodic(track.y) + periodic(np.cos(track.psi)) * ey''',
         '''    cx, cy, heading = path_frame_numpy(track, station)
    px = cx - np.sin(heading) * ey
    py = cy + np.cos(heading) * ey'''),
        ('        curvature = interp["kappa"](station)',
         '        curvature = interp["frame_heading_rate"](station)'),
        ('        den = 1.0 - curvature * ey[k]',
         '        den = interp["frame_tangent_metric"](station) - curvature * ey[k]'),
        ('''        horizontal_u, horizontal_v = _unity_horizontal_velocity_casadi(
            u[k], v[k], body_vertical, roll[k], pitch[k])''',
         '''        horizontal_u, horizontal_v = _unity_horizontal_velocity_casadi(
            u[k], v[k] - 0.15532 * yaw_rate[k],
            body_vertical + 0.15532 * pitch_rate[k], roll[k], pitch[k])'''),
        ('''        midpoint_horizontal_u, midpoint_horizontal_v = (
            _unity_horizontal_velocity_casadi(
                midpoint_u, midpoint_v, midpoint_body_vertical,
                midpoint_x[12], midpoint_x[14]))''',
         '''        midpoint_horizontal_u, midpoint_horizontal_v = (
            _unity_horizontal_velocity_casadi(
                midpoint_u, midpoint_v - 0.15532 * midpoint_yaw_rate,
                midpoint_body_vertical + 0.15532 * midpoint_x[15],
                midpoint_x[12], midpoint_x[14]))'''),
        ('''        midpoint_curvature = interp["kappa"](midpoint_station)
        midpoint_denominator = (
            1.0 - midpoint_curvature * midpoint_x[0])''',
         '''        midpoint_curvature = interp["frame_heading_rate"](midpoint_station)
        midpoint_denominator = (
            interp["frame_tangent_metric"](midpoint_station)
            - midpoint_curvature * midpoint_x[0])'''),
        ('''            midpoint_curvature = interp["kappa"](midpoint_station)
            midpoint_den = 1.0 - midpoint_curvature * midpoint_ey''',
         '''            midpoint_curvature = interp["frame_heading_rate"](midpoint_station)
            midpoint_den = (interp["frame_tangent_metric"](midpoint_station)
                            - midpoint_curvature * midpoint_ey)'''),
        ('''            midpoint_path_angle = ca.atan2(
                midpoint_ey_rate, midpoint_den)''',
         '''            midpoint_path_angle = ca.atan2(
                midpoint_ey_rate + interp["frame_normal_drift"](midpoint_station),
                midpoint_den)'''),
    ]
    for old, new in substitutions:
        # The preceding state-elimination pass emits canonical Python through
        # ast.unparse. Match statements in that representation explicitly.
        indentation = old[:len(old) - len(old.lstrip())]
        old = textwrap.indent(ast.unparse(ast.parse(textwrap.dedent(old))), indentation)
        new = textwrap.indent(ast.unparse(ast.parse(textwrap.dedent(new))), indentation)
        if source.count(old) != 1:
            raise RuntimeError("Cannot uniquely correct IPOPT path frame: " + old)
        source = source.replace(old, new, 1)
    old = 'path_lateral_speed / sdot,'
    if source.count(old) != 1:
        raise RuntimeError("Cannot uniquely correct lateral path-coordinate dynamics")
    source = source.replace(
        old, "path_lateral_speed / sdot - interp['frame_normal_drift'](station),", 1)
    # Positive drive has one spatial phase whose length is exactly the known
    # reference length. Keeping it as a decision lets infeasible NLP trials
    # move every evaluation across interpolation knots and switch theta'.
    # Make the equality an exact substitution, preserving all XY decisions.
    phase_constraints = '''    for phase, count in enumerate(node_counts):
        add_constraint(phase_lengths[phase] >= 0.005 * int(count), f'phase_length_min_{phase}')
        add_constraint(phase_lengths[phase] <= 0.25 * int(count), f'phase_length_max_{phase}')
    add_constraint(ca.sum1(phase_lengths) == track.length, 'periodic_phase_lengths_sum')'''
    if source.count(phase_constraints) != 1:
        raise RuntimeError('Cannot eliminate the fixed one-phase length constraints')
    source = source.replace(phase_constraints, '', 1)
    declaration = '    phase_lengths = opti.variable(phase_count)'
    if source.count(declaration) != 1:
        raise RuntimeError('Cannot eliminate the fixed one-phase length decision')
    # Snapshot/continuation helpers still carry the known phase length in
    # reports. They must not try to set a decision initializer on a parameter.
    source = source.replace('opti.set_initial(', 'set_initial(')
    source = source.replace(declaration, '''    phase_lengths = opti.parameter(phase_count)
    if phase_count != 1 or n != int(node_counts[0]):
        raise ValueError('fixed mesh requires the positive-drive single phase')
    phase_initial_lengths = np.asarray([track.length], dtype=np.float64)
    opti.set_value(phase_lengths, phase_initial_lengths)

    def set_initial(variable, values):
        if variable is not phase_lengths:
            opti.set_initial(variable, values)''', 1)
    # The flat primal cache must contain decision variables only. Keep the
    # fixed phase parameter in report snapshots without assigning it a slice.
    source = source.replace(
        'physical_vars = (x, qdelta, throttle, phase_lengths, *elastic_vars)',
        'physical_vars = (x, qdelta, throttle, *elastic_vars)', 1)
    snapshot = "return {var: flat[lo:hi].reshape((rows, cols), order='F').copy() for var, (lo, hi, rows, cols) in var_slices.items()}"
    if source.count(snapshot) != 1:
        raise RuntimeError('Cannot preserve fixed mesh in Phase-I snapshots')
    source = source.replace(snapshot,
        "return {phase_lengths: phase_initial_lengths.reshape(-1, 1).copy(), "
        "**{var: flat[lo:hi].reshape((rows, cols), order='F').copy() "
        "for var, (lo, hi, rows, cols) in var_slices.items()}}", 1)
    # Guarded trial dictionaries used by projected checkpoint search also
    # need the known parameter for report extraction after a failed solve.
    source = source.replace(
        'proposed = {var: reference[var] + alpha * (trial[var] - reference[var]) for var in physical_vars}',
        'proposed = {phase_lengths: phase_initial_lengths.reshape(-1, 1).copy(), '
        '**{var: reference[var] + alpha * (trial[var] - reference[var]) for var in physical_vars}}')
    # The condensed force/load channels are expressions of x/input, not
    # decisions. Bounding them again in a numerical trust box duplicates
    # nonlinear constraints and their expensive second derivatives. Physical
    # force/load equations and safety bounds remain unchanged. The trust box
    # now measures independent dynamic coordinates and controls only.
    trust = '''        time_trust_region_deviation = ca.vertcat(
            ca.vec((x - ca.DM(time_reference_x)) / state_scales),
            (qdelta - ca.DM(time_reference_qdelta)) / 3.2,
            throttle - ca.DM(time_reference_throttle),
            (phase_lengths - ca.DM(time_reference_phase_lengths)) / ca.DM(np.maximum(time_reference_phase_lengths, 1.0)),
            ca.vec((normal_load - ca.DM(time_reference_load)) / 10.0),
            ca.vec((accel_xy - ca.DM(time_reference_accel_xy)) / 10.0))'''
    reduced_trust = '''        time_trust_region_deviation = ca.vertcat(
            ca.vec((x - ca.DM(time_reference_x)) / state_scales),
            (qdelta - ca.DM(time_reference_qdelta)) / 3.2,
            throttle - ca.DM(time_reference_throttle))'''
    # Only the source-equality formulation has this component-wise box.
    if 'time_trust_region_deviation' in source:
        if source.count(trust) != 1:
            raise RuntimeError('Cannot remove derived coordinates from time-stage trust box')
        source = source.replace(trust, reduced_trust, 1)
        for name in ('normal_load', 'accel_xy', 'phase_lengths'):
            assignment = ('        time_reference_' +
                          {'normal_load':'load', 'accel_xy':'accel_xy',
                           'phase_lengths':'phase_lengths'}[name] +
                          ' = np.asarray(stage_values(stage1_solution, stage1_stats, ' +
                          name + '), dtype=np.float64)\n')
            if source.count(assignment) != 1:
                raise RuntimeError('Cannot remove unused derived trust reference: ' + name)
            source = source.replace(assignment, '', 1)
    ast.parse(source)
    return source
