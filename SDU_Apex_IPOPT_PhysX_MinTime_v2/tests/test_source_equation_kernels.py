"""Value/Jacobian/Hessian parity for the compact source-force kernel."""
from pathlib import Path
import ast
import sys

import casadi as ca
import numpy as np
import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from recover import load_modules
from source_vehicle import current_model, bind_current_wheel_moment_arms
from source_equation_kernels import (
    make_force_kernel, NodeValueCache, scale_solver_coordinates)


@pytest.fixture(scope='module')
def equations():
    old, wheel, _, _, _ = load_modules(
        PACKAGE.parent, PACKAGE, phase1_hessian='exact',
        phase1_formulation='source-equalities')
    model, _ = current_model(PACKAGE.parent, wheel)
    bind_current_wheel_moment_arms(PACKAGE.parent, old, wheel)
    return model, wheel.tire_forces_casadi


def flatten(output):
    return ca.vertcat(*output[:6], *[ca.vertcat(*v) for v in output[6:]])


def call(function, model, z):
    return function(model, *[z[i, :] for i in range(4)],
                    [z[i, :] for i in range(4, 8)],
                    [z[i, :] for i in range(8, 12)],
                    *[z[i, :] for i in range(12, 18)])


@pytest.mark.parametrize('jit', [False, True])
def test_force_values_and_exact_derivatives_match(equations, jit):
    model, source = equations
    compact = make_force_kernel(model, source, jit=jit)
    z = ca.MX.sym('z', 18)
    weights = ca.DM(np.linspace(.01, 1, 22))
    expressions = [flatten(call(fn, model, z)) for fn in (source, compact)]
    functions = [ca.Function(f'oracle_{i}', [z], [
        expression, ca.jacobian(expression, z),
        ca.hessian(ca.dot(weights, expression), z)[0]])
        for i, expression in enumerate(expressions)]
    rng = np.random.default_rng(1251)
    for speed in (0.0, .001, .1, 1., 4.9999, 5., 5.0001, 8., 12.):
        for steering in (-.5, -.2, 0., .2, .5):
            state = np.array([
                speed, rng.uniform(-.5, .5), rng.uniform(-3., 3.), steering,
                *np.maximum(0., speed/model.wheel_radius_m + rng.normal(0., 5., 4)),
                *model.static_normal_load_n,
                model.suspension_reference_body_height_m, .002,
                .01, .02, -.005, .01])
            expected, measured = (fn(state) for fn in functions)
            for left, right in zip(expected, measured):
                np.testing.assert_allclose(left, right, rtol=2.e-11, atol=2.e-8)


def test_batched_force_outputs_and_model_binding(equations):
    model, source = equations
    compact = make_force_kernel(model, source, jit=False)
    columns = ca.MX.sym('batch', 18, 3)
    measured = flatten(call(compact, model, columns))
    expected = ca.horzcat(*[flatten(call(source, model, columns[:, i]))
                            for i in range(3)])
    compare = ca.Function('compare_batch', [columns], [measured, expected])
    data = np.tile(np.array([5., .1, -.5, .2, *([85.]*4),
        *model.static_normal_load_n, model.suspension_reference_body_height_m,
        .002, .01, .02, -.005, .01])[:, None], (1, 3))
    a, b = compare(data)
    np.testing.assert_allclose(a, b, rtol=2.e-12, atol=2.e-10)
    with pytest.raises(ValueError, match='different model'):
        call(compact, object(), columns[:, 0])


def test_export_cache_preserves_rows_columns_and_reuses_evaluation():
    calls = []
    def evaluate(matrix):
        calls.append(matrix.shape)
        return matrix
    cache = NodeValueCache(evaluate)
    scalars = [ca.DM(1), ca.DM(2), ca.DM(3)]
    tuples = [(ca.DM(1), ca.DM(4)), (ca.DM(2), ca.DM(5)), (ca.DM(3), ca.DM(6))]
    vectors = [ca.vertcat(*values) for values in tuples]
    np.testing.assert_array_equal(cache(scalars), [[1, 2, 3]])
    np.testing.assert_array_equal(cache(tuples), [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(np.column_stack(cache(vectors).T), [[1, 2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(cache(tuples), cache(vectors))
    assert len(calls) == 3


def test_solver_coordinate_scaling_preserves_physical_values(monkeypatch):
    opti = ca.Opti()
    states, rate = opti.variable(3, 2), opti.variable(2)
    opti.minimize(ca.sumsqr(states)+ca.sumsqr(rate))
    opti.subject_to(states[0, :]+states[1, :]*states[2, :] >= 1.)
    values = np.array([[.01, .02], [7., 8.], [100., 120.]])
    opti.set_initial(states, values)
    before = opti.value(opti.g, opti.initial())
    monkeypatch.setenv('SDU_IPOPT_VARIABLE_SCALING', '1')
    scale_solver_coordinates(opti, states, ca.repmat(ca.DM([.02, 2., 150.]), 1, 2), rate)
    np.testing.assert_array_equal(opti.value(states, opti.initial()), values)
    np.testing.assert_array_equal(opti.value(opti.g, opti.initial()), before)


def test_generated_solver_does_not_reregister_callback_or_export_scalar_loops():
    old, _, _, _, _ = load_modules(
        PACKAGE.parent, PACKAGE, phase1_hessian='exact',
        phase1_formulation='source-equalities')
    tree = ast.parse(Path(old.__file__).read_text())
    solver = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'solve_bottom_up')
    loops = [node for node in ast.walk(solver) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == 'trust_factor']
    assert len(loops) == 1
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr == 'callback' for node in ast.walk(loops[0]))
    for node in ast.walk(solver):
        if isinstance(node, ast.ListComp):
            assert not any(isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                           and call.func.id == 'value' for call in ast.walk(node))
