"""Pin the native physics step-mode bug fix and byte-exact v2.3.5 cache upgrade."""
import hashlib
import numpy as np
import pytest

from solver_numerics import (
    guard_solver, SPATIAL_SOURCE, IPOPT_PHASE1_SOURCE,
)
from solver_overlay import (
    transform_solver, transform_solver_v235, prepare_working_solver,
)
from tests.test_phase1_overlay import _sample, LEGACY, SOLVER


def full_historical_sites():
    return ('    spatial_rhs = []\n' + SPATIAL_SOURCE + '\n' +
            '    def source_wheel_update(omega, *args):\n'
            '        return omega\n'
            '        interval_start_forward_speed = x[0, k]\n' +
            IPOPT_PHASE1_SOURCE + '\n')


def test_forward_speed_source_index_corrected_only_in_new_solver():
    original = full_historical_sites()
    latest = guard_solver(original)
    prior = guard_solver(original, fix_wheel_substep_speed=False)
    assert 'interval_start_forward_speed = x[2, k]' in latest
    assert 'interval_start_forward_speed = x[0, k]' not in latest
    assert 'interval_start_forward_speed = x[0, k]' in prior
    assert 'interval_start_forward_speed = x[2, k]' not in prior


def test_substep_selection_semantics_against_native_branch():
    # Historically ey=0 -> 5 wheel substeps regardless of car speed.
    # Corrected: u>=5 m/s -> 1 step, matching actual PhysX vehicle.
    ey = np.zeros(5)
    u = np.array([0.36, 2.4, 4.99, 5.0, 8.0])
    old = np.where(np.abs(ey) < 5, 5, 1)
    new = np.where(np.abs(u) < 5, 5, 1)
    np.testing.assert_array_equal(old, [5, 5, 5, 5, 5])
    np.testing.assert_array_equal(new, [5, 5, 5, 1, 1])


def test_mismatched_wheel_index_fails_closed():
    text = full_historical_sites().replace('x[0, k]', 'x[1, k]')
    with pytest.raises(RuntimeError, match='PhysX wheel-substep actual-forward-speed'):
        guard_solver(text)


def test_v235_cache_is_automatically_upgraded_without_editing_history(tmp_path):
    restored, original, src = _sample(tmp_path)
    target = tmp_path / 'working'
    dest = target / LEGACY / SOLVER
    dest.parent.mkdir(parents=True)
    prior = transform_solver_v235(src, 'limited-memory')
    dest.write_text(prior)
    _, meta = prepare_working_solver(restored, target, LEGACY)
    latest = transform_solver(src, 'limited-memory')
    assert dest.read_text() == latest
    assert meta['working_solver_sha256'] == hashlib.sha256(latest.encode()).hexdigest()
    assert 'u=x[2,k]' in meta['physical_dynamics_change']
    assert original.read_text() == src
    dest.write_text(prior + '\n# unrecognized local edit')
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_solver(restored, target, LEGACY)


def test_v236_diagnostic_uses_algebraic_projection_not_unprojected_seed():
    from pathlib import Path
    code = (Path(__file__).resolve().parents[1] /
            'diagnose_phase1_v236.sh').read_text()
    assert '--seed-file ./optimized_practice_v2_3_0/' in code
    assert '--no-seed-force-projection' not in code
    assert '--brake-zones 0' in code
    assert '--skip-native' in code
    assert '--feasibility-iterations 50' in code
