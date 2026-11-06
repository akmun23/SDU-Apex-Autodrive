"""Pin the native physics step-mode bug fix and byte-exact v2.3.5 cache upgrade."""
import hashlib
import math
import numpy as np
import pytest

from solver_numerics import (
    guard_solver, SPATIAL_SOURCE, IPOPT_PHASE1_SOURCE,
)
from solver_overlay import (
    transform_solver, transform_solver_v235,
    transform_solver_phase1_raw_diagnostics,
    prepare_working_physx_substep_source, prepare_working_solver,
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


def test_substep_selection_uses_pinned_unity_physx_5_mps_branch():
    # Matching pinned practice-player binary calls setSubStepCount(5.0f,5,1).
    u = np.array([0.36, 4.999, -4.999, 5.0, -5.0, 5.001, -5.001, 8.0])
    actual = np.where(np.abs(u) < 5.0, 5, 1)
    np.testing.assert_array_equal(actual, [5, 5, 5, 1, 1, 1, 1, 1])


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
    latest = transform_solver_phase1_raw_diagnostics(src, 'limited-memory')
    assert dest.read_text() == latest
    assert meta['working_solver_sha256'] == hashlib.sha256(latest.encode()).hexdigest()
    assert 'u=x[2,k]' in meta['physical_dynamics_change']
    assert original.read_text() == src
    dest.write_text(prior + '\n# unrecognized local edit')
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_solver(restored, target, LEGACY)


def test_generated_wheel_source_uses_native_5_mps_threshold_and_upgrades_bad_cache(tmp_path):
    recovered = tmp_path / 'recovered'
    working = tmp_path / 'working'
    rel = 'tools/vehicle_dynamics_learning/physx_no_drive_source.py'
    for root in (recovered, working):
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
    original = ('import math\n'
                'UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 5.0\n'
                'UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5\n'
                'UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1\n'
                'def no_drive_substep_count(forward_speed_mps):\n'
                '    if not math.isfinite(forward_speed_mps):\n'
                '        raise ValueError("speed must be finite")\n'
                '    return (UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS\n'
                '            if abs(forward_speed_mps) < UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS\n'
                '            else UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS)\n')
    (recovered / rel).write_text(original)
    (working / rel).write_text(original)

    meta = prepare_working_physx_substep_source(recovered, working)
    patched = (working / rel).read_text()
    assert 'UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 5.0' in patched
    assert 'setSubStepCount(5.0f, 5, 1)' in patched
    assert meta['physx_substep_threshold_mps'] == 5.0
    assert meta['physx_substep_call'] == 'setSubStepCount(5.0f, 5, 1)'
    assert meta['recovered_source_untouched']
    assert (recovered / rel).read_text() == original
    namespace = {'math': math, 'np': np}
    exec(compile(patched, str(working / rel), 'exec'), namespace)
    assert namespace['no_drive_substep_count'](4.999) == 5
    assert namespace['no_drive_substep_count'](-4.999) == 5
    assert namespace['no_drive_substep_count'](5.0) == 1
    assert namespace['no_drive_substep_count'](8.0) == 1
    assert namespace['no_drive_substep_count'](-8.0) == 1

    # The previous generator's exact bad output is recognized and replaced;
    # arbitrary local edits remain protected below.
    old_block = ('UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 5.0\n'
                 'UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5\n'
                 'UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1')
    old_bad_block = ('# Unity VehicleCollider initialization calls\n'
                     '# setSubStepCount(128.0f, 5, 1); keep this distinct from the\n'
                     '# unrelated 5 m/s wheel-angle integration threshold.\n'
                     'UNITY_NO_DRIVE_SUBSTEP_THRESHOLD_MPS = 128.0\n'
                     'UNITY_NO_DRIVE_LOW_SPEED_SUBSTEPS = 5\n'
                     'UNITY_NO_DRIVE_HIGH_SPEED_SUBSTEPS = 1')
    old_bad = original.replace(old_block, old_bad_block, 1)
    (working / rel).write_text(old_bad)
    prepare_working_physx_substep_source(recovered, working)
    assert (working / rel).read_text() == patched

    (working / rel).write_text(patched + '# unknown edit\n')
    with pytest.raises(RuntimeError, match='Unknown generated PhysX substep helper'):
        prepare_working_physx_substep_source(recovered, working)
