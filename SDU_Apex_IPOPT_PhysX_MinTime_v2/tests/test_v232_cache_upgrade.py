"""Regression: v2.3.1 must recognize the byte-exact v2.3.0 solver cache."""
from __future__ import annotations

import hashlib

import pytest

from solver_overlay import (prepare_working_solver, transform_solver,
                            transform_solver_v229, transform_solver_228)
from tests.test_phase1_overlay import mock_historical_source, LEGACY, SOLVER

# Canonical output of the retained v2.3.0 cache transformer on this fixture.
PREVIOUS_HASHES = {
    'limited-memory': '7912739376c8629b041482dc840956565651cb0b27fadc41534d38cd7d5337c9',
    'exact': '1fea8e0dd74f34fda9048d239382cf5e2cf3432428bb64278cd4103ac7356bec',
}


@pytest.mark.parametrize('hessian_mode', ['limited-memory', 'exact'])
def test_v230_solver_cache_accepted_and_upgraded(tmp_path, hessian_mode):
    raw = mock_historical_source()
    old_cache = transform_solver_v229(raw, hessian_mode)
    assert hashlib.sha256(old_cache.encode()).hexdigest() == PREVIOUS_HASHES[hessian_mode]
    assert transform_solver_228(raw, hessian_mode) == old_cache
    assert old_cache != transform_solver(raw, hessian_mode)

    restored = tmp_path / 'historical'
    original = restored / LEGACY / SOLVER
    original.parent.mkdir(parents=True)
    original.write_text(raw)

    working = tmp_path / 'working'
    cache = working / LEGACY / SOLVER
    cache.parent.mkdir(parents=True)
    cache.write_text(old_cache)
    root, meta = prepare_working_solver(restored, working, LEGACY, mode=hessian_mode)
    assert root == working.resolve()
    assert cache.read_text() == transform_solver(raw, hessian_mode)
    assert original.read_text() == raw
    assert meta['source_untouched']
    # A second invocation must be idempotent.
    root2, meta2 = prepare_working_solver(restored, working, LEGACY, mode=hessian_mode)
    assert root2 == root and meta2 == meta


def test_unknown_local_solver_edit_remains_protected(tmp_path):
    raw = mock_historical_source()
    recovered = tmp_path / 'historical'
    original = recovered / LEGACY / SOLVER
    original.parent.mkdir(parents=True)
    original.write_text(raw)
    target = tmp_path / 'working'
    cache = target / LEGACY / SOLVER
    cache.parent.mkdir(parents=True)
    cache.write_text(transform_solver_v229(raw, 'limited-memory') + "\n# user's patch\n")
    with pytest.raises(RuntimeError, match='unknown local edits'):
        prepare_working_solver(recovered, target, LEGACY)
    assert cache.read_text().endswith("# user's patch\n")
    assert original.read_text() == raw
