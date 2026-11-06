import numpy as np
import pytest
from causal_seed import control_profile,project_periodic_guess,propagate_source


def test_steering_profiles_are_cyclic_and_reject_large_changes():
    for amp, harmonic in ((.01,1),(-.02,3)):
        pattern=control_profile(80,amp,harmonic)
        assert len(pattern)==80
        assert np.max(np.abs(pattern))<=abs(amp)+1e-12
        assert abs(np.sum(pattern))<1e-10
    with pytest.raises(ValueError):
        control_profile(80,.12,2)


def test_exact_source_propagation_and_periodic_projection_are_distinct():
    initial=np.zeros(21);u=np.zeros((2,40))
    def exact(z):
        r=z[:21].copy()
        r[0]+=z[21]+.01
        r[2]+=.1
        r[3]+=z[22]+.02
        return r
    raw=propagate_source(exact,initial,u)
    assert raw[0,-1]==pytest.approx(.4)
    assert raw[3,-1]==pytest.approx(.8)
    projected,difference=project_periodic_guess(raw,1)
    assert difference[0]==pytest.approx(.4)
    assert projected[0,-1]==pytest.approx(projected[0,0])
    assert projected[3,-1]==pytest.approx(projected[3,0])
    assert projected[2,-1]-projected[2,0]==pytest.approx(2*np.pi)
    # Projections are *not* exact source dynamics and never imply certification.
    assert projected[0,1]-exact(np.r_[projected[:,0],u[:,0]])[0]!=pytest.approx(0.,abs=1e-10)
