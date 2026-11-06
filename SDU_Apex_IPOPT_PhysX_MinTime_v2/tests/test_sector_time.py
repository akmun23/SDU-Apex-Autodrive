import numpy as np
import pytest

from sector_time import periodic_sector_weights, predictor_timing, resample_source_time_initialization


def test_weights_are_periodic_nonnegative_and_preserve_a_uniform_solution():
    for n,k in [(40,1),(285,12),(40,7),(40,40)]:
        weights=periodic_sector_weights(n,k)
        assert weights.shape==(n,k)
        np.testing.assert_allclose(weights.sum(axis=1),1.,atol=1e-14)
        assert (weights>=0).all()
        np.testing.assert_allclose(weights @ np.full(k,.4),.4,atol=1e-14)
    with pytest.raises(ValueError):
        periodic_sector_weights(5,6)


def test_sector_time_is_locally_variable_and_bounded():
    dt,edges=predictor_timing(40,[0.,0.,1.,1.],4)
    assert np.ptp(dt)>.0009
    assert .024-1e-12 <= min(dt) <= max(dt) <= .025+1e-12
    assert edges[-1]==pytest.approx(.98)
    assert np.diff(edges).min()>0
    for bad in [[.1],[-.1,.2,.3,.4],[0.,.1,.2,float('nan')]]:
        with pytest.raises(ValueError):
            predictor_timing(40,bad,4)


def test_variable_time_resampler_preserves_endpoints_and_held_packet_ownership():
    n=100
    t=np.linspace(0,1,n+1)
    states=np.tile(t,(21,1))
    controls=np.vstack([np.arange(n),np.full(n,.1)])
    result=resample_source_time_initialization(states,controls,[0.,0.,1.,1.],4)
    assert result is not None
    x,u,lap=result
    assert lap==pytest.approx(2.45)
    assert u.shape==(2,98)
    np.testing.assert_array_equal(x[:,0],states[:,0])
    np.testing.assert_array_equal(x[:,-1],states[:,-1])
    _,edges=predictor_timing(n,[0.,0.,1.,1.],4)
    expected=np.searchsorted(edges,np.linspace(0,lap,99)[:-1],side='right')-1
    np.testing.assert_array_equal(u[0],expected)
    assert not np.array_equal(u[0],np.arange(98)*100//98)


def test_no_fractional_time_witness_when_rounding_cannot_remove_packet():
    x=np.zeros((21,101));u=np.zeros((2,100))
    assert resample_source_time_initialization(x,u,np.ones(4),4) is None
