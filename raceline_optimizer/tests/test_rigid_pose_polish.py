import math
import numpy as np
import pytest

from rigid_pose_polish import transform_pose,rigid_correction


def test_se2_transformation_preserves_body_variables_and_closed_winding():
    n=40
    a=np.zeros((21,n+1))
    t=np.linspace(0,2*np.pi,n+1)
    a[0]=2*np.cos(t);a[1]=2*np.sin(t);a[2]=t;a[3]=4.
    transformed=transform_pose(a,[.01,-.02],.003,np.mean(a[:2],axis=1))
    np.testing.assert_array_equal(transformed[3:],a[3:])
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(transformed[:2],axis=1),axis=0),
        np.linalg.norm(np.diff(a[:2],axis=1),axis=0),atol=1e-12)
    assert transformed[2,-1]-transformed[2,0]==pytest.approx(2*np.pi)
    assert np.max(np.abs(transformed[:2]-a[:2]))>0.


def test_rigid_recovery_lp_requires_actual_positive_margin():
    # Synthetic distance field (not a source-derived physical certificate).
    from map_track import _FOOTPRINT_LOCAL, _FOOTPRINT_UNSAMPLED_CORRECTION
    class Plane:
        def distance(self,pts):
            return np.asarray(pts)[...,0]+.08+_FOOTPRINT_UNSAMPLED_CORRECTION+.05-1e-10
    trace=np.zeros((21,100))
    result=rigid_correction(Plane(),trace,margin=.05)
    assert result is not None
    assert result['initial_slack_m']<0.
    assert result['shifted_slack_m']>0.
    assert result['numerical_only'] is False or result['shifted_slack_m']<1e-6
    class NegativeEverywhere:
        def distance(self,points):
            return np.zeros(np.asarray(points).shape[:-1])
    assert rigid_correction(NegativeEverywhere(),trace) is None
