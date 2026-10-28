"""Independent world-motion and production-frame checks, without solving an OCP."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from path_frame import path_frame_numpy, path_progress_numpy  # noqa: E402
from recover import load_modules  # noqa: E402


def test_generated_ocp_frame_reconstructs_world_motion_across_seam():
    optimizer, *_ = load_modules(
        PACKAGE.parent, PACKAGE, phase1_formulation="source-equalities")
    # Deliberately inconsistent chord/heading/curvature, including the seam.
    track = SimpleNamespace(
        s=np.array([0., 1., 2.2, 3.4]), length=4.8, count=4,
        x=np.array([0., 1., 1.2, -0.2]),
        y=np.array([0., 0.2, 1.1, 0.8]),
        psi=np.array([0.1, 1.5, 2.9, -1.4]),
        kappa=np.zeros(4), left=np.ones(4), right=np.ones(4))
    interp = optimizer._periodic_interpolants(track)
    h = 1.e-7
    for station in [0.2, 0.8, 1.3, 2.7, 3.7, 4.6]:
        for ey in [-0.08, 0.0, 0.12]:
            for epsi in [-0.2, 0.15]:
                u, v, yaw_rate = 4.1, -0.25, 0.7
                _, _, theta = path_frame_numpy(track, station)
                a = float(interp['frame_tangent_metric'](station))
                b = float(interp['frame_normal_drift'](station))
                dh = float(interp['frame_heading_rate'](station))
                a -= dh * ey
                if a < 0.25:  # Same nonsingular domain enforced by the OCP.
                    continue
                vt = u*np.cos(epsi) - v*np.sin(epsi)
                vn = u*np.sin(epsi) + v*np.cos(epsi)
                sdot = vt / a
                eydot = vn - b*sdot
                epsidot = yaw_rate - dh*sdot
                def world(t):
                    cx, cy, heading = path_frame_numpy(track, station + t*sdot)
                    offset = ey + t*eydot
                    return np.array([cx - np.sin(heading)*offset,
                                     cy + np.cos(heading)*offset,
                                     heading + epsi + t*epsidot])
                derivative = (world(h) - world(-h))/(2*h)
                yaw = theta + epsi
                expected = [u*np.cos(yaw) - v*np.sin(yaw),
                            u*np.sin(yaw) + v*np.cos(yaw), yaw_rate]
                np.testing.assert_allclose(derivative, expected, atol=2.e-8)
                assert abs(float(interp['cos_psi'](station))**2
                           + float(interp['sin_psi'](station))**2 - 1) < 1.e-14
    # Export and symbolic coordinates must give the identical pose.
    stations = np.array([0.3, 1.7, 2.8, 4.4])
    offsets = np.array([0.1, -0.07, 0.08, -0.1])
    x, y, *_ = optimizer._path_geometry_at(track, stations, offsets)
    for i, s in enumerate(stations):
        np.testing.assert_allclose([x[i], y[i]], [
            float(interp['x'](s) - offsets[i]*interp['sin_psi'](s)),
            float(interp['y'](s) + offsets[i]*interp['cos_psi'](s))], atol=1.e-14)
    # Heading derivative must come from heading, never the deliberately zero
    # curvature column. This catches the former coordinate substitution.
    assert abs(float(interp['frame_heading_rate'](0.2))) > 1.0


def test_rear_axle_progress_matches_rigid_point_velocity_with_tilt():
    track = SimpleNamespace(s=np.array([0.,1.,2.,3.]), length=4., count=4,
        x=np.array([0.,1.,1.,0.]), y=np.array([0.,0.,1.,1.]),
        psi=np.array([0.,0.,np.pi,-np.pi/2]))
    station = .4
    u,v,r,q,roll,pitch,height_rate = 5.,.4,2.,.3,.06,-.03,.1
    sr,cr,sp,cp = np.sin(roll),np.cos(roll),np.sin(pitch),np.cos(pitch)
    w = (height_rate+sp*u-cp*sr*v)/(cp*cr)
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    body_point_velocity=np.array([u,v,w])+np.cross([0,q,r],[-.15532,0,0])
    expected=(ry@rx@body_point_velocity)[0]
    actual=path_progress_numpy(track,station,0.,0.,u,v,roll=roll,
        pitch=pitch,height_rate=height_rate,yaw_rate=r,pitch_rate=q)
    np.testing.assert_allclose(actual,expected,atol=1.e-12)
