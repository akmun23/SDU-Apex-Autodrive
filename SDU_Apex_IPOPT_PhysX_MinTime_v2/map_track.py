"""PGM/ROS YAML map processing, closed corridor skeleton and footprint clearance.

No surrogate physical friction/vehicle parameters are used here.
"""
from __future__ import annotations
import math
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import yaml
from scipy import ndimage
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize


# Runtime trajectory / Unity GPS origin is the rear axle, 0.16 m behind
# the Unity root. Root bounds [-0.240, +0.270] therefore become these bounds.
# Match MPC_REAR_OVERHANG_M and MPC_REAR_AXLE_TO_FRONT_BUMPER_M.
# Precomputed once; do not allocate sampling grids in the control loop.
_EDGE_ALONG = np.array([-.080, .430])
_EDGE_SIDE = np.array([-.273/2, .273/2])
_FOOTPRINT_LOCAL = np.asarray(
    [(a,b) for a in _EDGE_ALONG for b in np.linspace(*_EDGE_SIDE,7)] +
    [(a,b) for b in _EDGE_SIDE for a in np.linspace(*_EDGE_ALONG,15)],
    dtype=float)
_FOOTPRINT_UNSAMPLED_CORRECTION = .5*max(
    (_EDGE_ALONG[1]-_EDGE_ALONG[0])/14,
    (_EDGE_SIDE[1]-_EDGE_SIDE[0])/6)
_FOOTPRINT_RADIUS = float(np.max(np.linalg.norm(_FOOTPRINT_LOCAL,axis=1)))
DEFAULT_WALL_MARGIN_M = 0.05
_CLEARANCE_POLICY_PATH = Path(__file__).resolve().parents[1]/'config/racing/raceline_clearance_policy.json'
_CLEARANCE_POLICY = json.loads(_CLEARANCE_POLICY_PATH.read_text())
CLEARANCE_NUMERICAL_TOLERANCE_M = float(_CLEARANCE_POLICY['numerical_tolerance_m'])
if (_CLEARANCE_POLICY.get('schema') != 'sdu-apex-raceline-clearance/v1' or
        not math.isfinite(CLEARANCE_NUMERICAL_TOLERANCE_M) or
        not 0. <= CLEARANCE_NUMERICAL_TOLERANCE_M <= 1.e-4):
    raise ValueError('invalid raceline numerical-clearance policy')


def clearance_passes(slack_m, target_m=0., *, tolerance_m=CLEARANCE_NUMERICAL_TOLERANCE_M):
    """Keep raw measurements; tolerate only the explicitly declared roundoff band."""
    return (math.isfinite(slack_m) and math.isfinite(target_m) and
            math.isfinite(tolerance_m) and 0. <= tolerance_m <= CLEARANCE_NUMERICAL_TOLERANCE_M
            and slack_m >= target_m-tolerance_m)


def declared_clearance_tolerance(report):
    value = report.get('clearance_numerical_tolerance_m',
        (report.get('rollout') or {}).get('clearance_numerical_tolerance_m', 0.))
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and 0. <= value <= CLEARANCE_NUMERICAL_TOLERANCE_M else None

@dataclass
class MapGrid:
    image: np.ndarray
    free: np.ndarray
    resolution: float
    origin: np.ndarray
    yaw: float
    clearance: np.ndarray

    @classmethod
    def load(cls, pgm: Path, yaml_path: Path | None = None,
             resolution: float | None = None, origin: tuple[float, float] | None = None):
        pgm = Path(pgm)
        if yaml_path is None and pgm.with_suffix('.yaml').exists():
            yaml_path = pgm.with_suffix('.yaml')
        cfg = yaml.safe_load(Path(yaml_path).read_text()) if yaml_path else {}
        resolution = resolution if resolution is not None else cfg.get('resolution')
        if resolution is None or float(resolution) <= 0:
            raise ValueError('PGM has no metric scale: provide adjacent map YAML or --resolution')
        pose = cfg.get('origin', [0., 0., 0.])
        if origin is not None:
            pose = [*origin, pose[2]]
        image = cv2.imread(str(pgm), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 2:
            raise ValueError(f'Invalid grayscale map image: {pgm}')
        if image.dtype != np.uint8:
            image = (image / np.iinfo(image.dtype).max * 255).astype(np.uint8)
        occ = image.astype(float) / 255.0
        if not bool(cfg.get('negate', 0)):
            occ = 1.0 - occ
        # Unknown cells are unsafe; require unequivocally free occupancy probability.
        free = occ < float(cfg.get('free_thresh', 0.25))
        # Include map boundary as occupied so distance values cannot cross it.
        free[[0, -1], :] = False
        free[:, [0, -1]] = False
        clearance = ndimage.distance_transform_edt(free) * float(resolution)
        return cls(image, free, float(resolution), np.array(pose[:2], dtype=float),
                   float(pose[2]), clearance)

    def pixel_to_world(self, col, row):
        x = (np.asarray(col) + 0.5) * self.resolution
        y = (self.image.shape[0] - np.asarray(row) - 0.5) * self.resolution
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.stack([self.origin[0] + c*x - s*y,
                         self.origin[1] + s*x + c*y], axis=-1)

    def world_to_pixel(self, xy):
        xy = np.asarray(xy, dtype=float)
        d = xy - self.origin
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        x, y = c*d[..., 0] + s*d[..., 1], -s*d[..., 0] + c*d[..., 1]
        return np.stack([self.image.shape[0] - y/self.resolution - 0.5,
                         x/self.resolution - 0.5], axis=-1)  # row, col

    def distance(self, xy):
        rc = self.world_to_pixel(xy)
        rc=np.asarray(rc)
        if rc.ndim==1:
            vals=ndimage.map_coordinates(self.clearance,rc.reshape(2,1),
                order=1,mode='constant',cval=0.0)
            return float(vals[0])
        vals=ndimage.map_coordinates(self.clearance, np.stack([rc[...,0],rc[...,1]]),
            order=1,mode='constant',cval=0.0)
        return vals

    def footprint_clearance(self, x, y, psi, *, margin=DEFAULT_WALL_MARGIN_M):
        # Same rear-axle origin as Unity GPS, production MPC and exported path.
        # Include densely sampled edges: do not rely on the four corners alone.
        pts=_FOOTPRINT_LOCAL
        c, s = math.cos(psi), math.sin(psi)
        world = np.column_stack([x + c*pts[:,0] - s*pts[:,1],
                                 y + s*pts[:,0] + c*pts[:,1]])
        # 1-Lipschitz clearance: half the largest probe spacing is a conservative
        # bound on unexamined points between perimeter probes.
        return float(np.min(self.distance(world)) -
                     _FOOTPRINT_UNSAMPLED_CORRECTION - margin)

    def side_clearances(self, xy, heading, *, maximum_distance=3.0):
        """Raw map distances along path normals, not car-adjusted bounds."""
        distances = np.arange(0., maximum_distance+.01, max(.0125, .5*self.resolution))
        normal = np.column_stack((-np.sin(heading), np.cos(heading)))
        widths = []
        for sign in (1, -1):
            probes = xy[:, None, :] + sign*distances[None, :, None]*normal[:, None, :]
            occupied = self.distance(probes) < .55*self.resolution
            first = np.argmax(occupied, axis=1)
            widths.append(np.where(occupied.any(axis=1), distances[first], distances[-1]))
        return np.column_stack(widths)


def _skeleton_cycle(free):
    mask = skeletonize(free)
    rr, cc = np.where(mask)
    nodes = {(int(r), int(c)) for r,c in zip(rr,cc)}
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    for r,c in nodes:
        for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
            p=(r+dr,c+dc)
            if p in nodes:
                # Prevent diagonal shortcuts across an already-connected pixel corner.
                if dr and dc and ((r,c+dc) in nodes or (r+dr,c) in nodes):
                    continue
                graph.add_edge((r,c),p,weight=math.hypot(dr,dc))
    # Iteratively remove spurs without touching the central closed-loop core.
    changed=True
    while changed:
        leaves=[n for n,d in graph.degree if d<=1]
        changed=bool(leaves)
        graph.remove_nodes_from(leaves)
    if len(graph)<12:
        raise ValueError('No closed free-space centerline found. Provide --centerline CSV.')
    # For a normal racetrack the 2-core is a single loop. If noisy branch cycles
    # exist, use the longest cycle in the cycle basis of the biggest component.
    components=sorted(nx.connected_components(graph),key=len,reverse=True)
    best=[]
    for comp in components[:3]:
        sub=graph.subgraph(comp)
        cycles=nx.cycle_basis(sub)
        for cyc in cycles:
            if len(cyc)>len(best):
                best=cyc
    if len(best)<12:
        raise ValueError('Ambiguous skeleton: no usable closed cycle; use --centerline.')
    return np.array([(c,r) for r,c in best], dtype=float)


@dataclass
class Track:
    xy: np.ndarray
    s: np.ndarray
    length: float
    heading: np.ndarray
    kappa: np.ndarray
    tree: cKDTree

    @classmethod
    def from_xy(cls, xy: np.ndarray, spacing=0.12, smoothing=0.02):
        xy=np.asarray(xy,dtype=float)
        if len(xy)>2 and np.linalg.norm(xy[0]-xy[-1]) < 1e-6:
            xy=xy[:-1]
        if len(xy)<12:
            raise ValueError('Need at least 12 centerline points')
        step=np.linalg.norm(np.diff(np.vstack([xy,xy[:1]]),axis=0),axis=1)
        keep=step>1e-5
        xy=xy[keep]
        p=np.concatenate([[0.],np.cumsum(np.linalg.norm(np.diff(np.vstack([xy,xy[:1]]),axis=0),axis=1))])
        L=float(p[-1]); n=max(80,int(math.ceil(L/spacing)))
        # Periodic cubic spline in arclength; modest smoothing of pixel-grid artifacts.
        from scipy.interpolate import splprep, splev
        try:
            tck,_=splprep([xy[:,0],xy[:,1]],u=p[:-1]/L,per=True,s=(smoothing**2)*len(xy),k=3)
            dense=np.stack(splev(np.linspace(0,1,4*n,endpoint=False),tck),axis=-1)
        except (ValueError,TypeError):
            dense=xy
        cum=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(np.vstack([dense,dense[:1]]),axis=0),axis=1))]
        LL=cum[-1]
        a=np.linspace(0.,LL,n,endpoint=False)
        # Interpolate periodic dense curve without edge artifacts.
        x=np.interp(a,cum,np.r_[dense[:,0],dense[0,0]])
        y=np.interp(a,cum,np.r_[dense[:,1],dense[0,1]])
        sampled=np.column_stack([x,y])
        actual=np.linalg.norm(np.diff(np.vstack([sampled,sampled[:1]]),axis=0),axis=1)
        s=np.r_[0.,np.cumsum(actual[:-1])]
        L=float(np.sum(actual))
        # Exact periodic cubic on uniform chord-length parametrization.
        csx=CubicSpline(np.r_[s,L],np.r_[x,x[0]],bc_type='periodic')
        csy=CubicSpline(np.r_[s,L],np.r_[y,y[0]],bc_type='periodic')
        ddx,ddy=csx(s,1),csy(s,1)
        heading=np.arctan2(ddy,ddx)
        kappa=(ddx*csy(s,2)-ddy*csx(s,2))/np.maximum((ddx*ddx+ddy*ddy)**1.5,1.e-12)
        return cls(sampled,s,L,heading,kappa,cKDTree(sampled))

    def at(self, s):
        q=float(s)%self.length
        x=np.interp(q,np.r_[self.s,self.length],np.r_[self.xy[:,0],self.xy[0,0]])
        y=np.interp(q,np.r_[self.s,self.length],np.r_[self.xy[:,1],self.xy[0,1]])
        j=max(0,min(len(self.s)-1,int(np.searchsorted(self.s,q,side='right')-1)))
        k=(j+1)%len(self.s)
        s0=float(self.s[j]);s1=float(self.s[k]) if k>j else self.length
        alpha=float(np.clip((q-s0)/max(s1-s0,1e-9),0.,1.))
        a=float(self.heading[j]+alpha*math.atan2(math.sin(self.heading[k]-self.heading[j]),math.cos(self.heading[k]-self.heading[j])))
        k=np.interp(q,np.r_[self.s,self.length],np.r_[self.kappa,self.kappa[0]])
        return x,y,a,k

    def project(self, x,y, previous_s=None):
        _,inds=self.tree.query([x,y],k=min(12,len(self.xy)))
        inds=np.atleast_1d(inds)
        best=None
        for i in inds:
            for ix in (int(i)-1,int(i)):
                j=ix%len(self.xy); k=(j+1)%len(self.xy)
                a,b=self.xy[j],self.xy[k]
                d=b-a; dd=float(d@d)
                if dd<1e-10: continue
                t=np.clip(np.dot([x-a[0],y-a[1]],d)/dd,0,1)
                p=a+t*d
                se=float(self.s[j]+t*np.linalg.norm(d))
                if previous_s is not None:
                    se+=self.length*round((previous_s-se)/self.length)
                    if abs(se-previous_s)>max(2.0,.12*self.length):
                        continue
                dist=float((x-p[0])**2+(y-p[1])**2)
                if best is None or dist<best[0]:
                    ey=float(d[0]*(y-p[1])-d[1]*(x-p[0]))/(math.sqrt(dd))
                    best=(dist,se,ey,math.atan2(d[1],d[0]))
        if best is None:
            return None
        return best[1:]


def load_centerline(path: Path):
    pts=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'): continue
        try:
            row=[float(q.strip()) for q in line.split(',')]
        except ValueError: continue
        if len(row)>=3: pts.append(row[1:3])   # s,x,y,...
        elif len(row)==2: pts.append(row)
    return np.asarray(pts)


# These are existing, checked-in racelines for the exact maintained map pair.
# They are geometric initial references ONLY: speed is optimized independently,
# and every resulting vehicle rollout is still checked against the input PGM.
# A different map filename does not silently select one of these references.
_KNOWN_MAP_SEEDS = {
    'autodrive_track_ftg_commit_20260909_025m.pgm':
        'trajectories/autodrive_mintime_sim_5p0_dense/autodrive_mintime_raceline.csv',
    'autodrive_practice_20260924_b.pgm':
        'trajectories/autodrive_practice_20260924_b/autodrive_practice_20260924_b_mintime_raceline.csv',
}


def _validate_initial_track(grid: MapGrid, xy: np.ndarray, spacing: float,
                            source: str, *, smoothing: float = 0.02) -> Track:
    """Never relax map clearance to make a seed fit.

    Sample between optimizer nodes as well: a node-only clearance check can
    miss interpolated path segments leaving the free corridor.
    """
    track = Track.from_xy(xy, spacing=spacing, smoothing=smoothing)
    step = np.linalg.norm(np.roll(track.xy, -1, axis=0) - track.xy, axis=1)
    fractions = np.linspace(0., 1., 5, endpoint=False)
    probe = (track.xy[:, None, :] + fractions[None, :, None] *
             (np.roll(track.xy, -1, axis=0) - track.xy)[:, None, :])
    values = np.asarray(grid.distance(probe.reshape(-1, 2)))
    minimum = float(np.min(values))
    if not np.isfinite(minimum) or minimum < .15:
        raise ValueError(
            f'{source}: candidate centerline crosses occupied/unknown or '
            f'out-of-bounds cells (minimum interpolated clearance '
            f'{minimum:.3f} m, required 0.150 m).')
    if not np.isfinite(track.length) or track.length <= 0. or not np.all(np.isfinite(step)):
        raise ValueError(f'{source}: nonfinite centerline geometry')
    return track


def map_to_track(grid: MapGrid, centerline: Path | None = None, spacing=0.12,
                 *, repo_root: Path | None = None, map_path: Path | None = None):
    """Get a map-verified reference loop.

    An explicit --centerline is authoritative. For maintained maps, prefer the
    same-map checked-in reference: whole-map skeleton cycle-basis enumeration
    can select the wrong loop on a complex occupied/unknown map. Otherwise
    recover the centerline from PGM free-space skeleton alone.

    Crucially, even the known-map reference MUST pass the actual provided
    map's clearance checks. This is not an occupancy/safety bypass.
    """
    if centerline is not None:
        path = Path(centerline)
        return _validate_initial_track(grid, load_centerline(path), spacing,
                                       f'explicit centerline {path}')

    errors = []
    if repo_root is not None and map_path is not None:
        relative = _KNOWN_MAP_SEEDS.get(Path(map_path).name)
        if relative:
            reference = Path(repo_root) / 'f1tenth_planning' / relative
            if reference.is_file():
                try:
                    result = _validate_initial_track(
                        grid, load_centerline(reference), spacing,
                        f'matched map raceline {reference}')
                    warnings.warn(
                        'Automatic PGM centerline: using existing same-map '
                        f'geometric reference {reference}. The optimizer will '
                        'still optimize its control inputs and lateral offsets, '
                        'and enforce map clearance on all vehicle rollouts.',
                        stacklevel=2)
                    return result
                except (OSError, ValueError, FloatingPointError) as exc:
                    errors.append(str(exc))
            else:
                errors.append(f'same-map reference not found: {reference}')

    try:
        rc = _skeleton_cycle(grid.free)
        xy = grid.pixel_to_world(rc[:, 0], rc[:, 1])
        # Prefer the mild smoothing, but attempt no smoothing if the spline
        # overshoots the free region. Both paths use identical map checks.
        try:
            return _validate_initial_track(grid, xy, spacing, 'map skeleton')
        except ValueError as exc:
            errors.append(str(exc))
            # No smoothing of pixel-loop vertices avoids B-spline overshoot.
            try:
                return _validate_initial_track(
                    grid, xy, spacing, 'map skeleton without smoothing',
                    smoothing=0.0)
            except ValueError as unsmoothed_error:
                errors.append(str(unsmoothed_error))
    except (ValueError, nx.NetworkXException) as exc:
        errors.append(str(exc))

    raise ValueError(
        'No safe closed centerline extracted from PGM. '
        'This usually means the skeleton selected a wrong/spurious loop, '
        'not that the vehicle itself cannot clear the track. '
        'Pass --centerline path/to/same-map-raceline.csv or provide a '
        'track-specific centerline. Checked candidates: ' + ' | '.join(errors))
