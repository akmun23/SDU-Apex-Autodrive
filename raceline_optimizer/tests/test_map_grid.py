import numpy as np
import cv2
import yaml
from map_track import MapGrid,map_to_track


def ring(tmp_path):
    h=w=250; rr,cc=np.indices((h,w));rad=np.hypot(rr-h/2,cc-w/2)
    image=np.zeros((h,w),np.uint8)
    image[(rad>45)&(rad<95)]=255
    path=tmp_path/'ring.pgm'
    assert cv2.imwrite(str(path),image)
    (tmp_path/'ring.yaml').write_text(yaml.safe_dump({'resolution':.05,'origin':[-6.25,-6.25,0.],
        'negate':0,'occupied_thresh':.65,'free_thresh':.25}))
    return path


def test_map_world_pixel_roundtrip(tmp_path):
    g=MapGrid.load(ring(tmp_path))
    xy=g.pixel_to_world(100,35)
    assert np.allclose(g.world_to_pixel(xy),[35,100],atol=1e-8)
    assert g.distance(xy)>0
    assert g.distance([-100,-100])==0


def test_closed_map_centerline(tmp_path):
    g=MapGrid.load(ring(tmp_path))
    t=map_to_track(g,spacing=.25)
    assert 20<t.length<25
    assert np.min(g.distance(t.xy))>.6
