#!/usr/bin/env python3
import argparse, importlib.util, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

def load_base(path):
    spec=importlib.util.spec_from_file_location('base_estimator',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv',type=Path); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    root=Path(__file__).parent; base=load_base(root/'train_benchmark_estimator.py')
    d=base.load_aligned(args.csv); d,X=base.make_features(d)
    global_model=lgb.Booster(model_file=str(root/'models/global_lgbm_speed_residual.txt'))
    brake_model=lgb.Booster(model_file=str(root/'models/braking_specialist.txt'))
    brake_features=json.loads((root/'models/braking_feature_names.json').read_text())
    baseline_column = (
        'odom_v2_base_speed_mps'
        if 'odom_v2_base_speed_mps' in d.columns
        else 'odom_diagnostics_speed_mps'
    )
    odom=d[baseline_column].to_numpy(float)
    global_speed=np.clip(odom+np.maximum(odom,0.5)*global_model.predict(X),0,30)
    brake_speed=np.clip(odom+brake_model.predict(X[brake_features]),0,30)
    a=X.acc_obs.to_numpy(float)
    blend=np.clip((-a-0.35)/0.50,0.0,1.0)
    speed=(1.0-blend)*global_speed+blend*brake_speed
    out=pd.DataFrame({'source_event_stamp_s':d.source_event_stamp_s,'estimated_speed_mps':speed,'global_speed_mps':global_speed,'braking_specialist_speed_mps':brake_speed,'braking_blend':blend})
    if 'truth_speed' in d: out['truth_speed_mps']=d.truth_speed
    args.output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(args.output,index=False)
if __name__=='__main__': main()
