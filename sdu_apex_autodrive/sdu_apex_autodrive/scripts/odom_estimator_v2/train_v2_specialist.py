#!/usr/bin/env python3
import argparse, importlib.util, json
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

def load_base(path):
    spec=importlib.util.spec_from_file_location('base_estimator',path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('training_csv',type=Path); args=ap.parse_args()
    root=Path(__file__).parent; base=load_base(root/'train_benchmark_estimator.py')
    d=base.load_aligned(args.training_csv); d,X=base.make_features(d)
    global_model=lgb.Booster(model_file=str(root/'models/global_lgbm_speed_residual.txt'))
    importance=pd.Series(global_model.feature_importance(importance_type='gain'),index=global_model.feature_name()).sort_values(ascending=False)
    features=list(importance.head(50).index)
    baseline_column = base.runtime_baseline_column(d)
    truth=d.truth_speed.to_numpy(float); odom=d[baseline_column].to_numpy(float); a=X.acc_obs.to_numpy(float)
    mask=(truth>=0.5)&np.isfinite(truth)&np.isfinite(odom)&(a < -0.35)
    target=truth-odom
    weights=np.ones(len(d)); weights[(truth>=1)&(truth<3)]*=10; weights[(truth>=3)&(truth<5)]*=5; weights[(truth>=0.5)&(truth<1)]*=2
    model=lgb.LGBMRegressor(objective='regression',n_estimators=350,learning_rate=0.025,num_leaves=31,min_child_samples=15,colsample_bytree=0.9,reg_lambda=2.0,reg_alpha=0.05,verbosity=-1,n_jobs=-1,random_state=32)
    model.fit(X.loc[mask,features],target[mask],sample_weight=weights[mask])
    model.booster_.save_model(str(root/'models/braking_specialist.txt'))
    (root/'models/braking_feature_names.json').write_text(json.dumps(features,indent=2))
    print(f'trained on {int(mask.sum())} braking/near-braking samples')
if __name__=='__main__': main()
