from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
"""Offline, narrowly scoped corrections for Academia Quantum AQ9.
Inputs: verified November 6 corrected archive + 999-permutation results.
No live QPU extraction; no reuse of obsolete pre-repair inference.
"""
import hashlib,io,json,zipfile, pathlib
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import mutual_info_regression
MASTER_SEED=20260817
CONTROL=['width','m','avg_degree','degree_std','degree_cv']
OUT=pathlib.Path(str(ROOT));OUT.mkdir(exist_ok=True)
def sseed(*parts):
    return int.from_bytes(hashlib.sha256('||'.join(map(str,parts)).encode('utf-8')).digest()[:8],'big')%(2**32-1)
z=zipfile.ZipFile(str(ROOT / 'inputs' / 'Academia_QPU_CORRECTED_600_UNIQUE.zip'))
read=lambda n: pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/'+n+'.csv')))
df=read('merged_patch_dataset');pairs=read('matched_pair_distances')
assert (len(df),len(pairs),df.patch_id.nunique(),pairs.group_id.nunique())==(600,2102,600,94)
re=zipfile.ZipFile(str(ROOT / 'inputs' / 'AQ13_supporting_results.zip'))
results=pd.read_csv(io.BytesIO(re.read('Academia_Q_FINAL_pairs.csv')))
assert len(results)==930
# Bootstrap already computed with vectorized, identical RNG in _aq9_fast_boot.py
# 2) target six fixed pairs were preselected in archived strict-disjoint diagnostic
panel=pd.read_csv(io.BytesIO(re.read('Academia_Q_FINAL_reviewer/strict_disjoint_MI_top6.csv')))[['structural_feature','calibration_feature']].drop_duplicates()
assert len(panel)==6
scols=sorted(set(panel.structural_feature));qcols=sorted(set(panel.calibration_feature))
features=scols+qcols
# Reproduce data transformations of reviewer_crossfit_controls, with unchanged 96-tree specification.
def oof_resid(kind):
    out=pd.DataFrame(index=df.index,columns=features,dtype=float)
    for device,g in df.groupby('device',sort=True):
        pos=np.asarray(g.index)
        X=g[CONTROL].apply(pd.to_numeric,errors='coerce')
        Y=g[features].apply(pd.to_numeric,errors='coerce')
        splitter=KFold(n_splits=min(5,len(g)//10),shuffle=True,random_state=sseed(MASTER_SEED,'reviewer_oof_split',device))
        for fold,(tr,te) in enumerate(splitter.split(X)):
            imp=SimpleImputer(strategy='median').fit(X.iloc[tr]);xx=imp.transform(X.iloc[tr]);xte=imp.transform(X.iloc[te])
            ytr=Y.iloc[tr]; med=ytr.median(axis=0).fillna(0.0);filled=ytr.fillna(med).to_numpy(float)
            if kind=='nonlinear_extra_trees_oof':
                model=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=5,max_features=1.0,random_state=sseed(MASTER_SEED,'reviewer_extra',device,fold),n_jobs=1)
            elif kind=='linear_within_device_oof':model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
            else: raise ValueError(kind)
            model.fit(xx,filled); pred=model.predict(xte)
            out.loc[pos[te],features]=Y.iloc[te].to_numpy(dtype=float)-pred
    return out
# Original in-sample within-device OLS; compare against the 999-permutation primary values.
def original_resid():
    out=pd.DataFrame(index=df.index,columns=features,dtype=float)
    for device,g in df.groupby('device'):
        idx=g.index
        Xraw=g[CONTROL].copy()
        Xraw=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(Xraw),columns=Xraw.columns,index=idx)
        X=np.column_stack([np.ones(len(g)),StandardScaler().fit_transform(Xraw)])
        for target in features:
            y=pd.to_numeric(g[target],errors='coerce').to_numpy(float)
            mask=np.isfinite(y)&np.isfinite(X).all(axis=1);r=np.full(len(g),np.nan)
            if mask.sum()>=X.shape[1]+5:
                beta,*_=np.linalg.lstsq(X[mask],y[mask],rcond=None)
                r[mask]=y[mask]-X[mask]@beta
            out.loc[idx,target]=r
    return out
modes={'original_linear_in_sample':original_resid(), 'linear_within_device_oof':oof_resid('linear_within_device_oof'), 'nonlinear_extra_trees_oof':oof_resid('nonlinear_extra_trees_oof')}
# This is deliberately an OBSERVED-MI-only analysis; no re-labelling null or FDR is inferred.
mirows=[]
for mode,fr in modes.items():
  for rec in panel.itertuples(index=False):
    a,b=rec.structural_feature,rec.calibration_feature
    x=fr[a].to_numpy(dtype=float);y=fr[b].to_numpy(dtype=float)
    good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
    assert len(x)>=590,(mode,a,b,len(x))
    x=(x-x.mean())/x.std();y=(y-y.mean())/y.std()
    # exactly fixed tie-breaking magnitude; separate diagnostic seed to avoid mislabelling an identity match to main run
    rng=np.random.default_rng(sseed(MASTER_SEED,'aq9_oof_mi',mode,a,b))
    x=x+rng.normal(0,1e-10,len(x));y=y+rng.normal(0,1e-10,len(y))
    for k in (3,5,7):
        m=float(mutual_info_regression(x.reshape(-1,1),y,n_neighbors=k,random_state=73)[0])
        mirows.append({'control_model':mode,'structural_feature':a,'calibration_feature':b,'k_MI':k,'n':len(x),'mi_nats_descriptive':m,'note':'selected post hoc strongest pairs; no permutation p or FDR; not independent-sample significance'})
mi=pd.DataFrame(mirows);mi.to_csv(OUT/'target_six_OOF_MI_descriptive.csv',index=False)
