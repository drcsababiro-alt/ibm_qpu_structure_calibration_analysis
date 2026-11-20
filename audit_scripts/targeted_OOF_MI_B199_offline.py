from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
"""Exploratory paired six-pair MI check on OOF residuals from archived cohort.
This is deliberately separate from the 310-test FDR family and does not
resolve physical region overlap. Only pre-existing selected panel, k_MI=5.
"""
import pandas as pd,numpy as np,zipfile,io,hashlib, time
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
BASE=str(ROOT) + '/'
seed=lambda *v:int.from_bytes(hashlib.sha256('||'.join(map(str,v)).encode()).digest()[:8],'big')%(2**32-1)
z=zipfile.ZipFile(str(ROOT / 'inputs' / 'Academia_QPU_CORRECTED_600_UNIQUE.zip'));df=pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/merged_patch_dataset.csv')))
r=zipfile.ZipFile(str(ROOT / 'inputs' / 'AQ13_supporting_results.zip'))
pairs=pd.read_csv(io.BytesIO(r.read('Academia_Q_FINAL_reviewer/strict_disjoint_MI_top6.csv')))[['structural_feature','calibration_feature']].drop_duplicates()
features=list(dict.fromkeys(list(pairs.structural_feature)+list(pairs.calibration_feature)));ctrl=['width','m','avg_degree','degree_std','degree_cv']
blocks=[g.index.to_numpy() for _,g in df.groupby(['device','width'],sort=True)]

def get_oof(mode):
 out=pd.DataFrame(index=df.index,columns=features,dtype=float)
 for device,g in df.groupby('device',sort=True):
  X=g[ctrl];Y=g[features];positions=np.asarray(g.index)
  cv=KFold(n_splits=5,shuffle=True,random_state=seed(20260817,'reviewer_oof_split',device))
  for fold,(tr,te) in enumerate(cv.split(X)):
   inp=SimpleImputer(strategy='median').fit(X.iloc[tr]);xx=inp.transform(X.iloc[tr]);xt=inp.transform(X.iloc[te]); yy=Y.iloc[tr];med=yy.median().fillna(0); yy=yy.fillna(med)
   if mode=='nonlinear_extra_trees_oof':m=ExtraTreesRegressor(n_estimators=96,min_samples_leaf=5,max_features=1.0,random_state=seed(20260817,'reviewer_extra',device,fold),n_jobs=1)
   else:m=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
   m.fit(xx,yy);out.loc[positions[te],features]=Y.iloc[te].to_numpy(float)-m.predict(xt)
 return out
B=199;k=5;outputs=[]
for mode in ['linear_within_device_oof','nonlinear_extra_trees_oof']:
 fr=get_oof(mode)
 for rec in pairs.itertuples(index=False):
  a,b=rec.structural_feature,rec.calibration_feature
  x=fr[a].to_numpy(float); y=fr[b].to_numpy(float);mask=np.isfinite(x)&np.isfinite(y)
  assert mask.all(),(mode,a,b)
  x=(x-x.mean())/x.std();y=(y-y.mean())/y.std()
  gen=np.random.default_rng(seed(20260817,'aq9_b199_mi',mode,a,b))
  x+=gen.normal(0,1e-10,len(x));y+=gen.normal(0,1e-10,len(y))
  def estimate(v):return float(mutual_info_regression(x.reshape(-1,1),v,n_neighbors=k,random_state=73)[0])
  obs=estimate(y); null=np.empty(B)
  for i in range(B):
   yp=y.copy()
   for block in blocks:yp[block]=y[gen.permutation(block)]
   null[i]=estimate(yp)
  p=(1+int((null>=obs).sum()))/(B+1)
  outputs.append({'control_model':mode,'structural_feature':a,'calibration_feature':b,'k_MI':k,'observed_mi':obs,'null_mean':float(null.mean()),'excess_mi':obs-float(null.mean()),'p_conditional_unadjusted':p,'B':B,'n':len(x),'inference':'exploratory selected-after-primary; block-null subject to physical overlap; not BH-corrected over 310'})
  print(mode,a,b,'MI',round(obs,4),'null',round(null.mean(),4),'p',p,flush=True)
res=pd.DataFrame(outputs)
res.to_csv(BASE+'target_six_OOF_MI_B199.csv',index=False)
