from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
"""Exploratory, selected six-pair k=5 full-QPU-environment structural sensitivity.
Region vertices and primary calibration targets unchanged; only structure's local
1/2-hop environment uses full-device topology. Not the 310-pair FDR family.
"""
import pandas as pd,numpy as np,zipfile,io,hashlib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
z=zipfile.ZipFile(str(ROOT / 'inputs' / 'Academia_QPU_CORRECTED_600_UNIQUE.zip'))
df=pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/merged_patch_dataset.csv')))
env=pd.read_csv(str(ROOT / 'results' / 'AQ13_supplement' / 'Table_S12_full_environment_per_region.csv'))
r=zipfile.ZipFile(str(ROOT / 'inputs' / 'AQ13_supporting_results.zip'))
sel=pd.read_csv(io.BytesIO(r.read('Academia_Q_FINAL_reviewer/strict_disjoint_MI_top6.csv')))[['structural_feature','calibration_feature']].drop_duplicates()
struct=list(dict.fromkeys(sel.structural_feature));q=list(dict.fromkeys(sel.calibration_feature));feat=struct+q
assert len(df)==600
fr=df[['patch_id']+[c for c in df.columns if c not in ['patch_id']]].merge(env[['patch_id']+['full_env__'+c for c in struct]],on='patch_id',validate='one_to_one')
ctrl=['width','m','avg_degree','degree_std','degree_cv']
for c in struct:fr[c]=fr['full_env__'+c]
# Original OLS residualization, within each QPU snapshot.
out=pd.DataFrame(index=fr.index,columns=feat,dtype=float)
for dev,g in fr.groupby('device'):
 idx=g.index;X=pd.DataFrame(SimpleImputer(strategy='median').fit_transform(g[ctrl]),index=idx)
 xx=np.column_stack([np.ones(len(g)),StandardScaler().fit_transform(X)])
 for c in feat:
  y=pd.to_numeric(g[c],errors='coerce').to_numpy(float);good=np.isfinite(y)
  fit=np.linalg.lstsq(xx[good],y[good],rcond=None)[0];res=np.full(len(g),np.nan);res[good]=y[good]-xx[good]@fit
  out.loc[idx,c]=res
seed=lambda *v:int.from_bytes(hashlib.sha256('||'.join(map(str,v)).encode()).digest()[:8],'big')%(2**32-1)
blocks=[g.index.to_numpy() for _,g in fr.groupby(['device','width'],sort=True)]
rows=[];B=199
for z in sel.itertuples(index=False):
 a,b=z.structural_feature,z.calibration_feature
 x=out[a].to_numpy(float);y=out[b].to_numpy(float);good=np.isfinite(x)&np.isfinite(y)
 assert good.sum()==600
 x=(x-x.mean())/x.std();y=(y-y.mean())/y.std()
 gen=np.random.default_rng(seed(20260817,'aq9_full_env_mi',a,b));x+=gen.normal(0,1e-10,len(x));y+=gen.normal(0,1e-10,len(y))
 est=lambda v:float(mutual_info_regression(x.reshape(-1,1),v,n_neighbors=5,random_state=73)[0])
 observed=est(y); perm=[]
 for _ in range(B):
  yp=y.copy()
  for block in blocks:yp[block]=y[gen.permutation(block)]
  perm.append(est(yp))
 nm=float(np.mean(perm));p=float((1+sum(q>=observed for q in perm))/(B+1))
 rows.append({'structural_feature':a,'calibration_feature':b,'observed_mi_full_env':observed,'null_mean':nm,'excess_mi':observed-nm,'p_conditional_unadjusted':p,'B':B,'k_MI':5,'n':600,'note':'posthoc six selected pairs; no FDR claim; same region roots, full-device k-hop environment, only local metrics redefined'})
 print(a,b,round(observed,4),round(nm,4),p,flush=True)
pd.DataFrame(rows).to_csv(str(ROOT / 'target_six_full_environment_MI_B199.csv'),index=False)
