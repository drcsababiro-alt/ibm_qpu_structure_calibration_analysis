from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
import numpy as np,pandas as pd,io,zipfile,hashlib,time,json
from scipy.stats import spearmanr,rankdata
z=zipfile.ZipFile(str(ROOT / 'inputs' / 'Academia_QPU_CORRECTED_600_UNIQUE.zip')); df=pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/matched_pair_distances.csv')))
group=[(n,g[['structural_distance','calibration_distance']].to_numpy(float)) for n,g in df.groupby('group_id')]; sizes=[len(x) for _,x in group]; X=[x[:,0] for _,x in group];Y=[x[:,1] for _,x in group]
seed=int.from_bytes(hashlib.sha256(b'20260817||matched_bootstrap').digest()[:8],'big')%(2**32-1)
rng=np.random.default_rng(seed); vals=[];now=time.time();N=2000
for t in range(N):
 ids=rng.integers(0,len(group),size=len(group));xx=np.concatenate([X[i] for i in ids]); yy=np.concatenate([Y[i] for i in ids]); vals.append(float(spearmanr(xx,yy).statistic))
 if t==1:print('first2 time',time.time()-now,flush=True)
obs=float(spearmanr(df.structural_distance,df.calibration_distance).statistic)
report={'observed_rho':obs,'bootstrap_mean':float(np.mean(vals)),'ci95_low':float(np.quantile(vals,0.025)),'ci95_high':float(np.quantile(vals,0.975)),'B':N,'bootstrap_unit':'exact_degree_group','warning':'Descriptive cluster bootstrap does not account for overlap across matched groups, not a coverage guarantee.'}
print(report,'secs',time.time()-now)
pd.DataFrame([report]).to_csv(str(ROOT / 'matched_group_bootstrap_corrected.csv'),index=False)
rows=[]
for held in sorted(df.device.unique()):
 p=df[df.device!=held];rows.append({'held_out_device':held,'n_pairs':len(p),'n_groups':p.group_id.nunique(),'rho':float(spearmanr(p.structural_distance,p.calibration_distance).statistic)})
pd.DataFrame(rows).to_csv(str(ROOT / 'matched_omit_device_sensitivity.csv'),index=False)
print(rows)
