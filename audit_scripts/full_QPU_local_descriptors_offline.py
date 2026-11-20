from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
"""Same physical region vertices, induced-vs-full-QPU node-local multiscale metrics.
No Qiskit calls; uses frozen corrected edge topology and region memberships.
A full-QPU environment can include paths/qubits outside the region.
Does not redefine global conventional (spectral) whole-region metrics.
"""
import numpy as np, pandas as pd, networkx as nx,zipfile,io,math,collections,time
from scipy.stats import spearmanr
OUT=str(ROOT) + '/'
z=zipfile.ZipFile(str(ROOT / 'inputs' / 'Academia_QPU_CORRECTED_600_UNIQUE.zip'))
read=lambda n:pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/'+n+'.csv')))
df=read('merged_patch_dataset'); edges=read('edges'); nodes=read('patch_nodes')

def ent(w):
 w=np.array(w,dtype=float);w=w[np.isfinite(w)&(w>0)]
 if len(w)==0:return 0.
 p=w/w.sum();return float(-sum(p*np.log2(p)))
def ent_disc(signatures):
 count=list(collections.Counter(signatures).values());q=np.asarray(count,dtype=float)/len(signatures)
 return float(-sum(q*np.log2(q)))
def per_root(G,roots):
 data={};rads=[1,2]
 for v in roots:
  lens=nx.single_source_shortest_path_length(G,v,cutoff=2)
  vmetrics={}
  for k in rads:
   near=[u for u,d in lens.items() if 1<=d<=k];H=G.subgraph(near);n=len(near)
   h=ent([G.degree(u) for u in near]); hl=ent([H.degree(u) for u in near])
   density=2*H.number_of_edges()/(n*(n-1)) if n>1 else 0.0
   shells=tuple(sum(1 for d in lens.values() if d==j) for j in range(1,k+1))
   vmetrics[k]=(h,hl*density,hl*density/n if n else 0.,shells,len(near))
  data[v]=vmetrics
 return data
def aggregate(kv,roots):
 out={};n=len(roots)
 for k in [1,2]:
  vals=[kv[v][k] for v in roots]
  for label,idx in [('KHEM',0),('EWR',1),('NED',2)]:
   x=np.asarray([vv[idx] for vv in vals],float)
   out[f'{label}_k{k}_mean']=float(x.mean());out[f'{label}_k{k}_std']=float(x.std())
  signatures=[v[3] for v in vals];sse=ent_disc(signatures)
  out[f'SSE_k{k}']=sse;out[f'NSSE_k{k}']=sse/math.log2(n) if n>1 else 0.
  growth=np.asarray([v[4] for v in vals],float);mu=growth.mean()
  out[f'GCV_k{k}']=float(growth.std()/mu) if mu else 0.
  out[f'growth_k{k}_mean']=float(mu);out[f'growth_k{k}_std']=float(growth.std())
 return out
G={}; cache={}
for dev,e in edges.groupby('device'):
 g=nx.Graph();g.add_edges_from([(int(x.u),int(x.v)) for x in e.itertuples(index=False)]);G[dev]=g
 cache[dev]=per_root(g,list(g.nodes()))
rows=[]; err=0;maxerr=0;now=time.time()
for pid,gg in nodes.groupby('patch_id',sort=True):
 dev=gg.device.iloc[0];roots=sorted(map(int,gg.node));induced=G[dev].subgraph(roots)
 ii=aggregate(per_root(induced,roots),roots)
 full=aggregate(cache[dev],roots)
 archive=df[df.patch_id==pid].iloc[0]
 for c,iv in ii.items():
  dif=abs(float(archive[c])-iv)
  if dif>1e-9:err+=1;print('MISMATCH',pid,c,iv,archive[c])
  maxerr=max(maxerr,dif)
 out={'patch_id':pid,'device':dev,'width':len(roots)}
 for c in ii:
  out['induced__'+c]=ii[c];out['full_env__'+c]=full[c];out['delta__'+c]=full[c]-ii[c]
 rows.append(out)
assert err==0,(err,maxerr)
combined=pd.DataFrame(rows);combined.to_csv(OUT+'full_device_environment_per_region.csv',index=False)
report=[]
for c in ii:
 x=combined['induced__'+c];y=combined['full_env__'+c];d=combined['delta__'+c]
 rho=float(spearmanr(x,y).statistic) if (np.std(x)>1e-12 and np.std(y)>1e-12) else np.nan
 report.append({'descriptor':c,'rho_induced_vs_full_env':rho,'median_absolute_change':float(np.median(abs(d))),'mean_absolute_change':float(np.mean(abs(d))),'fraction_regions_changed_at_1e-9':float(np.mean(abs(d)>1e-9)),'induced_sd':float(x.std(ddof=0)),'full_env_sd':float(y.std(ddof=0)),'n_regions':600,'interpretation':'same sampled root vertices; full-env includes external vertices; no permutation inference'})
summ=pd.DataFrame(report);summ.to_csv(OUT+'full_device_environment_summary.csv',index=False)
print('600x',len(ii),'validated induced maximum mismatch',maxerr,'seconds',time.time()-now)
print(summ.sort_values('mean_absolute_change',ascending=False).to_string(index=False)[:5000])
