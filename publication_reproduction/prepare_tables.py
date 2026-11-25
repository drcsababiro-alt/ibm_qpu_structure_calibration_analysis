#!/usr/bin/env python3
"""Regenerate the six manuscript table data summaries from archived computational outputs.
This derives DISPLAY TABLES from precomputed outputs; it does not rerun the original statistical tests.
"""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'publication_reproduction'/'generated'/'tables'
REV=ROOT/'results'/'Academia_Q_FINAL_reviewer'
SUP=ROOT/'results'/'AQ13_supplement'

def save(n,frame): frame.to_csv(OUT/f'Table_{n}_data.csv',index=False)
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 # 1: snapshot counts and coverage (the original snapshot CSV does not encode all displayed coverage columns).
 inv=pd.read_csv(SUP/'Table_S4_snapshot_inventory.csv').sort_values('device').copy()
 inv['snapshot']=inv['device'].str.replace('Fake','',regex=False)
 inv['node_observations']=inv['n_qubits']
 inv['edge_observations']=inv['n_edges']
 save(1,inv[['snapshot','n_qubits','n_edges','node_observations','edge_observations']])
 # 2: four distinct analyses; the displayed p-values use different nulls.
 matched=pd.read_csv(REV/'matched_group_threshold_sensitivity.csv').iloc[0]
 cca=pd.read_csv(REV/'CCA_permutation.csv').iloc[0]
 rv=pd.read_csv(REV/'rv_permutation.csv').iloc[0]
 knn=pd.read_csv(REV/'knn_permutation.csv').iloc[0]
 save(2,pd.DataFrame([
 {'statistic':'Matched distance rho','observed':matched.spearman_rho,'null_mean':matched.perm_mean,'p':matched.group_aware_centered_permutation_p,'B':matched.n_permutations},
 {'statistic':'First CCA r1','observed':cca.observed_conditional_cca1,'null_mean':cca.null_mean,'p':cca.block_permutation_p,'B':cca.n_permutations},
 {'statistic':'RV','observed':rv.conditional_rv,'null_mean':rv.null_mean,'p':rv.block_permutation_p,'B':rv.n_permutations},
 {'statistic':'5-NN mean overlap','observed':knn.observed_mean_neighbor_overlap,'null_mean':knn.null_mean,'p':knn.block_permutation_p,'B':knn.n_permutations},
 ]))
 sensitivity=pd.read_csv(REV/'matched_group_threshold_sensitivity.csv')
 save(3,sensitivity[['minimum_group_size','n_groups','n_pairs','spearman_rho','group_aware_centered_permutation_p']])
 lodo=pd.read_csv(REV/'external_lodo_ablation.csv')
 assert len(lodo)==250
 models=['controls_only','conventional_only','multiscale_only','full','pruned_train_only']
 t4=lodo.groupby('model').agg(median_r2=('r2','median'),median_nmae=('nmae_train_sd','median'),positive_r2=('r2',lambda s:int((s>0).sum())),n=('r2','size')).reindex(models).reset_index()
 assert t4.n.tolist()==[50]*5
 save(4,t4)
 t5=lodo[lodo.model.isin(['controls_only','full','pruned_train_only'])].pivot_table(index='held_out_device',columns='model',values='r2',aggfunc='median').reset_index()
 save(5,t5)
 mi=pd.read_csv(SUP/'Table_S1_MI_full_310_per_k.csv')
 assert len(mi)==930
 targets=['twoq_error_std','twoq_error_mean','oneq_error_std','oneq_error_mean','t2_mean','readout_error_std','readout_error_mean','t1_mean','t1_std','t2_std']
 t6=pd.DataFrame([{'calibration_target':name,**{f'k_{k}':int(((mi.calibration_feature==name)&(mi.k==k)&(mi.q_bh_310<.05)).sum()) for k in (3,5,7)}} for name in targets])
 assert [t6[f'k_{k}'].sum() for k in (3,5,7)]==[78,69,65]
 save(6,t6)
 print('Created Table 1–6 data extracts:',OUT)

if __name__=='__main__':main()
