#!/usr/bin/env python3
"""Create independent CSV inputs for the three published figures from frozen source data."""
from pathlib import Path
import zipfile, io
import pandas as pd
from generate_figures import ROOT, DATASET_ZIP, S1_TABLE, KS, CALIBRATION_ORDER, STRUCTURAL_ORDER
OUT = ROOT / 'publication_reproduction' / 'generated' / 'figure_data'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DATASET_ZIP) as z:
        data=pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/merged_patch_dataset.csv')))
    groups=[g for _,g in data.groupby(['device','width','degree_sequence']) if len(g)>=2]
    metrics=[('KHEM','KHEM_k2_mean'),('EWR','EWR_k2_mean'),('NED','NED_k2_mean'),('SSE','SSE_k2'),('NSSE','NSSE_k2'),('GCV','GCV_k2')]
    f1=pd.DataFrame([{'descriptor':name,'matched_groups_with_variation':sum((g[col].max()-g[col].min())>1e-10 for g in groups)} for name,col in metrics])
    assert len(groups)==94 and f1.matched_groups_with_variation.tolist()==[87,83,83,83,83,82]
    f1.to_csv(OUT/'Figure_1_data.csv',index=False)
    mi=pd.read_csv(S1_TABLE)
    assert len(mi)==930 and not mi.duplicated(['structural_feature','calibration_feature','k']).any()
    f2=pd.DataFrame([{'calibration_feature':cal,'k':k,'significant_pairs':int(((mi.k==k)&(mi.calibration_feature==cal)&(mi.q_bh_310<.05)).sum())} for k in KS for cal in CALIBRATION_ORDER])
    assert f2.groupby('k').significant_pairs.sum().reindex(KS).tolist()==[78,69,65]
    f2.to_csv(OUT/'Figure_2_data.csv',index=False)
    mi['fdr_significant']=mi.q_bh_310.lt(.05) & mi.status.eq('ok')
    f3=mi[['structural_feature','calibration_feature','k','status','q_bh_310','fdr_significant']].copy()
    f3['structural_order']=pd.Categorical(f3.structural_feature, categories=STRUCTURAL_ORDER,ordered=True)
    f3['calibration_order']=pd.Categorical(f3.calibration_feature,categories=CALIBRATION_ORDER,ordered=True)
    f3=f3.sort_values(['k','structural_order','calibration_order']).drop(columns=['structural_order','calibration_order'])
    assert len(f3)==930 and f3.groupby('k').fdr_significant.sum().reindex(KS).tolist()==[78,69,65]
    f3.to_csv(OUT/'Figure_3_data.csv',index=False)
    print('Created CSV data for Figures 1–3:',OUT)

if __name__=='__main__':main()
