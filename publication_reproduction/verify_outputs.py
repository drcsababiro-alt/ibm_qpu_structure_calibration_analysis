#!/usr/bin/env python3
"""Verify key published statistics and generated table/figure CSVs against frozen inputs."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
G=ROOT/'publication_reproduction'/'generated'

def main():
 f1=pd.read_csv(G/'figure_data'/'Figure_1_data.csv')
 assert f1.matched_groups_with_variation.tolist()==[87,83,83,83,83,82]
 f2=pd.read_csv(G/'figure_data'/'Figure_2_data.csv')
 f3=pd.read_csv(G/'figure_data'/'Figure_3_data.csv')
 for frame,col in [(f2,'significant_pairs'),(f3,'fdr_significant')]:
  assert frame.groupby('k')[col].sum().reindex([3,5,7]).tolist()==[78,69,65]
 t2=pd.read_csv(G/'tables'/'Table_2_data.csv')
 assert abs(float(t2.loc[t2.statistic=='Matched distance rho','observed'].iloc[0])-.0264)<.0001
 assert abs(float(t2.loc[t2.statistic=='First CCA r1','observed'].iloc[0])-.3611)<.0001
 assert abs(float(t2.loc[t2.statistic=='RV','observed'].iloc[0])-.01609)<.00001
 t4=pd.read_csv(G/'tables'/'Table_4_data.csv');assert (t4.n==50).all()
 t6=pd.read_csv(G/'tables'/'Table_6_data.csv');assert [int(t6[f'k_{k}'].sum()) for k in (3,5,7)]==[78,69,65]
 print('PASS: validated core figure counts, summary table statistics and LODO sample sizes')
 print('NOTE: verification of summaries is not a fresh rerun of all 999-permutation analyses')

if __name__=='__main__':main()
