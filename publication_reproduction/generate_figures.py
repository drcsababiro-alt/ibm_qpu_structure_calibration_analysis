#!/usr/bin/env python3
"""Regenerate all three published manuscript plots from frozen archived inputs.

These figures preserve the original axes, feature ordering, plotted statistics,
invalid-feature masking, and color semantics. They are newly rendered using the
installed Matplotlib version; byte-identical PNG output is not promised.

Run from the repository root:
    python publication_reproduction/generate_figures.py
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
DATASET_ZIP = ROOT / 'Academia_QPU_CORRECTED_600_UNIQUE.zip'
S1_TABLE = ROOT / 'results' / 'AQ13_supplement' / 'Table_S1_MI_full_310_per_k.csv'
DEFAULT_OUT = ROOT / 'publication_reproduction' / 'generated' / 'figures'
KS = (3, 5, 7)
CALIBRATION_ORDER = (
    'twoq_error_std', 'twoq_error_mean', 'oneq_error_std',
    'oneq_error_mean', 't2_mean', 'readout_error_std',
    'readout_error_mean', 't1_mean', 't2_std', 't1_std',
)
FIGURE2_TICK_LABELS = (
    'Two-qubit\nerror SD', 'Two-qubit\nerror mean', 'One-qubit\nerror SD',
    'One-qubit\nerror mean', '$T_2$ mean', 'Readout\nerror SD',
    'Readout\nerror mean', '$T_1$ mean', '$T_1$ SD', '$T_2$ SD',
)
STRUCTURAL_ORDER = (
    'growth_k2_mean', 'SSE_k1', 'NSSE_k1', 'diameter', 'avg_path',
    'NSSE_k2', 'KHEM_k1_mean', 'GCV_k2', 'KHEM_k2_std', 'SSE_k2',
    'growth_k2_std', 'NED_k2_std', 'KHEM_k1_std', 'EWR_k2_std',
    'KHEM_k2_mean', 'NED_k2_mean', 'laplacian_radius', 'spectral_radius',
    'adjacency_energy', 'global_efficiency', 'EWR_k2_mean',
    'algebraic_connectivity', 'growth_k1_mean', 'growth_k1_std',
    'EWR_k1_std', 'EWR_k1_mean', 'NED_k1_mean', 'GCV_k1',
    'avg_clustering', 'NED_k1_std', 'transitivity',
)
EXPECTED_FIG1 = [87, 83, 83, 83, 83, 82]
EXPECTED_TOTALS = [78, 69, 65]


def load_data():
    with zipfile.ZipFile(DATASET_ZIP) as z:
        data = pd.read_csv(io.BytesIO(z.read('Academia_QPU_CORRECTED/tables/merged_patch_dataset.csv')))
    mi = pd.read_csv(S1_TABLE)
    assert len(data) == 600, f'Expected 600 regions, found {len(data)}'
    assert len(mi) == 930, f'Expected 930 MI records, found {len(mi)}'
    assert set(mi['k']) == set(KS)
    assert set(mi['structural_feature']) == set(STRUCTURAL_ORDER)
    assert set(mi['calibration_feature']) == set(CALIBRATION_ORDER)
    keys = ['structural_feature', 'calibration_feature', 'k']
    assert not mi.duplicated(keys).any(), 'Duplicate feature-pair/estimator records'
    assert len(mi.groupby(keys)) == 930, 'Incomplete 31 x 10 x 3 feature registry'
    assert mi['q_bh_310'].notna().all(), 'Missing corrected q-values'
    return data, mi


def fig1(data, out):
    groups = [g for _, g in data.groupby(['device', 'width', 'degree_sequence']) if len(g) >= 2]
    metrics = [
        ('KHEM', 'KHEM_k2_mean'), ('EWR', 'EWR_k2_mean'),
        ('NED', 'NED_k2_mean'), ('SSE', 'SSE_k2'),
        ('NSSE', 'NSSE_k2'), ('GCV', 'GCV_k2'),
    ]
    counts = [sum((g[col].max() - g[col].min()) > 1e-10 for g in groups) for _, col in metrics]
    assert len(groups) == 94 and counts == EXPECTED_FIG1, (len(groups), counts)
    fig, ax = plt.subplots(figsize=(9.8, 4.76))
    bars = ax.bar([name for name, _ in metrics], counts)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Matched groups with nonzero range (of 94)')
    ax.set_xlabel('Primary two-hop structural descriptor')
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), ha='center', va='bottom')
    fig.tight_layout()
    fig.savefig(out / 'Figure_1_regenerated.png', dpi=150)
    plt.close(fig)
    return counts


def fig2(mi, out):
    counts = np.array([
        [int(((mi['k'] == k) & (mi['calibration_feature'] == name) & (mi['q_bh_310'] < .05)).sum())
         for name in CALIBRATION_ORDER]
        for k in KS
    ])
    assert counts.sum(axis=1).tolist() == EXPECTED_TOTALS
    assert counts.shape == (3, 10)
    x = np.arange(len(CALIBRATION_ORDER))
    w = 0.25
    fig, ax = plt.subplots(figsize=(13.5, 5.43))
    for idx, k in enumerate(KS):
        ax.bar(x + (idx - 1) * w, counts[idx], w, label=rf'$k_{{MI}} = {k}$')
    ax.set_xticks(x, FIGURE2_TICK_LABELS)
    ax.set_ylabel('BH-FDR significant descriptor pairs')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(out / 'Figure_2_regenerated.png', dpi=150)
    plt.close(fig)
    return counts


def fig3(mi, out):
    # IMPORTANT: pair_index is a significance-ranking/index, NOT a grid-cell index.
    # Correct matrix uses the named structural and calibration coordinates.
    # Invalid coordinates are masked WHITE, unlike nonsignificant valid records (PURPLE).
    panels = []
    valid_panels = []
    for k in KS:
        rows = mi[mi['k'] == k].set_index(['structural_feature', 'calibration_feature'])
        selected = np.empty((len(STRUCTURAL_ORDER), len(CALIBRATION_ORDER)), dtype=float)
        for i, structural in enumerate(STRUCTURAL_ORDER):
            for j, calibration in enumerate(CALIBRATION_ORDER):
                row = rows.loc[(structural, calibration)]
                selected[i, j] = float(row['q_bh_310'] < 0.05) if row['status'] == 'ok' else np.nan
        panels.append(selected)
        valid_panels.append(int(np.isfinite(selected).sum()))
    totals = [int(np.nansum(a)) for a in panels]
    assert totals == EXPECTED_TOTALS, totals
    assert valid_panels == [220, 220, 220], valid_panels

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('white')
    fig, axes = plt.subplots(1, 3, figsize=(16.1, 9.8), sharey=True)
    for ax, k, data in zip(axes, KS, panels):
        ax.imshow(np.ma.masked_invalid(data), aspect='auto', interpolation='nearest',
                  vmin=0, vmax=1, cmap=cmap)
        ax.set_title(f'MI FDR q<0.05 | k={k} (conditional)')
        ax.set_xticks(np.arange(len(CALIBRATION_ORDER)))
        ax.set_xticklabels(CALIBRATION_ORDER, rotation=90)
        ax.set_yticks(np.arange(len(STRUCTURAL_ORDER)))
        ax.set_yticklabels(STRUCTURAL_ORDER)
        ax.tick_params(axis='y', labelleft=True)
    fig.tight_layout()
    fig.savefig(out / 'Figure_3_regenerated.png', dpi=200)
    plt.close(fig)
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data, mi = load_data()
    print('Figure 1 values:', fig1(data, args.out))
    print('Figure 2 totals k=3,5,7:', fig2(mi, args.out).sum(axis=1).tolist())
    print('Figure 3 totals k=3,5,7:', fig3(mi, args.out))
    print('New images written to:', args.out)


if __name__ == '__main__':
    main()
