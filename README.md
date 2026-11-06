# IBM QPU Structure–Calibration Analysis

This repository accompanies the research article:

**Csaba Biró, “Beyond Connectivity: Complementary Local Structure and Calibration Heterogeneity in IBM Superconducting QPUs.”**

The study assesses whether local structural properties of superconducting quantum-processing-unit (QPU) coupling graphs and calibration heterogeneity contain overlapping or complementary information after controlling for region size and first-order connectivity organization. It uses **historical IBM fake-backend snapshots**, not live IBM Quantum hardware measurements.

> **Recommended for the revised manuscript:** use `academia_quantum_REVIEWER_COMPLETE.py` with `Academia_QPU_CORRECTED_600_UNIQUE.zip`. You do **not** need to run the original script first. The original analysis is retained for transparency but its earlier numerical results have been superseded.

## Which script should I run?

| Script | Purpose | Status |
| --- | --- | --- |
| `academia_quantum_REVIEWER_COMPLETE.py` | Corrected archived-data analysis, matched tests, full 310-pair mutual-information (MI) audit, and reviewer-requested robustness analyses | **Recommended for reproducing the revised results** |
| `ibm_qpu_structure_calibration_analysis.py` | Original Qiskit-based extraction and analysis workflow | **Historical version; retained for traceability** |

The corrected analysis uses its **own supplied input archive**. Do not generate its inputs by running the old script and do not combine earlier and corrected result tables.

## Files for the corrected analysis

- `academia_quantum_REVIEWER_COMPLETE.py` — integrated Python program for the revised analysis and reviewer robustness tests.
- `Academia_QPU_CORRECTED_600_UNIQUE.zip` — corrected input archive of 600 unique sampled physical qubit regions, regional descriptors, matched groups, and historical snapshot metadata. It corrects eight duplicate region node sets found in the earlier archived sample.
- `results/Academia_Q_FINAL_*.csv` and `results/Academia_Q_FINAL_metadata.json` — results from the completed 999-permutation MI run.
- `results/Academia_Q_FINAL_reviewer/` — descriptor redundancy and ablation, overlap and boundary diagnostics, matched-distance sensitivity tests, CCA/RV/kNN checks, and other reviewer analyses.
- `results/Academia_Q_FINAL_reviewer_supplement/` — all 310 MI pairs by estimator setting, family/scale summaries, and a heatmap.

The full revised analysis depends on both the **corrected script** and the **corrected archive**. The repository's original analysis remains available separately.

## Quick start: corrected reproducibility workflow

Install Python 3 and the scientific-Python dependencies:

```bash
python -m pip install numpy pandas networkx scipy scikit-learn matplotlib
```

The archived-data workflow does not require a Qiskit installation or an IBM Quantum account. A fresh backend-extraction workflow, if used, does require compatible `qiskit` and `qiskit-ibm-runtime` installations; this route was **not** independently validated by the archived-data audit.

Place the corrected Python script and corrected ZIP archive in the same directory. From that directory, first validate the input:

```bash
python academia_quantum_REVIEWER_COMPLETE.py --self-test --archive Academia_QPU_CORRECTED_600_UNIQUE.zip
```

Then reproduce the full statistical audit:

```bash
python academia_quantum_REVIEWER_COMPLETE.py \
  --mi-full-audit --reviewer-extras \
  --archive Academia_QPU_CORRECTED_600_UNIQUE.zip \
  --out Academia_Q_FINAL \
  --permutations 999 \
  --neighbors 3,5,7 \
  --checkpoint-every 5
```

**Spyder:** Open `academia_quantum_REVIEWER_COMPLETE.py`, select **Run → Configuration per file**, paste the options beginning with `--mi-full-audit` into **Command line options**, set **Working directory** to the directory containing both the script and corrected archive, and run with **F5**.

The `--out` value is an output-file prefix. Compatible checkpoints allow an interrupted MI run to resume without recomputing finished pairs. Use a new prefix when changing the input dataset or analysis options.

## Structural features and graph-neighborhood scales

The analysis includes conventional graph descriptors and six multiscale structural metric families:

- **KHEM:** entropy of the normalized local-degree distribution in an induced graph neighborhood.
- **EWR:** Entropy-Weighted Redundancy.
- **NED:** Normalized Entropy Density.
- **SSE:** Shell-Signature Entropy.
- **NSSE:** size-normalized Shell-Signature Entropy.
- **GCV:** coefficient of variation of cumulative neighborhood growth.

The principal graph-neighborhood radii are **1 and 2 hops**. Where available, the original pipeline also records radii 3 and 4 as secondary structural sensitivity analyses. At one hop, shell signatures may reduce to degree-based information; interpretation of higher-order local organization therefore requires appropriate connectivity controls.

**Notation:** The MI estimator's `--neighbors 3,5,7` are **numbers of nearest neighbors in statistical estimation**, not 3-, 5-, or 7-hop graph neighborhoods. In the manuscript these should be distinguished as graph radius `r` and MI neighbor count `k_MI`.

## Calibration features

The primary calibration feature block comprises the regional means and standard deviations of `T1`, `T2`, readout error, one-qubit gate error, and two-qubit gate error. Qubit frequency and two-qubit gate duration may be retained as secondary data.

Where available from the historical source extraction, finite directed two-qubit calibration entries were averaged when converted to undirected edges. **The corrected archived input does not restore those original direction-specific observations**; directed-gate sensitivity would require source-level re-extraction.

## Connectivity controls and inference

Two complementary controls are used:

1. **Exact degree-sequence matching:** compares regions with the same complete sorted degree sequence from the same historical QPU snapshot.
2. **Within-device residualization:** removes linear contributions of predefined region-size and basic connectivity summaries. This does not remove every possible first-order connectivity effect.

The analyses include matched structural–calibration distances, conditional multivariate correspondence (CCA and RV), local nearest-neighbor overlap, leave-one-device-out models, and pairwise nonlinear MI. Reviewer robustness modules assess group-size thresholds, L1 versus L2 distance, structural-descriptor redundancy/ablation, nonlinear control sensitivity, region boundaries, physical overlap, and selected strictly nonoverlapping subsets.

**Important inference limitation:** sampled regions on the same QPU can share physical qubits and edges. Block-restricted or within-group permutations do not by themselves establish exchangeability when regions overlap. The corresponding permutation p-values are conditional sensitivity evidence, not proof of statistical independence or of independent hardware replication. Strictly disjoint analyses have reduced sample sizes and power.

## Mutual-information estimation

The corrected MI audit evaluates **310 registered structural–calibration descriptor pairs** using scikit-learn nearest-neighbor MI estimation with `k_MI = 3, 5, 7` and 999 block permutations per estimable pair and setting. Nine degenerate structural coordinates yield 90 nonestimable pairs; these remain within the prespecified 310-hypothesis family, conservatively assigned `p = 1`. Benjamini–Hochberg FDR correction is performed on the full family separately for each `k_MI` setting.

These neighbor settings are a **post-review sensitivity analysis**, rather than values expressly prescribed by the reviewers. Estimator settings, data handling, and the exact testing family should be reported alongside numerical results. Do not interpret normalized excess MI as an absolute information-theoretic fraction when its numerator and entropy normalization come from different estimators.

## Outputs and reproducibility

The revised run produces prefixed CSV files and folders for the full MI test registry, summary statistics, feature diagnostics, reviewer tables, and MI figures. The supplied results are from the corrected 600-region dataset. Prior result counts from the historical code and datasets should not be transferred to the revised analysis.

For best reproducibility, retain the corrected input archive, exact script version, run metadata, generated results, and package versions together. A tagged GitHub release (optionally archived through Zenodo with a DOI) can identify the exact code version supporting the published manuscript.

The corrected archive supports reproducibility **from the frozen historical extracted data onward**. Reproducing extraction independently from Qiskit would additionally require compatible historical fake-backend package versions, and should not be conflated with the archived-data rerun.

## Scope and interpretation

The study tests whether structural and calibration representations are **largely nonredundant yet selectively coupled**. Absence of strong global correspondence does not imply statistical independence, and descriptor-level associations do not establish improvements in circuit compilation or execution fidelity. Measurements are historical snapshots rather than the current operational status of IBM devices.

## Related metric publications

**KHEM:** Biró C. *Structural Sensitivity in Graphs: An Entropy-Based k-Hop Metric and its Applications.* 2025 IEEE 19th International Symposium on Applied Computational Intelligence and Informatics (SACI), pp. 273–278. DOI: [10.1109/SACI66288.2025.11030155](https://doi.org/10.1109/SACI66288.2025.11030155).

**EWR and NED:** Biró C. *Hybrid entropy-based metrics for k-hop environment analysis in complex networks.* *Mathematics*. 2025;13(17):2902. DOI: [10.3390/math13172902](https://doi.org/10.3390/math13172902).

## Repository and citation

Repository: https://github.com/drcsababiro-alt/ibm_qpu_structure_calibration_analysis

When using this software in scientific work, cite the accompanying article once final bibliographic information is available, and cite the relevant original metric publications where appropriate. The revised code should be associated with a specific repository commit or release tag.
