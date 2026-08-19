# IBM QPU Structure–Calibration Analysis

This repository contains the reproducible Python analysis accompanying the research article:

**Csaba Biró, “Beyond Connectivity: Complementary Local Structure and Calibration Heterogeneity in IBM Superconducting QPUs.”**

The study examines whether local structural properties of superconducting quantum-processing-unit (QPU) coupling graphs and calibration heterogeneity provide overlapping or complementary information. The analysis uses historical IBM fake-backend snapshots distributed with Qiskit Runtime and does **not** query live IBM Quantum hardware.

## Main script

`ibm_qpu_structure_calibration_analysis.py`

The script performs the complete computational workflow used for the manuscript. It:

1. loads five calibration-valid historical IBM fake-backend snapshots (`FakeTorino`, `FakeFez`, `FakeMarrakesh`, `FakeBrisbane`, and `FakeSherbrooke`);
2. constructs undirected QPU coupling graphs and extracts available calibration properties;
3. samples deterministic connected local regions of 8, 12, 16, 24, and 32 qubits;
4. computes conventional graph descriptors and six multiscale metric families;
5. aggregates regional calibration descriptors;
6. forms exact degree-sequence matched groups;
7. evaluates pairwise, matched-distance, multivariate, local-neighborhood, cross-device, and nonlinear structure–calibration relationships;
8. applies dependence-aware permutation, bootstrap, and leave-one-device-out procedures;
9. exports intermediate datasets, result tables, diagnostic figures, environment information, configuration parameters, and SHA-256 checksums.

## Structural metrics

Three metrics follow previously published definitions:

- **KHEM** — entropy of the normalized local-degree distribution within the induced `k`-hop neighborhood;
- **EWR** — Entropy-Weighted Redundancy;
- **NED** — Normalized Entropy Density.

Three complementary patch-level descriptors are also evaluated:

- **SSE** — Shell-Signature Entropy;
- **NSSE** — size-normalized Shell-Signature Entropy;
- **GCV** — coefficient of variation of cumulative `k`-hop growth.

The primary local scales are `k = 1` and `k = 2`. The `k = 3` and `k = 4` results are retained as secondary sensitivity analyses. At `k = 1`, the shell signature reduces to vertex degree; consequently, higher-order structural interpretation is based primarily on the `k = 2` results after exact degree-sequence control.

## Calibration representation

The primary calibration feature block contains the regional mean and standard deviation of:

- `T1`;
- `T2`;
- readout error;
- one-qubit gate error;
- two-qubit gate error.

Qubit frequency and two-qubit gate duration are retained as additional data but are not included in the primary calibration-quality representation.

When multiple directed two-qubit instruction entries map to the same undirected coupling edge, finite error and duration values are combined using their arithmetic mean. Missing or non-finite values are not replaced by zero.

## Connectivity controls

Two complementary approaches are used.

**Exact degree-sequence matching** compares regions from the same QPU snapshot and region size with identical complete sorted degree sequences.

**Within-device residualization** removes the linear contributions of the predefined control summaries:

- region size;
- number of edges;
- mean degree;
- degree standard deviation;
- degree coefficient of variation.

Residualization is not interpreted as eliminating all first-order connectivity information. Exact degree-sequence matching provides the stricter combinatorial control for the matched analyses.

## Nonlinear mutual-information analysis

Pairwise mutual information is estimated with scikit-learn's `mutual_info_regression` using `n_neighbors = 3`. A deterministic small jitter is applied to reduce sensitivity to repeated values. Significance is evaluated using device-by-region-size block permutations, followed by Benjamini–Hochberg false-discovery-rate correction.

The scale-adjusted normalized excess-MI diagnostic uses one-dimensional histogram marginal entropies with Freedman–Diaconis binning. Because the mutual-information and marginal-entropy estimates use different estimators, this normalized quantity is treated only as a descriptive effect-size diagnostic.

## Requirements

Python 3 and the following packages are required:

```text
numpy
pandas
networkx
scipy
scikit-learn
matplotlib
qiskit
qiskit-ibm-runtime
```

No IBM Quantum account is required.

A typical installation is:

```bash
python -m pip install numpy pandas networkx scipy scikit-learn matplotlib qiskit qiskit-ibm-runtime
```

## Running the analysis

From the repository directory, run:

```bash
python ibm_qpu_structure_calibration_analysis.py
```

The script creates a timestamped output directory of the form:

```text
ibm_qpu_structure_calibration_results_YYYYMMDD_HHMMSS/
```

The analysis is computationally intensive because it includes thousands of permutations, bootstrap replicates, cross-device models, and pairwise mutual-information tests. Runtime depends on CPU resources and the installed scientific-Python stack.

## Output structure

The generated result directory contains:

```text
source_snapshots/   historical backend data extracted for the analysis
tables/             sampled regions, descriptors, statistical tests, and summaries
figures/            diagnostic figures produced by the analysis
CONFIG.json         frozen analysis parameters used in the run
RESULT_DIGEST.json  compact summary of the principal numerical results
environment.txt     Python, platform, and package-version information
README.txt          run-specific description of the generated dataset
checksums_sha256.txt SHA-256 checksums for generated files
```

Important tables include the exact degree-sequence groups, structural and calibration descriptors, matched-pair distances, permutation tests, CCA/RV/kNN analyses, leave-one-device-out predictions, structural-descriptor redundancy results, and permutation-corrected mutual-information results.

## Reproducibility

A fixed master seed (`20260817`) is used. Subordinate pseudorandom seeds are deterministically generated from SHA-256 hashes of the master seed and analysis identifiers. Sampled region definitions and generated-file checksums are exported so that a run can be audited and reproduced.

The script also records the software environment and package versions because Qiskit fake-backend availability and calibration snapshots are distributed through installed package versions rather than queried from a live service.

## Scope and interpretation

The analysis is based on **historical IBM fake-backend snapshots**. Results should therefore not be interpreted as measurements of the current operating state of the corresponding IBM processors.

The central scientific distinction is between **non-redundancy** and **independence**. Weak global representational correspondence does not imply that individual structural and calibration descriptors are statistically independent; the pipeline explicitly tests for selective pairwise and nonlinear dependence.

Sampled regions from the same QPU may overlap. For this reason, central inferential procedures use exact matched groups, group-aware resampling, device-by-region-size block permutations, and device-level holdout rather than assuming that all sampled regions are independent observations.

## Related metric publications

KHEM:

> Biró C. *Structural Sensitivity in Graphs: An Entropy-Based k-Hop Metric and its Applications.* 2025 IEEE 19th International Symposium on Applied Computational Intelligence and Informatics (SACI), pp. 273–278. DOI: 10.1109/SACI66288.2025.11030155.

EWR and NED:

> Biró C. *Hybrid entropy-based metrics for k-hop environment analysis in complex networks.* Mathematics. 2025;13(17):2902. DOI: 10.3390/math13172902.

## Repository

Project repository:

`https://github.com/drcsababiro-alt/ibm_qpu_structure_calibration_analysis`

## Citation

If this code is used in scientific work, please cite the accompanying article once its final bibliographic details are available, together with the relevant metric publications when KHEM, EWR, or NED are used directly.
