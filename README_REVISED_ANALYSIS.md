# Academia Quantum — corrected analysis and reviewer robustness audit

This repository retains the original analysis script for historical transparency. **The corrected reproducibility workflow for the revised manuscript** is `academia_quantum_REVIEWER_COMPLETE.py`; the original script is not the source of the revised statistical claims.

## Files added in this update

- `academia_quantum_REVIEWER_COMPLETE.py` — integrated corrected pipeline, 310-pair mutual-information audit, and reviewer robustness analyses.
- `Academia_QPU_CORRECTED_600_UNIQUE.zip` — corrected input archive containing 600 unique sampled physical regions and source-snapshot metadata. This supersedes the earlier input archive for the revised analysis.
- `results/Academia_Q_FINAL_*.csv` and `results/Academia_Q_FINAL_metadata.json` — completed local run (999 permutations; MI estimator neighbor counts 3, 5, 7).
- `results/Academia_Q_FINAL_reviewer/` — additional diagnostics and sensitivity checks.
- `results/Academia_Q_FINAL_reviewer_supplement/` — full MI results, scale summary and heatmap.

## Reproduce the analysis

Install Python packages used by the script; the archived-data audit can run without Qiskit. Run in the same directory as the script and input archive:

```bash
python academia_quantum_REVIEWER_COMPLETE.py --self-test --archive Academia_QPU_CORRECTED_600_UNIQUE.zip
python academia_quantum_REVIEWER_COMPLETE.py --mi-full-audit --reviewer-extras --archive Academia_QPU_CORRECTED_600_UNIQUE.zip --out Academia_Q_FINAL --permutations 999 --neighbors 3,5,7 --checkpoint-every 5
```

In Spyder, open the script and enter the arguments after the filename in **Run > Configuration per file > Command line options**. Set the working directory to the repository folder.

The `--out` prefix determines output file names. Running with the same output prefix and unchanged settings can resume from a compatible checkpoint. Use a different output prefix if input data or parameters change.

## Interpretation and cautions

- Graph neighborhood radii (typically 1 and 2 hops) are **not** the MI estimator neighbor settings (`--neighbors 3,5,7`).
- Results based on overlapping physical regions require exchangeability caveats for permutation-based inference.
- Degenerate structural coordinates are identified; their pairs remain in the 310-hypothesis multiplicity family, conservatively assigned p=1.
- The older dataset contained duplicate physical node sets under separate identifiers; the corrected data archive replaces these duplicates. Earlier result counts are not interchangeable with the corrected analysis.
- The analysis uses historical QPU snapshots; it does not demonstrate circuit-level gains.

The final manuscript statistics should be verified against the CSV files in `results/` and should not reuse figures from the superseded run.
