# AQ13 computational supplement and targeted sensitivity analyses

This supplement accompanies the *Academia Quantum* revised manuscript, "Beyond Connectivity: Complementary Local Structure and Calibration Heterogeneity in IBM Superconducting QPUs".

The file inventory was compared directly with a GitHub source ZIP supplied by the author, whose archive comment identifies commit `a40f294de84ad2c079cab23aca8a421de08aeea2`. This supplement contains later sensitivity analyses not present in that archive. The completed GitHub upload will create **a new commit**; the new full SHA (rather than the prior `a40f294...` SHA) should be cited in the final manuscript and Reviewer 2 response.

## Contents and provenance

- `results/AQ13_supplement/`: all 14 tables and two supplementary figures from the AQ13 submission package, including the S10 selected-pair and S11–S14 sensitivity analyses.
- `audit_scripts/`: five offline AQ9 analysis scripts, whose path references have been adapted to repository-relative inputs. All five passed Python syntax checks. A fresh end-to-end execution of these supplemental scripts is **not** claimed.
- `inputs/Academia_QPU_CORRECTED_600_UNIQUE.zip`: a second, script-local copy of the corrected 600-region archive already at the repository root; this is the *same* archive by SHA-256.
- `inputs/AQ13_supporting_results.zip`: final pair results and the preselected six-pair definitions needed for the offline sensitivity scripts.
- `results/Academia_Q_FINAL_pairs.csv` and `results/Academia_Q_FINAL_reviewer_supplement/MI_full_310_per_k.csv`: replacement CSVs that remove the obsolete, entirely empty `baseline_q` column. All other 930 rows and numeric strings remain unchanged. This is a schema cleanup, **not** a recomputation.

The principal analysis remains `academia_quantum_REVIEWER_COMPLETE.py` at the repository root, with its corrected region archive and precomputed 930 MI rows. The principal source script passed `--self-test` when paired with the archived cohort from commit `a40f294...`.

## Exact run and inferential limitations

The 930 MI rows represent 310 registered structural/calibration pairs at each of estimator k = 3, 5, 7, with 999 within-device-by-width block permutations for evaluable pairs. Per-setting BH-FDR discovery counts are 78/69/65. The additional six-pair sensitivity results are **exploratory**; they do not form an independent replication or a re-run of the full 310-test FDR family under nonlinear residualization.

Overlap between physical regions means conditional block permutations are not fully overlap-adjusted tests. The archived data are historical frozen QPU snapshots. The repository supports reproducibility from these extracted snapshots onward and not necessarily fresh extraction of native instruction-level measurements.

## Directory structure

Retain all paths unchanged when adding files to the repository root. These offline scripts use paths relative to the root and expect `inputs/` and `results/AQ13_supplement/`.
