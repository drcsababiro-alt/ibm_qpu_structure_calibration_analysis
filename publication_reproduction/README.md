# Publication figures and tables Beyond Connectivity: Complementary Local Structure and Calibration Heterogeneity in IBM Superconducting QPUs |

## Original published figures

Original, unchanged PNG files included here:

- `figures/matched_family_variability.png` Figure 2 (10 calibration targets, k = 3, 5, 7)
- `figures/MI_310_FDR_heatmap.png` S14), and related metadata.
- The Figure 3 matrix is keyed by **structural and calibration feature names**, never reshaped by `pair_index`. Invalid MI pairs remain distinct from valid non-significant pairs.
- `prepare_tables.py` outputs numerical **display-table summaries**; it is not a claim of recalculating the CCA, RV or MI tests themselves.
- The raw historical backend instruction-name lists and original Qiskit versions are unavailable from the frozen archive. End-to-end reproduction starts with the deposited corrected 600-region dataset, not fresh extraction from live QPUs.

## Manuscript release reference

The manuscript cites commit `91c2cc65ecef7c0ba9692e29d7e08e0448e37a19`, which anchors the archived computational data and analysis. This added publication-rendering folder and a new README update will appear in a later commit on `main`; it **does not retroactively change** the contents of the cited commit.
