"""
IBM QPU STRUCTURE-CALIBRATION ANALYSIS
======================================

Reproducible analysis accompanying the manuscript on the conditional
non-redundancy of local structural and calibration representations in IBM
superconducting quantum processing units (QPUs).

Scientific question
-------------------
Do local structural and calibration representations provide empirically
non-redundant information about IBM superconducting QPU regions after
controlling for region size and first-order connectivity organization?

Data source and scope
---------------------
The analysis uses historical IBM QPU snapshots distributed with the Qiskit
Runtime fake-backend provider. These snapshots provide reproducible coupling
maps and calibration properties and are not interpreted as live-current
device calibrations.

Calibration-valid cohort:
    FakeTorino
    FakeFez
    FakeMarrakesh
    FakeBrisbane
    FakeSherbrooke

Analysis outline
----------------
1. Extract coupling graphs and calibration properties.
2. Sample deterministic connected local regions.
3. Compute conventional and multiscale structural descriptors.
4. Aggregate region-level calibration descriptors.
5. Form exact degree-sequence matched groups.
6. Quantify structural variability under exact connectivity controls.
7. Evaluate controlled pairwise structure-calibration associations.
8. Compare matched structural and calibration distances with group-aware
   permutation and bootstrap procedures.
9. Evaluate conditional CCA, RV, and local-neighborhood correspondence.
10. Perform leave-one-device-out bidirectional reconstruction.
11. Estimate permutation-corrected pairwise nonlinear mutual information.
12. Perform PCA and device-exclusion robustness diagnostics.
13. Export analysis tables, figures, metadata, and SHA-256 checksums.

Scientific guardrails
---------------------
* Empirical non-redundancy is not interpreted as statistical independence.
* Central conditional analyses use within-device residualization on region
  size and first-order connectivity variables.
* Exact degree-sequence matching provides a complementary strict control.
* Primary local interpretation uses k=1 and k=2; k=3 and k=4 are retained
  only as secondary sensitivity scales.
* Overlapping sampled regions induce dependence; group-aware or block-aware
  resampling is used for central inferential tests.
* Cross-device analyses are interpreted as transfer tests after device-specific
  conditioning, not as fully train-only external validation.

Reproducibility
---------------
All stochastic operations use deterministic SHA-256-derived seeds. The script
records software versions, analysis constants, source snapshots, derived tables,
figures, and file checksums in the output directory.

Requirements
------------
Python 3 with numpy, pandas, networkx, scipy, scikit-learn, matplotlib,
qiskit, and qiskit-ibm-runtime. No IBM account is required.
"""


from __future__ import annotations

import sys
import json
import math
import hashlib
import platform
import warnings
import importlib
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from itertools import combinations

# =============================================================================
# Dependency check
# =============================================================================

REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "networkx": "networkx",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "qiskit": "qiskit",
    "qiskit_ibm_runtime": "qiskit-ibm-runtime",
}

_missing = []
for _module, _package in REQUIRED.items():
    try:
        importlib.import_module(_module)
    except Exception:
        _missing.append(_package)

if _missing:
    raise RuntimeError(
        "Missing required Python packages: "
        + ", ".join(sorted(set(_missing)))
        + ". Install them in the active Python environment before running this analysis."
    )

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

import qiskit_ibm_runtime.fake_provider as fake_provider

warnings.filterwarnings("ignore")

# =============================================================================
# Frozen analysis configuration
# =============================================================================

ANALYSIS_VERSION = "ibm-qpu-structure-calibration-analysis-1.0"
MASTER_SEED = 20260817

CALIBRATION_VALID_BACKENDS = [
    "FakeTorino",
    "FakeFez",
    "FakeMarrakesh",
    "FakeBrisbane",
    "FakeSherbrooke",
]

PATCH_WIDTHS = [8, 12, 16, 24, 32]
PATCHES_PER_DEVICE_WIDTH = 24

# Primary local scales for interpretation.
PRIMARY_K = [1, 2]
# Secondary sensitivity only.
SECONDARY_K = [3, 4]
ALL_K = PRIMARY_K + SECONDARY_K

N_MATCHED_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
N_MI_PERMUTATIONS = 1000
N_CCA_PERMUTATIONS = 2000
N_RV_PERMUTATIONS = 2000
N_KNN_PERMUTATIONS = 2000
KNN_K = 5

# Exact-degree groups with fewer patches are not sufficiently informative.
MIN_MATCHED_GROUP_SIZE = 2

# Structural features used for the central non-redundancy tests.
# Connectivity-control variables are intentionally NOT part of the "higher-order S"
# set in matched analyses.
STRUCTURAL_CORE = [
    "avg_path",
    "diameter",
    "global_efficiency",
    "avg_clustering",
    "transitivity",
    "spectral_radius",
    "algebraic_connectivity",
    "laplacian_radius",
    "adjacency_energy",
]

STRUCTURAL_K_PRIMARY = [
    "KHEM_k1", "NED_k1", "EWR_k1", "growth_k1_mean", "growth_k1_std",
    "KHEM_k2", "NED_k2", "EWR_k2", "growth_k2_mean", "growth_k2_std",
]

STRUCTURAL_K_SECONDARY = [
    "KHEM_k3", "NED_k3", "EWR_k3", "growth_k3_mean", "growth_k3_std",
    "KHEM_k4", "NED_k4", "EWR_k4", "growth_k4_mean", "growth_k4_std",
]

# Primary calibration QUALITY layer used in the central claim.
# Gate duration is kept as a SECONDARY timing sensitivity analysis because it can
# carry strong device-generation scale information that is not itself an error/fidelity
# measure.
CALIBRATION_PRIMARY = [
    "t1_mean", "t1_std",
    "t2_mean", "t2_std",
    "readout_error_mean", "readout_error_std",
    "oneq_error_mean", "oneq_error_std",
    "twoq_error_mean", "twoq_error_std",
]

CALIBRATION_SECONDARY = [
    "twoq_duration_mean", "twoq_duration_std",
]

# Backward-compatible name used by some table-generation code.
CALIBRATION_CORE = CALIBRATION_PRIMARY

CONNECTIVITY_CONTROLS = [
    "width",
    "m",
    "avg_degree",
    "degree_std",
    "degree_cv",
]

# =============================================================================
# Helpers
# =============================================================================

def now_utc():
    return datetime.now(timezone.utc).isoformat()

def package_version(name):
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"

def stable_seed(*parts):
    text = "||".join(map(str, parts)).encode("utf-8")
    digest = hashlib.sha256(text).digest()
    return int.from_bytes(digest[:8], "big") % (2**32 - 1)

def safe_float(x):
    try:
        y = float(x)
        return y if math.isfinite(y) else np.nan
    except Exception:
        return np.nan

def finite_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else np.nan

def finite_std(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=0)) if len(x) else np.nan

def entropy_discrete(values):
    vals = list(values)
    if not vals:
        return 0.0
    counts = np.array(list(Counter(vals).values()), dtype=float)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))

def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)

def zscore_frame(df):
    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(df)
    sc = StandardScaler()
    return sc.fit_transform(X)

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# =============================================================================
# Backend extraction
# =============================================================================

def instantiate_backend(class_name):
    if not hasattr(fake_provider, class_name):
        raise RuntimeError(
            f"{class_name} is not available in the installed "
            "qiskit_ibm_runtime.fake_provider."
        )
    return getattr(fake_provider, class_name)()

def backend_graph_and_properties(class_name):
    backend = instantiate_backend(class_name)
    target = backend.target
    n = int(getattr(backend, "num_qubits", 0) or getattr(target, "num_qubits", 0))

    G = nx.Graph()
    G.add_nodes_from(range(n))

    cmap = getattr(backend, "coupling_map", None)
    if cmap is None:
        cmap = target.build_coupling_map()

    edges = cmap.get_edges() if hasattr(cmap, "get_edges") else list(cmap)
    for u, v in edges:
        G.add_edge(int(u), int(v))

    # Qubit-level calibration.
    qrows = {q: {"node": q} for q in range(n)}
    try:
        qprops = target.qubit_properties
        if qprops is not None:
            for q, qp in enumerate(qprops):
                if qp is None:
                    continue
                qrows[q]["t1"] = safe_float(getattr(qp, "t1", np.nan))
                qrows[q]["t2"] = safe_float(getattr(qp, "t2", np.nan))
                qrows[q]["frequency"] = safe_float(getattr(qp, "frequency", np.nan))
    except Exception:
        pass

    # Instruction errors and durations.
    oneq_errors = {q: [] for q in range(n)}
    readout_errors = {q: [] for q in range(n)}
    twoq_errors = {}
    twoq_durations = {}

    for opname in getattr(target, "operation_names", []):
        try:
            props = target[opname]
        except Exception:
            continue

        if not hasattr(props, "items"):
            continue

        for qargs, ip in props.items():
            if ip is None:
                continue
            qargs = tuple(int(x) for x in qargs)
            err = safe_float(getattr(ip, "error", np.nan))
            dur = safe_float(getattr(ip, "duration", np.nan))

            if len(qargs) == 1:
                q = qargs[0]
                if np.isfinite(err):
                    if opname == "measure":
                        readout_errors[q].append(err)
                    else:
                        oneq_errors[q].append(err)

            elif len(qargs) == 2:
                u, v = qargs
                key = tuple(sorted((u, v)))
                if np.isfinite(err):
                    twoq_errors.setdefault(key, []).append(err)
                if np.isfinite(dur):
                    twoq_durations.setdefault(key, []).append(dur)

    for q in range(n):
        qrows[q]["readout_error"] = finite_mean(readout_errors[q])
        qrows[q]["oneq_error"] = finite_mean(oneq_errors[q])

    for u, v in G.edges():
        key = tuple(sorted((u, v)))
        G[u][v]["twoq_error"] = finite_mean(twoq_errors.get(key, []))
        G[u][v]["twoq_duration"] = finite_mean(twoq_durations.get(key, []))

    # Also attach qubit properties to the graph.
    for q, row in qrows.items():
        for k, v in row.items():
            if k != "node":
                G.nodes[q][k] = v

    metadata = {
        "backend_class": class_name,
        "num_qubits": n,
        "num_edges": G.number_of_edges(),
        "connected": nx.is_connected(G),
    }

    return backend, G, pd.DataFrame(list(qrows.values())), metadata

# =============================================================================
# Structural descriptors
# =============================================================================

def khop_signatures(G, k):
    sig = []
    shell_totals = []
    for v in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=k)
        shells = tuple(
            sum(1 for d in lengths.values() if d == j)
            for j in range(1, k + 1)
        )
        sig.append(shells)
        shell_totals.append(sum(shells))
    return sig, np.asarray(shell_totals, dtype=float)

def structural_descriptors(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.asarray([d for _, d in G.degree()], dtype=float)

    out = {
        "n": n,
        "m": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "avg_degree": finite_mean(deg),
        "degree_std": finite_std(deg),
        "degree_cv": finite_std(deg) / finite_mean(deg)
            if finite_mean(deg) not in (0, np.nan) and np.isfinite(finite_mean(deg))
            else 0.0,
        "min_degree": float(np.min(deg)) if len(deg) else 0.0,
        "max_degree": float(np.max(deg)) if len(deg) else 0.0,
        "avg_clustering": nx.average_clustering(G) if n > 1 else 0.0,
        "transitivity": nx.transitivity(G) if n > 2 else 0.0,
        "degree_entropy": entropy_discrete(deg.astype(int)),
    }

    if n > 1 and nx.is_connected(G):
        out["avg_path"] = nx.average_shortest_path_length(G)
        out["diameter"] = nx.diameter(G)
        out["global_efficiency"] = nx.global_efficiency(G)
    else:
        out["avg_path"] = np.nan
        out["diameter"] = np.nan
        out["global_efficiency"] = nx.global_efficiency(G) if n > 1 else 0.0

    if n > 1:
        A = nx.to_numpy_array(G, dtype=float)
        aev = np.linalg.eigvalsh(A)
        L = nx.laplacian_matrix(G).astype(float).toarray()
        lev = np.sort(np.linalg.eigvalsh(L))

        out["spectral_radius"] = float(np.max(np.abs(aev)))
        out["algebraic_connectivity"] = float(lev[1]) if len(lev) > 1 else 0.0
        out["laplacian_radius"] = float(lev[-1])
        out["adjacency_energy"] = float(np.sum(np.abs(aev)))
    else:
        out["spectral_radius"] = 0.0
        out["algebraic_connectivity"] = 0.0
        out["laplacian_radius"] = 0.0
        out["adjacency_energy"] = 0.0

    for k in ALL_K:
        sig, totals = khop_signatures(G, k)
        H = entropy_discrete(sig)
        maxH = math.log2(max(2, n))
        out[f"KHEM_k{k}"] = H
        out[f"NED_k{k}"] = H / maxH if maxH > 0 else 0.0
        mu = finite_mean(totals)
        out[f"EWR_k{k}"] = finite_std(totals) / mu if mu > 0 else 0.0
        out[f"growth_k{k}_mean"] = mu
        out[f"growth_k{k}_std"] = finite_std(totals)

    # WL graph hash provides a graph-structure fingerprint under identical node labels ignored.
    try:
        out["wl_hash"] = nx.weisfeiler_lehman_graph_hash(G)
    except Exception:
        out["wl_hash"] = ""

    out["degree_sequence"] = "|".join(
        map(str, sorted((int(d) for _, d in G.degree()), reverse=True))
    )

    return out

# =============================================================================
# Patch sampling
# =============================================================================

def connected_patch(G, width, rng):
    nodes = list(G.nodes())

    for _ in range(500):
        start = nodes[int(rng.integers(len(nodes)))]
        chosen = [start]
        seen = {start}
        frontier = [start]

        while frontier and len(chosen) < width:
            idx = int(rng.integers(len(frontier)))
            x = frontier.pop(idx)

            neigh = list(G.neighbors(x))
            rng.shuffle(neigh)

            for y in neigh:
                if y not in seen:
                    seen.add(y)
                    chosen.append(y)
                    frontier.append(y)
                    if len(chosen) >= width:
                        break

        if len(chosen) == width:
            P = G.subgraph(chosen).copy()
            if nx.is_connected(P):
                return P, chosen

    raise RuntimeError(f"Could not sample connected patch width={width}")

def generate_patches(device, G):
    rows = []
    patch_graphs = {}

    for width in PATCH_WIDTHS:
        for j in range(PATCHES_PER_DEVICE_WIDTH):
            seed = stable_seed(MASTER_SEED, device, width, j)
            rng = np.random.default_rng(seed)

            P, nodes = connected_patch(G, width, rng)
            pid = f"{device}__w{width}__p{j:03d}"

            rows.append({
                "patch_id": pid,
                "device": device,
                "width": width,
                "patch_index": j,
                "seed": seed,
                "nodes": "|".join(map(str, nodes)),
            })
            patch_graphs[pid] = P

    return pd.DataFrame(rows), patch_graphs

# =============================================================================
# Calibration aggregation
# =============================================================================

def calibration_descriptors(P):
    node_attrs = [
        "t1", "t2", "frequency", "readout_error", "oneq_error"
    ]
    edge_attrs = ["twoq_error", "twoq_duration"]

    out = {}

    for attr in node_attrs:
        vals = [safe_float(P.nodes[q].get(attr, np.nan)) for q in P.nodes()]
        out[f"{attr}_mean"] = finite_mean(vals)
        out[f"{attr}_std"] = finite_std(vals)

    for attr in edge_attrs:
        vals = [safe_float(P[u][v].get(attr, np.nan)) for u, v in P.edges()]
        out[f"{attr}_mean"] = finite_mean(vals)
        out[f"{attr}_std"] = finite_std(vals)

    # Coverage flags are essential for reproducibility and sensitivity checks.
    out["node_calibration_coverage"] = float(np.mean([
        np.isfinite(P.nodes[q].get("t1", np.nan))
        and np.isfinite(P.nodes[q].get("readout_error", np.nan))
        for q in P.nodes()
    ]))
    out["edge_calibration_coverage"] = float(np.mean([
        np.isfinite(P[u][v].get("twoq_error", np.nan))
        for u, v in P.edges()
    ])) if P.number_of_edges() else np.nan

    return out

# =============================================================================
# Residualization / controls
# =============================================================================

# =============================================================================
# Exact degree-sequence groups
# =============================================================================

def exact_degree_groups(df):
    grouped = []

    for keys, g in df.groupby(["device", "width", "degree_sequence"]):
        if len(g) < MIN_MATCHED_GROUP_SIZE:
            continue

        grouped.append({
            "group_id": hashlib.sha256(
                "||".join(map(str, keys)).encode()
            ).hexdigest()[:16],
            "device": keys[0],
            "width": keys[1],
            "degree_sequence": keys[2],
            "n_patches": len(g),
            "n_unique_wl_hash": g["wl_hash"].nunique(),
            "patch_ids": "|".join(g["patch_id"].astype(str)),
        })

    return pd.DataFrame(grouped)

def group_variability(df, groups, structural_cols, calibration_cols):
    rows = []

    gmap = {}
    for _, r in groups.iterrows():
        for pid in str(r["patch_ids"]).split("|"):
            gmap[pid] = r["group_id"]

    z = df[df.patch_id.isin(gmap)].copy()
    z["group_id"] = z.patch_id.map(gmap)

    for gid, g in z.groupby("group_id"):
        row = {
            "group_id": gid,
            "device": g.device.iloc[0],
            "width": int(g.width.iloc[0]),
            "n_patches": len(g),
            "n_unique_wl_hash": g.wl_hash.nunique(),
        }

        for c in structural_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"range_S__{c}"] = vals.max() - vals.min()

        for c in calibration_cols:
            vals = pd.to_numeric(g[c], errors="coerce")
            row[f"range_Q__{c}"] = vals.max() - vals.min()

        rows.append(row)

    return pd.DataFrame(rows), z

# =============================================================================
# Correlation analyses
# =============================================================================

def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg FDR correction implemented without extra dependencies."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full(len(p), np.nan)
    good = np.isfinite(p)
    pg = p[good]
    if len(pg) == 0:
        return out

    order = np.argsort(pg)
    ranked = pg[order]
    m = len(ranked)

    q = ranked * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)

    tmp = np.empty(m)
    tmp[order] = q
    out[np.where(good)[0]] = tmp
    return out


def pairwise_correlations(S, Q):
    rows = []

    for s in S.columns:
        for q in Q.columns:
            x = pd.to_numeric(S[s], errors="coerce")
            y = pd.to_numeric(Q[q], errors="coerce")
            m = x.notna() & y.notna()

            if m.sum() < 20:
                continue

            rho, ps = spearmanr(x[m], y[m])
            r, pp = pearsonr(x[m], y[m])

            rows.append({
                "structural_feature": s,
                "calibration_feature": q,
                "n": int(m.sum()),
                "spearman_rho": float(rho),
                "spearman_p": float(ps),
                "pearson_r": float(r),
                "pearson_p": float(pp),
                "abs_spearman": abs(float(rho)),
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["spearman_q_fdr"] = benjamini_hochberg(out["spearman_p"].values)
        out["pearson_q_fdr"] = benjamini_hochberg(out["pearson_p"].values)
        out = out.sort_values("abs_spearman", ascending=False)
    return out


# =============================================================================
# Matched-pair distance analysis
# =============================================================================

def standardized_distance_matrix(df, cols):
    X = df[cols].replace([np.inf, -np.inf], np.nan)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    return X

def matched_pair_distances(matched_df, structural_cols, calibration_cols):
    rows = []

    for gid, g in matched_df.groupby("group_id"):
        if len(g) < 2:
            continue

        gs = g.reset_index(drop=True)
        XS = standardized_distance_matrix(gs, structural_cols)
        XQ = standardized_distance_matrix(gs, calibration_cols)

        for i, j in combinations(range(len(gs)), 2):
            ds = float(np.linalg.norm(XS[i] - XS[j]) / math.sqrt(XS.shape[1]))
            dq = float(np.linalg.norm(XQ[i] - XQ[j]) / math.sqrt(XQ.shape[1]))

            rows.append({
                "group_id": gid,
                "device": gs.loc[i, "device"],
                "width": int(gs.loc[i, "width"]),
                "patch_a": gs.loc[i, "patch_id"],
                "patch_b": gs.loc[j, "patch_id"],
                "structural_distance": ds,
                "calibration_distance": dq,
            })

    return pd.DataFrame(rows)

def matched_pair_permutation_test(pairs, n_perm=N_MATCHED_PERMUTATIONS):
    observed, p_naive = spearmanr(
        pairs.structural_distance,
        pairs.calibration_distance
    )

    rg = np.random.default_rng(MASTER_SEED)
    perm_stats = []

    # Group-aware: shuffle calibration distances only within matched groups.
    grouped = list(pairs.groupby("group_id"))

    for _ in range(n_perm):
        yp = []
        xp = []

        for _, g in grouped:
            x = g.structural_distance.to_numpy()
            y = g.calibration_distance.to_numpy().copy()
            rg.shuffle(y)
            xp.extend(x)
            yp.extend(y)

        stat, _ = spearmanr(xp, yp)
        perm_stats.append(stat)

    perm_stats = np.asarray(perm_stats, dtype=float)
    p_perm = float(
        (1 + np.sum(np.abs(perm_stats) >= abs(observed)))
        / (1 + len(perm_stats))
    )

    return pd.DataFrame([{
        "n_pairs": len(pairs),
        "n_groups": pairs.group_id.nunique(),
        "spearman_rho": float(observed),
        "naive_p": float(p_naive),
        "group_aware_permutation_p": p_perm,
        "n_permutations": n_perm,
        "perm_mean": float(np.nanmean(perm_stats)),
        "perm_sd": float(np.nanstd(perm_stats)),
    }])


def matched_pair_group_bootstrap(pairs, n_boot=N_BOOTSTRAP):
    """Bootstrap matched groups, not individual dependent pairs."""
    groups = {gid: g.copy() for gid, g in pairs.groupby("group_id")}
    gids = np.asarray(list(groups.keys()), dtype=object)
    rg = np.random.default_rng(stable_seed(MASTER_SEED, "matched_bootstrap"))

    vals = []
    for _ in range(n_boot):
        sampled = rg.choice(gids, size=len(gids), replace=True)
        pieces = [groups[g] for g in sampled]
        z = pd.concat(pieces, ignore_index=True)
        rho, _ = spearmanr(z.structural_distance, z.calibration_distance)
        vals.append(rho)

    vals = np.asarray(vals, dtype=float)
    obs, _ = spearmanr(pairs.structural_distance, pairs.calibration_distance)

    return pd.DataFrame([{
        "observed_rho": float(obs),
        "bootstrap_mean": float(np.nanmean(vals)),
        "ci95_low": float(np.nanquantile(vals, 0.025)),
        "ci95_high": float(np.nanquantile(vals, 0.975)),
        "n_group_bootstrap": n_boot,
    }])


def matched_pair_leave_one_device_out(pairs):
    rows = []
    for held in sorted(pairs.device.unique()):
        z = pairs[pairs.device != held]
        rho, p = spearmanr(z.structural_distance, z.calibration_distance)
        rows.append({
            "held_out_device": held,
            "n_pairs": len(z),
            "n_groups": z.group_id.nunique(),
            "spearman_rho": float(rho),
            "naive_p": float(p),
        })
    return pd.DataFrame(rows)


# =============================================================================
# CCA
# =============================================================================

def residualize_within_device(df, target_cols,
                              numeric_controls=("width", "m", "avg_degree",
                                                "degree_std", "degree_cv")):
    """
    Remove first-order connectivity effects separately within each device.

    This is the central conditioning transform for the article:
        higher-order structure | device, width, connectivity/degree organization
        calibration          | device, width, connectivity/degree organization

    Device-specific fitting intentionally prevents device-generation offsets from
    masquerading as structure-calibration association.
    """
    out = pd.DataFrame(index=df.index)

    for device, g in df.groupby("device"):
        idx = g.index
        Xraw = g[list(numeric_controls)].copy()
        Xraw = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(Xraw),
            columns=Xraw.columns,
            index=idx,
        )
        X = np.column_stack([
            np.ones(len(g)),
            StandardScaler().fit_transform(Xraw),
        ])

        for target in target_cols:
            y = pd.to_numeric(g[target], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
            r = np.full(len(g), np.nan)

            if mask.sum() >= X.shape[1] + 5:
                beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
                r[mask] = y[mask] - X[mask] @ beta

            out.loc[idx, target] = r

    return out


def _prepare_residual_matrices(df, s_cols, q_cols):
    Sres = residualize_within_device(df, s_cols)
    Qres = residualize_within_device(df, q_cols)

    S = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(Sres[s_cols]),
        columns=s_cols, index=df.index
    )
    Q = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(Qres[q_cols]),
        columns=q_cols, index=df.index
    )
    return S, Q


def _cca_from_arrays(XS_tr, XQ_tr, XS_te, XQ_te, n_components=3):
    scS = StandardScaler()
    scQ = StandardScaler()

    XS_tr = scS.fit_transform(XS_tr)
    XQ_tr = scQ.fit_transform(XQ_tr)
    XS_te = scS.transform(XS_te)
    XQ_te = scQ.transform(XQ_te)

    ncomp = min(
        n_components,
        XS_tr.shape[1], XQ_tr.shape[1],
        max(1, len(XS_tr) - 1)
    )

    cca = CCA(n_components=ncomp, max_iter=4000)
    cca.fit(XS_tr, XQ_tr)

    Us_tr, Uq_tr = cca.transform(XS_tr, XQ_tr)
    Us_te, Uq_te = cca.transform(XS_te, XQ_te)

    tr_corr, te_corr = [], []
    for i in range(ncomp):
        tr_corr.append(float(np.corrcoef(Us_tr[:, i], Uq_tr[:, i])[0, 1]))
        te_corr.append(float(np.corrcoef(Us_te[:, i], Uq_te[:, i])[0, 1]))

    return tr_corr, te_corr


def conditional_cca_analysis(df, s_cols, q_cols):
    """
    CCA on WITHIN-DEVICE, CONNECTIVITY-RESIDUALIZED feature spaces.

    Device-specific conditioning prevents device-generation scale and first-order
    connectivity differences from dominating the central comparison.
    """
    S, Q = _prepare_residual_matrices(df, s_cols, q_cols)

    trc, tec = _cca_from_arrays(S.values, Q.values, S.values, Q.values)

    summary = pd.DataFrame([{
        "analysis": "conditional_full_sample_diagnostic",
        "canonical_component": i + 1,
        "canonical_correlation": trc[i],
        "squared_canonical_correlation": trc[i] ** 2,
    } for i in range(len(trc))])

    rows = []
    for held in sorted(df.device.unique()):
        tr = df.device != held
        te = df.device == held

        trc, tec = _cca_from_arrays(
            S.loc[tr].values, Q.loc[tr].values,
            S.loc[te].values, Q.loc[te].values
        )

        for i in range(len(trc)):
            rows.append({
                "held_out_device": held,
                "canonical_component": i + 1,
                "train_correlation": trc[i],
                "test_correlation": tec[i],
                "squared_test_canonical_correlation": tec[i] ** 2,
            })

    return summary, pd.DataFrame(rows), S, Q


def conditional_cca_permutation_test(df, S, Q, n_perm=N_CCA_PERMUTATIONS):
    """
    Test CCA1 against a null that preserves device × width blocks while destroying
    row-level structural-calibration correspondence.
    """
    observed, _ = _cca_from_arrays(S.values, Q.values, S.values, Q.values, 1)
    observed = float(observed[0])

    rg = np.random.default_rng(MASTER_SEED)
    blocks = [
        np.asarray(idx, dtype=int)
        for _, idx in df.reset_index(drop=True).groupby(["device", "width"]).groups.items()
    ]

    S0 = S.reset_index(drop=True)
    Q0 = Q.reset_index(drop=True)

    stats = []
    for _ in range(n_perm):
        perm_index = np.arange(len(df))
        for idx in blocks:
            shuffled = idx.copy()
            rg.shuffle(shuffled)
            perm_index[idx] = shuffled

        qperm = Q0.iloc[perm_index].to_numpy()
        c, _ = _cca_from_arrays(
            S0.values, qperm, S0.values, qperm, 1
        )
        stats.append(c[0])

    stats = np.asarray(stats, dtype=float)
    p = float((1 + np.sum(stats >= observed)) / (1 + len(stats)))

    return pd.DataFrame([{
        "observed_conditional_cca1": observed,
        "squared_canonical_correlation": observed ** 2,
        "block_permutation_p": p,
        "n_permutations": n_perm,
        "null_mean": float(np.mean(stats)),
        "null_sd": float(np.std(stats)),
        "null_q95": float(np.quantile(stats, 0.95)),
    }])


def conditional_lodo_prediction(df, Sres, Qres, source_cols, target_cols, direction):
    """
    Predict CONTROL-ADJUSTED WITHIN-DEVICE VARIATION across a held-out QPU snapshot.

    This deliberately does NOT ask a model trained on four devices to reproduce
    the absolute calibration scale of a fifth device. It asks the scientifically
    relevant question: does one information layer predict local deviations in the
    other layer after first-order connectivity effects have been removed?
    """
    rows = []

    source = Sres if direction == "structure_to_calibration" else Qres
    target = Qres if direction == "structure_to_calibration" else Sres

    for target_col in target_cols:
        for held in sorted(df.device.unique()):
            tr = df.device != held
            te = df.device == held

            Xtr = source.loc[tr, source_cols]
            Xte = source.loc[te, source_cols]
            ytr = pd.to_numeric(target.loc[tr, target_col], errors="coerce")
            yte = pd.to_numeric(target.loc[te, target_col], errors="coerce")

            good_tr = ytr.notna()
            good_te = yte.notna()

            if good_tr.sum() < 40 or good_te.sum() < 20:
                continue

            models = {
                "Ridge": Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]),
                "ExtraTrees": Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("model", ExtraTreesRegressor(
                        n_estimators=500,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        random_state=MASTER_SEED,
                        n_jobs=-1,
                    )),
                ]),
            }

            for name, model in models.items():
                model.fit(Xtr.loc[good_tr], ytr.loc[good_tr])
                pred = model.predict(Xte.loc[good_te])

                rows.append({
                    "direction": direction,
                    "target": target_col,
                    "held_out_device": held,
                    "model": name,
                    "n_train": int(good_tr.sum()),
                    "n_test": int(good_te.sum()),
                    "r2": float(r2_score(yte.loc[good_te], pred)),
                    "mae": float(mean_absolute_error(yte.loc[good_te], pred)),
                    "target_sd": float(np.std(yte.loc[good_te], ddof=0)),
                    "mae_over_target_sd":
                        float(mean_absolute_error(yte.loc[good_te], pred) /
                              max(np.std(yte.loc[good_te], ddof=0), 1e-15)),
                })

    return pd.DataFrame(rows)


def rv_coefficient(X, Y):
    """Escoufier RV coefficient between two centered multivariate data blocks."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    Sxy = X.T @ Y
    Sxx = X.T @ X
    Syy = Y.T @ Y

    num = float(np.sum(Sxy * Sxy))
    den = math.sqrt(
        float(np.sum(Sxx * Sxx)) * float(np.sum(Syy * Syy))
    )
    return num / den if den > 0 else np.nan


def conditional_rv_test(df, S, Q, n_perm=N_RV_PERMUTATIONS):
    """Block-permutation RV test on the residualized feature spaces."""
    X = StandardScaler().fit_transform(S)
    Y = StandardScaler().fit_transform(Q)
    observed = rv_coefficient(X, Y)

    work = df.reset_index(drop=True)
    blocks = [
        np.asarray(idx, dtype=int)
        for _, idx in work.groupby(["device", "width"]).groups.items()
    ]

    rg = np.random.default_rng(stable_seed(MASTER_SEED, "rv"))
    stats = []
    for _ in range(n_perm):
        yp = Y.copy()
        for idx in blocks:
            perm = idx.copy()
            rg.shuffle(perm)
            yp[idx] = Y[perm]
        stats.append(rv_coefficient(X, yp))

    stats = np.asarray(stats)
    p = float((1 + np.sum(stats >= observed)) / (1 + len(stats)))

    return pd.DataFrame([{
        "conditional_rv": float(observed),
        "block_permutation_p": p,
        "n_permutations": n_perm,
        "null_mean": float(np.mean(stats)),
        "null_sd": float(np.std(stats)),
        "null_q95": float(np.quantile(stats, 0.95)),
    }])


def knn_overlap_statistic(df, S, Q, k=KNN_K):
    """
    Local-neighborhood redundancy:
    if S and Q encode similar local organization, their nearest-neighbor sets
    should overlap more than chance within the same device × width block.
    """
    X = StandardScaler().fit_transform(S)
    Y = StandardScaler().fit_transform(Q)

    work = df.reset_index(drop=True)
    overlaps = []

    for _, idx in work.groupby(["device", "width"]).groups.items():
        idx = np.asarray(idx, dtype=int)
        if len(idx) <= k:
            continue

        XS = X[idx]
        XQ = Y[idx]

        DS = np.linalg.norm(XS[:, None, :] - XS[None, :, :], axis=2)
        DQ = np.linalg.norm(XQ[:, None, :] - XQ[None, :, :], axis=2)
        np.fill_diagonal(DS, np.inf)
        np.fill_diagonal(DQ, np.inf)

        for i in range(len(idx)):
            ns = set(np.argsort(DS[i])[:k])
            nq = set(np.argsort(DQ[i])[:k])
            overlaps.append(len(ns & nq) / k)

    return float(np.mean(overlaps)) if overlaps else np.nan


def conditional_knn_overlap_test(df, S, Q, n_perm=N_KNN_PERMUTATIONS):
    observed = knn_overlap_statistic(df, S, Q, KNN_K)

    work = df.reset_index(drop=True)
    blocks = [
        np.asarray(idx, dtype=int)
        for _, idx in work.groupby(["device", "width"]).groups.items()
    ]
    Q0 = Q.reset_index(drop=True)

    rg = np.random.default_rng(stable_seed(MASTER_SEED, "knn"))
    stats = []

    for _ in range(n_perm):
        perm_index = np.arange(len(work))
        for idx in blocks:
            perm = idx.copy()
            rg.shuffle(perm)
            perm_index[idx] = perm

        qperm = Q0.iloc[perm_index].reset_index(drop=True)
        stats.append(knn_overlap_statistic(work, S.reset_index(drop=True), qperm, KNN_K))

    stats = np.asarray(stats, dtype=float)
    p = float((1 + np.sum(stats >= observed)) / (1 + len(stats)))

    # theoretical chance overlap = k/(block_size-1); empirical permutation is primary.
    return pd.DataFrame([{
        "k": KNN_K,
        "observed_mean_neighbor_overlap": observed,
        "block_permutation_p": p,
        "n_permutations": n_perm,
        "null_mean": float(np.nanmean(stats)),
        "null_sd": float(np.nanstd(stats)),
        "excess_overlap_over_null": float(observed - np.nanmean(stats)),
    }])


# =============================================================================
# Mutual-information redundancy
# =============================================================================

def _entropy_histogram_1d(x, bins="fd"):
    """
    Histogram entropy in nats. Used ONLY as a normalization scale for MI.
    The MI estimator itself remains sklearn's kNN mutual_info_regression.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    edges = np.histogram_bin_edges(x, bins=bins)
    if len(edges) < 3:
        return 0.0
    counts, _ = np.histogram(x, bins=edges)
    p = counts[counts > 0].astype(float)
    p /= p.sum()
    return float(-np.sum(p * np.log(p)))


def residual_mutual_information(df, s_cols, q_cols,
                                n_perm=N_MI_PERMUTATIONS):
    """
    Permutation-corrected and normalized nonlinear redundancy diagnostic.

    Step 1
    ------
    Residualize BOTH structural and calibration features WITHIN DEVICE on
    width, edge count, average degree, degree SD and degree CV.

    Step 2
    ------
    Estimate pairwise kNN mutual information (MI) on those residuals.

    Step 3
    ------
    Construct a device × width BLOCK-PERMUTATION null. Calibration residuals
    are shuffled only within the same device and patch width, preserving the
    main sampling/block structure while destroying row-level S-Q correspondence.

    Step 4
    ------
    Report:
      * raw MI;
      * permutation-null mean and SD;
      * excess MI = raw MI - null mean;
      * permutation p-value;
      * z-score relative to the permutation null;
      * normalized raw MI = MI / min(H(S), H(Q));
      * normalized excess MI = max(0, excess MI) / min(H(S), H(Q));
      * null-adjusted fraction = max(0, excess MI) / raw MI.

    IMPORTANT
    ---------
    Normalized MI is a sensitivity/interpretability diagnostic because the
    marginal entropies are histogram estimates whereas MI uses a kNN estimator.
    The PRIMARY inferential quantities are permutation p and excess MI.
    """
    controls = ("width", "m", "avg_degree", "degree_std", "degree_cv")
    Sres = residualize_within_device(df, s_cols, controls)
    Qres = residualize_within_device(df, q_cols, controls)

    work = df.reset_index(drop=True)
    Sres = Sres.reset_index(drop=True)
    Qres = Qres.reset_index(drop=True)

    blocks = [
        np.asarray(idx, dtype=int)
        for _, idx in work.groupby(["device", "width"]).groups.items()
    ]

    rows = []

    for s in s_cols:
        xs_all = pd.to_numeric(Sres[s], errors="coerce").to_numpy(dtype=float)

        for q in q_cols:
            yq_all = pd.to_numeric(Qres[q], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(xs_all) & np.isfinite(yq_all)

            if mask.sum() < 50:
                continue

            # Work only with rows where both variables exist.
            valid_idx = np.where(mask)[0]
            x = xs_all[valid_idx]
            y = yq_all[valid_idx]

            # Deterministic tiny jitter avoids pathological ties.
            local = np.random.default_rng(
                stable_seed(MASTER_SEED, "mi_observed", s, q)
            )
            xj = x + local.normal(0, 1e-8, size=x.shape)
            yj = y + local.normal(0, 1e-8, size=y.shape)

            raw_mi = float(mutual_info_regression(
                xj.reshape(-1, 1),
                yj,
                random_state=stable_seed(MASTER_SEED, "mi_raw", s, q)
            )[0])

            # Marginal entropy normalization (nats).
            hx = _entropy_histogram_1d(xj)
            hy = _entropy_histogram_1d(yj)
            hmin = min(hx, hy) if np.isfinite(hx) and np.isfinite(hy) else np.nan
            normalized_raw = raw_mi / hmin if np.isfinite(hmin) and hmin > 0 else np.nan

            # Map original row indices -> compact valid-row positions.
            pos = {orig: i for i, orig in enumerate(valid_idx)}
            valid_blocks = []
            for block in blocks:
                vb = np.asarray([pos[i] for i in block if i in pos], dtype=int)
                if len(vb) >= 2:
                    valid_blocks.append(vb)

            null = []
            rg = np.random.default_rng(
                stable_seed(MASTER_SEED, "mi_perm", s, q)
            )

            for rep in range(n_perm):
                yp = y.copy()

                for vb in valid_blocks:
                    perm = vb.copy()
                    rg.shuffle(perm)
                    yp[vb] = y[perm]

                # Independent deterministic jitter per permutation.
                rj = np.random.default_rng(
                    stable_seed(MASTER_SEED, "mi_perm_jitter", s, q, rep)
                )
                x_perm = x + rj.normal(0, 1e-8, size=x.shape)
                y_perm = yp + rj.normal(0, 1e-8, size=yp.shape)

                mi0 = mutual_info_regression(
                    x_perm.reshape(-1, 1),
                    y_perm,
                    random_state=stable_seed(
                        MASTER_SEED, "mi_perm_estimator", s, q, rep
                    )
                )[0]
                null.append(float(mi0))

            null = np.asarray(null, dtype=float)
            null_mean = float(np.mean(null))
            null_sd = float(np.std(null))
            excess = float(raw_mi - null_mean)
            excess_positive = max(0.0, excess)

            p_perm = float(
                (1 + np.sum(null >= raw_mi))
                / (1 + len(null))
            )

            z_perm = (
                float((raw_mi - null_mean) / null_sd)
                if null_sd > 0 else np.nan
            )

            normalized_excess = (
                excess_positive / hmin
                if np.isfinite(hmin) and hmin > 0 else np.nan
            )

            null_adjusted_fraction = (
                excess_positive / raw_mi
                if raw_mi > 0 else 0.0
            )

            rows.append({
                "structural_feature": s,
                "calibration_feature": q,
                "n": int(mask.sum()),
                "raw_mi_nats": raw_mi,
                "null_mi_mean": null_mean,
                "null_mi_sd": null_sd,
                "excess_mi": excess,
                "excess_mi_positive": excess_positive,
                "permutation_p": p_perm,
                "permutation_z": z_perm,
                "H_struct_hist_nats": hx,
                "H_calibration_hist_nats": hy,
                "normalization_entropy_min": hmin,
                "normalized_raw_mi": normalized_raw,
                "normalized_excess_mi": normalized_excess,
                "null_adjusted_fraction": null_adjusted_fraction,
                "n_permutations": n_perm,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out["permutation_q_fdr"] = benjamini_hochberg(
            out["permutation_p"].values
        )
        out = out.sort_values(
            ["normalized_excess_mi", "excess_mi"],
            ascending=False
        )
    return out


# =============================================================================
# PCA variance diagnostics
# =============================================================================

def pca_variance(df, cols, family):
    X = zscore_frame(df[cols])
    pca = PCA()
    pca.fit(X)

    rows = []
    cum = np.cumsum(pca.explained_variance_ratio_)

    for i, (ev, c) in enumerate(zip(pca.explained_variance_ratio_, cum), start=1):
        rows.append({
            "family": family,
            "component": i,
            "explained_variance_ratio": float(ev),
            "cumulative_variance": float(c),
        })

    return pd.DataFrame(rows)

# =============================================================================
# Main pipeline
# =============================================================================

def main():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(f"ibm_qpu_structure_calibration_results_{stamp}")

    dirs = {
        "snapshots": root / "source_snapshots",
        "tables": root / "tables",
        "figures": root / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("IBM QPU STRUCTURE-CALIBRATION ANALYSIS")
    print("=" * 100)
    print("Output:", root.resolve())

    # -------------------------------------------------------------------------
    # 1. Source extraction
    # -------------------------------------------------------------------------
    print("\n[1/14] Extracting calibration-valid IBM QPU snapshots")

    device_rows = []
    all_node_rows = []
    all_edge_rows = []
    backend_graphs = {}

    for device in CALIBRATION_VALID_BACKENDS:
        print("  ", device)
        backend, G, qdf, meta = backend_graph_and_properties(device)
        backend_graphs[device] = G

        device_rows.append({
            "device": device,
            "n": G.number_of_nodes(),
            "m": G.number_of_edges(),
            "avg_degree": 2 * G.number_of_edges() / G.number_of_nodes(),
            "connected": nx.is_connected(G),
        })

        qdf.insert(0, "device", device)
        all_node_rows.append(qdf)

        for u, v, attrs in G.edges(data=True):
            all_edge_rows.append({
                "device": device,
                "u": u,
                "v": v,
                "twoq_error": attrs.get("twoq_error", np.nan),
                "twoq_duration": attrs.get("twoq_duration", np.nan),
            })

        snapshot = {
            "analysis_version": ANALYSIS_VERSION,
            "device": device,
            "metadata": meta,
            "nodes": [
                {"id": q, **{
                    k: (float(v) if isinstance(v, (int, float, np.number)) and np.isfinite(v) else None)
                    for k, v in G.nodes[q].items()
                }}
                for q in G.nodes()
            ],
            "edges": [
                {"u": u, "v": v, **{
                    k: (float(vv) if isinstance(vv, (int, float, np.number)) and np.isfinite(vv) else None)
                    for k, vv in attrs.items()
                }}
                for u, v, attrs in G.edges(data=True)
            ],
        }

        (dirs["snapshots"] / f"{device}.json").write_text(
            json.dumps(snapshot, indent=2)
        )

    pd.DataFrame(device_rows).to_csv(dirs["tables"] / "devices.csv", index=False)
    pd.concat(all_node_rows, ignore_index=True).to_csv(
        dirs["tables"] / "nodes.csv", index=False
    )
    pd.DataFrame(all_edge_rows).to_csv(
        dirs["tables"] / "edges.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 2. Patch generation
    # -------------------------------------------------------------------------
    print("[2/14] Sampling reproducible local regions")

    patch_tables = []
    patch_graphs = {}

    for device, G in backend_graphs.items():
        ptab, pgraphs = generate_patches(device, G)
        patch_tables.append(ptab)
        patch_graphs.update(pgraphs)

    patch_meta = pd.concat(patch_tables, ignore_index=True)
    patch_meta.to_csv(dirs["tables"] / "patches.csv", index=False)

    patch_node_rows = []
    for _, r in patch_meta.iterrows():
        for node in str(r.nodes).split("|"):
            patch_node_rows.append({
                "patch_id": r.patch_id,
                "device": r.device,
                "width": r.width,
                "node": int(node),
            })
    pd.DataFrame(patch_node_rows).to_csv(
        dirs["tables"] / "patch_nodes.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 3. Descriptor computation
    # -------------------------------------------------------------------------
    print("[3/14] Computing structural and calibration descriptors")

    srows = []
    qrows = []

    for _, meta in patch_meta.iterrows():
        pid = meta.patch_id
        P = patch_graphs[pid]

        s = {
            "patch_id": pid,
            "device": meta.device,
            "width": int(meta.width),
        }
        s.update(structural_descriptors(P))
        srows.append(s)

        q = {
            "patch_id": pid,
            "device": meta.device,
            "width": int(meta.width),
        }
        q.update(calibration_descriptors(P))
        qrows.append(q)

    sdf = pd.DataFrame(srows)
    qdf = pd.DataFrame(qrows)

    sdf.to_csv(dirs["tables"] / "structural_descriptors.csv", index=False)
    qdf.to_csv(dirs["tables"] / "calibration_descriptors.csv", index=False)

    merged = sdf.merge(
        qdf.drop(columns=["device", "width"]),
        on="patch_id",
        how="inner",
        validate="one_to_one",
    )

    merged.to_csv(
        dirs["tables"] / "merged_patch_dataset.csv", index=False
    )

    # Feature availability.
    s_core = [c for c in STRUCTURAL_CORE if c in merged]
    s_primary = [c for c in STRUCTURAL_K_PRIMARY if c in merged]
    s_secondary = [c for c in STRUCTURAL_K_SECONDARY if c in merged]
    s_all = s_core + s_primary + s_secondary

    q_core = [
        c for c in CALIBRATION_PRIMARY
        if c in merged and merged[c].notna().sum() >= 0.6 * len(merged)
    ]
    q_secondary = [
        c for c in CALIBRATION_SECONDARY
        if c in merged and merged[c].notna().sum() >= 0.6 * len(merged)
    ]

    if len(q_core) < 4:
        raise RuntimeError(
            "Insufficient calibration coverage for the planned non-redundancy analysis."
        )

    # -------------------------------------------------------------------------
    # 4. Exact degree-sequence matching
    # -------------------------------------------------------------------------
    print("[4/14] Exact degree-sequence matched structural analysis")

    groups = exact_degree_groups(merged)
    groups.to_csv(
        dirs["tables"] / "exact_degree_groups.csv", index=False
    )

    variability, matched_df = group_variability(
        merged, groups, s_all, q_core
    )
    variability.to_csv(
        dirs["tables"] / "exact_degree_group_variability.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 5. Raw and controlled correlations
    # -------------------------------------------------------------------------
    print("[5/14] Structure-calibration association")

    corr = pairwise_correlations(
        merged[s_all],
        merged[q_core],
    )
    corr.to_csv(
        dirs["tables"] / "structure_calibration_correlations.csv",
        index=False,
    )

    # Controlled pairwise associations use the same central
    # within-device conditioning transform as CCA, RV, kNN and MI.
    Sres = residualize_within_device(
        merged,
        s_all,
        numeric_controls=("width", "m", "avg_degree", "degree_std", "degree_cv"),
    )
    Qres = residualize_within_device(
        merged,
        q_core,
        numeric_controls=("width", "m", "avg_degree", "degree_std", "degree_cv"),
    )

    pcorr = pairwise_correlations(Sres, Qres)
    pcorr.to_csv(
        dirs["tables"] / "partial_residual_correlations.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # 6. Matched-pair distances + permutation
    # -------------------------------------------------------------------------
    print("[6/14] Matched-pair structural vs calibration distance test")

    pair_df = matched_pair_distances(
        matched_df,
        s_core + s_primary,
        q_core,
    )
    pair_df.to_csv(
        dirs["tables"] / "matched_pair_distances.csv", index=False
    )

    pair_test = matched_pair_permutation_test(pair_df)
    pair_test.to_csv(
        dirs["tables"] / "matched_pair_test.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 7. Corrected conditional multivariate redundancy tests
    # -------------------------------------------------------------------------
    print("[7/14] Conditional CCA + RV + local-neighborhood overlap")

    cca_summary, cca_lodo, Scond, Qcond = conditional_cca_analysis(
        merged,
        s_core + s_primary,
        q_core,
    )
    cca_summary.to_csv(
        dirs["tables"] / "cca_conditional_summary.csv", index=False
    )
    cca_lodo.to_csv(
        dirs["tables"] / "cca_conditional_leave_one_device_out.csv", index=False
    )

    cca_perm = conditional_cca_permutation_test(
        merged, Scond, Qcond
    )
    cca_perm.to_csv(
        dirs["tables"] / "cca_conditional_permutation_test.csv", index=False
    )

    rv_test = conditional_rv_test(
        merged, Scond, Qcond
    )
    rv_test.to_csv(
        dirs["tables"] / "conditional_rv_test.csv", index=False
    )

    knn_test = conditional_knn_overlap_test(
        merged, Scond, Qcond
    )
    knn_test.to_csv(
        dirs["tables"] / "conditional_knn_overlap_test.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 8. Corrected conditional bidirectional prediction
    # -------------------------------------------------------------------------
    print("[8/14] Conditional bidirectional leave-one-device-out prediction")

    s_to_q = conditional_lodo_prediction(
        merged,
        Scond,
        Qcond,
        s_core + s_primary,
        q_core,
        direction="structure_to_calibration",
    )
    s_to_q.to_csv(
        dirs["tables"] / "structure_to_calibration_conditional_prediction.csv",
        index=False,
    )

    structural_targets = [
        c for c in [
            "avg_path",
            "diameter",
            "global_efficiency",
            "algebraic_connectivity",
            "KHEM_k2",
            "NED_k2",
            "EWR_k2",
        ]
        if c in merged
    ]

    q_to_s = conditional_lodo_prediction(
        merged,
        Scond,
        Qcond,
        q_core,
        structural_targets,
        direction="calibration_to_structure",
    )
    q_to_s.to_csv(
        dirs["tables"] / "calibration_to_structure_conditional_prediction.csv",
        index=False,
    )

    # Matched-pair robustness.
    pair_boot = matched_pair_group_bootstrap(pair_df)
    pair_boot.to_csv(
        dirs["tables"] / "matched_pair_bootstrap.csv", index=False
    )

    pair_lodo = matched_pair_leave_one_device_out(pair_df)
    pair_lodo.to_csv(
        dirs["tables"] / "matched_pair_leave_one_device_out.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 9. Nonlinear MI diagnostic
    # -------------------------------------------------------------------------
    print("[9/14] Permutation-corrected / normalized residual mutual-information diagnostic")

    mi = residual_mutual_information(
        merged,
        s_core + s_primary,
        q_core,
    )
    mi.to_csv(
        dirs["tables"] / "mutual_information_permutation_normalized.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 10. PCA variance diagnostics
    # -------------------------------------------------------------------------
    print("[10/14] PCA variance diagnostics")

    pca_s = pca_variance(
        merged,
        s_core + s_primary,
        "structural",
    )
    pca_q = pca_variance(
        merged,
        q_core,
        "calibration",
    )
    pca_df = pd.concat([pca_s, pca_q], ignore_index=True)
    pca_df.to_csv(
        dirs["tables"] / "pca_variance_summary.csv", index=False
    )


    # -------------------------------------------------------------------------
    # 11. Device-exclusion and calibration-definition robustness
    # -------------------------------------------------------------------------
    print("[11/14] Device-exclusion and calibration-definition robustness")

    robustness_rows = []

    for held in ["NONE"] + sorted(merged.device.unique()):
        z = merged.copy() if held == "NONE" else merged[merged.device != held].copy()

        # Controlled pairwise max association using PRIMARY quality calibration.
        Sr = residualize_within_device(z, s_core + s_primary)
        Qr = residualize_within_device(z, q_core)
        cc = pairwise_correlations(Sr, Qr)

        # Conditional CCA1.
        csum, _, _, _ = conditional_cca_analysis(
            z, s_core + s_primary, q_core
        )
        cca1 = float(csum.loc[
            csum.canonical_component == 1, "canonical_correlation"
        ].iloc[0])

        robustness_rows.append({
            "held_out_device": held,
            "n_patches": len(z),
            "max_abs_controlled_spearman":
                float(cc.abs_spearman.max()) if len(cc) else np.nan,
            "conditional_cca1": cca1,
            "conditional_cca1_squared_correlation": cca1 ** 2,
        })

    robustness_df = pd.DataFrame(robustness_rows)
    robustness_df.to_csv(
        dirs["tables"] / "robustness_by_device.csv", index=False
    )

    # Secondary timing sensitivity is deliberately separated from the central
    # calibration-quality claim.
    if q_secondary:
        secondary_corr = pairwise_correlations(
            residualize_within_device(merged, s_core + s_primary),
            residualize_within_device(merged, q_secondary),
        )
        secondary_corr.to_csv(
            dirs["tables"] / "secondary_timing_associations.csv", index=False
        )

    # -------------------------------------------------------------------------
    # 12. Compact evidence concordance table
    # -------------------------------------------------------------------------
    print("[12/14] Evidence concordance audit")

    evidence_rows = [
        {
            "evidence_family": "exact_degree_structural_variability",
            "statistic": "fraction matched groups with >1 WL hash",
            "value": float((groups.n_unique_wl_hash > 1).mean()) if len(groups) else np.nan,
            "supports_nonredundancy_if": "high",
        },
        {
            "evidence_family": "matched_pair_distance_association",
            "statistic": "abs Spearman(structural distance, calibration distance)",
            "value": abs(float(pair_test.spearman_rho.iloc[0])),
            "supports_nonredundancy_if": "low",
        },
        {
            "evidence_family": "conditional_pairwise_association",
            "statistic": "max abs controlled Spearman",
            "value": float(pcorr.abs_spearman.max()) if len(pcorr) else np.nan,
            "supports_nonredundancy_if": "low/modest",
        },
        {
            "evidence_family": "conditional_CCA",
            "statistic": "CCA1 squared canonical correlation",
            "value": float(cca_summary.loc[
                cca_summary.canonical_component == 1, "squared_canonical_correlation"
            ].iloc[0]),
            "supports_nonredundancy_if": "limited",
        },
        {
            "evidence_family": "conditional_RV",
            "statistic": "RV coefficient",
            "value": float(rv_test.conditional_rv.iloc[0]),
            "supports_nonredundancy_if": "limited",
        },
        {
            "evidence_family": "local_neighborhood_overlap",
            "statistic": "excess kNN overlap over block-permutation null",
            "value": float(knn_test.excess_overlap_over_null.iloc[0]),
            "supports_nonredundancy_if": "near zero",
        },
    ]

    pd.DataFrame(evidence_rows).to_csv(
        dirs["tables"] / "evidence_concordance.csv", index=False
    )

    # -------------------------------------------------------------------------
    # 13. Figures
    # -------------------------------------------------------------------------
    print("[13/14] Figures")

    # Fig 1: degree-matched structural variability.
    range_cols = [c for c in variability if c.startswith("range_S__")]
    vr = variability[range_cols].median().sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(
        [x.replace("range_S__", "") for x in vr.index[::-1]],
        vr.values[::-1],
    )
    ax.set_xlabel("Median within-group range")
    ax.set_title("Higher-order structural variability under exact degree-sequence matching")
    savefig(fig, dirs["figures"] / "fig01_patch_structural_variability.png")

    # Fig 2: controlled association heatmap.
    pivot = pcorr.pivot(
        index="structural_feature",
        columns="calibration_feature",
        values="spearman_rho",
    )
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_title("Controlled structural–calibration Spearman associations")
    fig.colorbar(im, ax=ax, label="Spearman rho")
    savefig(fig, dirs["figures"] / "fig02_structure_calibration_heatmap.png")

    # Fig 3: matched-pair distances.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        pair_df.structural_distance,
        pair_df.calibration_distance,
        s=10,
        alpha=0.35,
    )
    ax.set_xlabel("Structural distance (exact degree-matched pairs)")
    ax.set_ylabel("Calibration distance")
    ax.set_title("Matched structural and calibration distances")
    savefig(fig, dirs["figures"] / "fig03_matched_pair_distance_scatter.png")

    # Fig 4: CCA LODO first component.
    z = cca_lodo[cca_lodo.canonical_component == 1].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(z.held_out_device, z.test_correlation)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Held-out canonical correlation")
    ax.tick_params(axis="x", rotation=35)
    ax.set_title("Conditional CCA generalization to unseen IBM QPU snapshots")
    savefig(fig, dirs["figures"] / "fig04_cca_generalization.png")

    # Fig 5: prediction generalization.
    zz = s_to_q.groupby(["target", "model"], as_index=False).r2.mean()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{r.target}\n{r.model}" for _, r in zz.iterrows()]
    ax.bar(labels, zz.r2)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("Mean leave-one-device-out R²")
    ax.tick_params(axis="x", rotation=55)
    ax.set_title("Can controlled local structure predict controlled calibration variation?")
    savefig(fig, dirs["figures"] / "fig05_prediction_generalization.png")

    # -------------------------------------------------------------------------
    # 14. Digest, provenance, checksums
    # -------------------------------------------------------------------------
    print("[14/14] Result digest and reproducibility metadata")

    exact_multi_hash_rate = (
        float((groups.n_unique_wl_hash > 1).mean())
        if len(groups) else np.nan
    )

    pair_rho = float(pair_test.spearman_rho.iloc[0])
    pair_p = float(pair_test.group_aware_permutation_p.iloc[0])

    cca_first = cca_summary[cca_summary.canonical_component == 1].iloc[0]
    cca_lodo_first = cca_lodo[cca_lodo.canonical_component == 1]

    mean_lodo_cca = float(np.nanmean(np.abs(cca_lodo_first.test_correlation)))

    s2q_mean_r2 = (
        s_to_q.groupby("model").r2.mean().to_dict()
        if len(s_to_q) else {}
    )
    q2s_mean_r2 = (
        q_to_s.groupby("model").r2.mean().to_dict()
        if len(q_to_s) else {}
    )

    digest = {
        "analysis_version": ANALYSIS_VERSION,
        "claim_under_test":
            "Local structural and calibration heterogeneity constitute empirically "
            "non-redundant dimensions of IBM superconducting QPU characterization.",
        "n_devices": int(merged.device.nunique()),
        "n_patches": int(len(merged)),
        "n_exact_degree_groups": int(len(groups)),
        "n_patches_in_exact_degree_groups": int(
            len(set(
                pid
                for ids in groups.patch_ids.astype(str)
                for pid in ids.split("|")
            ))
        ) if len(groups) else 0,
        "exact_degree_groups_with_multiple_wl_hash_rate": exact_multi_hash_rate,
        "matched_pair_test": {
            "n_pairs": int(pair_test.n_pairs.iloc[0]),
            "spearman_rho": pair_rho,
            "group_aware_permutation_p": pair_p,
        },
        "cca": {
            "conditional_first_canonical_correlation":
                float(cca_first.canonical_correlation),
            "conditional_first_squared_canonical_correlation":
                float(cca_first.squared_canonical_correlation),
            "conditional_cca_block_permutation_p":
                float(cca_perm.block_permutation_p.iloc[0]),
            "mean_abs_lodo_first_canonical_correlation":
                mean_lodo_cca,
        },
        "structure_to_calibration_mean_lodo_r2_by_model": {
            str(k): float(v) for k, v in s2q_mean_r2.items()
        },
        "calibration_to_structure_mean_lodo_r2_by_model": {
            str(k): float(v) for k, v in q2s_mean_r2.items()
        },
        "max_abs_controlled_spearman":
            float(pcorr.abs_spearman.max()) if len(pcorr) else np.nan,
        "conditional_rv": float(rv_test.conditional_rv.iloc[0]),
        "conditional_rv_permutation_p": float(rv_test.block_permutation_p.iloc[0]),
        "knn_overlap_excess_over_null":
            float(knn_test.excess_overlap_over_null.iloc[0]),
        "knn_overlap_permutation_p":
            float(knn_test.block_permutation_p.iloc[0]),
        "matched_pair_bootstrap_ci95": [
            float(pair_boot.ci95_low.iloc[0]),
            float(pair_boot.ci95_high.iloc[0]),
        ],
        "max_raw_residual_mutual_information":
            float(mi.raw_mi_nats.max()) if len(mi) else np.nan,
        "max_permutation_corrected_excess_mi":
            float(mi.excess_mi_positive.max()) if len(mi) else np.nan,
        "max_normalized_excess_mi":
            float(mi.normalized_excess_mi.max()) if len(mi) else np.nan,
        "n_mi_pairs_fdr_q_lt_0_05":
            int((mi.permutation_q_fdr < 0.05).sum()) if len(mi) else 0,
        "interpretation_guardrail":
            "Empirical non-redundancy does not imply statistical independence.",
    }

    (root / "RESULT_DIGEST.json").write_text(
        json.dumps(digest, indent=2)
    )

    config = {
        "analysis_version": ANALYSIS_VERSION,
        "master_seed": MASTER_SEED,
        "backends": CALIBRATION_VALID_BACKENDS,
        "patch_widths": PATCH_WIDTHS,
        "patches_per_device_width": PATCHES_PER_DEVICE_WIDTH,
        "primary_k": PRIMARY_K,
        "secondary_k": SECONDARY_K,
        "n_matched_permutations": N_MATCHED_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "n_mi_permutations": N_MI_PERMUTATIONS,
        "n_cca_permutations": N_CCA_PERMUTATIONS,
        "n_rv_permutations": N_RV_PERMUTATIONS,
        "n_knn_permutations": N_KNN_PERMUTATIONS,
        "knn_k": KNN_K,
        "structural_core": STRUCTURAL_CORE,
        "calibration_primary_requested": CALIBRATION_PRIMARY,
        "calibration_primary_used": q_core,
        "calibration_secondary_timing": q_secondary,
        "connectivity_controls": CONNECTIVITY_CONTROLS,
    }
    (root / "CONFIG.json").write_text(json.dumps(config, indent=2))

    env_lines = [
        f"created_utc={now_utc()}",
        f"python={platform.python_version()}",
        f"platform={platform.platform()}",
        f"python_executable={sys.executable}",
    ]
    for pkg in [
        "numpy", "pandas", "networkx", "scipy", "scikit-learn",
        "matplotlib", "qiskit", "qiskit-ibm-runtime"
    ]:
        env_lines.append(f"{pkg}={package_version(pkg)}")

    (root / "environment.txt").write_text("\n".join(env_lines) + "\n")

    readme = f"""IBM QPU Structural–Calibration Non-Redundancy Dataset
====================================================

Analysis version: {ANALYSIS_VERSION}

Scientific claim under test
---------------------------
Local structural and calibration heterogeneity constitute empirically
non-redundant dimensions of IBM superconducting QPU characterization.

Scope
-----
Calibration-valid historical IBM fake-backend snapshots:
{chr(10).join("- " + x for x in CALIBRATION_VALID_BACKENDS)}

The study uses reproducibly sampled connected local regions at widths:
{PATCH_WIDTHS}

Primary local k-hop interpretation: k={PRIMARY_K}
Secondary sensitivity only: k={SECONDARY_K}

Central controls
----------------
1. region size;
2. edge/connectivity budget;
3. exact degree sequence for matched analyses;
4. device, width, and first-order connectivity variables for residual analyses.

Complementary evidence families
-------------------------------
* exact degree-sequence structural heterogeneity;
* pairwise and controlled structural-calibration association;
* exact-degree matched structural/calibration distance association;
* group-aware permutation testing;
* connectivity-residualized canonical correlation analysis;
* block-permutation significance for conditional CCA;
* leave-one-device-out conditional CCA;
* Escoufier RV multivariate redundancy with block permutation;
* local k-nearest-neighbor overlap between structural and calibration spaces;
* bidirectional prediction of controlled within-device variation;
* permutation-corrected and normalized residual mutual-information diagnostic;
* leave-one-device sensitivity analysis;
* FDR-corrected pairwise association tables.

Important limitation
--------------------
These are historical IBM QPU snapshots distributed through the Qiskit Runtime
fake-backend provider, not live longitudinal calibration streams. The central
calibration claim is therefore scoped to the available snapshot cohort.

Reproduction
------------
Run the accompanying Python script in a compatible Python environment.
No IBM account is required.
"""
    (root / "README.txt").write_text(readme)

    # Checksums last.
    checksum_rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name != "checksums_sha256.txt":
            checksum_rows.append(
                f"{sha256_file(p)}  {p.relative_to(root)}"
            )

    (root / "checksums_sha256.txt").write_text(
        "\n".join(checksum_rows) + "\n"
    )

    print("\n" + "=" * 100)
    print("PIPELINE COMPLETE")
    print("=" * 100)
    print("Patches:", len(merged))
    print("Exact degree-sequence groups:", len(groups))
    print("Matched pairs:", len(pair_df))
    print("Matched distance rho:", round(pair_rho, 5))
    print("Matched permutation p:", round(pair_p, 5))
    print("Max |controlled Spearman|:",
          round(float(pcorr.abs_spearman.max()), 5) if len(pcorr) else "NA")
    print("Conditional CCA1:",
          round(float(cca_first.canonical_correlation), 5))
    print("Conditional CCA1 squared correlation:",
          round(float(cca_first.squared_canonical_correlation), 5))
    print("Conditional CCA block-permutation p:",
          round(float(cca_perm.block_permutation_p.iloc[0]), 5))
    print("Mean |held-out conditional CCA1|:", round(mean_lodo_cca, 5))
    print("Conditional RV:", round(float(rv_test.conditional_rv.iloc[0]), 5))
    print("kNN excess overlap over null:",
          round(float(knn_test.excess_overlap_over_null.iloc[0]), 5))
    if len(mi):
        best_mi = mi.iloc[0]
        print("Max normalized excess MI:",
              round(float(best_mi.normalized_excess_mi), 5))
        print("Best MI pair:",
              best_mi.structural_feature, "<->", best_mi.calibration_feature)
        print("Best MI permutation p:",
              round(float(best_mi.permutation_p), 5))
        print("MI pairs surviving FDR q<0.05:",
              int((mi.permutation_q_fdr < 0.05).sum()))
    print("\nOutput:", root.resolve())


if __name__ == "__main__":
    main()
