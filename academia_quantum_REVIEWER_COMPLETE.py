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
8. Compare matched structural and calibration distances with matrix-coherent
   region-label permutations and group bootstrap procedures.
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
* Primary local interpretation uses k=1 and k=2; k=3 is retained as a
  secondary sensitivity scale. All six metric families are computed for k=1,2,3.
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

if _missing and not any(flag in sys.argv for flag in ("--overlap-cca-rv", "--overlap-mi", "--matched-threshold-audit", "--matched-global-disjoint", "--mi-full-audit", "--reviewer-audit", "--repair-archive", "--self-test")):
    raise RuntimeError(
        "Missing required Python packages: "
        + ", ".join(sorted(set(_missing)))
        + ". Install them in the active Python environment before running this analysis."
    )

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error

if not any(flag in sys.argv for flag in ("--overlap-cca-rv", "--overlap-mi", "--matched-threshold-audit", "--matched-global-disjoint", "--mi-full-audit", "--reviewer-audit", "--repair-archive", "--self-test")):
    import qiskit_ibm_runtime.fake_provider as fake_provider


# =============================================================================
# Frozen analysis configuration
# =============================================================================

ANALYSIS_VERSION = "reviewer-complete-v5-20261106"
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

# Primary local scales for central interpretation.
PRIMARY_K = [1, 2]
# k=3 is retained as a secondary sensitivity scale.
SECONDARY_K = [3]
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
    "KHEM_k1_mean", "KHEM_k1_std",
    "EWR_k1_mean", "EWR_k1_std",
    "NED_k1_mean", "NED_k1_std",
    "SSE_k1", "NSSE_k1", "GCV_k1",
    "growth_k1_mean", "growth_k1_std",

    "KHEM_k2_mean", "KHEM_k2_std",
    "EWR_k2_mean", "EWR_k2_std",
    "NED_k2_mean", "NED_k2_std",
    "SSE_k2", "NSSE_k2", "GCV_k2",
    "growth_k2_mean", "growth_k2_std",
]

STRUCTURAL_K_SECONDARY = [
    "KHEM_k3_mean", "KHEM_k3_std",
    "EWR_k3_mean", "EWR_k3_std",
    "NED_k3_mean", "NED_k3_std",
    "SSE_k3", "NSSE_k3", "GCV_k3",
    "growth_k3_mean", "growth_k3_std",
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

def khop_nodes(G, v, k):
    """Return nodes at distance 1..k from v, excluding v itself."""
    lengths = nx.single_source_shortest_path_length(G, v, cutoff=k)
    return [u for u, d in lengths.items() if 1 <= d <= k]


def shannon_entropy_from_weights(weights):
    """Shannon entropy in bits after normalizing positive finite weights."""
    w = np.asarray(weights, dtype=float)
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) == 0:
        return 0.0
    p = w / w.sum()
    return float(-np.sum(p * np.log2(p)))


def published_khem_node(G, v, k):
    """
    KHEM as defined in the SACI paper:
      p_u = deg_G(u) / sum_{x in N_k(v)} deg_G(x)
      KHEM_k(v) = -sum p_u log2(p_u)
    where N_k(v) contains nodes reachable from v in at most k steps,
    excluding v.
    """
    neigh = khop_nodes(G, v, k)
    if not neigh:
        return 0.0
    return shannon_entropy_from_weights([G.degree(u) for u in neigh])


def published_ewr_ned_node(G, v, k):
    """
    EWR and NED as defined in the Mathematics paper.

    The entropy component uses degrees inside the induced neighborhood
    subgraph G[N_k(v)]. Let H_local be that entropy and let
      R_k(v) = |E(G[N_k(v)])| / C(|N_k(v)|, 2).
    Then
      EWR_k(v) = H_local * R_k(v)
      NED_k(v) = H_local * R_k(v) / |N_k(v)|.
    """
    neigh = khop_nodes(G, v, k)
    nk = len(neigh)
    if nk == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    Hsub = G.subgraph(neigh)
    local_entropy = shannon_entropy_from_weights(
        [Hsub.degree(u) for u in neigh]
    )

    max_edges = nk * (nk - 1) / 2.0
    density = (
        float(Hsub.number_of_edges() / max_edges)
        if max_edges > 0 else 0.0
    )

    ewr = float(local_entropy * density)
    ned = float((local_entropy / nk) * density) if nk > 0 else 0.0
    return ewr, ned, local_entropy, density, nk


def khop_signatures(G, k):
    """
    Radial shell signatures for SSE/NSSE/GCV.

    s_k(v) = (n_1(v), ..., n_k(v)),
    where n_j(v) is the number of vertices at exact distance j.
    """
    signatures = []
    cumulative_growth = []

    for v in G.nodes():
        lengths = nx.single_source_shortest_path_length(G, v, cutoff=k)
        shell = tuple(
            sum(1 for d in lengths.values() if d == j)
            for j in range(1, k + 1)
        )
        signatures.append(shell)
        cumulative_growth.append(sum(shell))

    return signatures, np.asarray(cumulative_growth, dtype=float)


def six_metric_descriptors(G, return_node_rows=False):
    """
    Compute all six agreed metric families for k=1,2,3.

    Published metrics:
      KHEM, EWR, NED.

    New descriptors:
      SSE, NSSE, GCV.

    KHEM/EWR/NED are node-level metrics and are summarized at graph/region
    level by mean and population standard deviation. SSE/NSSE/GCV are
    graph/region-level descriptors.
    """
    n = G.number_of_nodes()
    out = {}
    node_rows = []

    for k in ALL_K:
        khem_vals, ewr_vals, ned_vals = [], [], []

        for v in G.nodes():
            khem = published_khem_node(G, v, k)
            ewr, ned, hlocal, density, nk = published_ewr_ned_node(G, v, k)

            khem_vals.append(khem)
            ewr_vals.append(ewr)
            ned_vals.append(ned)

            if return_node_rows:
                node_rows.append({
                    "node": int(v),
                    "k": int(k),
                    "KHEM": float(khem),
                    "EWR": float(ewr),
                    "NED": float(ned),
                    "EWR_NED_local_entropy": float(hlocal),
                    "neighborhood_density": float(density),
                    "neighborhood_size": int(nk),
                })

        out[f"KHEM_k{k}_mean"] = finite_mean(khem_vals)
        out[f"KHEM_k{k}_std"] = finite_std(khem_vals)
        out[f"EWR_k{k}_mean"] = finite_mean(ewr_vals)
        out[f"EWR_k{k}_std"] = finite_std(ewr_vals)
        out[f"NED_k{k}_mean"] = finite_mean(ned_vals)
        out[f"NED_k{k}_std"] = finite_std(ned_vals)

        signatures, growth = khop_signatures(G, k)
        sse = entropy_discrete(signatures)
        nsse = (
            float(sse / math.log2(n))
            if n > 1 and math.log2(n) > 0 else 0.0
        )
        growth_mean = finite_mean(growth)
        growth_std = finite_std(growth)
        gcv = (
            float(growth_std / growth_mean)
            if np.isfinite(growth_mean) and growth_mean > 0 else 0.0
        )

        out[f"SSE_k{k}"] = float(sse)
        out[f"NSSE_k{k}"] = float(nsse)
        out[f"GCV_k{k}"] = float(gcv)
        out[f"growth_k{k}_mean"] = float(growth_mean)
        out[f"growth_k{k}_std"] = float(growth_std)

    return (out, node_rows) if return_node_rows else out


def metric_definition_self_tests():
    """Fail fast if any metric definition drifts from the agreed formulas."""
    # SACI illustrative example: star center with three leaf neighbors.
    Gs = nx.Graph()
    Gs.add_edges_from([("B", "A"), ("B", "C"), ("B", "D")])
    assert np.isclose(
        published_khem_node(Gs, "B", 1), math.log2(3), atol=1e-12
    )

    # Triangle at k=1: KHEM=1, local entropy=1, density=1, EWR=1, NED=1/2.
    Gt = nx.complete_graph(3)
    assert np.isclose(published_khem_node(Gt, 0, 1), 1.0, atol=1e-12)
    ewr, ned, hloc, dens, nk = published_ewr_ned_node(Gt, 0, 1)
    assert nk == 2
    assert np.isclose(hloc, 1.0, atol=1e-12)
    assert np.isclose(dens, 1.0, atol=1e-12)
    assert np.isclose(ewr, 1.0, atol=1e-12)
    assert np.isclose(ned, 0.5, atol=1e-12)

    # Path-4: hand-checkable SSE/NSSE/GCV at k=1,2,3.
    Gp = nx.path_graph(4)
    m = six_metric_descriptors(Gp)
    assert np.isclose(m["SSE_k1"], 1.0, atol=1e-12)
    assert np.isclose(m["NSSE_k1"], 0.5, atol=1e-12)
    assert np.isclose(m["GCV_k1"], 1.0 / 3.0, atol=1e-12)
    assert np.isclose(m["SSE_k2"], 1.0, atol=1e-12)
    assert np.isclose(m["GCV_k2"], 0.2, atol=1e-12)
    assert np.isclose(m["SSE_k3"], 1.0, atol=1e-12)
    assert np.isclose(m["GCV_k3"], 0.0, atol=1e-12)


def structural_descriptors(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg = np.asarray([d for _, d in G.degree()], dtype=float)
    deg_mean = finite_mean(deg)
    deg_std = finite_std(deg)

    out = {
        "n": n,
        "m": m,
        "density": nx.density(G) if n > 1 else 0.0,
        "avg_degree": deg_mean,
        "degree_std": deg_std,
        "degree_cv": deg_std / deg_mean
            if np.isfinite(deg_mean) and deg_mean > 0 else 0.0,
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

    out.update(six_metric_descriptors(G))

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
        used_physical_sets = set()
        for j in range(PATCHES_PER_DEVICE_WIDTH):
            for retry in range(20001):
                seed = (stable_seed(MASTER_SEED, device, width, j)
                        if retry == 0 else stable_seed(MASTER_SEED, device, width, j, 'deduplicate', retry))
                rng = np.random.default_rng(seed)
                P, nodes = connected_patch(G, width, rng)
                physical = frozenset(nodes)
                if physical not in used_physical_sets:
                    used_physical_sets.add(physical)
                    break
            else:
                raise RuntimeError(f'Duplicate-free patch sampling failed: {device}, width={width}')
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
    """Region-label (matrix-conjugation) permutation within exact-degree groups.

    Pairwise distances are NOT exchangeable as individual observations.
    The null relabels the *regions* on the calibration side, equivalently
    D_Q -> P D_Q P.T, while retaining each group's distance geometry.

    IMPORTANT: overlapping physical regions can still violate region-level
    exchangeability. Treat this as a matrix-coherent sensitivity test, not a
    complete correction for overlapping-region dependence.
    """
    observed, p_naive = spearmanr(
        pairs.structural_distance, pairs.calibration_distance
    )
    if not np.isfinite(observed):
        raise ValueError("Matched-distance Spearman statistic is not finite")

    rng = np.random.default_rng(stable_seed(MASTER_SEED, "region_label_permutation"))
    group_arrays = []
    group_size_counts = Counter()
    for gid, g in pairs.groupby("group_id", sort=True):
        ids = sorted(set(g.patch_a.astype(str)) | set(g.patch_b.astype(str)))
        n = len(ids)
        group_size_counts[n] += 1
        index = {pid: i for i, pid in enumerate(ids)}
        ds = np.full((n, n), np.nan)
        dq = np.full((n, n), np.nan)
        np.fill_diagonal(ds, 0.0)
        np.fill_diagonal(dq, 0.0)
        for r in g.itertuples(index=False):
            i, j = index[str(r.patch_a)], index[str(r.patch_b)]
            ds[i, j] = ds[j, i] = float(r.structural_distance)
            dq[i, j] = dq[j, i] = float(r.calibration_distance)
        ii, jj = np.triu_indices(n, 1)
        if len(g) != len(ii) or not np.isfinite(ds[ii, jj]).all() or not np.isfinite(dq[ii, jj]).all():
            raise ValueError(f"Incomplete pairwise distance matrix in group {gid}")
        group_arrays.append((ds[ii, jj], dq, ii, jj, n))

    x = np.concatenate([a[0] for a in group_arrays])
    y = np.concatenate([a[1][a[2], a[3]] for a in group_arrays])
    if not np.isclose(spearmanr(x, y).statistic, observed, atol=1e-12):
        raise AssertionError("Reconstructed pair ordering changed observed rho")

    perm_stats = np.empty(n_perm, dtype=float)
    for rep in range(n_perm):
        y_parts = []
        for _, dq, ii, jj, n in group_arrays:
            perm = rng.permutation(n)
            y_parts.append(dq[perm[ii], perm[jj]])
        perm_stats[rep] = spearmanr(x, np.concatenate(y_parts)).statistic

    p_perm = float((1 + np.count_nonzero(np.abs(perm_stats) >= abs(observed))) /
                   (n_perm + 1))
    perm_center = float(np.mean(perm_stats))
    p_centered = float((1 + np.count_nonzero(np.abs(perm_stats - perm_center) >= abs(observed - perm_center))) / (n_perm + 1))
    return pd.DataFrame([{
        "n_pairs": len(pairs),
        "n_groups": pairs.group_id.nunique(),
        "n_groups_size_2": group_size_counts.get(2, 0),
        "n_groups_size_ge_3": sum(v for k, v in group_size_counts.items() if k >= 3),
        "spearman_rho": float(observed),
        "naive_p": float(p_naive),
        "group_aware_permutation_p": p_perm,
        "group_aware_centered_permutation_p": p_centered,
        "permutation_scheme": "within_group_region_label_matrix_conjugation",
        "overlap_exchangeability_caveat": True,
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
    Residualize on predefined region-size and first-order connectivity summaries separately within each device.

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
    CCA on WITHIN-DEVICE, RESIDUALIZED feature spaces.

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
    other layer after residualization for the predefined region-size and first-order connectivity summaries?
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
                n_neighbors=3,
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
                    n_neighbors=3,
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
    metric_definition_self_tests()

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

    directed_gate_rows = []

    for device in CALIBRATION_VALID_BACKENDS:
        print("  ", device)
        backend, G, qdf, meta = backend_graph_and_properties(device)
        backend_graphs[device] = G
        # Record native two-qubit instruction orientations BEFORE undirected averaging.
        # Gate-family and direction sensitivity requires a fresh Qiskit snapshot extraction.
        for op in getattr(backend.target,'operation_names',[]):
            try: entries=backend.target[op]
            except Exception: continue
            if not hasattr(entries,'items'):continue
            for qubits,properties in entries.items():
                if not qubits or len(qubits)!=2:continue
                try: a,b=map(int,qubits)
                except Exception:continue
                err=safe_float(getattr(properties,'error',np.nan)) if properties is not None else np.nan
                duration=safe_float(getattr(properties,'duration',np.nan)) if properties is not None else np.nan
                directed_gate_rows.append(dict(device=device,gate=str(op),source=a,target=b,
                    u=min(a,b),v=max(a,b),error=err,duration=duration))

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
    directed=pd.DataFrame(directed_gate_rows,columns=['device','gate','source','target','u','v','error','duration'])
    directed.to_csv(dirs['tables']/'directed_twoqubit_native_instructions.csv',index=False)
    # Pair orientation-specific entries only within identical gate families.
    direction_rows=[]
    if len(directed):
        for key,g in directed.groupby(['device','gate','u','v']):
            means=g.groupby(['source','target']).error.mean().dropna()
            orientation_gap=(float(abs(means.iloc[0]-means.iloc[1]))
                             if len(means)==2 else np.nan)
            direction_rows.append(dict(device=key[0],gate=key[1],u=key[2],v=key[3],
                n_orientations=int(len(means)),abs_orientation_error_gap=orientation_gap))
    pd.DataFrame(direction_rows,columns=['device','gate','u','v','n_orientations','abs_orientation_error_gap']).to_csv(
        dirs['tables']/'directed_gate_orientation_gaps.csv',index=False)


    # Full-QPU six-metric exports (k=1,2,3).
    qpu_metric_rows = []
    qpu_node_metric_rows = []
    for device, G in backend_graphs.items():
        metric_values, node_values = six_metric_descriptors(
            G, return_node_rows=True
        )
        row = {
            "device": device,
            "n_qubits": G.number_of_nodes(),
            "n_couplings": G.number_of_edges(),
        }
        row.update(metric_values)
        qpu_metric_rows.append(row)

        for r in node_values:
            qpu_node_metric_rows.append({"device": device, **r})

    pd.DataFrame(qpu_metric_rows).to_csv(
        dirs["tables"] / "qpu_six_metrics_k1_k2_k3.csv", index=False
    )
    pd.DataFrame(qpu_node_metric_rows).to_csv(
        dirs["tables"] / "qpu_published_khem_ewr_ned_node_metrics_k1_k2_k3.csv",
        index=False,
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
    patch_node_metric_rows = []

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

        _, node_metric_rows = six_metric_descriptors(P, return_node_rows=True)
        for r in node_metric_rows:
            patch_node_metric_rows.append({
                "patch_id": pid,
                "device": meta.device,
                "width": int(meta.width),
                **r,
            })

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

    # Compact six-metric patch table, in addition to the full structural table.
    six_metric_cols = [
        c for c in sdf.columns
        if c.startswith(("KHEM_", "EWR_", "NED_", "SSE_", "NSSE_", "GCV_"))
    ]
    sdf[["patch_id", "device", "width"] + six_metric_cols].to_csv(
        dirs["tables"] / "patch_six_metrics_k1_k2_k3.csv", index=False
    )
    pd.DataFrame(patch_node_metric_rows).to_csv(
        dirs["tables"] / "published_khem_ewr_ned_node_metrics_k1_k2_k3.csv",
        index=False,
    )

    merged = sdf.merge(
        qdf.drop(columns=["device", "width"]),
        on="patch_id",
        how="inner",
        validate="one_to_one",
    )

    merged.to_csv(
        dirs["tables"] / "merged_patch_dataset.csv", index=False
    )

    descriptor_dictionary = pd.DataFrame([
        {"family": "KHEM", "status": "published", "level": "node -> region mean/std",
         "definition": "SACI KHEM; probabilities proportional to original-graph degrees within N_k(v)"},
        {"family": "EWR", "status": "published", "level": "node -> region mean/std",
         "definition": "Mathematics EWR; local induced-neighborhood degree entropy multiplied by neighborhood density"},
        {"family": "NED", "status": "published", "level": "node -> region mean/std",
         "definition": "Mathematics NED; EWR additionally normalized by neighborhood size"},
        {"family": "SSE", "status": "new", "level": "region",
         "definition": "Shannon entropy of empirical radial shell-signature classes"},
        {"family": "NSSE", "status": "new", "level": "region",
         "definition": "SSE divided by log2(region size)"},
        {"family": "GCV", "status": "new", "level": "region",
         "definition": "population SD / mean of cumulative k-hop reach across region vertices"},
    ])
    descriptor_dictionary.to_csv(
        dirs["tables"] / "descriptor_dictionary.csv", index=False
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
            "KHEM_k2_mean",
            "EWR_k2_mean",
            "NED_k2_mean",
            "SSE_k2",
            "NSSE_k2",
            "GCV_k2",
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
        "metric_families": ["KHEM", "EWR", "NED", "SSE", "NSSE", "GCV"],
        "all_k": ALL_K,
        "mi_n_neighbors": 3,
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
Created UTC: {now_utc()}

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

Primary local k-hop interpretation: k={PRIMARY_K}\nSecondary sensitivity: k={SECONDARY_K}\nSix metric families: KHEM, EWR, NED, SSE, NSSE, GCV

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





# ================================================================
# INTEGRATED OVERLAP ROBUSTNESS EXTENSION (2026-11-06)
# Run on archived outputs, without requiring Qiskit:
#   python this_file.py --overlap-cca-rv --archive ARCHIVE.zip --out DIR --seeds 40 --permutations 999
#   python this_file.py --overlap-mi --archive ARCHIVE.zip --selections DIR/overlap_40seeds_selected_patches.csv --out DIR --seeds 8 --permutations 99
# Default invocation (no overlap flag) retains the original full pipeline.
# Caution: within-device/width permutations do not prove exchangeability;
# disjoint selections are sensitivity checks, not independent replicates.
# ================================================================
"""Standalone overlap sensitivity recheck using archived IBM QPU patch descriptors.
Does not require Qiskit when the archived tables are available.
"""
import io, zipfile, hashlib, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

S_CORE=['avg_path','diameter','global_efficiency','avg_clustering','transitivity','spectral_radius','algebraic_connectivity','laplacian_radius','adjacency_energy']
S_K=[f'{m}_k{k}{suffix}' for k in (1,2) for m,suffixes in [('KHEM',['_mean','_std']),('EWR',['_mean','_std']),('NED',['_mean','_std']),('SSE',['']),('NSSE',['']),('GCV',['']),('growth',['_mean','_std'])] for suffix in suffixes]
Q=['t1_mean','t1_std','t2_mean','t2_std','readout_error_mean','readout_error_std','oneq_error_mean','oneq_error_std','twoq_error_mean','twoq_error_std']
CTRL=['width','m','avg_degree','degree_std','degree_cv']
def overlap_load(z, name):
 with zipfile.ZipFile(z) as a:
  members=[x for x in a.namelist() if x.endswith('/tables/'+name+'.csv')]
  if len(members)!=1:raise ValueError(f'{name}: {members}')
  return pd.read_csv(io.BytesIO(a.read(members[0])))
def overlap_residual(df, cols):
 out=pd.DataFrame(index=df.index,columns=cols,dtype=float)
 for _,g in df.groupby('device'):
  xx=SimpleImputer(strategy='median').fit_transform(g[CTRL]);xx=np.column_stack([np.ones(len(g)),StandardScaler().fit_transform(xx)])
  for c in cols:
   y=pd.to_numeric(g[c],errors='coerce').to_numpy(float);valid=np.isfinite(y)
   if valid.sum()>=xx.shape[1]+5:
    beta=np.linalg.lstsq(xx[valid],y[valid],rcond=None)[0];out.loc[g.index[valid],c]=y[valid]-xx[valid]@beta
 return pd.DataFrame(SimpleImputer(strategy='median').fit_transform(out),columns=cols,index=df.index)
def overlap_rv(x,y):
 x=x-x.mean(0);y=y-y.mean(0)
 return float(np.sum((x.T@y)**2)/np.sqrt(np.sum((x.T@x)**2)*np.sum((y.T@y)**2)))
def overlap_cca3(x,y):
 # Inputs already PCA-reduced to three components per block.
 a=np.cov(x,rowvar=False);b=np.cov(y,rowvar=False);c=np.cov(x.T,y.T)[:3,3:]
 def invsqrt(m):
  e,v=np.linalg.eigh(m);return (v*(1/np.sqrt(np.maximum(e,1e-12))))@v.T
 return float(min(1.,np.linalg.svd(invsqrt(a)@c@invsqrt(b),compute_uv=False)[0]))
def overlap_cca_rv_main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--out',default='/mnt/data');p.add_argument('--seeds',type=int,default=40);p.add_argument('--permutations',type=int,default=499);args=p.parse_args()
 df=overlap_load(args.archive,'merged_patch_dataset').reset_index(drop=True);patch=overlap_load(args.archive,'patches')
 S=overlap_residual(df,S_CORE+S_K);T=overlap_residual(df,Q)
 patch['node_set']=patch.nodes.astype(str).map(lambda x:set(x.split('|')))
 patch=patch.set_index('patch_id').loc[df.patch_id].reset_index()
 rows=[]; selections=[]
 for seed in range(args.seeds):
  rng=np.random.default_rng(20261106+seed);chosen=[]
  # Enforce physical disjointness ACROSS ALL WIDTHS within each device.
  for device,g in patch.groupby('device',sort=True):
   used=set()
   for ix in rng.permutation(g.index.to_numpy()):
    nodes=patch.at[ix,'node_set']
    if used.isdisjoint(nodes):
     chosen.append(ix);used.update(nodes)
  idx=np.sort(np.asarray(chosen,dtype=int));sub=df.iloc[idx].reset_index(drop=True)
  # Fail loudly if any selected regions share physical qubits.
  for device,g in patch.iloc[idx].groupby('device'):
   seen=set()
   for nodes in g.node_set:
    assert seen.isdisjoint(nodes), f'Overlapping selected regions on {device}'
    seen.update(nodes)
  x=StandardScaler().fit_transform(S.iloc[idx]);y=StandardScaler().fit_transform(T.iloc[idx])
  blocks=[np.asarray(list(v),dtype=int) for v in sub.groupby(['device','width']).groups.values()]
  if len(idx)<10 or not any(len(b)>1 for b in blocks):
   raise ValueError(f'No informative permutation blocks: seed={seed}')
  xp=PCA(n_components=3).fit_transform(x); ypca=PCA(n_components=3).fit_transform(y)
  ob_rv=overlap_rv(x,y);ob_cca=overlap_cca3(xp,ypca);nr=[];nc=[]
  for j in range(args.permutations):
   order=np.arange(len(idx))
   for block in blocks:order[block]=rng.permutation(block)
   yp=y[order];nr.append(overlap_rv(x,yp));nc.append(overlap_cca3(xp,ypca[order]))
  rows.append(dict(seed=seed,n=len(idx),min_block=min(map(len,blocks)),permutable_regions=sum(len(b) for b in blocks if len(b)>1),singleton_blocks=sum(len(b)==1 for b in blocks),rv=ob_rv,rv_null_mean=np.mean(nr),rv_excess=ob_rv-np.mean(nr),rv_p=(1+sum(t>=ob_rv for t in nr))/(1+args.permutations),cca_pca3=ob_cca,cca_null_mean=np.mean(nc),cca_excess=ob_cca-np.mean(nc),cca_p=(1+sum(t>=ob_cca for t in nc))/(1+args.permutations)))
  selections.extend(dict(seed=seed,patch_id=pid) for pid in sub.patch_id)
  if seed%5==0:print('completed',seed+1,'/',args.seeds,flush=True)
 out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
 pd.DataFrame(rows).to_csv(out/'overlap_40seeds_cca_rv.csv',index=False)
 pd.DataFrame(selections).to_csv(out/'overlap_40seeds_selected_patches.csv',index=False)
 print(pd.DataFrame(rows)[['n','rv','rv_p','cca_pca3','cca_p']].describe().to_string())



"""Pre-specified exploratory MI overlap sensitivity on archived patch descriptors."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

PAIRS=[('growth_k2_mean','twoq_error_std'),('SSE_k1','twoq_error_std'),('NSSE_k1','twoq_error_std'),('growth_k2_mean','twoq_error_mean'),('growth_k2_mean','oneq_error_std'),('avg_clustering','t1_mean')]
def overlap_mi_main():
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--selections',required=True);p.add_argument('--out',default='/mnt/data');p.add_argument('--seeds',type=int,default=12);p.add_argument('--permutations',type=int,default=199);a=p.parse_args()
 df=overlap_load(a.archive,'merged_patch_dataset'); sels=pd.read_csv(a.selections)
 sc=sorted(set(x for x,y in PAIRS));qc=sorted(set(y for x,y in PAIRS))
 S=overlap_residual(df,sc);Q=overlap_residual(df,qc)
 rows=[]
 for seed in sorted(sels.seed.unique())[:a.seeds]:
  ids=sels[sels.seed==seed].patch_id.tolist();sub=df.set_index('patch_id').loc[ids].reset_index()
  ix=pd.Index(df.patch_id).get_indexer(ids); assert (ix>=0).all() and len(set(ix))==len(ix)
  blocks=[np.array(list(v),dtype=int) for v in sub.groupby(['device','width']).groups.values()]
  for pair_id,(s,q) in enumerate(PAIRS):
   x=S.iloc[ix][s].to_numpy(float);y=Q.iloc[ix][q].to_numpy(float)
   for k in (3,5):
    rng=np.random.default_rng(20261106+seed*1000+pair_id*10+k)
    def est(yy):return float(mutual_info_regression(x.reshape(-1,1),yy,n_neighbors=k,random_state=73)[0])
    obs=est(y);null=[]
    for _ in range(a.permutations):
     order=np.arange(len(y))
     for b in blocks:
      if len(b)>1:order[b]=rng.permutation(b)
     null.append(est(y[order]))
    rows.append(dict(seed=int(seed),structural_feature=s,calibration_feature=q,k_mi=k,n=len(x),observed_mi=obs,null_mean=float(np.mean(null)),excess_mi=obs-np.mean(null),p=(1+np.count_nonzero(np.array(null)>=obs))/(a.permutations+1)))
  print('MI completed seed',seed,flush=True)
 res=pd.DataFrame(rows)
 # FDR correction within each selection and estimator setting, six candidate pairs.
 for (seed,k),g in res.groupby(['seed','k_mi']):
  res.loc[g.index,'q_bh_six_pairs']=benjamini_hochberg(g.p.to_numpy(float))
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
 res.to_csv(out/'overlap_MI_robustness.csv',index=False)
 summary=res.groupby(['structural_feature','calibration_feature','k_mi']).agg(seeds=('seed','size'),median_excess=('excess_mi','median'),median_p=('p','median'),nominal_significant=('p',lambda x:int((x<.05).sum())),fdr_significant=('q_bh_six_pairs',lambda x:int((x<.05).sum()))).reset_index()
 summary.to_csv(out/'overlap_MI_summary.csv',index=False)
 print(summary.to_string(index=False))



def matched_group_threshold_audit(pairs, thresholds=(2, 3, 4, 5), n_perm=5000):
    """Recompute exact-degree matched-distance inference at increasing group sizes.

    The calibration distance matrix is relabeled by one common region permutation
    per group; individual condensed distance entries are never shuffled.
    """
    group_sizes = pairs.groupby('group_id').apply(
        lambda g: len(set(g.patch_a) | set(g.patch_b)), include_groups=False)
    results = []
    for min_size in thresholds:
        subset = pairs[pairs.group_id.isin(group_sizes[group_sizes >= min_size].index)]
        result = matched_pair_permutation_test(subset, n_perm=n_perm)
        result.insert(0, 'minimum_group_size', min_size)
        results.append(result)
    return pd.concat(results, ignore_index=True)


def matched_threshold_archive_cli():
    """Run the sensitivity audit directly from a saved analysis archive."""
    import argparse, zipfile, io
    parser = argparse.ArgumentParser()
    parser.add_argument('--matched-threshold-audit', action='store_true')
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', default='matched_threshold_sensitivity_results.csv')
    parser.add_argument('--permutations', type=int, default=5000)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        matches = [name for name in archive.namelist() if name.endswith('/tables/matched_pair_distances.csv')]
        if len(matches) != 1:
            raise ValueError('Expected exactly one matched_pair_distances.csv in archive')
        pairs = pd.read_csv(io.BytesIO(archive.read(matches[0])))
    result = matched_group_threshold_audit(pairs, n_perm=args.permutations)
    result.to_csv(args.out, index=False)
    print(result.to_string(index=False))


def global_disjoint_audit(pairs,nodes,seeds=20,permutations=499):
    meta=nodes.groupby('patch_id').agg(device=('device','first'),qubits=('node',lambda x:frozenset(x))).to_dict('index')
    groups={gid:g for gid,g in pairs.groupby('group_id',sort=True)}
    patch_to_group={p:gid for gid,g in groups.items() for p in set(g.patch_a)|set(g.patch_b)}
    devices={d:[p for p in patch_to_group if meta[p]['device']==d] for d in sorted(set(x['device'] for x in meta.values()))}
    results=[];selected_rows=[]
    for seed in range(seeds):
        rng=np.random.default_rng(20261106+seed);selected=[]
        for dev,ids in devices.items():
            used=set()
            for p in rng.permutation(ids):
                qubits=meta[p]['qubits']
                if used.isdisjoint(qubits):selected.append(p);used.update(qubits)
        chosen=set(selected); mats=[]
        for gid,g in groups.items():
            ids=sorted((set(g.patch_a)|set(g.patch_b))&chosen);n=len(ids)
            if n<2:continue
            gg=g[g.patch_a.isin(ids)&g.patch_b.isin(ids)];ii,jj=np.triu_indices(n,1)
            if len(gg)!=len(ii):raise AssertionError('incomplete')
            ix={p:i for i,p in enumerate(ids)};ds=np.zeros((n,n));dq=np.zeros((n,n))
            for r in gg.itertuples(index=False):
                i,j=ix[r.patch_a],ix[r.patch_b];ds[i,j]=ds[j,i]=r.structural_distance;dq[i,j]=dq[j,i]=r.calibration_distance
            mats.append((ds[ii,jj],dq,ii,jj,n))
        if not mats:
            results.append(dict(seed=seed,n_regions=len(selected),n_groups=0,n_pairs=0,rho=np.nan,null_mean=np.nan,p_centered=np.nan));continue
        x=np.concatenate([m[0] for m in mats]);y=np.concatenate([m[1][m[2],m[3]] for m in mats]);rx=rankdata(x);rx=(rx-rx.mean())/np.linalg.norm(rx-rx.mean());ry=rankdata(y);ry=(ry-ry.mean())/np.linalg.norm(ry-ry.mean());obs=float(rx@ry)
        null=np.empty(permutations)
        for b in range(permutations):
            yy=np.concatenate([dq[(p:=rng.permutation(n))[ii],p[jj]] for _,dq,ii,jj,n in mats]);rr=rankdata(yy);rr=(rr-rr.mean())/np.linalg.norm(rr-rr.mean());null[b]=rx@rr
        m=float(null.mean());pv=(1+np.count_nonzero(abs(null-m)>=abs(obs-m)))/(permutations+1)
        results.append(dict(seed=seed,n_regions=len(selected),n_groups=len(mats),n_pairs=len(x),rho=obs,null_mean=m,p_centered=pv,B=permutations))
        selected_rows.extend(dict(seed=seed,patch_id=p,device=meta[p]['device'],group_id=patch_to_group[p]) for p in selected)
    return pd.DataFrame(results),pd.DataFrame(selected_rows)



# =============================================================================
# Reviewer-facing archived-data diagnostics (no Qiskit dependency)
# =============================================================================

def reviewer_archive_tables(archive_path):
    """Load all data needed for reviewer diagnostics from one frozen archive."""
    import zipfile, io
    with zipfile.ZipFile(archive_path) as z:
        def get(name):
            matches = [f for f in z.namelist() if f.endswith('/tables/'+name+'.csv')]
            if len(matches) != 1:
                raise ValueError(f'Reviewer archive: expected exactly one {name}.csv, got {len(matches)}')
            return pd.read_csv(io.BytesIO(z.read(matches[0])))
        data = {name:get(name) for name in ('merged_patch_dataset', 'patch_nodes', 'edges',
                  'devices', 'patches')}
        data['mutual_information_permutation_normalized'] = (get('mi_pair_registry')
            if any(f.endswith('/tables/mi_pair_registry.csv') for f in z.namelist())
            else get('mutual_information_permutation_normalized'))
    data['merged_patch_dataset'] = data['merged_patch_dataset'].reset_index(drop=True)
    if data['merged_patch_dataset'].patch_id.duplicated().any():
        raise ValueError('Duplicate patch_id in merged dataset')
    return data


def reviewer_residual_redundancy(df, s_cols, threshold=0.95):
    """Descriptive correlation audit; reduced sets for LODO are fitted on TRAIN folds."""
    R = residualize_within_device(df, s_cols)
    rows=[]
    for i,a in enumerate(s_cols):
        x=R[a].to_numpy(dtype=float)
        for b in s_cols[i+1:]:
            y=R[b].to_numpy(dtype=float); valid=np.isfinite(x)&np.isfinite(y)
            rho=(float(spearmanr(x[valid],y[valid]).statistic)
                 if valid.sum()>10 and np.std(x[valid])>1e-9 and np.std(y[valid])>1e-9
                 else np.nan)
            rows.append(dict(feature_a=a,feature_b=b,n=int(valid.sum()),
                             spearman_rho=rho,abs_rho=abs(rho) if np.isfinite(rho) else np.nan,
                             redundant_candidate=bool(np.isfinite(rho) and abs(rho)>=threshold)))
    return pd.DataFrame(rows)


def reviewer_prune_train_only(frame, cols, threshold=0.95):
    """Greedy high-correlation filter learned on the TRAIN device folds only."""
    from scipy.stats import spearmanr as _spearman
    keep=[]
    for c in cols:
        x=pd.to_numeric(frame[c],errors='coerce').to_numpy(float)
        if np.isfinite(x).sum()<10 or np.nanstd(x)<1e-9:continue
        redundant=False
        for previous in keep:
            y=pd.to_numeric(frame[previous],errors='coerce').to_numpy(float)
            ok=np.isfinite(x)&np.isfinite(y)
            if ok.sum()>10 and np.nanstd(y[ok])>1e-9:
                rho=_spearman(x[ok],y[ok]).statistic
                if np.isfinite(rho) and abs(rho)>=threshold:
                    redundant=True;break
        if not redundant:keep.append(c)
    return keep


def reviewer_ablation_external_lodo(df, s_cols, q_cols):
    """TRUE external held-device reconstruction: raw targets and train-only transforms.

    Unlike conditional within-held-device residualization, this never uses held-out
    calibration targets for data preprocessing, feature selection, or scaling.
    """
    from sklearn.pipeline import make_pipeline
    controls=list(CONNECTIVITY_CONTROLS)
    standard=[c for c in STRUCTURAL_CORE if c in s_cols]
    multiscale=[c for c in STRUCTURAL_K_PRIMARY if c in s_cols]
    records=[]
    for held in sorted(df.device.unique()):
        train=df[df.device!=held];test=df[df.device==held]
        pruned=reviewer_prune_train_only(train,s_cols)
        models={'controls_only':[], 'conventional_only':standard,
                'multiscale_only':multiscale,'full':s_cols,
                'pruned_train_only':pruned}
        for model, features in models.items():
            inp=list(dict.fromkeys(controls+features))
            for outcome in q_cols:
                ytr=pd.to_numeric(train[outcome],errors='coerce').to_numpy(float)
                yte=pd.to_numeric(test[outcome],errors='coerce').to_numpy(float)
                tr_good=np.isfinite(ytr);te_good=np.isfinite(yte)
                if tr_good.sum()<30 or te_good.sum()<10:
                    records.append(dict(held_out_device=held,model=model,target=outcome,
                          n_test=int(te_good.sum()),n_features=len(features),r2=np.nan,
                          nmae_train_sd=np.nan,status='insufficient_target'));continue
                Xtr=train.loc[tr_good,inp].apply(pd.to_numeric,errors='coerce')
                Xte=test.loc[te_good,inp].apply(pd.to_numeric,errors='coerce')
                if Xtr.isna().all(axis=0).any():
                    good_columns=list(Xtr.columns[~Xtr.isna().all(axis=0)])
                    Xtr=Xtr[good_columns];Xte=Xte[good_columns]
                if not Xtr.shape[1]:continue
                pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),Ridge(alpha=10.0))
                pipe.fit(Xtr,ytr[tr_good])
                prediction=pipe.predict(Xte)
                records.append(dict(held_out_device=held,model=model,target=outcome,
                    n_test=int(te_good.sum()),n_features=len(features),
                    r2=float(r2_score(yte[te_good],prediction)),
                    nmae_train_sd=float(mean_absolute_error(yte[te_good],prediction)/max(np.std(ytr[tr_good]),1e-15)),
                    status='ok'))
    return pd.DataFrame(records)


def reviewer_crossfit_controls(df, features, kind, folds=5, trees=96):
    """Within-device out-of-fold control fitting, avoiding in-sample overfitting.

    Note: OOF residuals remain within-snapshot diagnostics and do not establish
    exchangeability of physically overlapping regions.
    """
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    result=pd.DataFrame(index=df.index,columns=features,dtype=float)
    for device,g in df.groupby('device',sort=True):
        positions=np.asarray(g.index)
        X=g[list(CONNECTIVITY_CONTROLS)].apply(pd.to_numeric,errors='coerce')
        Y=g[features].apply(pd.to_numeric,errors='coerce')
        splitter=KFold(n_splits=min(folds,len(g)//10),shuffle=True,
                        random_state=stable_seed(MASTER_SEED,'reviewer_oof_split',device))
        for fold,(tr,te) in enumerate(splitter.split(X)):
            ctrl=SimpleImputer(strategy='median').fit(X.iloc[tr])
            xx=ctrl.transform(X.iloc[tr]);xte=ctrl.transform(X.iloc[te])
            # Multi-output fitting is efficient; impute training labels ONLY.
            targets=Y.iloc[tr].copy()
            med=targets.median(axis=0).fillna(0.0)
            filled=targets.fillna(med).to_numpy(float)
            if kind=='nonlinear':
                model=ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=5,
                    max_features=1.0,random_state=stable_seed(MASTER_SEED,'reviewer_extra',device,fold),n_jobs=1)
            elif kind=='linear':
                model=make_pipeline(StandardScaler(),Ridge(alpha=1.0))
            else:raise ValueError(kind)
            model.fit(xx,filled)
            predictions=model.predict(xte)
            actual=Y.iloc[te].to_numpy(float)
            result.loc[positions[te],features]=actual-predictions
    return result


def reviewer_residual_sensitivity(df,s_cols,q_cols):
    """Compare in-sample linear and OOF linear/nonlinear control regimes descriptively."""
    results=[]
    methods={
       'linear_within_device_in_sample':(residualize_within_device(df,s_cols),residualize_within_device(df,q_cols)),
       'linear_within_device_oof':(reviewer_crossfit_controls(df,s_cols,'linear'),reviewer_crossfit_controls(df,q_cols,'linear')),
       'nonlinear_extra_trees_oof':(reviewer_crossfit_controls(df,s_cols,'nonlinear'),reviewer_crossfit_controls(df,q_cols,'nonlinear'))
    }
    for label,(S,Q) in methods.items():
        # Guard against constant coordinates: RV is descriptive, dimensionality-dependent.
        valid_s=[c for c in s_cols if np.nanstd(S[c].to_numpy(float))>=1e-9]
        valid_q=[c for c in q_cols if np.nanstd(Q[c].to_numpy(float))>=1e-9]
        def clean(frame,cols):
            Z=SimpleImputer(strategy='median').fit_transform(frame[cols])
            return StandardScaler().fit_transform(Z)
        a=clean(S,valid_s);b=clean(Q,valid_q)
        correlations=[]
        for i in range(a.shape[1]):
            for j in range(b.shape[1]):
                correlations.append(abs(float(spearmanr(a[:,i],b[:,j]).statistic)))
        results.append(dict(control_model=label,n_structural=len(valid_s),n_calibration=len(valid_q),
               rv_descriptive=float(rv_coefficient(a,b)),
               median_abs_pairwise_spearman=float(np.nanmedian(correlations)),
               p90_abs_pairwise_spearman=float(np.nanpercentile(correlations,90)),
               max_abs_pairwise_spearman=float(np.nanmax(correlations)),
               inference='descriptive_only_not_permutation_p'))
    return pd.DataFrame(results)


def reviewer_overlap_inventory(patch_nodes):
    """Physical overlap across all region widths; not merely within each width."""
    records=[]
    for device,g in patch_nodes.groupby('device',sort=True):
        nodes=sorted(set(int(u) for u in g.node))
        index={n:i for i,n in enumerate(nodes)}
        patches=list(g.groupby('patch_id',sort=True))
        mat=np.zeros((len(patches),len(nodes)),dtype=np.int16)
        widths=[];sets=[]
        for i,(name,p) in enumerate(patches):
            qubits=set(map(int,p.node));sets.append(frozenset(qubits))
            mat[i,[index[u] for u in qubits]]=1;widths.append(int(p.width.iloc[0]))
        inter=mat@mat.T
        sizes=mat.sum(axis=1)
        ii,jj=np.triu_indices(len(patches),1)
        ww=np.asarray(widths)
        for scale,sel in [('all_widths',np.ones(len(ii),dtype=bool)),
                          ('same_width',ww[ii]==ww[jj]),
                          ('different_widths',ww[ii]!=ww[jj])]:
            common=inter[ii[sel],jj[sel]].astype(float)
            un=(sizes[ii[sel]]+sizes[jj[sel]]-common).astype(float)
            jac=common/np.maximum(un,1)
            records.append(dict(device=device,comparison=scale,n_regions=len(patches),
                n_region_pairs=len(common),n_distinct_region_node_sets=len(set(sets)),
                n_exact_duplicates=int(np.sum(jac==1)),
                fraction_pairs_overlap=float(np.mean(common>0)) if len(common) else np.nan,
                median_jaccard_if_overlapping=float(np.median(jac[common>0])) if np.any(common>0) else 0.0,
                max_jaccard=float(np.max(jac)) if len(jac) else np.nan))
    return pd.DataFrame(records)


def reviewer_boundary_inventory(df,patch_nodes,edges):
    """Compare induced-region topology with its full-device external boundary."""
    graphs={}
    for d,ed in edges.groupby('device'):
        G=nx.Graph()
        G.add_edges_from((int(r.u),int(r.v)) for r in ed.itertuples(index=False))
        graphs[d]=G
    rows=[]
    for patch,g in patch_nodes.groupby('patch_id',sort=True):
        device=g.device.iloc[0];G=graphs[device];nodes=set(map(int,g.node))
        induced=G.subgraph(nodes)
        full_degrees=np.asarray([G.degree(u) for u in nodes],dtype=float)
        internal_degrees=np.asarray([induced.degree(u) for u in nodes],dtype=float)
        cut_edges=int(np.sum(full_degrees-internal_degrees))
        outgoing=set(v for u in nodes for v in G.neighbors(u) if v not in nodes)
        full_reach=[];region_reach=[]
        for u in nodes:
            full_reach.append(len(nx.single_source_shortest_path_length(G,u,cutoff=2))-1)
            region_reach.append(len(nx.single_source_shortest_path_length(induced,u,cutoff=2))-1)
        rows.append(dict(patch_id=patch,device=device,width=len(nodes),n_induced_edges=induced.number_of_edges(),
            boundary_cut_edges=cut_edges,external_neighbors=len(outgoing),
            full_degree_mean=float(full_degrees.mean()),induced_degree_mean=float(internal_degrees.mean()),
            external_degree_fraction=float(cut_edges/max(full_degrees.sum(),1)),
            full_minus_induced_k2_reach=float(np.mean(np.asarray(full_reach)-np.asarray(region_reach)))))
    patches=pd.DataFrame(rows)
    merged=patches.merge(df[['patch_id']+CALIBRATION_PRIMARY],on='patch_id',validate='one_to_one')
    corr=[]
    for d,g in [('ALL',merged),*list(merged.groupby('device'))]:
        for structure in ['external_degree_fraction','full_minus_induced_k2_reach']:
            for calibration in CALIBRATION_PRIMARY:
                xy=g[[structure,calibration]].dropna()
                if len(xy)>=10 and xy[structure].std()>1e-12 and xy[calibration].std()>1e-12:
                    rho=float(spearmanr(xy[structure],xy[calibration]).statistic)
                else:rho=np.nan
                corr.append(dict(device=d,boundary_feature=structure,calibration_feature=calibration,
                                 n=len(xy),rho=rho,kind='exploratory_descriptive_no_significance_claim'))
    return patches,pd.DataFrame(corr)


def reviewer_device_mi(df,baseline,ks=(3,5,7),max_pairs=0):
    """Within-device MI point estimates; NO p-values and NO FDR (diagnostic only)."""
    s_cols=list(dict.fromkeys(baseline.structural_feature))
    q_cols=list(dict.fromkeys(baseline.calibration_feature))
    S=residualize_within_device(df,s_cols)
    Q=residualize_within_device(df,q_cols)
    output=[]
    limit=max_pairs or len(baseline)
    for device,g in df.groupby('device'):
        ids=g.index
        for ix,rec in enumerate(baseline.itertuples(index=False)):
            if ix>=limit:break
            s,q=rec.structural_feature,rec.calibration_feature
            x=S.loc[ids,s].to_numpy(float);y=Q.loc[ids,q].to_numpy(float)
            good=np.isfinite(x)&np.isfinite(y);x=x[good];y=y[good]
            bad=(len(x)<40 or np.std(x)<1e-9 or np.std(y)<1e-9)
            if not bad:
                x=(x-np.mean(x))/np.std(x);y=(y-np.mean(y))/np.std(y)
                rg=np.random.default_rng(stable_seed(MASTER_SEED,'reviewer_mi_dev',device,s,q))
                x=x+rg.normal(0,1e-10,len(x));y=y+rg.normal(0,1e-10,len(y))
            for k in ks:
                score=(float(mutual_info_regression(x.reshape(-1,1),y,n_neighbors=k,random_state=73)[0])
                       if not bad and k<len(x) else np.nan)
                output.append(dict(device=device,structural_feature=s,calibration_feature=q,
                    k=k,n=len(x),mi_nats=score,status='degenerate' if bad else 'ok',
                    inference='within_device_descriptive_only'))
    return pd.DataFrame(output)



def repair_legacy_duplicate_regions(source_archive, output_archive):
    """Deterministically repair duplicate physical regions using FROZEN snapshot tables.

    The original generator did NOT reject duplicate node sets despite claims in
    manuscript prose. This rebuilds only duplicate patches using deterministic
    re-draws, then updates their structural/calibration descriptors and all
    exact-degree pair distances. It deliberately discards obsolete inferential
    outputs from the source archive.
    """
    import zipfile, io, json
    source=reviewer_archive_tables(source_archive)
    # The node table is not part of the lightweight reviewer table map.
    with zipfile.ZipFile(source_archive) as original:
        qmatches=[n for n in original.namelist() if n.endswith('/tables/nodes.csv')]
        if len(qmatches)!=1:raise ValueError('Cannot repair: missing original qubit calibration table')
        qtable=pd.read_csv(io.BytesIO(original.read(qmatches[0])))
        snapshots={n.split('/')[-1]:original.read(n) for n in original.namelist()
                   if '/source_snapshots/' in n and n.endswith('.json')}
    patches=source['patches'].copy()
    merged=source['merged_patch_dataset'].copy().set_index('patch_id',drop=False)
    edge_table=source['edges']; graphs={}
    for d,nd in qtable.groupby('device'):
        G=nx.Graph()
        for row in nd.itertuples(index=False):
            G.add_node(int(row.node),t1=float(row.t1) if pd.notna(row.t1) else np.nan,
                       t2=float(row.t2) if pd.notna(row.t2) else np.nan,
                       frequency=float(row.frequency) if pd.notna(row.frequency) else np.nan,
                       readout_error=float(row.readout_error) if pd.notna(row.readout_error) else np.nan,
                       oneq_error=float(row.oneq_error) if pd.notna(row.oneq_error) else np.nan)
        for row in edge_table.loc[edge_table.device==d].itertuples(index=False):
            G.add_edge(int(row.u),int(row.v),
                twoq_error=float(row.twoq_error) if pd.notna(row.twoq_error) else np.nan,
                twoq_duration=float(row.twoq_duration) if pd.notna(row.twoq_duration) else np.nan)
        graphs[d]=G
    # Strong archived-data invariant: unchanged region descriptors should agree
    # with recomputation from frozen nodes/edges before any repair is accepted.
    for d in sorted(graphs):
        row=patches[patches.device==d].iloc[0]
        P=graphs[d].subgraph(set(map(int,str(row.nodes).split('|')))).copy()
        S=structural_descriptors(P);C=calibration_descriptors(P)
        for col in ['m','avg_path','global_efficiency','KHEM_k2_mean','SSE_k2','twoq_error_mean','t1_mean']:
            orig=merged.loc[row.patch_id,col];new=(S|C).get(col)
            if pd.isna(orig) and pd.isna(new):continue
            if not np.isclose(float(orig),float(new),rtol=1e-7,atol=1e-8,equal_nan=True):
                raise ValueError(f'Frozen source cannot reconstruct {d}:{col}: archived={orig}, recomputed={new}')
    changes=[]
    for (device,width),idx in patches.groupby(['device','width'],sort=True).groups.items():
        seen=set();G=graphs[device]
        for i in idx:
            r=patches.loc[i]
            old=frozenset(int(x) for x in str(r.nodes).split('|'))
            if old not in seen:
                seen.add(old);continue
            # Restore required uniqueness by changing only the duplicate region,
            # never the first occurrence or a different device/width block.
            for retry in range(1,20001):
                seed=stable_seed(MASTER_SEED,device,int(width),int(r.patch_index),'deduplicate',retry)
                rng=np.random.default_rng(seed)
                P,nodes=connected_patch(G,int(width),rng)
                key=frozenset(map(int,nodes))
                if key not in seen:break
            else:raise RuntimeError(f'Unable to find distinct {device} w={width}')
            seen.add(key)
            old_str='|'.join(map(str,sorted(old)))
            patches.at[i,'nodes']='|'.join(map(str,nodes))
            patches.at[i,'seed']=seed
            desc=structural_descriptors(P)|calibration_descriptors(P)
            for col,value in desc.items():
                if col in merged.columns:merged.at[r.patch_id,col]=value
            changes.append(dict(device=device,width=int(width),patch_id=r.patch_id,
                old_node_set=old_str,new_node_set='|'.join(map(str,sorted(key))),
                retry=retry,new_seed=seed))
    source['patches']=patches
    source['merged_patch_dataset']=merged.reset_index(drop=True)
    node_rows=[]
    for r in patches.itertuples(index=False):
        node_rows.extend(dict(patch_id=r.patch_id,device=r.device,width=int(r.width),node=int(q))
                         for q in str(r.nodes).split('|'))
    source['patch_nodes']=pd.DataFrame(node_rows)
    group=exact_degree_groups(source['merged_patch_dataset'])
    _,matched=group_variability(source['merged_patch_dataset'],group,
         STRUCTURAL_CORE+STRUCTURAL_K_PRIMARY+STRUCTURAL_K_SECONDARY,CALIBRATION_PRIMARY)
    source['exact_degree_groups']=group
    source['matched_pair_distances']=matched_pair_distances(matched,
                      STRUCTURAL_CORE+STRUCTURAL_K_PRIMARY,CALIBRATION_PRIMARY)
    original_pairs=source.pop('mutual_information_permutation_normalized')
    source['mi_pair_registry']=original_pairs[['structural_feature','calibration_feature']].copy()
    node_sets=(source['patch_nodes'].groupby(['device','width','patch_id']).node
              .apply(lambda x:tuple(sorted(x))).reset_index(name='node_set'))
    if node_sets.duplicated(['device','width','node_set']).any():
        raise AssertionError('Canonical repair failed to eliminate duplicate regions')
    if len(source['merged_patch_dataset'])!=600 or len(source['mi_pair_registry'])!=310:
        raise AssertionError('Unexpected cohort/feature cardinality after repair')
    out=Path(output_archive);out.parent.mkdir(parents=True,exist_ok=True)
    meta=dict(status='CORRECTED_600_DISTINCT_PHYSICAL_REGIONS',source_archive_sha256=sha256_file(source_archive),
         generation_rule='original seed retained for first unique occurrence; SHA256-derived deterministic retry for duplicate patches',
         n_replaced=int(len(changes)),changes=changes,
         obsolete_outputs_discarded='ALL source inference and MI results. Original 84/310 significance and all old matched/CCA/RV estimates are NOT valid for this cohort.',
         n_regions=600,n_unique_physical_region_sets=600,
         n_exact_degree_groups=len(group),n_matched_pairs=len(source['matched_pair_distances']),
         directionality='Archived instruction directions irretrievably collapsed; re-extract via current script on fake backends if available.')
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for name,frame in source.items():
            z.writestr(f'Academia_QPU_CORRECTED/tables/{name}.csv',frame.to_csv(index=False))
        for name,content in snapshots.items():
            z.writestr('Academia_QPU_CORRECTED/source_snapshots/'+name,content)
        z.writestr('Academia_QPU_CORRECTED/SOURCE_REPAIR.json',json.dumps(meta,indent=2,ensure_ascii=False))
        z.writestr('Academia_QPU_CORRECTED/README.txt',
          'Canonical input dataset only. 600 unique regions; 8 deterministic replacements.\n'
          'No inferential output from the obsolete archive is carried forward.\n'
          'Rerun ALL tests, including MI, CCA, RV, matched distance, overlap, using these corrected records.\n')
    print('Repaired source:',str(out),'patches repaired:',len(changes),
          'matched groups:',len(group),'matched pairs:',len(source['matched_pair_distances']),flush=True)
    return meta


def reviewer_multivariate_corrected(df, s_cols, q_cols, n_perm):
    """Recompute manuscript's controlled CCA, RV, kNN on corrected 600-region data.

    Block permutations are CONDITIONAL diagnostics. Physical region overlap
    prevents interpreting sampled regions as independent QPU observations.
    """
    R= residualize_within_device(df, s_cols)
    valid_s=[c for c in s_cols if np.nanstd(R[c].to_numpy(dtype=float))>=1e-9]
    Qr=residualize_within_device(df,q_cols)
    valid_q=[c for c in q_cols if np.nanstd(Qr[c].to_numpy(dtype=float))>=1e-9]
    if len(valid_s)<3 or len(valid_q)<3:raise ValueError('Too few nondegenerate coordinates for multivariate analysis')
    summary,lodo,S,Q=conditional_cca_analysis(df,valid_s,valid_q)
    cca_perm=conditional_cca_permutation_test(df,S,Q,n_perm=n_perm)
    rv_perm=conditional_rv_test(df,S,Q,n_perm=n_perm)
    knn_perm=conditional_knn_overlap_test(df,S,Q,n_perm=n_perm)
    summary['n_active_s']=len(valid_s);summary['n_active_q']=len(valid_q)
    return dict(cca_descriptive=summary,cca_held_device=lodo,cca_permutation=cca_perm,
                rv_permutation=rv_perm,knn_permutation=knn_perm,
                multivariate_active_s=pd.DataFrame({'feature':valid_s}),
                multivariate_active_q=pd.DataFrame({'feature':valid_q}))


def reviewer_disjoint_multivariate(df,patch_nodes,s_cols,q_cols,select_seeds=20,n_perm=199):
    """Global physical-disjoint subsampling across sizes, PCA-3 CCA/RV sensitivity."""
    from sklearn.decomposition import PCA
    R= residualize_within_device(df,s_cols)
    Q= residualize_within_device(df,q_cols)
    s_cols=[c for c in s_cols if np.nanstd(R[c].to_numpy(float))>=1e-9]
    q_cols=[c for c in q_cols if np.nanstd(Q[c].to_numpy(float))>=1e-9]
    psets=patch_nodes.groupby(['device','patch_id']).node.apply(lambda x:frozenset(map(int,x))).to_dict()
    ordered={d:list(df.loc[df.device==d,'patch_id']) for d in sorted(df.device.unique())}
    lookup={p:i for i,p in enumerate(df.patch_id)}
    records=[];choices=[]
    for seed in range(select_seeds):
        rng=np.random.default_rng(stable_seed(MASTER_SEED,'strict_multivariate_selection',seed))
        selected=[]
        for device,ids in ordered.items():
            used=set()
            for patch in rng.permutation(ids):
                nodes=psets[(device,patch)]
                if used.isdisjoint(nodes):
                    used.update(nodes);selected.append(str(patch))
        ii=[lookup[p] for p in selected];sub=df.iloc[ii].reset_index(drop=True)
        for p in selected:choices.append(dict(selection_seed=seed,patch_id=p))
        if len(ii)<15:continue
        xs=R.iloc[ii][s_cols].copy();ys=Q.iloc[ii][q_cols].copy()
        for frame in (xs,ys):frame.fillna(frame.median().fillna(0),inplace=True)
        aa=StandardScaler().fit_transform(xs)
        bb=StandardScaler().fit_transform(ys)
        nd=min(3,len(sub)-2,aa.shape[1],bb.shape[1])
        aa=PCA(n_components=nd).fit_transform(aa)
        bb=PCA(n_components=nd).fit_transform(bb)
        def cc(x,y):
            m=CCA(n_components=1,max_iter=2000)
            m.fit(x,y)
            a,b=m.transform(x,y)
            return float(np.corrcoef(a[:,0],b[:,0])[0,1])
        obs_cca=cc(aa,bb);obs_rv=rv_coefficient(aa,bb)
        blocks=[np.asarray(list(v),dtype=int) for v in sub.groupby(['device','width']).groups.values()]
        rng2=np.random.default_rng(stable_seed(MASTER_SEED,'strict_multivariate_null',seed))
        cca_null=[];rv_null=[]
        for repeat in range(n_perm):
            order=np.arange(len(sub))
            for block in blocks:
                if len(block)>1:order[block]=rng2.permutation(block)
            cca_null.append(cc(aa,bb[order]));rv_null.append(rv_coefficient(aa,bb[order]))
        records.append(dict(selection_seed=seed,n_regions=len(ii),
           n_permutable_blocks=sum(len(block)>1 for block in blocks),
           cca1_pca3=obs_cca,cca_null_mean=float(np.mean(cca_null)),
           cca_p_conditional=float((1+np.sum(np.asarray(cca_null)>=obs_cca))/(n_perm+1)),
           rv_pca3=obs_rv,rv_null_mean=float(np.mean(rv_null)),
           rv_p_conditional=float((1+np.sum(np.asarray(rv_null)>=obs_rv))/(n_perm+1)),
           p_scope='conditional_exchangeability_not_independence',B=n_perm))
    return pd.DataFrame(records),pd.DataFrame(choices)


def reviewer_disjoint_mi_top(df,patch_nodes,baseline,selections=8,permutations=99):
    """Six *pre-specified* archived top pairs; hypothesis-generating overlap sensitivity.

    Original-pair ranking selected on the source data, so no independent
    confirmatory discovery claims are warranted from this diagnostic.
    """
    candidates=baseline.iloc[:6][['structural_feature','calibration_feature']].drop_duplicates()
    sc=list(dict.fromkeys(candidates.structural_feature));qc=list(dict.fromkeys(candidates.calibration_feature))
    S=residualize_within_device(df,sc);Q=residualize_within_device(df,qc)
    psets=patch_nodes.groupby(['device','patch_id']).node.apply(lambda x:frozenset(map(int,x))).to_dict()
    perdev={d:list(df.loc[df.device==d,'patch_id']) for d in sorted(df.device.unique())}
    lookup={p:i for i,p in enumerate(df.patch_id)}
    records=[]
    for seed in range(selections):
        rg=np.random.default_rng(stable_seed(MASTER_SEED,'mi_disjoint_chooser',seed))
        chosen=[]
        for device,ids in perdev.items():
            used=set()
            for pid in rg.permutation(ids):
                setp=psets[(device,pid)]
                if used.isdisjoint(setp):used.update(setp);chosen.append(str(pid))
        pos=[lookup[p] for p in chosen]
        subset=df.iloc[pos].reset_index(drop=True)
        blocks=[np.asarray(list(v),dtype=int) for v in subset.groupby(['device','width']).groups.values()]
        for a,b in candidates.itertuples(index=False):
            x=S.iloc[pos][a].to_numpy(float);y=Q.iloc[pos][b].to_numpy(float)
            good=np.isfinite(x)&np.isfinite(y)
            if good.sum()<20 or np.nanstd(x[good])<1e-9 or np.nanstd(y[good])<1e-9:
                for k in (3,5):
                    records.append(dict(selection_seed=seed,structural_feature=a,
                        calibration_feature=b,k=k,n=int(good.sum()),mi_obs=np.nan,
                        excess=np.nan,p=1.0,status='degenerate'))
                continue
            # Use complete-case positions to guarantee valid exchangeable blocks.
            old=np.flatnonzero(good);posmap={int(v):i for i,v in enumerate(old)}
            validblocks=[np.asarray([posmap[int(v)] for v in block if int(v) in posmap],int)
                         for block in blocks]
            validblocks=[vb for vb in validblocks if len(vb)>1]
            x=x[good];y=y[good]
            x=(x-x.mean())/x.std();y=(y-y.mean())/y.std()
            rand=np.random.default_rng(stable_seed(MASTER_SEED,'mi_disjoint_test',seed,a,b))
            x=x+rand.normal(0,1e-10,len(x));y=y+rand.normal(0,1e-10,len(y))
            nullorders=[]
            for rep in range(permutations):
                order=np.arange(len(y))
                for block in validblocks:order[block]=rand.permutation(block)
                nullorders.append(order)
            for k in (3,5):
                def estimate(values):
                    return float(mutual_info_regression(x.reshape(-1,1),values,
                          n_neighbors=k,random_state=73)[0])
                obs=estimate(y)
                null=np.asarray([estimate(y[order]) for order in nullorders])
                records.append(dict(selection_seed=seed,structural_feature=a,
                    calibration_feature=b,k=k,n=len(x),mi_obs=obs,
                    excess=float(obs-null.mean()),p=float((1+np.sum(null>=obs))/(permutations+1)),
                    status='ok'))
    out=pd.DataFrame(records)
    for (seed,k),inds in out.groupby(['selection_seed','k']).groups.items():
        out.loc[inds,'q_bh_six_exploratory']=benjamini_hochberg(out.loc[inds,'p'].to_numpy(float))
    return out


def reviewer_extra_audit(archive_path, out, preflight=False, include_device_mi=True):
    """Run reviewer-required diagnostics before the long permutation MI calculation.

    Outputs: feature redundancy, LODO ablations, nonlinear control sensitivity,
    physical overlap, induced/full boundary, device-level MI, metadata.
    These exports are diagnostic unless explicitly indicated otherwise.
    """
    import importlib.metadata as md
    from pathlib import Path as _Path
    output=_Path(out);output.mkdir(parents=True,exist_ok=True)
    manifest_file=output/'audit_manifest.json'
    if manifest_file.exists():
        old=json.loads(manifest_file.read_text(encoding='utf8'))
        required=['external_lodo_ablation.csv','nonlinear_control_sensitivity.csv',
                  'matched_group_threshold_sensitivity.csv','CCA_permutation.csv',
                  'strict_disjoint_CCA_RV.csv','strict_disjoint_MI_top6.csv']
        if (old.get('analysis_revision')=='reviewer_complete_v5'
                and old.get('source_archive_sha256')==sha256_file(archive_path)
                and old.get('run_type')==('PREFLIGHT' if preflight else 'COMPLETE_REVIEWER_DIAGNOSTICS')
                and all((output/x).exists() for x in required)):
            print('Reviewer suite already complete; validated same source fingerprint, skipping recomputation.',flush=True)
            return old
    data=reviewer_archive_tables(archive_path)
    df=data['merged_patch_dataset']
    baseline=data['mutual_information_permutation_normalized']
    if len(baseline)!=310 or baseline[['structural_feature','calibration_feature']].duplicated().any():
        raise ValueError('The preregistered MI family must contain 310 distinct pairs')
    s_cols=list(dict.fromkeys(baseline.structural_feature));q_cols=list(dict.fromkeys(baseline.calibration_feature))
    if len(s_cols)!=31 or len(q_cols)!=10:
        raise ValueError(f'Unexpected feature-family cardinality {len(s_cols)} x {len(q_cols)}')
    if set(s_cols+q_cols+CONNECTIVITY_CONTROLS)-set(df.columns):
        raise ValueError('Missing required descriptor/control fields')
    if df.device.nunique()!=5 or len(df)!=600 or df.width.isna().any():
        raise ValueError('Archive cohort differs from frozen five-device, 600-region data')
    patch_sets=(data['patch_nodes'].groupby(['device','width','patch_id']).node
                .apply(lambda x:tuple(sorted(map(int,x)))).reset_index(name='node_set'))
    if patch_sets.duplicated(['device','width','node_set']).any():
        raise ValueError('Duplicate physical region sets: first repair the frozen source archive')
    redundancy=reviewer_residual_redundancy(df,s_cols)
    redundancy.to_csv(output/'feature_redundancy.csv',index=False)
    ablation=reviewer_ablation_external_lodo(df,s_cols,q_cols)
    ablation.to_csv(output/'external_lodo_ablation.csv',index=False)
    residual=reviewer_residual_sensitivity(df,s_cols,q_cols)
    residual.to_csv(output/'nonlinear_control_sensitivity.csv',index=False)
    overlap=reviewer_overlap_inventory(data['patch_nodes'])
    overlap.to_csv(output/'physical_overlap.csv',index=False)
    boundary,bcorr=reviewer_boundary_inventory(df,data['patch_nodes'],data['edges'])
    boundary.to_csv(output/'full_graph_boundary_per_region.csv',index=False)
    bcorr.to_csv(output/'boundary_calibration_correlations.csv',index=False)
    devices=reviewer_device_mi(df,baseline,ks=(3,5,7),max_pairs=12 if preflight else 0) if include_device_mi else pd.DataFrame()
    if len(devices):devices.to_csv(output/'within_device_mi_descriptive.csv',index=False)
    # Re-run the EXACT-degree inference and sensitivity suite on the corrected
    # 600-region cohort. Old CSVs with 2134 pairs are not reused.
    import zipfile as _zipfile, io as _io
    with _zipfile.ZipFile(archive_path) as z:
        def _table(name):
            matches=[v for v in z.namelist() if v.endswith('/tables/'+name+'.csv')]
            if len(matches)!=1:raise ValueError('Missing corrected '+name+'.csv')
            return pd.read_csv(_io.BytesIO(z.read(matches[0])))
        corrected_pairs=_table('matched_pair_distances')
        corrected_groups=_table('exact_degree_groups')
    nmatch=19 if preflight else 5000
    t=matched_group_threshold_audit(corrected_pairs,thresholds=(2,3,4,5),n_perm=nmatch)
    t.to_csv(output/'matched_group_threshold_sensitivity.csv',index=False)
    groupmap={p:r.group_id for r in corrected_groups.itertuples(index=False)
              for p in str(r.patch_ids).split('|')}
    matched_df=df[df.patch_id.isin(groupmap)].copy()
    matched_df['group_id']=matched_df.patch_id.map(groupmap)
    lnorm=matched_distance_norm_sensitivity(matched_df,
         STRUCTURAL_CORE+STRUCTURAL_K_PRIMARY,CALIBRATION_PRIMARY,
         n_perm=19 if preflight else 1000)
    lnorm.to_csv(output/'matched_L1_L2_distance_sensitivity.csv',index=False)
    disjoint,chosen=global_disjoint_audit(corrected_pairs,data['patch_nodes'],
           seeds=3 if preflight else 20,permutations=19 if preflight else 499)
    disjoint.to_csv(output/'strictly_disjoint_matched_results.csv',index=False)
    chosen.to_csv(output/'strictly_disjoint_matched_selected_regions.csv',index=False)
    # The archived snapshot JSON lacks instruction-level orientation. Export its
    # actual metadata rather than inventing directed instruction measurements.
    snapshots=[]
    with _zipfile.ZipFile(archive_path) as z:
        for path in sorted(z.namelist()):
            if '/source_snapshots/' not in path or not path.endswith('.json'):continue
            raw=json.loads(z.read(path)); m=raw.get('metadata',{})
            snapshots.append(dict(device=raw.get('device'),backend_class=m.get('backend_class'),
                n_qubits=m.get('num_qubits'),n_edges=m.get('num_edges'),
                snapshot_date='unknown_in_export',
                two_qubit_native_directions_available=False))
    pd.DataFrame(snapshots).to_csv(output/'snapshot_inventory.csv',index=False)
    # Corrected 600-region CCA/RV/kNN: do not reuse tests from 8-duplicate archive.
    for label,frame in reviewer_multivariate_corrected(df,s_cols,q_cols,
          n_perm=9 if preflight else 999).items():
        frame.to_csv(output/('CCA_' + label[4:] + '.csv') if label.startswith('cca_')
                     else output/(label+'.csv'),index=False)
    strict_mv,strict_ids=reviewer_disjoint_multivariate(df,data['patch_nodes'],s_cols,q_cols,
          select_seeds=3 if preflight else 20,n_perm=9 if preflight else 199)
    strict_mv.to_csv(output/'strict_disjoint_CCA_RV.csv',index=False)
    strict_ids.to_csv(output/'strict_disjoint_CCA_RV_selections.csv',index=False)
    strict_mi=reviewer_disjoint_mi_top(df,data['patch_nodes'],baseline,
          selections=2 if preflight else 8,permutations=9 if preflight else 99)
    strict_mi.to_csv(output/'strict_disjoint_MI_top6.csv',index=False)
    descriptor=pd.concat([pd.DataFrame({'layer':'structural','descriptor':s_cols}),
                          pd.DataFrame({'layer':'calibration','descriptor':q_cols})],ignore_index=True)
    descriptor.to_csv(output/'frozen_feature_dictionary.csv',index=False)
    try:
        versions={pkg:md.version(pkg) for pkg in ('numpy','pandas','scipy','scikit-learn','networkx','matplotlib')}
    except Exception:versions={}
    metadata={'analysis_revision':'reviewer_complete_v5',
        'source_archive_sha256':sha256_file(archive_path),
        'run_type':'PREFLIGHT' if preflight else 'COMPLETE_REVIEWER_DIAGNOSTICS',
        'n_regions':len(df),'n_devices':int(df.device.nunique()),
        'n_frozen_mi_pairs':len(baseline),'n_structural_features':len(s_cols),
        'n_calibration_features':len(q_cols),'n_device_mi_pairs':(12 if preflight else 310) if include_device_mi else 0,
        'degree_of_overlap_warning':'Shared physical qubits induce statistical dependence; neither block permutation nor disjoint selection proves independence.',
        'directionality':'Archived edges are undirected means. Native gate names and orientation-specific errors cannot be recovered from these archive tables.',
        'nonlinear_control_warning':'OOF residual model comparison descriptive, not proof of adjusted independence.',
        'CCA_RV_kNN_p_values':'conditional device-by-width block permutation only; physical overlap remains',
        'MI_disjoint_top6':'original-data-selected hypotheses, NOT independent confirmatory family',
        'external_lodo':'Train-only preprocessing and raw held-out calibration targets; no held-out-target residualization.',
        'versions':versions}
    metadata['n_corrected_matched_pairs']=int(len(corrected_pairs))
    metadata['matched_permutations']=nmatch
    metadata['strict_overlap_seeds']=3 if preflight else 20
    (output/'audit_manifest.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
    print('REVIEWER DIAGNOSTICS',len(redundancy),'redundancy pairs; ',len(ablation),'LODO rows; ',
          len(overlap),'overlap rows; ',len(boundary),'patch-boundary rows; ',len(devices),'device MI estimates',flush=True)
    return metadata


def reviewer_self_test(archive_path):
    """Fail-fast invariant suite for the canonical corrected archive and MI family."""
    import zipfile
    dataset=reviewer_archive_tables(archive_path)
    df=dataset['merged_patch_dataset']
    pairs=dataset['mutual_information_permutation_normalized']
    if len(df)!=600 or df.device.nunique()!=5 or df.patch_id.duplicated().any():
        raise AssertionError('Wrong cohort cardinality')
    if set(df.width)!=set(PATCH_WIDTHS):
        raise AssertionError('Wrong patch widths')
    if not (df.groupby(['device','width']).size()==24).all():
        raise AssertionError('Expected exactly 24 regions per device and width')
    nodes=dataset['patch_nodes']
    sets=nodes.groupby(['device','width','patch_id']).node.apply(lambda x:tuple(sorted(map(int,x))))
    patch_defs=sets.reset_index(name='node_set')
    if patch_defs.patch_id.nunique()!=600 or patch_defs.duplicated(['device','width','node_set']).any():
        raise AssertionError('Duplicate or missing physical patch definition')
    if len(pairs)!=310 or pairs[['structural_feature','calibration_feature']].duplicated().any():
        raise AssertionError('Wrong MI pair registry')
    structural=set(pairs.structural_feature);calibration=set(pairs.calibration_feature)
    if len(structural)!=31 or len(calibration)!=10 or set(map(tuple,pairs[['structural_feature','calibration_feature']].to_numpy())) != {(s,c) for s in structural for c in calibration}:
        raise AssertionError('MI registry is not the full 31x10 Cartesian product')
    with zipfile.ZipFile(archive_path) as z:
        if not any(n.endswith('/SOURCE_REPAIR.json') for n in z.namelist()):
            raise AssertionError('Missing provenance of corrected duplicate regions')
        import io
        gp=[n for n in z.namelist() if n.endswith('/tables/matched_pair_distances.csv')]
        if len(gp)!=1:raise AssertionError('Missing corrected matched distances')
        mp=pd.read_csv(io.BytesIO(z.read(gp[0])))
    if mp.duplicated(['group_id','patch_a','patch_b']).any() or mp.isna().any().any():
        raise AssertionError('Duplicate/missing matched-distance entries')
    for gid,g in mp.groupby('group_id'):
        regions=set(g.patch_a)|set(g.patch_b); n=len(regions)
        if len(g)!=n*(n-1)//2:
            raise AssertionError('Broken pairwise distance matrix '+str(gid))
    if not np.allclose(benjamini_hochberg(np.array([0.01,0.02,1.0])),[0.03,0.03,1.0]):
        raise AssertionError('BH FDR implementation inconsistent')
    if stable_seed(MASTER_SEED,'self-test')!=stable_seed(MASTER_SEED,'self-test'):
        raise AssertionError('Unstable random seed')
    print(f'SELF TEST OK: {len(df)} distinct regions, {len(pairs)} complete registry pairs, '
          f'{mp.group_id.nunique()} matched groups, {len(mp)} matched distances, BH verified.',flush=True)
    return True


def reviewer_mi_exports(audit_pairs,out):
    """Publish supplementary full-pair/radius tables ONLY after complete 310-pair run."""
    from pathlib import Path as _Path
    output=_Path(out);output.mkdir(parents=True,exist_ok=True)
    if audit_pairs.groupby('k').size().nunique()!=1 or not (audit_pairs.groupby('k').size()==310).all():
        raise ValueError('Complete MI supplement requires all 310 rows for every requested k')
    pairs=audit_pairs.copy()
    def family(s):
        for name in ('KHEM','EWR','NED','SSE','NSSE','GCV','growth'):
            if s.startswith(name):return name
        return 'conventional'
    pairs['structural_family']=pairs.structural_feature.map(family)
    pairs['radius']=pairs.structural_feature.str.extract(r'_k(\d+)').iloc[:,0].fillna('not_applicable')
    pairs.to_csv(output/'MI_full_310_per_k.csv',index=False)
    counts=pairs.groupby(['k','structural_family','radius','calibration_feature']).agg(
        n_pairs=('p','size'),n_nominal=('p',lambda z:int((z<0.05).sum())),
        n_fdr_310=('q_bh_310',lambda z:int((z<0.05).sum())),
        median_excess_mi=('excess_mi','median')).reset_index()
    counts.to_csv(output/'MI_scale_family_summary.csv',index=False)
    # Supplementary visualization: complete 31 x 10 flag matrix per k, with
    # disabled/degenerate coordinates shown as NA, not spurious discoveries.
    ss=list(dict.fromkeys(pairs.structural_feature));qq=list(dict.fromkeys(pairs.calibration_feature))
    levels=sorted(pairs.k.unique());fig,axes=plt.subplots(1,len(levels),
         figsize=(max(9,6*len(levels)),11),squeeze=False)
    for ax,k in zip(axes[0],levels):
        slab=pairs[pairs.k==k].set_index(['structural_feature','calibration_feature'])
        heat=np.full((len(ss),len(qq)),np.nan)
        for i,a in enumerate(ss):
            for j,b in enumerate(qq):
                rec=slab.loc[(a,b)]
                if rec.status=='ok':heat[i,j]=float(rec.q_bh_310<0.05)
        image=ax.imshow(np.ma.masked_invalid(heat),aspect='auto',vmin=0,vmax=1)
        ax.set_title(f'MI FDR q<0.05 | k={k} (conditional)')
        ax.set_xticks(range(len(qq)),qq,rotation=90,fontsize=8)
        ax.set_yticks(range(len(ss)),ss,fontsize=7)
    fig.tight_layout()
    fig.savefig(output/'MI_310_FDR_heatmap.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    return counts


def mi_full_robustness_archive_cli():
    """Reproducible, resumable 310-pair sensitivity audit of archived MI descriptors.

    This is a block-conditional exploratory sensitivity analysis, not a valid
    independent-region inferential test when regions physically overlap.
    """
    import argparse, zipfile, io, hashlib, json, os
    parser = argparse.ArgumentParser()
    parser.add_argument('--mi-full-audit', action='store_true')
    parser.add_argument('--archive', required=True)
    parser.add_argument('--out', default='mi_full_audit')
    parser.add_argument('--permutations', type=int, default=999)
    parser.add_argument('--neighbors', default='3,5,7')
    parser.add_argument('--max-pairs', type=int, default=0,
                        help='For preflight only; 0 runs all 310')
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--preflight', action='store_true',
                        help='Run validation and a short test, not final inference')
    parser.add_argument('--reviewer-extras',action='store_true',
                        help='Run other reviewer-requested audits in the same invocation, BEFORE long MI')
    args = parser.parse_args()
    ks = [int(x) for x in args.neighbors.split(',')]
    if not ks or len(set(ks)) != len(ks) or any(k < 1 for k in ks):
        raise ValueError('neighbors must be unique positive integers')
    if args.permutations < 1 or args.checkpoint_every < 1 or args.max_pairs < 0:
        raise ValueError('Invalid permutation/checkpoint/pair counts')
    with zipfile.ZipFile(args.archive) as z:
        def read(name):
            files = [f for f in z.namelist() if f.endswith('/tables/'+name+'.csv')]
            if len(files) != 1:
                raise ValueError('Missing or ambiguous table: '+name)
            return pd.read_csv(io.BytesIO(z.read(files[0])))
        df = read('merged_patch_dataset').reset_index(drop=True)
        baseline = (read('mi_pair_registry') if any(f.endswith('/tables/mi_pair_registry.csv')
                    for f in z.namelist()) else read('mutual_information_permutation_normalized'))
    if len(baseline) != 310 or baseline[['structural_feature','calibration_feature']].duplicated().any():
        raise ValueError('Expected exactly 310 unique baseline pairs')
    # Failure-fast: do NOT waste a long MI run on the 8 duplicated physical regions.
    with zipfile.ZipFile(args.archive) as zcheck:
        corrected=any(f.endswith('/SOURCE_REPAIR.json') for f in zcheck.namelist())
    if not corrected:
        raise ValueError('The source archive has uncorrected duplicate region sets. First run --repair-archive. Do not rerun 310 MI on obsolete 600-region data.')
    keypath=[]
    with zipfile.ZipFile(args.archive) as zz:
        keypath=[f for f in zz.namelist() if f.endswith('/SOURCE_REPAIR.json')]
        repair_info=json.loads(zz.read(keypath[0]))
    if repair_info.get('status')!='CORRECTED_600_DISTINCT_PHYSICAL_REGIONS':
        raise ValueError('Archive lacks validated duplicate-repair marker')
    s_cols = list(dict.fromkeys(baseline.structural_feature))
    q_cols = list(dict.fromkeys(baseline.calibration_feature))
    needed = set(s_cols+q_cols+['device','width','m','avg_degree','degree_std','degree_cv'])
    if not needed.issubset(df.columns):
        raise ValueError('Missing data columns: '+str(sorted(needed-set(df.columns))))
    if args.reviewer_extras:
        reviewer_extra_audit(args.archive,args.out+'_reviewer',preflight=args.preflight)
    S = residualize_within_device(df,s_cols)
    Q = residualize_within_device(df,q_cols)
    blocks = [np.asarray(ix,dtype=int) for ix in df.groupby(['device','width']).groups.values()]
    audit=[]; degenerate=set()
    for layer,frame in [('S',S),('Q',Q)]:
        for c in frame:
            a=frame[c].to_numpy(float); a=a[np.isfinite(a)]
            std=float(np.std(a)) if len(a) else float('nan')
            # The original exploratory audit identified 1e-9 as the near-constant cutoff.
            bad=bool(len(a)<50 or not np.isfinite(std) or std<1e-9)
            audit.append(dict(layer=layer,feature=c,n=len(a),std=std,
                              n_unique=int(len(np.unique(a))),near_constant=bad))
            if bad:degenerate.add((layer,c))
    pd.DataFrame(audit).to_csv(args.out+'_features.csv',index=False)
    # Fingerprint the data and analysis settings. Never silently reuse an incompatible checkpoint.
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes()
                               +pd.util.hash_pandas_object(baseline,index=True).values.tobytes()).hexdigest()
    meta=dict(data_fingerprint=fingerprint,permutations=args.permutations,
              neighbors=ks,seed=MASTER_SEED,near_constant_std=1e-9,
              conditioning='within-device OLS: width,m,avg_degree,degree_std,degree_cv',
              permutation='within device x width; region overlap not eliminated',
              sklearn_mi='mutual_info_regression, standardized x/y, fixed 1e-10 jitter, continuous target, version 3')
    meta_path=args.out+'_metadata.json'; checkpoint=args.out+'_checkpoint.csv'
    if os.path.exists(meta_path):
        with open(meta_path,encoding='utf8') as f: prior=json.load(f)
        if prior != meta:raise ValueError('Checkpoint settings/data differ. Use a new --out prefix.')
    elif os.path.exists(checkpoint):
        raise ValueError('Orphan checkpoint without metadata; use a new --out prefix')
    else:
        with open(meta_path,'w',encoding='utf8') as f:json.dump(meta,f,indent=2)
    done={}
    if os.path.exists(checkpoint):
        old=pd.read_csv(checkpoint)
        for rec in old.to_dict('records'):
            key=(int(rec['pair_index']),int(rec['k']))
            if key in done:raise ValueError('Duplicate checkpoint key '+str(key))
            done[key]=rec
    limit=min(len(baseline),args.max_pairs or len(baseline))
    if args.preflight:limit=min(limit,5)
    rows=[]
    for pair_no,rec in enumerate(baseline.itertuples(index=False)):
        if pair_no>=limit:break
        s,q=rec.structural_feature,rec.calibration_feature
        keys=[(pair_no,k) for k in ks]
        if all(key in done for key in keys):
            rows.extend(done[key] for key in keys)
            continue
        xx=S[s].to_numpy(float); yy=Q[q].to_numpy(float)
        valid=np.isfinite(xx)&np.isfinite(yy)
        ids=np.flatnonzero(valid);x=xx[valid];y=yy[valid]
        invalid_reason=('near_constant' if ('S',s) in degenerate or ('Q',q) in degenerate
                        else 'insufficient_n' if len(x)<50 else '')
        if not invalid_reason and (np.std(x)<1e-9 or np.std(y)<1e-9):
            invalid_reason='pairwise_near_constant'
        if not invalid_reason:
            index_map={int(old):i for i,old in enumerate(ids)}
            vb=[np.asarray([index_map[int(i)] for i in b if int(i) in index_map],dtype=int) for b in blocks]
            vb=[b for b in vb if len(b)>1]
            rg=np.random.default_rng(stable_seed(MASTER_SEED,'mi_audit_v3_standardized_jitter',s,q))
            # Noise is fixed across observed/permuted targets and all k values.
            # Fixed stochastic tie-breaking in standard-deviation units.
            # Apply an identical jittered observation vector to every permuted test.
            x=(x-np.mean(x))/np.std(x); y=(y-np.mean(y))/np.std(y)
            x=x+rg.normal(0,1e-10,len(x)); y=y+rg.normal(0,1e-10,len(y))
            null_orders=[]
            for rep in range(args.permutations):
                order=np.arange(len(y))
                for b in vb:order[b]=rg.permutation(b)
                null_orders.append(order)
        for k in ks:
            key=(pair_no,k)
            if key in done:
                rows.append(done[key]);continue
            result=dict(pair_index=pair_no,structural_feature=s,calibration_feature=q,k=k,
                        n=len(x),status=invalid_reason or 'ok',observed_mi=np.nan,
                        null_mean=np.nan,excess_mi=np.nan,p=1.0,B=args.permutations,
                        )
            if not invalid_reason and k<len(x):
                def estimate(v):
                    return float(mutual_info_regression(x.reshape(-1,1),v,
                                 n_neighbors=k,random_state=73)[0])
                observed=estimate(y)
                null=np.array([estimate(y[order]) for order in null_orders])
                result.update(observed_mi=observed,null_mean=float(null.mean()),
                              excess_mi=observed-float(null.mean()),
                              p=(1+int(np.sum(null>=observed)))/(1+len(null)))
            elif not invalid_reason:result['status']='k_exceeds_n'
            rows.append(result)
        if (pair_no+1)%args.checkpoint_every==0 or pair_no+1==limit:
            pd.DataFrame(rows).to_csv(checkpoint+'.tmp',index=False)
            os.replace(checkpoint+'.tmp',checkpoint)
            print('MI audited',pair_no+1,'/',len(baseline),flush=True)
    result=pd.DataFrame(rows)
    # ``baseline_q`` was an obsolete pre-repair cohort statistic from
    # the archived pair registry. It is not part of the corrected
    # 600-region analysis and is excluded from all public outputs,
    # including results reloaded from an existing checkpoint.
    result=result.drop(columns=['baseline_q'], errors='ignore')
    # BH family size stays 310: excluded/invalid pairs receive conservative p=1.
    if limit==len(baseline):
        for k,g in result.groupby('k'):
            result.loc[g.index,'q_bh_310']=benjamini_hochberg(g.p.to_numpy(float))
        result['q_bh_all_k']=benjamini_hochberg(result.p.to_numpy(float))
        reviewer_mi_exports(result,args.out+'_reviewer_supplement')
    else:
        result['q_bh_310']=np.nan
        result['q_bh_all_k']=np.nan  # Never report partial FDR as a full-family result.
    result.to_csv(args.out+'_pairs.csv',index=False)
    summary=result.groupby('k').agg(n_pairs=('p','size'),valid_pairs=('status',lambda v:int((v=='ok').sum())),
        median_mi=('observed_mi','median'),nominal_p05=('p',lambda v:int((v<.05).sum())),
        fdr_q05=('q_bh_310',lambda v:int((v<.05).sum()))).reset_index()
    summary.to_csv(args.out+'_summary.csv',index=False)
    print(summary.to_string(index=False))
    if limit==len(baseline):
        print('Conservative q values available (310 per k and all k combined).')
    if limit<len(baseline):print('PARTIAL RUN; do not interpret BH as final 310-pair inference')
    else:print('FULL 310-PAIR RUN COMPLETE (conditional block-permutation sensitivity only)')

# REV sensitivity: compare L1 and L2 distance definitions in exact-degree groups.
# This function is part of the single integrated analysis script; call it with
# the matched-region table and the same structural/calibration column lists as
# matched_pair_distances(). It uses the matrix-coherent permutation function.
def matched_distance_norm_sensitivity(matched_df, structural_cols, calibration_cols,
                                      n_perm=1000):
    from scipy.spatial.distance import pdist
    results = []
    for norm in ('l2', 'l1'):
        rows = []
        for gid, g in matched_df.groupby('group_id'):
            if len(g) < 2:
                continue
            g = g.reset_index(drop=True)
            xs = standardized_distance_matrix(g, structural_cols)
            xq = standardized_distance_matrix(g, calibration_cols)
            metric = 'euclidean' if norm == 'l2' else 'cityblock'
            ds = pdist(xs, metric=metric) / (math.sqrt(xs.shape[1]) if norm == 'l2' else xs.shape[1])
            dq = pdist(xq, metric=metric) / (math.sqrt(xq.shape[1]) if norm == 'l2' else xq.shape[1])
            for (i, j), sv, qv in zip(combinations(range(len(g)), 2), ds, dq):
                rows.append(dict(group_id=gid, device=g.loc[i,'device'], width=int(g.loc[i,'width']),
                                 patch_a=g.loc[i,'patch_id'], patch_b=g.loc[j,'patch_id'],
                                 structural_distance=float(sv), calibration_distance=float(qv)))
        result = matched_pair_permutation_test(pd.DataFrame(rows), n_perm=n_perm)
        result['distance_norm'] = norm
        results.append(result)
    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    if "--mi-full-audit" in sys.argv:
        mi_full_robustness_archive_cli()
    elif "--self-test" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--self-test',action='store_true')
        parser.add_argument('--archive',required=True)
        args=parser.parse_args()
        reviewer_self_test(args.archive)
    elif "--repair-archive" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--repair-archive',action='store_true')
        parser.add_argument('--source',required=True)
        parser.add_argument('--out',required=True)
        args=parser.parse_args()
        repair_legacy_duplicate_regions(args.source,args.out)
    elif "--reviewer-audit" in sys.argv:
        import argparse
        parser=argparse.ArgumentParser()
        parser.add_argument('--reviewer-audit',action='store_true')
        parser.add_argument('--archive',required=True)
        parser.add_argument('--out',default='reviewer_diagnostics')
        parser.add_argument('--preflight',action='store_true')
        args=parser.parse_args()
        reviewer_extra_audit(args.archive,args.out,preflight=args.preflight)
    elif "--matched-global-disjoint" in sys.argv:
        import argparse, zipfile, io
        parser = argparse.ArgumentParser()
        parser.add_argument('--matched-global-disjoint', action='store_true')
        parser.add_argument('--archive', required=True)
        parser.add_argument('--out', default='matched_global_disjoint')
        parser.add_argument('--seeds', type=int, default=20)
        parser.add_argument('--permutations', type=int, default=499)
        args = parser.parse_args()
        with zipfile.ZipFile(args.archive) as archive:
            def load_table(name):
                matches = [n for n in archive.namelist() if n.endswith('/tables/'+name+'.csv')]
                if len(matches) != 1: raise ValueError('Archive table missing or ambiguous: '+name)
                return pd.read_csv(io.BytesIO(archive.read(matches[0])))
            pairs = load_table('matched_pair_distances')
            nodes = load_table('patch_nodes')
        results, selected = global_disjoint_audit(pairs, nodes, args.seeds, args.permutations)
        results.to_csv(args.out+'_results.csv', index=False)
        selected.to_csv(args.out+'_selected.csv', index=False)
        print(results.to_string(index=False))
    elif "--matched-threshold-audit" in sys.argv:
        matched_threshold_archive_cli()
    elif "--overlap-cca-rv" in sys.argv:
        sys.argv.remove("--overlap-cca-rv")
        overlap_cca_rv_main()
    elif "--overlap-mi" in sys.argv:
        sys.argv.remove("--overlap-mi")
        overlap_mi_main()
    else:
        main()
