"""
Low-level math + I/O utilities shared across the package.
"""
from __future__ import annotations

import os
import json
import gc
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────
# Numeric helpers
# ──────────────────────────────────────────────────────────────

def l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation (in-place safe copy)."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size for two proportions."""
    p1 = float(np.clip(p1, 1e-12, 1 - 1e-12))
    p2 = float(np.clip(p2, 1e-12, 1 - 1e-12))
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))


def effect_size_label(h: float) -> str:
    ah = abs(h)
    if ah >= 0.8:
        return "large"
    if ah >= 0.5:
        return "medium"
    if ah >= 0.2:
        return "small"
    return "negligible"


# ──────────────────────────────────────────────────────────────
# FAISS helpers
# ──────────────────────────────────────────────────────────────

def build_hnsw_ip_index(d: int, M: int, ef_construction: int, ef_search: int) -> faiss.Index:
    idx = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = ef_construction
    idx.hnsw.efSearch = ef_search
    return idx


def knn_search_chunked(
    index: faiss.Index,
    X: np.ndarray,
    k: int,
    chunk: int = 100_000,
    desc: str = "FAISS search",
) -> Tuple[np.ndarray, np.ndarray]:
    """Search index in chunks; returns (I, D) both shape (n, k)."""
    n = X.shape[0]
    I = np.empty((n, k), dtype=np.int64)
    D = np.empty((n, k), dtype=np.float32)
    for s in tqdm(range(0, n, chunk), desc=desc, dynamic_ncols=True):
        e = min(s + chunk, n)
        d_block, i_block = index.search(X[s:e], k)
        I[s:e] = i_block.astype(np.int64, copy=False)
        D[s:e] = d_block.astype(np.float32, copy=False)
    return I, D


# ──────────────────────────────────────────────────────────────
# Graph construction helpers
# ──────────────────────────────────────────────────────────────

def edges_from_knn(
    I: np.ndarray,
    D: np.ndarray,
    tau: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (src, dst, sim) edges from KNN arrays filtered by tau."""
    n, k = I.shape
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = I.reshape(-1).astype(np.int64, copy=False)
    sim = D.reshape(-1).astype(np.float32, copy=False)
    ok = (dst >= 0) & (src != dst) & (sim >= tau)
    return src[ok], dst[ok], sim[ok]


def mutual_filter(
    rows: np.ndarray,
    cols: np.ndarray,
    sims: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only mutually-present edges (i→j AND j→i)."""
    keys = rows.astype(np.uint64) * np.uint64(n) + cols.astype(np.uint64)
    keys_rev = cols.astype(np.uint64) * np.uint64(n) + rows.astype(np.uint64)
    s = np.sort(keys)
    pos = np.searchsorted(s, keys_rev)
    ok = (pos < s.size) & (s[pos] == keys_rev)
    return rows[ok], cols[ok], sims[ok]


def cap_degree(
    rows: np.ndarray,
    cols: np.ndarray,
    sims: np.ndarray,
    m: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep top-m outgoing edges per node by similarity."""
    order = np.lexsort((-sims, rows))
    rows, cols, sims = rows[order], cols[order], sims[order]
    keep = np.zeros(rows.size, dtype=bool)
    _, start = np.unique(rows, return_index=True)
    start = np.append(start, rows.size)
    for a, b in zip(start[:-1], start[1:]):
        keep[a : min(a + m, b)] = True
    return rows[keep], cols[keep], sims[keep]


# ──────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────

def site_acceptance_table(
    df: pd.DataFrame,
    accepted_mask: np.ndarray,
    site_col: str,
) -> Tuple[float, pd.DataFrame]:
    """Compute per-site acceptance rates vs global rate with Cohen's h."""
    accepted_mask = np.asarray(accepted_mask, dtype=bool)
    if len(accepted_mask) != len(df):
        raise ValueError(
            f"accepted_mask length {len(accepted_mask)} != len(df) {len(df)}"
        )
    global_rate = float(np.mean(accepted_mask))
    site_vals = df[site_col].astype(str).values
    vc = pd.Series(site_vals).value_counts()
    rows = []
    for site, n_site in vc.items():
        m = site_vals == site
        acc = int(np.sum(accepted_mask[m]))
        rate = acc / max(int(n_site), 1)
        h = cohens_h(rate, global_rate)
        rows.append(
            dict(
                site=site,
                count=int(n_site),
                accepted_count=acc,
                accept_rate=rate,
                abs_diff_from_global=rate - global_rate,
                cohens_h=float(h),
                effect_size=effect_size_label(h),
            )
        )
    tbl = (
        pd.DataFrame(rows)
        .set_index("site")
        .sort_values("accept_rate", ascending=False)
    )
    return global_rate, tbl


# ──────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
