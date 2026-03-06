"""
NarrativeClusterer — sklearn-style estimator for claim-level narrative clustering.

Workflow
--------
1. fit(df)           — full fit on a deduplicated DataFrame
2. partial_fit(df)   — add new rows to an existing fit (incremental HNSW + re-Leiden)
3. predict(df)       — neighbour-vote assignment for any DataFrame (uses existing index)
4. fit_predict(df)   — convenience: fit then predict on the same data
"""
from __future__ import annotations

import gc
import os
import json
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import faiss
import igraph as ig
import leidenalg
from tqdm import tqdm

from .config import ClusterConfig
from ._utils import (
    l2_normalize,
    build_hnsw_ip_index,
    knn_search_chunked,
    edges_from_knn,
    mutual_filter,
    cap_degree,
    site_acceptance_table,
    ensure_dir,
    save_json,
    load_json,
)


class NarrativeClusterer:
    """
    Claim-level narrative clustering with HNSW + Leiden + neighbour-vote assignment.

    Parameters
    ----------
    config : ClusterConfig, optional
        Hyperparameter bundle. Defaults are sensible for ~millions of posts.
    checkpoint_dir : str | Path, optional
        If given, all artefacts (index, KNN arrays, labels, metadata) are
        persisted here so subsequent calls are idempotent.

    Attributes (set after fit)
    --------------------------
    index_          : faiss.Index — HNSW index (mean-centred + L2-normalised space)
    mu_             : np.ndarray shape (1, D) — per-dimension mean used for centring
    labels_         : np.ndarray shape (N,) — Leiden cluster id for each training row
    df_unique_      : pd.DataFrame — deduplicated training data (with leiden_cluster col)
    n_clusters_     : int
    n_samples_fit_  : int — total rows that have been fit (including partial_fit calls)
    """

    # ── State-file names inside checkpoint_dir ──────────────
    _INDEX_FNAME  = "faiss.index"
    _MU_FNAME     = "mu.npy"
    _I_KNN_FNAME  = "I_knn.npy"
    _D_KNN_FNAME  = "D_knn.npy"
    _LABELS_FNAME = "leiden_labels.npy"
    _META_FNAME   = "meta.json"
    _DF_FNAME     = "df_unique.parquet"

    def __init__(
        self,
        config: Optional[ClusterConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
    ):
        self.config = config or ClusterConfig()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # fitted attributes
        self.index_: Optional[faiss.Index] = None
        self.mu_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.df_unique_: Optional[pd.DataFrame] = None
        self.n_clusters_: int = 0
        self.n_samples_fit_: int = 0

        # internal: raw KNN arrays (needed for partial_fit graph merge)
        self._I_knn: Optional[np.ndarray] = None
        self._D_knn: Optional[np.ndarray] = None

    # ────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame, deduplicate: bool = True) -> "NarrativeClusterer":
        """
        Full fit on *df*.

        Parameters
        ----------
        df : DataFrame with columns for text, site, and embeddings (see config).
        deduplicate : if True (default) drop duplicate texts before fitting.
        """
        cfg = self.config
        if deduplicate:
            df = self._dedup(df, cfg.text_col)

        # Try loading from checkpoint first
        if self.checkpoint_dir and self._checkpoint_exists():
            print("[NarrativeClusterer.fit] Checkpoint found — loading instead of re-fitting.")
            self._load_checkpoint(df)
            return self

        X_mc, mu = self._embed_matrix(df, cfg.embed_col)
        index = self._build_index(X_mc, cfg)
        I, D = knn_search_chunked(index, X_mc, cfg.k_graph, cfg.query_chunk, "KNN search")
        labels = self._run_leiden(len(df), I, D, cfg)

        # Persist state
        self.index_ = index
        self.mu_ = mu
        self._I_knn = I
        self._D_knn = D
        self.labels_ = labels
        self.n_clusters_ = int(labels.max()) + 1
        self.n_samples_fit_ = len(df)

        df = df.copy()
        df["leiden_cluster"] = labels.astype(np.int32)
        self.df_unique_ = df

        if self.checkpoint_dir:
            self._save_checkpoint()

        print(f"[NarrativeClusterer] fit complete — {len(df)} rows, {self.n_clusters_} clusters")
        return self

    def partial_fit(self, df: pd.DataFrame, deduplicate: bool = True) -> "NarrativeClusterer":
        """
        Incrementally add new rows to an existing fit.

        Strategy
        --------
        1. Deduplicate *df* against both itself and existing data (by text_col).
        2. Mean-centre with the *existing* mu (no refit of mu — keeps the space stable).
        3. Add new vectors to the live HNSW index.
        4. KNN-search the new rows to get their neighbourhood.
        5. Concatenate old + new KNN arrays, rebuild the Leiden graph, re-run Leiden.
           Cluster IDs may shift; downstream consumers should re-read `labels_`.
        6. Append new rows to df_unique_.

        Raises
        ------
        RuntimeError if called before fit().
        """
        if self.index_ is None:
            raise RuntimeError("Call fit() before partial_fit().")

        cfg = self.config

        # 1. Filter out already-seen texts
        if deduplicate:
            df = self._dedup(df, cfg.text_col)
        if self.df_unique_ is not None:
            existing_texts = set(self.df_unique_[cfg.text_col].values)
            df = df[~df[cfg.text_col].isin(existing_texts)].reset_index(drop=True)

        if len(df) == 0:
            print("[partial_fit] No new rows after deduplication — skipping.")
            return self

        print(f"[partial_fit] Adding {len(df)} new rows…")

        # 2. Centre with existing mu (no mu update to keep space stable)
        X_new = np.vstack(df[cfg.embed_col].values).astype(np.float32)
        X_new_mc = l2_normalize(X_new - self.mu_)

        # 3. Add to index
        offset = self.n_samples_fit_   # global ids for new rows start here
        self.index_.add(X_new_mc)

        # 4. KNN for new rows (search full index including the new vectors)
        I_new, D_new = knn_search_chunked(
            self.index_, X_new_mc, cfg.k_graph, cfg.query_chunk, "KNN (new rows)"
        )

        # 5. Merge KNN arrays and re-run Leiden
        # Existing rows keep their KNN arrays unchanged.
        # New rows get freshly computed KNN (neighbour ids are already global).
        n_old = self.n_samples_fit_
        n_new = len(df)
        n_total = n_old + n_new

        I_merged = np.concatenate([self._I_knn, I_new], axis=0)
        D_merged = np.concatenate([self._D_knn, D_new], axis=0)

        labels = self._run_leiden(n_total, I_merged, D_merged, cfg)

        # 6. Update state
        self._I_knn = I_merged
        self._D_knn = D_merged
        self.labels_ = labels
        self.n_clusters_ = int(labels.max()) + 1
        self.n_samples_fit_ = n_total

        df = df.copy()
        df["leiden_cluster"] = labels[n_old:].astype(np.int32)

        # Update leiden_cluster for existing rows too (ids may shift after re-Leiden)
        existing = self.df_unique_.copy()
        existing["leiden_cluster"] = labels[:n_old].astype(np.int32)
        self.df_unique_ = pd.concat([existing, df], ignore_index=True)

        if self.checkpoint_dir:
            self._save_checkpoint()

        print(
            f"[partial_fit] Done — total {self.n_samples_fit_} rows, "
            f"{self.n_clusters_} clusters"
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Neighbour-vote cluster assignment for *df*.

        Returns a copy of *df* with extra columns:
          nv_pred_cluster, nv_accepted, nv_votes_top, nv_votes_tot,
          nv_vote_frac, nv_vote_margin, nv_best_sim
        """
        if self.index_ is None:
            raise RuntimeError("Call fit() before predict().")

        cfg = self.config
        result = self._neighbor_vote(df, cfg)
        return result

    def fit_predict(self, df: pd.DataFrame, deduplicate: bool = True) -> pd.DataFrame:
        """Convenience: fit then predict on the same data."""
        self.fit(df, deduplicate=deduplicate)
        return self.predict(self.df_unique_)

    def site_stats(self, df_pred: pd.DataFrame) -> Tuple[float, pd.DataFrame]:
        """
        Compute per-site acceptance rates from a predict() result.

        Returns (global_rate, site_table).
        """
        if self.config.site_col not in df_pred.columns:
            raise KeyError(f"site_col='{self.config.site_col}' not in df_pred")
        mask = df_pred["nv_accepted"].values.astype(bool)
        return site_acceptance_table(df_pred, mask, self.config.site_col)

    def cluster_diagnostics(self) -> pd.DataFrame:
        """
        Return a small DataFrame summarising cluster size distribution.
        Requires fit() to have been called.
        """
        if self.labels_ is None:
            raise RuntimeError("No labels — call fit() first.")
        sizes = np.bincount(self.labels_)
        cfg = self.config
        rows = []
        for cid, sz in enumerate(sizes):
            rows.append(dict(cluster_id=cid, size=int(sz),
                             eligible=sz >= cfg.min_cluster_size))
        return (pd.DataFrame(rows)
                  .sort_values("size", ascending=False)
                  .reset_index(drop=True))

    # ────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
        before = len(df)
        df = (
            df.assign(_dup_count=df.groupby(text_col)[text_col].transform("count"))
              .drop_duplicates(subset=[text_col])
              .reset_index(drop=True)
        )
        print(f"[dedup] {before} → {len(df)} rows (removed {before - len(df)} duplicates)")
        return df

    @staticmethod
    def _embed_matrix(
        df: pd.DataFrame, embed_col: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack(df[embed_col].values).astype(np.float32)
        mu = X.mean(axis=0, keepdims=True).astype(np.float32)
        X_mc = l2_normalize(X - mu)
        return X_mc, mu

    @staticmethod
    def _build_index(X_mc: np.ndarray, cfg: ClusterConfig) -> faiss.Index:
        index = build_hnsw_ip_index(
            X_mc.shape[1], cfg.hnsw_m, cfg.ef_construction, cfg.ef_search
        )
        index.add(X_mc)
        print(f"[index] Built — ntotal={index.ntotal}, d={index.d}")
        return index

    @staticmethod
    def _run_leiden(
        n: int, I: np.ndarray, D: np.ndarray, cfg: ClusterConfig
    ) -> np.ndarray:
        rows, cols, sims = edges_from_knn(I, D, cfg.tau_graph)
        if cfg.do_mutual:
            rows, cols, sims = mutual_filter(rows, cols, sims, n)
        if cfg.cap_m is not None:
            rows, cols, sims = cap_degree(rows, cols, sims, cfg.cap_m)

        # undirected graph — keep one direction
        mask = rows < cols
        edges = list(zip(rows[mask].tolist(), cols[mask].tolist()))
        weights = sims[mask].astype(float).tolist()

        g = ig.Graph(n=n, edges=edges, directed=False)
        g.es["weight"] = weights

        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=cfg.leiden_resolution,
        )
        labels = np.array(part.membership, dtype=np.int32)
        n_clusters = int(labels.max()) + 1
        print(f"[leiden] {n_clusters} clusters | {len(edges)} edges")
        return labels

    def _neighbor_vote(self, df: pd.DataFrame, cfg: ClusterConfig) -> pd.DataFrame:
        index = self.index_
        mu = self.mu_
        labels_all = np.asarray(self.labels_, dtype=np.int32)
        cluster_sizes = np.bincount(labels_all)
        eligible = cluster_sizes >= cfg.min_cluster_size
        node_eligible = eligible[labels_all]

        n = len(df)
        pred_cluster   = np.full(n, -1, dtype=np.int32)
        pred_ok        = np.zeros(n, dtype=bool)
        vtop           = np.zeros(n, dtype=np.int16)
        vtot           = np.zeros(n, dtype=np.int16)
        vfrac          = np.full(n, np.nan, dtype=np.float32)
        vmarg          = np.full(n, np.nan, dtype=np.float32)
        bsim           = np.full(n, np.nan, dtype=np.float32)

        for s in tqdm(range(0, n, cfg.query_chunk), desc="Neighbour-vote", dynamic_ncols=True):
            e = min(s + cfg.query_chunk, n)
            Q = np.vstack(df.iloc[s:e][cfg.embed_col].values).astype(np.float32)
            Q = l2_normalize(Q - mu)

            Dq, Iq = index.search(Q, cfg.kq)
            row_ids = np.arange(s, e, dtype=np.int64)

            ok = (
                (Dq >= cfg.tau_edge)
                & (Iq >= 0)
                & (Iq != row_ids[:, None])
                & node_eligible[np.clip(Iq, 0, len(node_eligible) - 1)]
            )
            # Zero out invalid entries to avoid indexing issues
            Iq_safe = np.where(ok, Iq, 0)

            for r in range(e - s):
                m = ok[r]
                if not m.any():
                    continue
                nbr_ids  = Iq[r, m].astype(np.int64)
                nbr_sims = Dq[r, m].astype(np.float32)
                nbr_cids = labels_all[nbr_ids]
                total    = nbr_cids.size

                if total >= cfg.strict_votes_at:
                    min_sup  = cfg.min_support_strict
                    vote_tau = cfg.vote_tau_strict
                else:
                    min_sup  = cfg.min_support_relaxed
                    vote_tau = cfg.vote_tau_relaxed

                if total < min_sup:
                    continue

                u, cts = np.unique(nbr_cids, return_counts=True)
                order  = np.argsort(-cts)
                top_cid    = int(u[order[0]])
                top_votes  = int(cts[order[0]])
                sec_votes  = int(cts[order[1]]) if order.size > 1 else 0
                frac       = top_votes / total
                margin     = top_votes - sec_votes

                if frac >= vote_tau and margin >= cfg.vote_margin:
                    j = s + r
                    pred_cluster[j] = top_cid
                    pred_ok[j]      = True
                    vtop[j]         = top_votes
                    vtot[j]         = total
                    vfrac[j]        = float(frac)
                    vmarg[j]        = float(margin)
                    bsim[j]         = float(np.max(nbr_sims))

            del Q, Dq, Iq, ok
            gc.collect()

        out = df.copy()
        out["nv_pred_cluster"]  = pred_cluster
        out["nv_accepted"]      = pred_ok
        out["nv_votes_top"]     = vtop
        out["nv_votes_tot"]     = vtot
        out["nv_vote_frac"]     = vfrac
        out["nv_vote_margin"]   = vmarg
        out["nv_best_sim"]      = bsim
        return out

    # ────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ────────────────────────────────────────────────────────────

    def _checkpoint_exists(self) -> bool:
        d = self.checkpoint_dir
        return all(
            (d / f).exists()
            for f in [
                self._INDEX_FNAME,
                self._MU_FNAME,
                self._LABELS_FNAME,
                self._META_FNAME,
                self._DF_FNAME,
            ]
        )

    def _save_checkpoint(self) -> None:
        d = self.checkpoint_dir
        ensure_dir(d)
        faiss.write_index(self.index_, str(d / self._INDEX_FNAME))
        np.save(str(d / self._MU_FNAME), self.mu_.astype(np.float32))
        np.save(str(d / self._LABELS_FNAME), self.labels_.astype(np.int32))
        np.save(str(d / self._I_KNN_FNAME), self._I_knn.astype(np.int64))
        np.save(str(d / self._D_KNN_FNAME), self._D_knn.astype(np.float32))
        self.df_unique_.to_parquet(str(d / self._DF_FNAME), engine="pyarrow", compression="zstd")
        meta = {
            "n_samples_fit": self.n_samples_fit_,
            "n_clusters": self.n_clusters_,
            "config": self.config.to_dict(),
        }
        save_json(meta, d / self._META_FNAME)
        print(f"[checkpoint] Saved to {d}")

    def _load_checkpoint(self, df_fallback: Optional[pd.DataFrame] = None) -> None:
        d = self.checkpoint_dir
        print(f"[checkpoint] Loading from {d}")
        self.index_ = faiss.read_index(str(d / self._INDEX_FNAME))
        self.index_.hnsw.efSearch = self.config.ef_search
        self.mu_    = np.load(str(d / self._MU_FNAME)).astype(np.float32)
        self.labels_ = np.load(str(d / self._LABELS_FNAME))
        self._I_knn  = np.load(str(d / self._I_KNN_FNAME), mmap_mode="r")
        self._D_knn  = np.load(str(d / self._D_KNN_FNAME), mmap_mode="r")
        self.df_unique_ = pd.read_parquet(str(d / self._DF_FNAME), engine="pyarrow")
        meta = load_json(d / self._META_FNAME)
        self.n_samples_fit_ = int(meta["n_samples_fit"])
        self.n_clusters_    = int(meta["n_clusters"])
        # Restore config from checkpoint (caller's config overrides)
        saved_cfg = ClusterConfig.from_dict(meta.get("config", {}))
        # Prefer the live config over the saved one (allows hyper-param tweaks)
        print(f"[checkpoint] Loaded — {self.n_samples_fit_} rows, {self.n_clusters_} clusters")

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted model to *path* (a directory)."""
        prev_dir = self.checkpoint_dir
        self.checkpoint_dir = Path(path)
        self._save_checkpoint()
        self.checkpoint_dir = prev_dir

    @classmethod
    def load(cls, path: Union[str, Path], config: Optional[ClusterConfig] = None) -> "NarrativeClusterer":
        """Load a previously saved model from *path*."""
        obj = cls(config=config, checkpoint_dir=path)
        obj._load_checkpoint()
        return obj
