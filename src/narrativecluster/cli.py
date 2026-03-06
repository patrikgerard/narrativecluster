"""
CLI entry point: `narrativecluster`

Thin wrapper around NarrativeClusterer that mirrors the original script's
argument surface so existing workflows don't need changes.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from .clusterer import NarrativeClusterer
from .config import ClusterConfig
from ._utils import ensure_dir


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="narrativecluster",
        description="Claim-level narrative clustering (HNSW + Leiden + neighbour-vote)",
    )
    ap.add_argument("--raw", dest="raw_parquet", default=None,
                    help="Raw parquet path (only needed if dedup file doesn't exist).")
    ap.add_argument("--dedup", dest="dedup_path",
                    default="annotation_inputs/df_claim_level.parquet")
    ap.add_argument("--out", dest="out_dir",
                    default="annotation_inputs/checkpoints/hnsw_claim_level")
    ap.add_argument("--partial_fit", dest="partial_parquet", default=None,
                    help="Parquet with new rows to incrementally add via partial_fit.")

    ap.add_argument("--text_col",  default="cleaned_text")
    ap.add_argument("--site_col",  default="site")
    ap.add_argument("--embed_col", default="embedding_aligned")

    ap.add_argument("--hnsw_m",      type=int,   default=32)
    ap.add_argument("--ef_c",        type=int,   default=200)
    ap.add_argument("--ef_s",        type=int,   default=128)
    ap.add_argument("--k_graph",     type=int,   default=64)
    ap.add_argument("--tau_graph",   type=float, default=0.68)
    ap.add_argument("--do_mutual",   action="store_true", default=False)
    ap.add_argument("--cap_m",       type=int,   default=20)
    ap.add_argument("--leiden_res",  type=float, default=1.0)
    ap.add_argument("--kq",          type=int,   default=50)
    ap.add_argument("--tau_edge",    type=float, default=0.68)
    ap.add_argument("--query_chunk", type=int,   default=50_000)
    ap.add_argument("--seed",        type=int,   default=42)

    ap.add_argument("--force_repredict", action="store_true",
                    help="Ignore saved predictions and rerun neighbour-vote.")
    args = ap.parse_args(argv)

    ensure_dir(args.out_dir)

    cfg = ClusterConfig(
        embed_col=args.embed_col,
        text_col=args.text_col,
        site_col=args.site_col,
        hnsw_m=args.hnsw_m,
        ef_construction=args.ef_c,
        ef_search=args.ef_s,
        k_graph=args.k_graph,
        tau_graph=args.tau_graph,
        do_mutual=args.do_mutual,
        cap_m=args.cap_m,
        leiden_resolution=args.leiden_res,
        kq=args.kq,
        tau_edge=args.tau_edge,
        query_chunk=args.query_chunk,
        random_seed=args.seed,
    )

    # ── Load data ────────────────────────────────────────────
    import os
    if os.path.exists(args.dedup_path):
        print(f"[cli] Loading dedup df: {args.dedup_path}")
        df = pd.read_parquet(args.dedup_path, engine="pyarrow")
    elif args.raw_parquet:
        print(f"[cli] Loading raw parquet: {args.raw_parquet}")
        df = pd.read_parquet(args.raw_parquet, engine="pyarrow")
    else:
        print("[cli] ERROR: provide --raw or --dedup", file=sys.stderr)
        sys.exit(1)

    # ── Fit ──────────────────────────────────────────────────
    model = NarrativeClusterer(config=cfg, checkpoint_dir=args.out_dir)
    model.fit(df)

    # ── Optional partial_fit ─────────────────────────────────
    if args.partial_parquet:
        print(f"[cli] partial_fit from: {args.partial_parquet}")
        df_new = pd.read_parquet(args.partial_parquet, engine="pyarrow")
        model.partial_fit(df_new)

    # ── Predict ──────────────────────────────────────────────
    pred_path = os.path.join(args.out_dir, f"pred_FULL_tauEdge{args.tau_edge}_kq{args.kq}.parquet")

    if os.path.exists(pred_path) and not args.force_repredict:
        print(f"[cli] Loading saved predictions: {pred_path}")
        df_pred = pd.read_parquet(pred_path, engine="pyarrow")
    else:
        df_pred = model.predict(model.df_unique_)
        df_pred.to_parquet(pred_path, engine="pyarrow", compression="zstd")
        print(f"[cli] Saved predictions: {pred_path}")

    # ── Site stats ───────────────────────────────────────────
    if cfg.site_col in df_pred.columns:
        global_rate, site_tbl = model.site_stats(df_pred)
        stats_csv = os.path.join(args.out_dir, f"site_acceptance_tauEdge{args.tau_edge}_kq{args.kq}.csv")
        site_tbl.to_csv(stats_csv)
        print(f"\n[cli] Global acceptance rate: {global_rate:.4f}")
        print(f"[cli] Site table saved: {stats_csv}")
        with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
            print(site_tbl)

    # ── Cluster diagnostics ───────────────────────────────────
    diag = model.cluster_diagnostics()
    print("\n[cli] Cluster size diagnostics:")
    print(diag.describe())


if __name__ == "__main__":
    main()
