"""
Centralised hyperparameter config — pass one object around instead of dozens of kwargs.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ClusterConfig:
    # ── Embedding ────────────────────────────────────────────
    embed_col: str = "embedding_aligned"
    text_col: str = "cleaned_text"
    site_col: str = "site"

    # ── HNSW index ───────────────────────────────────────────
    hnsw_m: int = 32
    ef_construction: int = 200
    ef_search: int = 128

    # ── KNN graph (for clustering) ───────────────────────────
    k_graph: int = 64        # neighbours stored per node for the Leiden graph
    tau_graph: float = 0.68  # cosine-sim threshold for graph edges
    do_mutual: bool = False   # keep only mutual edges
    cap_m: Optional[int] = 20  # max out-degree per node; None = uncapped

    # ── Leiden ───────────────────────────────────────────────
    leiden_resolution: float = 1.0

    # ── Neighbour-vote prediction ────────────────────────────
    kq: int = 50              # neighbours to query at prediction time
    tau_edge: float = 0.68    # min sim for a vote to count
    min_cluster_size: int = 4 # clusters smaller than this are ineligible

    strict_votes_at: int = 20
    min_support_strict: int = 5
    min_support_relaxed: int = 4
    vote_tau_strict: float = 0.68
    vote_tau_relaxed: float = 0.65
    vote_margin: int = 2

    # ── Runtime ──────────────────────────────────────────────
    query_chunk: int = 50_000
    random_seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterConfig":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in fields})
