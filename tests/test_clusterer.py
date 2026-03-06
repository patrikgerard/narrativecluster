"""
Unit tests for narrativecluster.

These run without real embeddings — we synthesise random float vectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from narrativecluster import NarrativeClusterer, ClusterConfig


# ── Fixtures ──────────────────────────────────────────────────

DIM = 64
N_TRAIN = 500
N_NEW   = 100
N_QUERY = 50


def _make_df(n: int, seed: int = 0, text_prefix: str = "doc") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    embeds = rng.standard_normal((n, DIM)).astype(np.float32)
    return pd.DataFrame({
        "text":      [f"{text_prefix}_{i}" for i in range(n)],
        "platform":  rng.choice(["twitter", "telegram", "gab", "4chan"], size=n).tolist(),
        "embedding": list(embeds),
    })


@pytest.fixture
def small_cfg():
    return ClusterConfig(
        embed_col="embedding",
        text_col="text",
        site_col="platform",
        k_graph=8,
        tau_graph=0.0,   # low tau so we get edges on random data
        tau_edge=0.0,
        kq=8,
        leiden_resolution=1.0,
        cap_m=4,
        min_cluster_size=2,
        min_support_relaxed=2,
        min_support_strict=2,
        query_chunk=100,
    )


# ── Tests ──────────────────────────────────────────────────────

class TestFit:
    def test_fit_runs(self, small_cfg, tmp_path):
        df = _make_df(N_TRAIN)
        model = NarrativeClusterer(config=small_cfg, checkpoint_dir=tmp_path)
        model.fit(df)
        assert model.labels_ is not None
        assert len(model.labels_) == N_TRAIN
        assert model.n_clusters_ >= 1
        assert model.n_samples_fit_ == N_TRAIN

    def test_fit_idempotent_via_checkpoint(self, small_cfg, tmp_path):
        df = _make_df(N_TRAIN)
        m1 = NarrativeClusterer(config=small_cfg, checkpoint_dir=tmp_path)
        m1.fit(df)

        m2 = NarrativeClusterer(config=small_cfg, checkpoint_dir=tmp_path)
        m2.fit(df)  # should load from checkpoint, not re-fit

        np.testing.assert_array_equal(m1.labels_, m2.labels_)

    def test_dedup_removes_duplicates(self, small_cfg):
        df = _make_df(N_TRAIN)
        df_with_dups = pd.concat([df, df.iloc[:10]], ignore_index=True)
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df_with_dups, deduplicate=True)
        assert model.n_samples_fit_ == N_TRAIN


class TestPartialFit:
    def test_partial_fit_increases_n(self, small_cfg):
        df = _make_df(N_TRAIN)
        df_new = _make_df(N_NEW, seed=99, text_prefix="new_doc")
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df)
        model.partial_fit(df_new)
        assert model.n_samples_fit_ == N_TRAIN + N_NEW
        assert len(model.labels_) == N_TRAIN + N_NEW

    def test_partial_fit_ignores_already_seen(self, small_cfg):
        df = _make_df(N_TRAIN)
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df)
        model.partial_fit(df)  # all already seen
        assert model.n_samples_fit_ == N_TRAIN

    def test_partial_fit_raises_before_fit(self, small_cfg):
        model = NarrativeClusterer(config=small_cfg)
        with pytest.raises(RuntimeError, match="fit()"):
            model.partial_fit(_make_df(10))


class TestPredict:
    def test_predict_returns_correct_cols(self, small_cfg):
        df = _make_df(N_TRAIN)
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df)
        result = model.predict(df)
        for col in ["nv_pred_cluster", "nv_accepted", "nv_vote_frac",
                    "nv_votes_top", "nv_votes_tot", "nv_vote_margin", "nv_best_sim"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_predict_raises_before_fit(self, small_cfg):
        model = NarrativeClusterer(config=small_cfg)
        with pytest.raises(RuntimeError, match="fit()"):
            model.predict(_make_df(10))


class TestSaveLoad:
    def test_save_and_load(self, small_cfg, tmp_path):
        df = _make_df(N_TRAIN)
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df)
        model.save(tmp_path / "saved_model")

        loaded = NarrativeClusterer.load(tmp_path / "saved_model", config=small_cfg)
        assert loaded.n_samples_fit_ == model.n_samples_fit_
        assert loaded.n_clusters_    == model.n_clusters_
        np.testing.assert_array_equal(loaded.labels_, model.labels_)


class TestDiagnostics:
    def test_cluster_diagnostics(self, small_cfg):
        df = _make_df(N_TRAIN)
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df)
        diag = model.cluster_diagnostics()
        assert "cluster_id" in diag.columns
        assert "size" in diag.columns
        assert diag["size"].sum() == N_TRAIN

    def test_site_stats(self, small_cfg):
        df = _make_df(N_TRAIN)
        model = NarrativeClusterer(config=small_cfg)
        model.fit(df)
        df_pred = model.predict(df)
        global_rate, tbl = model.site_stats(df_pred)
        assert 0.0 <= global_rate <= 1.0
        assert "accept_rate" in tbl.columns
