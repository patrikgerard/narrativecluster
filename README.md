# narrativecluster

Claim-level narrative clustering for large social-media corpora.

**Stack:** HNSW (FAISS) → Leiden community detection → neighbour-vote cluster assignment

---

## Why this approach?

Clustering millions of social-media posts is an unsupervised problem with no clean ground truth, no fixed number of narratives, and a distribution that shifts every week as new stories emerge.  Most standard algorithms break down in one of a few predictable ways.

**K-means and its variants** require you to specify *k* up front.  For a corpus spanning thousands of distinct claims across a multi-year election cycle, the right *k* is unknowable in advance — and the algorithm will happily partition noise into fake clusters to fill whatever *k* you gave it.

**Gaussian Mixture Models and DP-Means** are theoretically appealing (they can infer *k* from the data) but brittle at scale.  DP-Means is sensitive to its bandwidth parameter in high-dimensional embedding space, where "distance" is dominated by the curse of dimensionality rather than semantic content, and it tends to either over-fragment coherent narratives or lump disparate ones together depending on how that single parameter lands.  GMMs have similar issues and become computationally expensive as *k* grows into the thousands.

**DBSCAN / HDBSCAN** handle arbitrary cluster shapes well on low-dimensional data but struggle in the intrinsic geometry of dense embedding spaces — nearly every point becomes a "core point" or noise depending on a single `eps` that is nearly impossible to calibrate in cosine space.  They also have no natural mechanism for assigning *new* documents to existing clusters without re-running the full algorithm.

### What narrativecluster does instead

The approach decouples **structure discovery** (Leiden on a cosine-thresholded KNN graph) from **point assignment** (neighbour-vote), which buys three things simultaneously: no fixed *k*, graceful handling of noise, and fast incremental updates.

**Structure discovery via graph community detection.** Rather than treating each post as a point in metric space, we build a sparse *k*-NN similarity graph where an edge only exists if two posts exceed a cosine similarity threshold (`tau_graph`). Leiden community detection then finds densely-connected subgraphs — natural narrative clusters — without any assumption about their number, shape, or size. Leiden is resolution-parameterised, which is semantically interpretable: higher resolution → finer-grained narratives, lower → coarser topical buckets. Critically, isolated posts and borderline cases simply have no edges and fall into small singleton clusters rather than being force-assigned to the nearest centroid.

**Neighbour-vote assignment for new data.** Once the graph structure is learned, assigning a new post is cheap and explainable: query the HNSW index for its nearest neighbours, filter to neighbours above `tau_edge`, and take a plurality vote over their cluster memberships. The vote fraction and margin give a per-post confidence score, so you can dial your own precision/recall tradeoff *after* fitting — without touching the model. This is O(log N) per new post rather than O(N²).

**A concrete example.** Suppose you have five posts:

```
A: "Ballots were found in a dumpster in Maricopa County — proof of fraud"
B: "Election officials in Arizona caught destroying ballots"
C: "The mRNA vaccine is causing myocarditis in young athletes"
D: "Pfizer trial data shows spike protein accumulates in organs"
E: "My dog had a great walk this morning :)"
```

- *K-means with k=2* groups {A,B,C,D} vs {E} — conflating two distinct narratives because it must fill both centroids.
- *DP-Means* may correctly split {A,B} from {C,D}, but will almost certainly pull post E into whichever centroid is geometrically closest rather than leaving it unclaimed.
- *narrativecluster*: A↔B have high cosine similarity and form one Leiden community; C↔D form another; E has no neighbours above `tau_graph` and lands in a singleton — it is simply *not assigned* when you call `predict()`.  That is the correct answer for noise.

**Speed.** HNSW graph construction is O(N log N) and approximate-neighbour search at prediction time is effectively O(log N). On a 12M-post corpus, `fit()` takes ~20 minutes on a single CPU; `predict()` on a new 100K-post batch takes under 2 minutes. DP-Means or a full GMM on the same corpus would be intractable.

---

## Install

```bash
pip install narrativecluster
# GPU variant (requires faiss-gpu):
pip install "narrativecluster[gpu]"
```

From source:

```bash
git clone https://github.com/yourorg/narrativecluster
cd narrativecluster
pip install -e ".[dev]"
```

---

## Quick start

```python
from narrativecluster import NarrativeClusterer, ClusterConfig

cfg = ClusterConfig(
    embed_col="embedding",   # column holding np.ndarray / list embeddings
    text_col="text",
    site_col="platform",
)

model = NarrativeClusterer(config=cfg, checkpoint_dir="./checkpoints")

# Full fit — idempotent: loads from checkpoint if it exists
model.fit(df_train)

# Incremental update — adds new rows, re-runs Leiden, keeps cluster space stable
model.partial_fit(df_new_batch)

# Neighbour-vote assignment for any DataFrame
df_pred = model.predict(df_query)
# df_pred now has: nv_pred_cluster, nv_accepted, nv_vote_frac, …

# Per-site acceptance stats
global_rate, site_tbl = model.site_stats(df_pred)
print(site_tbl)

# Save / load
model.save("./my_model")
model2 = NarrativeClusterer.load("./my_model", config=cfg)
```

---

## CLI

```bash
# Full fit + predict
narrativecluster --raw raw_posts.parquet --out ./checkpoints

# Incremental update
narrativecluster --dedup ./checkpoints/df_unique.parquet \
                 --partial_fit new_posts.parquet \
                 --out ./checkpoints

# Force re-predict
narrativecluster --dedup ./checkpoints/df_unique.parquet \
                 --out ./checkpoints \
                 --force_repredict
```

---

## partial_fit semantics

`partial_fit` is designed for **streaming / incremental ingestion**:

1. New rows are deduplicated against themselves *and* all previously seen texts.
2. They are mean-centred using the **original** µ (the embedding space is not re-centred — this keeps the geometry stable across batches).
3. New vectors are added to the live HNSW index.
4. KNN arrays are extended; Leiden is re-run on the combined graph.

> **Note:** Cluster IDs may shift after each `partial_fit` because Leiden is non-deterministic and the graph changes. If you need stable IDs across incremental updates, pin `random_seed` and store a cluster-label mapping separately.

---

## Configuration

All hyper-parameters live in `ClusterConfig`:

| Parameter | Default | Description |
|---|---|---|
| `embed_col` | `"embedding_aligned"` | DataFrame column with embeddings |
| `hnsw_m` | 32 | HNSW M parameter |
| `ef_construction` | 200 | HNSW efConstruction |
| `ef_search` | 128 | HNSW efSearch |
| `k_graph` | 64 | KNN neighbours for Leiden graph |
| `tau_graph` | 0.68 | Cosine-sim threshold for graph edges |
| `do_mutual` | False | Keep only mutual edges |
| `cap_m` | 20 | Max out-degree per node (None = uncapped) |
| `leiden_resolution` | 1.0 | Leiden resolution parameter |
| `kq` | 50 | Neighbours queried at prediction time |
| `tau_edge` | 0.68 | Min sim for a vote to count |
| `min_cluster_size` | 4 | Clusters smaller than this are ineligible |
| `vote_margin` | 2 | Min vote margin (top − second) to accept |

---

## Output columns (predict)

| Column | Type | Description |
|---|---|---|
| `nv_pred_cluster` | int32 | Predicted Leiden cluster id (-1 = rejected) |
| `nv_accepted` | bool | Whether the vote passed quality thresholds |
| `nv_votes_top` | int16 | Votes for winning cluster |
| `nv_votes_tot` | int16 | Total eligible neighbour votes |
| `nv_vote_frac` | float32 | Top votes / total votes |
| `nv_vote_margin` | float32 | top_votes − second_votes |
| `nv_best_sim` | float32 | Max cosine similarity among winning-cluster neighbours |

---

## License

MIT
