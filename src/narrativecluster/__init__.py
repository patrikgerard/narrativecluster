"""
narrativecluster
~~~~~~~~~~~~~~~~
Claim-level narrative clustering with HNSW + Leiden + neighbour-vote assignment.

Quick start
-----------
>>> from narrativecluster import NarrativeClusterer, ClusterConfig
>>> cfg = ClusterConfig(embed_col="embedding", text_col="text", site_col="platform")
>>> model = NarrativeClusterer(config=cfg, checkpoint_dir="./my_checkpoints")
>>> model.fit(df_train)                  # full fit
>>> model.partial_fit(df_new_batch)      # incremental update
>>> df_pred = model.predict(df_query)    # neighbour-vote assignment
"""

from .clusterer import NarrativeClusterer
from .config import ClusterConfig
from ._utils import site_acceptance_table, cohens_h, effect_size_label

__version__ = "0.1.0"
__all__ = [
    "NarrativeClusterer",
    "ClusterConfig",
    "site_acceptance_table",
    "cohens_h",
    "effect_size_label",
]
