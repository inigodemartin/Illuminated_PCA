#!/usr/bin/env python3
"""
Shared helpers for future standalone-HTML general UMAP scripts (the UMAP
counterpart of general_pca_common.py, which serves presence_absence_pca.py
and general_pca_abundance.py).

UMAP has no linear components_ the way PCA/TruncatedSVD does, so there is
no UMAP equivalent of top_loadings_by_pc / write_top_loadings_tsv /
compute_species_contributions -- per an explicit decision, UMAP-based
scripts built on top of this module have no "top GO terms per axis"
sidebar (a post-hoc analog, e.g. correlating each GO term's abundance with
each embedding axis, would not be a true loading and risked being read as
one). If cluster-level GO enrichment is wanted later, add it as a new
function here rather than resurrecting the PCA loadings functions.

The GO id/description lookups and the GO search/illumination payload
builder don't depend on the fitting algorithm at all, so they're reused
as-is from general_pca_common.py rather than duplicated here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from general_pca_common import (
    DEFAULT_IC_PATH,
    rgb_to_hex,
    load_go_ic_and_descriptions,
    load_go_descriptions,
    load_go_ic,
    build_go_search_payload,
)

TEMPLATE_PATH = Path(__file__).parent / "templates" / "general_umap_template.html"
DATA_MARKER = "__GENERAL_UMAP_DATA__"
TITLE_MARKER = "__GENERAL_UMAP_TITLE__"

__all__ = [
    "DEFAULT_IC_PATH",
    "TEMPLATE_PATH",
    "DATA_MARKER",
    "TITLE_MARKER",
    "rgb_to_hex",
    "load_go_ic_and_descriptions",
    "load_go_descriptions",
    "load_go_ic",
    "build_go_search_payload",
    "run_umap",
]


def run_umap(matrix, n_components=2, n_neighbors=15, min_dist=0.1,
             metric="euclidean", random_state=42):
    """
    Fit UMAP on an already-preprocessed (species x feature) matrix and
    return the embedding as a DataFrame indexed like `matrix`, with
    columns UMAP1..UMAPn.

    Preprocessing (rare-GO-term filtering, presence/absence binarization
    vs relative-abundance normalization, StandardScaler) stays in each
    branch script, mirroring how run_pca_on_presence_absence /
    run_pca_on_abundance own their own preprocessing in the PCA branch --
    this only wraps the actual UMAP fit, which is identical across
    branches.
    """
    import umap

    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(values)

    columns = [f"UMAP{i + 1}" for i in range(n_components)]
    index = matrix.index if isinstance(matrix, pd.DataFrame) else None
    return pd.DataFrame(embedding, columns=columns, index=index)
