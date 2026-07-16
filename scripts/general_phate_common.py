#!/usr/bin/env python3
"""
Shared helpers for the standalone-HTML general PHATE scripts (the PHATE
counterpart of general_pca_common.py / general_umap_common.py, which serve
presence_absence_pca.py/general_pca_abundance.py and
presence_absence_umap.py/general_umap_abundance.py respectively).

PHATE, like UMAP, has no linear components_ -- no PHATE equivalent of
top_loadings_by_pc / write_top_loadings_tsv / compute_species_contributions.
Same explicit decision as general_umap_common.py: no "top GO terms per
axis" sidebar for PHATE-based scripts built on this module.

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

TEMPLATE_PATH = Path(__file__).parent / "templates" / "general_phate_template.html"
DATA_MARKER = "__GENERAL_PHATE_DATA__"
TITLE_MARKER = "__GENERAL_PHATE_TITLE__"

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
    "run_phate",
]


def run_phate(matrix, n_components=2, knn=5, decay=40, t="auto",
              metric="euclidean", random_state=42):
    """
    Fit PHATE on an already-preprocessed (species x feature) matrix and
    return the embedding as a DataFrame indexed like `matrix`, with
    columns PHATE1..PHATEn.

    PHATE builds a diffusion-based affinity graph (knn/decay/t control the
    same "how local vs how global" tradeoff n_neighbors/min_dist control in
    UMAP) and is explicitly designed to preserve long-range/global
    structure in addition to local neighborhoods -- see
    [[project_umap_branch_design]] for why UMAP alone doesn't. Preprocessing
    stays in each branch script's own run_phate_on_* function, mirroring
    the UMAP/PCA branches -- this only wraps the actual fit.
    """
    import phate

    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    operator = phate.PHATE(
        n_components=n_components,
        knn=knn,
        decay=decay,
        t=t,
        knn_dist=metric,
        random_state=random_state,
        verbose=0,
    )
    embedding = operator.fit_transform(values)

    columns = [f"PHATE{i + 1}" for i in range(n_components)]
    index = matrix.index if isinstance(matrix, pd.DataFrame) else None
    return pd.DataFrame(embedding, columns=columns, index=index)
