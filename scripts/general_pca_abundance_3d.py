#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a 3-component (PC1/PC2/PC3)
PCA of GO term relative abundance across species -- rotate with mouse drag,
zoom with the wheel. Simple companion to general_pca_abundance.py: same CLR
+ StandardScaler fit, but TruncatedSVD(n_components=3) and a canvas-based 3D
scatter instead of the 2D SVG view (no GO search/illumination or top-loadings
sidebar in this first version).
"""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map
from interactive_go_tree import load_species_stats
from general_pca_common import DEFAULT_IC_PATH, load_go_ic, rgb_to_hex

TEMPLATE_PATH = Path(__file__).parent / "templates" / "general_pca_3d_template.html"
DATA_MARKER = "__GENERAL_PCA_3D_DATA__"
TITLE_MARKER = "__GENERAL_PCA_3D_TITLE__"


def run_pca_3d(raw_df, total_prots):
    """
    Same rare-GO-term filter and CLR conversion as
    interactive_go_tree.run_pca_on_relative_abundance, StandardScaler, then
    TruncatedSVD with 3 components instead of 2.
    """
    species = [s for s in raw_df.index if s in total_prots.index]
    raw_df = raw_df.loc[species]

    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    counts = pca_input.to_numpy(dtype="float64") + 1.0  # pseudo-count: log(0) is undefined
    log_counts = np.log(counts)
    clr_values = log_counts - log_counts.mean(axis=1, keepdims=True)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(clr_values)

    model = TruncatedSVD(n_components=3)
    components = model.fit_transform(normalized)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2", "PC3"], index=species)
    return pca_df, model.explained_variance_ratio_


def remove_outliers_3d(pca_df, low=5, high=95):
    """Percentile filtering across all three axes (PC1/PC2/PC3)."""
    mask = pd.Series(True, index=pca_df.index)
    for col in ("PC1", "PC2", "PC3"):
        mask &= (pca_df[col] >= np.percentile(pca_df[col], low)) & (pca_df[col] <= np.percentile(pca_df[col], high))
    return pca_df[mask]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive 3D PCA (PC1/PC2/PC3) of GO term relative abundance"
    )
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix, species x GO terms (TSV)")
    parser.add_argument("--species-stats", required=True, help="TSV with a Species index and a Total_prots column")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    parser.add_argument("--output", default="general_pca_abundance_3d.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> IC TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--ic-threshold", type=float, default=None,
                        help="Minimum IC to include a GO term in the PCA; GOs below this value are dropped from the matrix before fitting")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    taxon_dict = load_taxonomy(args.taxonomy)

    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    if args.ic_threshold is not None:
        go_ic = load_go_ic(args.ic_file)
        n_before = raw_full.shape[1]
        raw_full = raw_full[[c for c in raw_full.columns if go_ic.get(c, 0.0) >= args.ic_threshold]]
        print(f"IC filter (≥ {args.ic_threshold}): kept {raw_full.shape[1]} / {n_before} GO terms")

    pca_df, explained_variance = run_pca_3d(raw_full, total_prots)
    pca_df = remove_outliers_3d(pca_df, low=5, high=95)

    pca_df = pca_df.copy()
    pca_df["Group"] = pca_df.index.map(taxon_dict)
    pca_df = pca_df.dropna(subset=["Group"])

    species = list(pca_df.index)
    color_map = build_global_color_map(taxon_dict)

    species_records = [
        {
            "name": name,
            "pc1": float(pca_df.loc[name, "PC1"]),
            "pc2": float(pca_df.loc[name, "PC2"]),
            "pc3": float(pca_df.loc[name, "PC3"]),
            "group": pca_df.loc[name, "Group"],
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    title = "General PCA (3D): GO term relative abundance"
    if args.ic_threshold is not None:
        title += f" (IC ≥ {args.ic_threshold})"

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "meta": {
            "explained_variance": [float(v) for v in explained_variance],
            "title": title,
        },
    }

    template = TEMPLATE_PATH.read_text()
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title).replace(DATA_MARKER, data_json)

    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(species_records)} species)")


if __name__ == "__main__":
    main()
