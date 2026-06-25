#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a single general PCA of
GO term relative abundance across species -- no GO tree, no illumination.

Same idea as presence_absence_pca.py (same template, same live GO-term
search/illumination), but the PCA itself is fit on relative abundance
(count / Total_prots per species), reusing interactive_go_tree.py's PCA
step, instead of binarized presence/absence.
"""

from pathlib import Path
import argparse
import json

import pandas as pd

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers
from interactive_go_tree import run_pca_on_relative_abundance, load_species_stats
from general_pca_common import (
    TEMPLATE_PATH,
    DEFAULT_IC_PATH,
    DATA_MARKER,
    TITLE_MARKER,
    rgb_to_hex,
    load_go_descriptions,
    top_loadings_by_pc,
    write_top_loadings_tsv,
    build_go_search_payload,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive PCA of GO term relative abundance (no GO tree, no illumination)"
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
    parser.add_argument("--output", default="general_pca_abundance.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--top-loadings-n", type=int, default=15,
                         help="Number of most-influential GO terms to report per PC (default: 15)")
    parser.add_argument("--loadings-output", default=None,
                         help="Top-loadings TSV path (default: alongside --output, with _top_loadings.tsv)")
    parser.add_argument("-o", "--no_outliers", action="store_true",
                         help="Robust (percentile-clipped) scaling instead of log scaling when illuminating a searched GO term")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    taxon_dict = load_taxonomy(args.taxonomy)

    # Restrict to the requested taxa *before* running PCA, not after: the
    # whole point of -t/--taxa is to compute the PCA only from variance
    # among those species, not to compute it on everyone and crop the plot
    # to a sub-region of the same global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    pca_df, explained_variance, loadings = run_pca_on_relative_abundance(raw_full, total_prots)
    n_go_used = loadings.shape[0]
    # How many of the same GO columns the PCA was fit on (loadings.index)
    # each species has any annotation for, for the tooltip -- reuses the
    # PCA's own rare-term filter instead of re-deriving it.
    richness = (raw_full[loadings.index] > 0).sum(axis=1)
    pca_df = remove_outliers(pca_df, low=5, high=95)

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
            "group": pca_df.loc[name, "Group"],
            "go_terms_present": int(richness.get(name, 0)),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    go_desc = load_go_descriptions(args.ic_file)
    top_loadings = top_loadings_by_pc(loadings, go_desc, args.top_loadings_n)
    go_search = build_go_search_payload(raw_full, species, go_desc)

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "top_loadings": top_loadings,
        "go_search": go_search,
        "meta": {
            "n_go_terms_used": int(n_go_used),
            "explained_variance": [float(v) for v in explained_variance],
            "no_outliers": bool(args.no_outliers),
            "title": "General PCA: GO term relative abundance",
            "mode_label": "relative abundance, not presence/absence",
            "filename_base": "general_pca_abundance",
        },
    }

    template = TEMPLATE_PATH.read_text()
    title = payload["meta"]["title"]
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title).replace(DATA_MARKER, data_json)

    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(species_records)} species, {n_go_used} GO columns used)")

    loadings_output = args.loadings_output or f"{Path(args.output).with_suffix('')}_top_loadings.tsv"
    write_top_loadings_tsv(top_loadings, loadings_output)
    print(f"Wrote {loadings_output} (top {args.top_loadings_n} GO terms per PC)")


if __name__ == "__main__":
    main()
