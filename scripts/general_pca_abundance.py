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
    load_go_ic_and_descriptions,
    top_loadings_by_pc,
    write_top_loadings_tsv,
    build_go_search_payload,
    compute_species_contributions,
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
    parser.add_argument("--ic-threshold", type=float, default=None,
                        help="Minimum IC to include a GO term in the PCA; GOs below this value are dropped from the matrix before fitting")
    parser.add_argument("--top-loadings-n", type=int, default=10,
                         help="Number of GO terms to report per direction (positive/negative) per PC (default: 10)")
    parser.add_argument("--loadings-output", default=None,
                         help="Top-loadings TSV path (default: alongside --output, with _top_loadings.tsv)")
    parser.add_argument("--outlier-percentile", type=float, nargs=2, default=[0, 100], metavar=("LOW", "HIGH"),
                         help="Drop species whose PC1 or PC2 falls outside this percentile range before "
                              "rendering (default: 0 100, i.e. no trimming). Pass e.g. '5 95' to trim -- "
                              "with taxonomically diverse datasets the most divergent species are often "
                              "the most interesting ones, so trimming is opt-in, not the default.")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    taxon_dict = load_taxonomy(args.taxonomy)

    go_ic, go_desc_raw = load_go_ic_and_descriptions(args.ic_file)
    # Embed IC in the description string so it surfaces everywhere the
    # description is shown: GO search suggestions, top-loadings sidebar, etc.
    go_desc = {
        go_id: f"{desc} (IC: {go_ic[go_id]:.2f})" if go_id in go_ic else desc
        for go_id, desc in go_desc_raw.items()
    }

    # Restrict to the requested taxa *before* running PCA, not after: the
    # whole point of -t/--taxa is to compute the PCA only from variance
    # among those species, not to compute it on everyone and crop the plot
    # to a sub-region of the same global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    # Drop GO terms below the IC threshold before fitting the PCA so that
    # overly general terms (present in nearly all species, low information
    # content) don't dominate the variance.
    n_absent_ic = sum(1 for c in raw_full.columns if c not in go_ic)
    if n_absent_ic:
        print(f"Warning: {n_absent_ic} GO terms in matrix have no IC value in {args.ic_file}")

    if args.ic_threshold is not None:
        n_before = raw_full.shape[1]
        raw_full = raw_full[[c for c in raw_full.columns if go_ic.get(c, 0.0) >= args.ic_threshold]]
        print(f"IC filter (≥ {args.ic_threshold}): kept {raw_full.shape[1]} / {n_before} GO terms")

    pca_df, explained_variance, loadings, normalized_df = run_pca_on_relative_abundance(raw_full, total_prots)
    n_go_used = loadings.shape[0]
    # How many of the same GO columns the PCA was fit on (loadings.index)
    # each species has any annotation for, for the tooltip -- reuses the
    # PCA's own rare-term filter instead of re-deriving it.
    richness = (raw_full[loadings.index] > 0).sum(axis=1)
    outlier_low, outlier_high = args.outlier_percentile
    n_before_outliers = pca_df.shape[0]
    pca_df = remove_outliers(pca_df, low=outlier_low, high=outlier_high)
    n_dropped = n_before_outliers - pca_df.shape[0]
    if n_dropped:
        print(f"Outlier trim (percentile {outlier_low}-{outlier_high}): dropped {n_dropped} / {n_before_outliers} species")

    pca_df = pca_df.copy()
    pca_df["Group"] = pca_df.index.map(taxon_dict)
    pca_df = pca_df.dropna(subset=["Group"])

    species = list(pca_df.index)
    color_map = build_global_color_map(taxon_dict)

    contributions = compute_species_contributions(
        normalized_df.loc[[s for s in species if s in normalized_df.index]],
        loadings,
        n=args.top_loadings_n,
    )
    del normalized_df

    species_records = [
        {
            "name": name,
            "pc1": float(pca_df.loc[name, "PC1"]),
            "pc2": float(pca_df.loc[name, "PC2"]),
            "group": pca_df.loc[name, "Group"],
            "go_terms_present": int(richness.get(name, 0)),
            "contributions": contributions.get(name, {}),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    top_loadings = top_loadings_by_pc(loadings, go_desc, args.top_loadings_n)
    go_search = build_go_search_payload(raw_full, species, go_desc)

    title = "General PCA: GO term relative abundance"
    mode_label = "relative abundance, not presence/absence"
    if args.ic_threshold is not None:
        title += f" (IC ≥ {args.ic_threshold})"
        mode_label += f", IC ≥ {args.ic_threshold}"

    payload = {
        "species": species_records,
        "groups": groups_hex,
        "top_loadings": top_loadings,
        "go_search": go_search,
        "meta": {
            "n_go_terms_used": int(n_go_used),
            "explained_variance": [float(v) for v in explained_variance],
            "title": title,
            "mode_label": mode_label,
            "filename_base": "general_pca_abundance",
        },
    }

    template = TEMPLATE_PATH.read_text()
    data_json = json.dumps(payload).replace("</", "<\\/")
    html = template.replace(TITLE_MARKER, title).replace(DATA_MARKER, data_json)

    Path(args.output).write_text(html)
    print(f"Wrote {args.output} ({len(species_records)} species, {n_go_used} GO columns used)")

    loadings_output = args.loadings_output or f"{Path(args.output).with_suffix('')}_top_loadings.tsv"
    write_top_loadings_tsv(top_loadings, loadings_output)
    print(f"Wrote {loadings_output} (top {args.top_loadings_n} GO terms per PC)")


if __name__ == "__main__":
    main()
