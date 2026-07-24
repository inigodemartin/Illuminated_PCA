#!/usr/bin/env python3
"""
Generate a standalone interactive HTML page with a single general PCA of
GO term presence/absence across species -- no GO tree, no illumination.

Same input matrix as interactive_go_tree.py (raw GO counts, species x GO
terms), but each retained GO column is binarized (count > 0 -> 1) instead
of converted to relative abundance, since presence/absence doesn't need
the Total_prots normalization the abundance PCA uses.
"""

from pathlib import Path
import argparse
import json

import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers, assign_taxonomy_group
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


def run_pca_on_presence_absence(raw_df):
    """
    Same rare-GO-term filter as the relative-abundance PCA (keep columns
    with raw count sum > 5 across species), but binarize the retained
    columns to presence/absence (count > 0 -> 1) instead of dividing by
    Total_prots, before StandardScaler + TruncatedSVD.

    Also returns each species' GO-term richness (how many of the retained
    columns it has any annotation for) for the tooltip, and the per-GO-term
    loadings (model.components_, transposed so rows are GO ids): how much
    each retained GO column contributes to PC1/PC2.
    """
    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    presence = (pca_input > 0).astype(float)

    scaler = StandardScaler()
    normalized = scaler.fit_transform(presence.values)

    model = TruncatedSVD(n_components=2)
    components = model.fit_transform(normalized)

    pca_df = pd.DataFrame(components, columns=["PC1", "PC2"], index=raw_df.index)
    richness = presence.sum(axis=1)
    loadings = pd.DataFrame(model.components_.T, columns=["PC1", "PC2"], index=pca_input.columns)
    return pca_df, model.explained_variance_ratio_, richness, loadings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standalone interactive PCA of GO term presence/absence (no GO tree, no illumination)"
    )
    parser.add_argument("--matrix", "-m", required=True, help="Raw GO counts matrix, species x GO terms (TSV)")
    parser.add_argument("--taxonomy", required=True, help="TSV with Species and Group columns")
    parser.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    parser.add_argument("--output", default="general_pca_presence_absence.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--top-loadings-n", type=int, default=10,
                         help="Number of GO terms to report per direction (positive/negative) per PC (default: 10)")
    parser.add_argument("--loadings-output", default=None,
                         help="Top-loadings TSV path (default: alongside --output, with _top_loadings.tsv)")
    parser.add_argument("--outlier-percentile", type=float, nargs=2, default=[0, 100], metavar=("LOW", "HIGH"),
                         help="Drop species whose PC1 or PC2 falls outside this percentile range "
                              "(default: 0 100, i.e. no trimming). Pass e.g. '5 95' to trim.")
    return parser.parse_args()


def main():
    args = parse_args()

    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    taxon_dict = load_taxonomy(args.taxonomy)

    # Restrict to the requested taxa *before* running PCA, not after: the
    # whole point of -t/--taxa is to compute the PCA only from variance
    # among those species, not to compute it on everyone and crop the plot
    # to a sub-region of the same global layout.
    if args.taxa:
        raw_full = raw_full[raw_full.index.map(taxon_dict).isin(args.taxa)]

    pca_df, explained_variance, richness, loadings = run_pca_on_presence_absence(raw_full)
    n_go_used = loadings.shape[0]
    outlier_low, outlier_high = args.outlier_percentile
    n_before_outliers = pca_df.shape[0]
    pca_df = remove_outliers(pca_df, low=outlier_low, high=outlier_high)
    n_dropped = n_before_outliers - pca_df.shape[0]
    if n_dropped:
        print(f"Outlier trim (percentile {outlier_low}-{outlier_high}): dropped {n_dropped} / {n_before_outliers} species")

    pca_df = assign_taxonomy_group(pca_df, taxon_dict)

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
            "title": "General PCA: GO term presence/absence",
            "mode_label": "presence/absence, not abundance",
            "filename_base": "general_pca_presence_absence",
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
