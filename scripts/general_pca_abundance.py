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
import time

import pandas as pd

from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers, assign_taxonomy_group
from interactive_go_tree import run_pca_on_relative_abundance, load_species_stats
from general_pca_common import (
    TEMPLATE_PATH,
    DEFAULT_IC_PATH,
    DEFAULT_SPECIES_ACCESSION_PATH,
    DEFAULT_METAZOA_TAXONOMY_PATH,
    METAZOA_SUBDIVIDE_RANKS,
    DATA_MARKER,
    TITLE_MARKER,
    rgb_to_hex,
    load_go_ic_and_descriptions,
    load_species_ncbi_accessions,
    load_metazoa_phylum_lineage,
    build_lineage_label_colors,
    top_loadings_by_pc,
    write_top_loadings_tsv,
    write_full_loadings_tsv,
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
    parser.add_argument("--min-go-terms", type=int, default=None,
                         help="Drop species with fewer than this many GO terms present before fitting the PCA "
                              "-- weakly-annotated species are a major non-biological confound (they alone can "
                              "dominate PC1/PC2). Off by default. See --min-go-terms-exempt for groups to spare.")
    parser.add_argument(
        "--min-go-terms-exempt",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=[],
        help="Comma-separated taxonomy groups exempt from --min-go-terms (e.g. groups where thin annotation "
             "is normal for the whole group, not a red flag for that one species -- 'Asgard,Protists')",
    )
    parser.add_argument("--output", default="general_pca_abundance.html", help="Output HTML path")
    parser.add_argument("--ic-file", default=str(DEFAULT_IC_PATH), help="GO id -> description TSV (default: bundled data/All_GOs_ic.tsv)")
    parser.add_argument("--species-accession-file", default=str(DEFAULT_SPECIES_ACCESSION_PATH),
                         help="Species -> NCBI genome accession TSV, for the click-through NCBI link "
                              "(default: bundled data/species_ncbi_accession.tsv, see "
                              "scripts/build_ncbi_accession_lookup.py; species missing from it fall back "
                              "to a text search)")
    parser.add_argument("--metazoa-taxonomy", default=str(DEFAULT_METAZOA_TAXONOMY_PATH),
                         help="Metazoa species -> Phylum TSV (default: bundled data/metadata_metazoa.txt), used "
                              "to add a client-side 'expand Metazoa into phyla' accordion to the legend -- pass "
                              "a nonexistent path to omit it")
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

    # No output at all can print for a couple of minutes right here on the
    # real ~191MB matrix (species x ~29,700 GO columns) -- pandas.read_csv
    # itself has no progress indicator. Printed before/after so a slow-but-
    # working run is distinguishable from an actually-frozen one.
    print(f"Reading matrix {args.matrix} ...")
    t0 = time.monotonic()
    raw_full = pd.read_csv(args.matrix, sep="\t", index_col=0).fillna(0)
    print(f"  {raw_full.shape[0]} species x {raw_full.shape[1]} GO columns ({time.monotonic() - t0:.0f}s)")
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

    # Drop species with too few GO terms annotated *before* fitting, same
    # reasoning as -t/--taxa above: a weakly-annotated species should not
    # get to pull on the PCA's variance at all, not be fit in and cropped
    # out of the plot afterward. Computed on the full matrix as loaded
    # (before the IC-threshold column filter just below), since annotation
    # completeness is a property of the species, independent of this run's
    # own GO-term cutoff.
    if args.min_go_terms is not None:
        richness_pre_fit = (raw_full > 0).sum(axis=1)
        species_group = raw_full.index.map(taxon_dict)
        exempt = set(args.min_go_terms_exempt)
        keep_mask = (richness_pre_fit >= args.min_go_terms) | species_group.isin(exempt)
        n_before = raw_full.shape[0]
        raw_full = raw_full[keep_mask]
        n_dropped = n_before - raw_full.shape[0]
        exempt_note = f" (exempt: {', '.join(sorted(exempt))})" if exempt else ""
        print(f"GO-terms-present filter (< {args.min_go_terms}){exempt_note}: dropped {n_dropped} / {n_before} species")

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

    # n_components=3 so the browser can offer an optional 3D (PC1/PC2/PC3)
    # view -- PC1/PC2 themselves are unaffected (see run_pca_on_relative_abundance's
    # docstring: this relies on algorithm="arpack" being an exact, nested
    # solver), so the default 2D page is the same PCA as before. loadings/
    # top-loadings below keep all three PCs, so the sidebar's "Top GO terms
    # per PC" also covers PC3.
    # normalized_df isn't used here (it only fed the now-removed per-species
    # contributions modal) -- discarded immediately, it's the single biggest
    # in-memory array in this pipeline (n_species x n_go floats).
    print(f"Fitting PCA on {raw_full.shape[0]} species ...")
    t0 = time.monotonic()
    pca_df, explained_variance, loadings_3d, normalized_df = run_pca_on_relative_abundance(
        raw_full, total_prots, n_components=3
    )
    print(f"  done ({time.monotonic() - t0:.0f}s)")
    del normalized_df
    loadings = loadings_3d
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

    pca_df = assign_taxonomy_group(pca_df, taxon_dict)

    species = list(pca_df.index)
    color_map = build_global_color_map(taxon_dict)
    ncbi_accessions = load_species_ncbi_accessions(args.species_accession_file)

    species_records = [
        {
            "name": name,
            "pc1": float(pca_df.loc[name, "PC1"]),
            "pc2": float(pca_df.loc[name, "PC2"]),
            "pc3": float(pca_df.loc[name, "PC3"]),
            "group": pca_df.loc[name, "Group"],
            "go_terms_present": int(richness.get(name, 0)),
            "total_prots": int(total_prots[name]) if name in total_prots.index else None,
            "ncbi_id": ncbi_accessions.get(name),
        }
        for name in species
    ]
    groups_used = sorted({rec["group"] for rec in species_records})
    groups_hex = {g: rgb_to_hex(color_map[g]) for g in groups_used}

    species_lineage = load_metazoa_phylum_lineage(args.metazoa_taxonomy)
    for rec in species_records:
        rec["base_group"] = rec["group"]
        lineage = species_lineage.get(rec["name"], {})
        if "phylum" in lineage:
            rec["phylum"] = lineage["phylum"]
    groups_hex.update(build_lineage_label_colors(species_lineage))
    has_lineage_data = any(
        species_lineage.get(rec["name"]) for rec in species_records
    )

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
            "has_lineage_data": has_lineage_data,
            "lineage_ranks": METAZOA_SUBDIVIDE_RANKS,
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

    full_loadings_output = f"{Path(args.output).with_suffix('')}_full_loadings.tsv"
    write_full_loadings_tsv(loadings_3d, go_desc, full_loadings_output)
    print(f"Wrote {full_loadings_output} ({loadings_3d.shape[0]} GO terms x {loadings_3d.shape[1]} PCs)")


if __name__ == "__main__":
    main()
