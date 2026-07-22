#!/usr/bin/env python3
"""
Diagnostic figures for the abundance PCA pipeline (interactive_go_tree.
run_pca_on_relative_abundance): (1) the data distribution at each
mathematical step -- rare-column filter, pseudo-count, log, CLR
row-centering, column z-score -- and (2) an actual PCA (StandardScaler +
TruncatedSVD, the same recipe used everywhere else in this repo -- see
run_pca_on_relative_abundance and general_pca_abundance_raw.
run_pca_on_raw_counts) fit independently on the matrix at each stage,
colored by taxon, so PC1/PC2 explained variance is directly comparable
across stages.

IMPORTANT: every per-stage PCA here ends in StandardScaler before
TruncatedSVD. Fitting TruncatedSVD directly on an un-centered, un-scaled
matrix (an earlier version of this script did that to "isolate" each raw
step) does NOT give a meaningful PC1 -- TruncatedSVD orders components by
singular value, not by explained variance, and on non-negative uncentered
data the top singular vector is often dominated by the mean/scale
direction rather than by the axis of highest between-sample variance. That
produces nonsensical results (e.g. PC2 > PC1, or a raw-counts PC1 in the
80%s) that don't match the properly-scaled numbers this project already
reports elsewhere (general_pca_abundance_raw.py). Always compare stages
through the same final StandardScaler + TruncatedSVD step, never SVD alone.

Not a production tool: a one-off exploratory companion to the real
pipeline, kept in sync with it by construction (same steps, same order,
run against the same real matrix), used to answer "what does each
transformation do to the numbers" while iterating on the PCA methodology.
"""

from pathlib import Path
import argparse
import gc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

from interactive_go_tree import load_species_stats
from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers

# Real matrix is ~1542 species x ~23700 GO columns; keeping several float64
# copies alive at once (one per pipeline stage) does not fit in a small
# machine's RAM. Stats (skew/mean/sd) are computed on the full-precision
# stage array, then only a downsampled slice is kept for plotting, and each
# stage's array is freed as soon as the next stage's input has been derived
# from it.
MAX_PLOT_SAMPLE = 2_000_000

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "figure.facecolor": "white",
})

BLUE = "#4C9BE8"
GREY = "#888888"
AMBER = "#F5A623"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", "-m", default="merged_PCA_fantasia.tsv", help="Raw GO counts matrix (species x GO, TSV)")
    ap.add_argument("--species-stats", default="merged_species_stats.tsv", help="TSV with a Species index and Total_prots column")
    ap.add_argument("--taxonomy", default="merged_taxons.tsv", help="TSV with Species and Group columns, used to color the PCA scatter plots")
    ap.add_argument("--output", default="pca_transform_steps.pdf", help="Output figure path (also controls the companion figures' names)")
    ap.add_argument("--sample-cols", type=int, default=None,
                     help="If set, randomly subsample this many GO columns before running the "
                          "pipeline (speeds up iteration; full matrix is used by default)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def stage_stats(values, rng):
    """Full-precision skew/mean/sd, plus a bounded random sample for plotting."""
    v = values[np.isfinite(values)]
    stats = {"skew": float(skew(v)), "mean": float(v.mean()), "sd": float(v.std())}
    if v.size > MAX_PLOT_SAMPLE:
        idx = rng.choice(v.size, size=MAX_PLOT_SAMPLE, replace=False)
        v = v[idx]
    return v, stats


def hist_panel(ax, sample, stats, title, color, log_x=False, mark_zero=False):
    v = sample
    if log_x:
        v_plot = v[v > 0]
        bins = np.logspace(np.log10(max(v_plot.min(), 1e-3)), np.log10(v_plot.max()), 80)
        ax.hist(v_plot, bins=bins, color=color, edgecolor="none")
        ax.set_xscale("log")
    else:
        ax.hist(v, bins=80, color=color, edgecolor="none")
    if mark_zero:
        ax.axvline(0, color=GREY, linewidth=1, linestyle="--")
    ax.set_title(f"{title}\nskew={stats['skew']:.2f}  mean={stats['mean']:.2f}  sd={stats['sd']:.2f}")
    ax.set_ylabel("frecuencia")


def run_stage_pca(matrix, species, taxon_dict):
    """
    StandardScaler + TruncatedSVD(2) -- the same recipe as
    interactive_go_tree.run_pca_on_relative_abundance and
    general_pca_abundance_raw.run_pca_on_raw_counts. Never skip the
    StandardScaler step: see the module docstring for why comparing raw
    SVD output across stages is meaningless.

    Returns a plotting-ready DataFrame (species with a known taxon,
    percentile outliers removed -- same filtering general_pca_abundance.py
    applies) and the PC1/PC2 explained-variance percentages computed over
    ALL fitted species (not just the plotted subset), so the numbers match
    what the production scripts would report.
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(matrix)
    model = TruncatedSVD(n_components=2)
    pc = model.fit_transform(normalized)
    explained = model.explained_variance_ratio_ * 100
    del normalized
    gc.collect()

    pca_df = pd.DataFrame(pc, columns=["PC1", "PC2"], index=species)
    pca_df["Group"] = pca_df.index.map(taxon_dict)
    pca_df = pca_df.dropna(subset=["Group"])
    pca_df = remove_outliers(pca_df, low=5, high=95)
    return pca_df, explained


def scatter_panel(ax, pca_df, color_map, title, explained):
    for group, sub in pca_df.groupby("Group"):
        ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.6, color=color_map[group], edgecolor="none")
    ax.set_title(f"{title}\nPC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print(f"Loading {args.matrix} ...")
    # dtype=float32 at parse time (not a post-hoc .astype) keeps the C parser
    # from ever materializing a float64 copy of the whole matrix. The dtype
    # map excludes the first (species-name) column, which read_csv consumes
    # as the index before dtype coercion -- passing a single dtype for all
    # columns tries to parse species names as floats and fails.
    header_cols = pd.read_csv(args.matrix, sep="\t", nrows=0).columns
    dtype_map = {c: "float32" for c in header_cols[1:]}
    raw_df = pd.read_csv(args.matrix, sep="\t", index_col=0, dtype=dtype_map).fillna(0)
    total_prots = load_species_stats(args.species_stats)
    taxon_dict = load_taxonomy(args.taxonomy)
    color_map = build_global_color_map(taxon_dict)

    if args.sample_cols is not None and args.sample_cols < raw_df.shape[1]:
        cols = rng.choice(raw_df.columns, size=args.sample_cols, replace=False)
        raw_df = raw_df[cols]
        print(f"Subsampled to {args.sample_cols} GO columns for speed")

    # --- replicate interactive_go_tree.run_pca_on_relative_abundance step by step ---
    species = [s for s in raw_df.index if s in total_prots.index]
    raw_df = raw_df.loc[species]

    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    print(f"Matrix: {raw_df.shape[0]} species x {raw_df.shape[1]} GO columns "
          f"-> {pca_input.shape[1]} kept after rare-column filter (colsum > 5)")
    del raw_df
    gc.collect()

    raw_counts = pca_input.to_numpy(dtype="float32")
    del pca_input
    gc.collect()

    pca_results = {}  # label -> (pca_df, explained[PC1%, PC2%])

    print("Fitting PCA (StandardScaler + TruncatedSVD) on: raw counts ...")
    pca_results["1. Raw counts"] = run_stage_pca(raw_counts, species, taxon_dict)

    total_prots_col = total_prots.loc[species].to_numpy(dtype="float32")[:, None]
    relative_abundance = raw_counts / total_prots_col           # old normalization, count / Total_prots
    print("Fitting PCA (StandardScaler + TruncatedSVD) on: relative abundance (count / Total_prots) ...")
    pca_results["2. Relative abundance"] = run_stage_pca(relative_abundance, species, taxon_dict)
    del relative_abundance, total_prots_col
    gc.collect()

    # distribution diagnostics on the still-untouched raw counts / their row totals
    row_totals = raw_counts.sum(axis=1, dtype="float64")        # per-species scale (~ Total_prots effect)
    raw_sample, raw_stats = stage_stats(raw_counts.ravel(), rng)
    row_sample, row_stats = stage_stats(row_totals, rng)

    log_counts = np.log1p(raw_counts)                            # new array; raw_counts left untouched above
    del raw_counts
    gc.collect()
    log_sample, log_stats = stage_stats(log_counts.ravel(), rng)

    print("Fitting PCA (StandardScaler + TruncatedSVD) on: log(count+1), no row-centering ...")
    pca_results["3. log(count+1)"] = run_stage_pca(log_counts, species, taxon_dict)

    row_means = log_counts.mean(axis=1, keepdims=True)
    clr_values = log_counts - row_means                          # CLR row-centering -- new array
    del log_counts
    gc.collect()
    clr_sample, clr_stats = stage_stats(clr_values.ravel(), rng)

    print("Fitting PCA (StandardScaler + TruncatedSVD) on: CLR (log + row-centered) ...")
    pca_results["4. CLR"] = run_stage_pca(clr_values, species, taxon_dict)

    # histogram-only view of what StandardScaler does to the CLR values
    # (this is the same array run_stage_pca just standardized internally
    # for stage 4 -- redone here only to keep a sample for the histogram)
    scaled_for_hist = StandardScaler().fit_transform(clr_values)
    scaled_sample, scaled_stats = stage_stats(scaled_for_hist.ravel(), rng)
    del scaled_for_hist, clr_values
    gc.collect()

    print("\nPC1 / PC2 explained variance by pipeline stage (StandardScaler + TruncatedSVD, same recipe every stage):")
    for label, (_, ev) in pca_results.items():
        print(f"  {label:<24s} PC1={ev[0]:5.1f}%  PC2={ev[1]:5.1f}%")

    # --- figure 1: distributions at each step ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    hist_panel(axes[0, 0], raw_sample, raw_stats, "1. Conteos crudos (raw counts, ceros excluidos de la vista log)", BLUE, log_x=True)
    axes[0, 0].set_xlabel("count (escala log)")

    hist_panel(axes[0, 1], row_sample, row_stats, "2. Total de conteos por especie\n(efecto de escala / Total_prots)", AMBER)
    axes[0, 1].set_xlabel("Σ counts por fila")

    hist_panel(axes[0, 2], log_sample, log_stats, "3. Tras pseudo-count (+1) y log", BLUE)
    axes[0, 2].set_xlabel("log(count + 1)")

    hist_panel(axes[1, 0], clr_sample, clr_stats, "4. Tras CLR (centrado por fila)", BLUE, mark_zero=True)
    axes[1, 0].set_xlabel("log(count+1) - media_fila")

    hist_panel(axes[1, 1], scaled_sample, scaled_stats, "5. Tras StandardScaler\n(z-score por columna)", BLUE, mark_zero=True)
    axes[1, 1].set_xlabel("z-score")

    # skew trend across the same four stages, instead of a duplicate PCA
    # scatter (that now lives in its own colored-by-taxon figure below)
    skew_stages = ["Raw", "Log", "CLR", "CLR+\nScaler"]
    skew_vals = [raw_stats["skew"], log_stats["skew"], clr_stats["skew"], scaled_stats["skew"]]
    ax = axes[1, 2]
    bars = ax.bar(skew_stages, skew_vals, color=BLUE)
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.2f}", (b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    ax.set_title("6. Asimetría (skew) por etapa\n(cuanto más cerca de 0, más simétrica)")
    ax.set_ylabel("skew")
    ax.axhline(0, color=GREY, linewidth=1)

    fig.suptitle("Transformaciones de los datos a lo largo del pipeline de PCA de abundancia", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.output)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")
    if out_path.suffix != ".png":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight")
        print(f"Wrote {png_path}")

    # --- figure 2: PCA scatter per stage, colored by taxon ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 11))
    panel_axes = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
    panel_titles = {
        "1. Raw counts": "1. Raw counts + StandardScaler",
        "2. Relative abundance": "2. Abundancia relativa (count/Total_prots) + StandardScaler",
        "3. log(count+1)": "3. log(count+1), sin centrar por fila + StandardScaler",
        "4. CLR": "4. CLR (log + centrado por fila) + StandardScaler  [pipeline real]",
    }
    for ax, (label, (pca_df, explained)) in zip(panel_axes, pca_results.items()):
        scatter_panel(ax, pca_df, color_map, panel_titles[label], explained)

    groups_sorted = sorted(color_map.keys())
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color_map[g], label=g)
        for g in groups_sorted
    ]
    fig2.legend(handles=legend_handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
                frameon=False, title="Taxón")
    fig2.suptitle("PCA (StandardScaler + TruncatedSVD) en cada etapa, coloreado por taxón", fontsize=14)
    fig2.tight_layout(rect=[0, 0, 0.86, 0.96])

    pca_out = out_path.with_name(out_path.stem + "_pca_by_stage" + out_path.suffix)
    fig2.savefig(pca_out, bbox_inches="tight")
    print(f"Wrote {pca_out}")
    if pca_out.suffix != ".png":
        pca_png = pca_out.with_suffix(".png")
        fig2.savefig(pca_png, bbox_inches="tight")
        print(f"Wrote {pca_png}")

    # --- figure 3: PC1/PC2 explained variance by pipeline stage (same numbers as the scatter titles) ---
    labels = list(pca_results.keys())
    pc1_vals = [pca_results[l][1][0] for l in labels]
    pc2_vals = [pca_results[l][1][1] for l in labels]

    fig3, ax3 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    width = 0.32
    bars1 = ax3.bar(x - width / 2, pc1_vals, width, label="PC1", color=BLUE)
    bars2 = ax3.bar(x + width / 2, pc2_vals, width, label="PC2", color=AMBER)
    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax3.annotate(f"{h:.1f}%", (b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9)

    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("% varianza explicada")
    ax3.set_title("Varianza explicada por PC1/PC2 en cada etapa\n(StandardScaler + TruncatedSVD, misma receta en las 4 etapas)")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(frameon=False)
    fig3.tight_layout()

    var_out = out_path.with_name(out_path.stem + "_variance_by_stage" + out_path.suffix)
    fig3.savefig(var_out, bbox_inches="tight")
    print(f"Wrote {var_out}")
    if var_out.suffix != ".png":
        var_png = var_out.with_suffix(".png")
        fig3.savefig(var_png, bbox_inches="tight")
        print(f"Wrote {var_png}")


if __name__ == "__main__":
    main()
