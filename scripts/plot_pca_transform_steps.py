#!/usr/bin/env python3
"""
Diagnostic figure: shows the data distribution at every mathematical step of
the abundance PCA pipeline (interactive_go_tree.run_pca_on_relative_abundance),
so the effect of each transformation -- rare-column filter, pseudo-count,
log, CLR row-centering, column z-score, SVD -- can be inspected visually
instead of taken on faith.

Not a production tool: a one-off exploratory companion to the real pipeline,
kept in sync with it by construction (same steps, same order, run against
the same real matrix), used to answer "what does each transformation do to
the numbers" while iterating on the PCA methodology.
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

# Real matrix is ~1542 species x ~23700 GO columns; keeping several float64
# copies alive at once (one per pipeline stage) does not fit in a small
# machine's RAM. Stats (skew/mean/sd) are computed on the full-precision
# stage array, then only a downsampled slice is kept for plotting, and the
# full array is freed before the next stage is built.
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
    ap.add_argument("--output", default="pca_transform_steps.pdf", help="Output figure path")
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

    row_totals = raw_counts.sum(axis=1, dtype="float64")       # per-species scale (~ Total_prots effect)
    raw_sample, raw_stats = stage_stats(raw_counts.ravel(), rng)
    row_sample, row_stats = stage_stats(row_totals, rng)

    np.add(raw_counts, 1.0, out=raw_counts)                    # pseudo-count, in place
    np.log(raw_counts, out=raw_counts)                         # log transform, in place -- now log_counts
    log_sample, log_stats = stage_stats(raw_counts.ravel(), rng)

    row_means = raw_counts.mean(axis=1, keepdims=True)
    raw_counts -= row_means                                    # CLR row-centering, in place -- now clr_values
    clr_sample, clr_stats = stage_stats(raw_counts.ravel(), rng)

    scaler = StandardScaler(copy=False)
    scaled_values = scaler.fit_transform(raw_counts)            # column z-score, reuses the array (copy=False)
    scaled_sample, scaled_stats = stage_stats(scaled_values.ravel(), rng)

    model = TruncatedSVD(n_components=2)
    pc = model.fit_transform(scaled_values)                     # SVD projection
    explained = model.explained_variance_ratio_ * 100
    del scaled_values, raw_counts
    gc.collect()

    # --- figure ---
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

    ax = axes[1, 2]
    ax.scatter(pc[:, 0], pc[:, 1], s=10, alpha=0.5, color=BLUE, edgecolor="none")
    ax.set_title(f"6. PCA final (SVD)\nPC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    fig.suptitle("Transformaciones de los datos a lo largo del pipeline de PCA de abundancia", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.output)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")

    # also drop a PNG alongside for quick viewing
    if out_path.suffix != ".png":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight")
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
