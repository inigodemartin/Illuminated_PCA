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
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

from interactive_go_tree import load_species_stats, filter_species_by_stats
from illuminate_PCA import load_taxonomy, build_global_color_map, remove_outliers, assign_taxonomy_group

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
    ap.add_argument(
        "-t", "--taxa",
        type=lambda s: [item.strip() for item in s.split(",")],
        default=None,
        help="Comma-separated taxonomic groups to restrict to",
    )
    ap.add_argument("--sample-cols", type=int, default=None,
                     help="If set, randomly subsample this many GO columns before running the "
                          "pipeline (speeds up iteration; full matrix is used by default)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-ellipses", action="store_true",
                     help="Disable the shaded per-taxon confidence ellipses on the PCA scatter panels")
    ap.add_argument("--ellipse-std", type=float, default=2.0,
                     help="Ellipse radius in standard deviations from each taxon's centroid "
                          "(default: 2.0 -- the ellipse follows the group's own spread, so a few "
                          "far-flung points don't drag it out; smaller = tighter core cluster only)")
    ap.add_argument("--outlier-percentile", type=float, nargs=2, default=[0, 100], metavar=("LOW", "HIGH"),
                     help="Drop species whose PC1 or PC2 falls outside this percentile range "
                          "(default: 0 100, i.e. no trimming). Pass e.g. '5 95' to trim.")
    return ap.parse_args()


def stage_stats(values, rng):
    """Full-precision skew/mean/sd, plus a bounded random sample for plotting."""
    v = values[np.isfinite(values)]
    stats = {"skew": float(skew(v)), "mean": float(v.mean()), "sd": float(v.std())}
    if v.size > MAX_PLOT_SAMPLE:
        idx = rng.choice(v.size, size=MAX_PLOT_SAMPLE, replace=False)
        v = v[idx]
    return v, stats


def scaled_stage_stats(matrix, rng):
    """
    Distribution stats (sample + skew/mean/sd) of `matrix` AFTER a
    StandardScaler pass -- i.e. exactly what run_stage_pca hands to
    TruncatedSVD, recomputed here standalone since run_stage_pca discards
    its own internal scaled copy. The StandardScaler fit itself is cheap
    (one mean/std pass), so recomputing it is negligible next to the SVD.
    """
    scaled = StandardScaler().fit_transform(matrix)
    sample, stats = stage_stats(scaled.ravel(), rng)
    del scaled
    gc.collect()
    return sample, stats


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
    ax.set_title(f"{title}\nskew={stats['skew']:.2f}  mean={stats['mean']:.3g}  sd={stats['sd']:.3g}")
    ax.set_ylabel("frecuencia")


def run_stage_pca(matrix, species, taxon_dict, outlier_percentile=(0, 100)):
    """
    StandardScaler + TruncatedSVD(2) -- the same recipe as
    interactive_go_tree.run_pca_on_relative_abundance and
    general_pca_abundance_raw.run_pca_on_raw_counts. Never skip the
    StandardScaler step: see the module docstring for why comparing raw
    SVD output across stages is meaningless.

    Returns a plotting-ready DataFrame (species with a known taxon,
    optionally percentile-trimmed via outlier_percentile -- same filtering
    general_pca_abundance.py applies) and the PC1/PC2 explained-variance
    percentages computed over
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
    pca_df = assign_taxonomy_group(pca_df, taxon_dict)
    pca_df = remove_outliers(pca_df, low=outlier_percentile[0], high=outlier_percentile[1])
    return pca_df, explained


def taxon_ellipse_params(pca_df, min_points=3):
    """group -> (mean[2], cov[2x2]) for every taxon with enough points to estimate a covariance."""
    params = {}
    for group, sub in pca_df.groupby("Group"):
        x = sub["PC1"].to_numpy()
        y = sub["PC2"].to_numpy()
        if len(x) < min_points:
            continue
        cov = np.cov(x, y)
        if not np.all(np.isfinite(cov)) or cov[0, 0] <= 0 or cov[1, 1] <= 0:
            continue
        params[group] = (np.array([x.mean(), y.mean()]), cov)
    return params


def draw_taxon_ellipse(ax, mean, cov, color, n_std=2.0, fill_alpha=0.15, edge_alpha=0.55):
    """
    Shaded region for one taxon: a covariance ellipse centered on the
    group's own mean, sized by its own spread (n_std standard deviations
    along each principal axis of the group's point cloud). This is
    deliberately NOT a convex hull -- a hull would stretch out to touch
    every last far-flung point; an ellipse derived from the covariance
    matrix is dominated by where most of the group's mass actually is, so
    a handful of stragglers barely move it.
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    pearson = np.clip(pearson, -0.999, 0.999)
    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)

    face = mcolors.to_rgba(color, alpha=fill_alpha)
    edge = mcolors.to_rgba(color, alpha=edge_alpha)
    ellipse = Ellipse((0, 0), width=radius_x * 2, height=radius_y * 2,
                       facecolor=face, edgecolor=edge, linewidth=1.4, zorder=1)
    scale_x = np.sqrt(cov[0, 0]) * n_std
    scale_y = np.sqrt(cov[1, 1]) * n_std
    transf = (mtransforms.Affine2D()
              .rotate_deg(45)
              .scale(scale_x, scale_y)
              .translate(mean[0], mean[1]))
    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)


def taxon_silhouette(pca_df):
    """
    Mean silhouette score (sklearn) of the PC1/PC2 points, labeled by
    taxon: for each species, how much closer it sits to its own taxon's
    points than to the nearest other taxon's points, averaged over all
    species. Range [-1, 1]; > 0 means taxa are, on average, better
    separated than confounded. Unlike the ellipse-overlap idea, this
    doesn't assume an elliptical/Gaussian cluster shape and has no
    arbitrary size parameter (no equivalent of --ellipse-std) -- it's
    computed directly from the actual point-to-point distances. Requires
    >= 2 taxa with >= 2 points; returns None otherwise.
    """
    counts = pca_df["Group"].value_counts()
    usable = pca_df[pca_df["Group"].isin(counts[counts >= 2].index)]
    if usable["Group"].nunique() < 2:
        return None
    return float(silhouette_score(usable[["PC1", "PC2"]].to_numpy(), usable["Group"].to_numpy()))


def scatter_panel(ax, pca_df, color_map, title, explained, ellipses=True, ellipse_std=2.0):
    if ellipses:
        params = taxon_ellipse_params(pca_df)
        for group, (mean, cov) in params.items():
            draw_taxon_ellipse(ax, mean, cov, color_map[group], n_std=ellipse_std)

    for group, sub in pca_df.groupby("Group"):
        ax.scatter(sub["PC1"], sub["PC2"], s=8, alpha=0.7, color=color_map[group], edgecolor="none", zorder=2)

    sil = taxon_silhouette(pca_df)
    title_line2 = f"PC1={explained[0]:.1f}%  PC2={explained[1]:.1f}%"
    if sil is not None:
        title_line2 += f"   |   silhouette = {sil:.2f}"
    ax.set_title(f"{title}\n{title_line2}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    return sil


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

    # Restrict to the requested taxa *before* running the pipeline, not
    # after: the whole point of -t/--taxa is to compute each stage's PCA
    # only from variance among those species, not to compute it on everyone
    # and crop the plot to a sub-region of the same global layout (same
    # rationale as general_pca_abundance.py's -t/--taxa).
    if args.taxa:
        n_before = raw_df.shape[0]
        raw_df = raw_df[raw_df.index.map(taxon_dict).isin(args.taxa)]
        print(f"Taxa filter ({', '.join(args.taxa)}): kept {raw_df.shape[0]} / {n_before} species")

    if args.sample_cols is not None and args.sample_cols < raw_df.shape[1]:
        cols = rng.choice(raw_df.columns, size=args.sample_cols, replace=False)
        raw_df = raw_df[cols]
        print(f"Subsampled to {args.sample_cols} GO columns for speed")

    # --- replicate interactive_go_tree.run_pca_on_relative_abundance step by step ---
    raw_df = filter_species_by_stats(raw_df, total_prots)
    species = list(raw_df.index)

    pca_input = raw_df.loc[:, raw_df.sum(axis=0) > 5]
    print(f"Matrix: {raw_df.shape[0]} species x {raw_df.shape[1]} GO columns "
          f"-> {pca_input.shape[1]} kept after rare-column filter (colsum > 5)")
    del raw_df
    gc.collect()

    raw_counts = pca_input.to_numpy(dtype="float32")
    del pca_input
    gc.collect()

    pca_results = {}  # label -> (pca_df, explained[PC1%, PC2%])

    # Distribution diagnostics captured on the EXACT matrix each PCA panel
    # fits, both right BEFORE and right AFTER run_stage_pca's own internal
    # StandardScaler call -- the "after" version is recomputed here via a
    # throwaway StandardScaler call (run_stage_pca's own copy is discarded
    # internally), which costs a cheap mean/std pass, not the expensive SVD.
    # Captured inline, stage by stage, rather than kept as separate named
    # arrays, since some stages otherwise need multiple ~150MB float32
    # copies alive in memory at once.
    raw_sample, raw_stats = stage_stats(raw_counts.ravel(), rng)
    raw_scaled_sample, raw_scaled_stats = scaled_stage_stats(raw_counts, rng)
    print("Fitting PCA (StandardScaler + TruncatedSVD) on: raw counts ...")
    pca_results["1. Raw counts"] = run_stage_pca(raw_counts, species, taxon_dict, args.outlier_percentile)

    total_prots_col = total_prots.loc[species].to_numpy(dtype="float32")[:, None]
    relative_abundance = raw_counts / total_prots_col           # old normalization, count / Total_prots
    rel_sample, rel_stats = stage_stats(relative_abundance.ravel(), rng)
    rel_scaled_sample, rel_scaled_stats = scaled_stage_stats(relative_abundance, rng)
    print("Fitting PCA (StandardScaler + TruncatedSVD) on: relative abundance (count / Total_prots) ...")
    pca_results["2. Relative abundance"] = run_stage_pca(relative_abundance, species, taxon_dict, args.outlier_percentile)
    del relative_abundance, total_prots_col
    gc.collect()

    log_counts = np.log1p(raw_counts)                            # new array; raw_counts left untouched above
    del raw_counts
    gc.collect()
    log_sample, log_stats = stage_stats(log_counts.ravel(), rng)
    log_scaled_sample, log_scaled_stats = scaled_stage_stats(log_counts, rng)

    print("Fitting PCA (StandardScaler + TruncatedSVD) on: log(count+1), no row-centering ...")
    pca_results["3. log(count+1)"] = run_stage_pca(log_counts, species, taxon_dict, args.outlier_percentile)

    row_means = log_counts.mean(axis=1, keepdims=True)
    clr_values = log_counts - row_means                          # CLR row-centering -- new array
    del log_counts
    gc.collect()
    clr_sample, clr_stats = stage_stats(clr_values.ravel(), rng)
    clr_scaled_sample, clr_scaled_stats = scaled_stage_stats(clr_values, rng)

    print("Fitting PCA (StandardScaler + TruncatedSVD) on: CLR (log + row-centered) ...")
    pca_results["4. CLR"] = run_stage_pca(clr_values, species, taxon_dict, args.outlier_percentile)
    del clr_values
    gc.collect()

    print("\nPC1 / PC2 explained variance by pipeline stage (StandardScaler + TruncatedSVD, same recipe every stage):")
    for label, (_, ev) in pca_results.items():
        print(f"  {label:<24s} PC1={ev[0]:5.1f}%  PC2={ev[1]:5.1f}%")

    # --- figure 1: distribution of the exact matrix each pca_by_stage.png
    # panel fits, captured right before that panel's own internal
    # StandardScaler call -- a 1:1 companion to figure 2, same 2x2 layout,
    # same stage order/titles, so the two figures can be read side by side. ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    hist_panel(axes[0, 0], raw_sample, raw_stats,
               "1. Raw counts (antes de StandardScaler)", BLUE, log_x=True)
    axes[0, 0].set_xlabel("count (escala log; ceros excluidos de la vista log)")

    hist_panel(axes[0, 1], rel_sample, rel_stats,
               "2. Abundancia relativa (count/Total_prots)\n(antes de StandardScaler)", BLUE, log_x=True)
    axes[0, 1].set_xlabel("count / Total_prots (escala log; ceros excluidos de la vista log)")

    hist_panel(axes[1, 0], log_sample, log_stats,
               "3. log(count+1), sin centrar por fila\n(antes de StandardScaler)", BLUE)
    axes[1, 0].set_xlabel("log(count + 1)")

    hist_panel(axes[1, 1], clr_sample, clr_stats,
               "4. CLR (log + centrado por fila)\n(antes de StandardScaler)  [pipeline real]", BLUE, mark_zero=True)
    axes[1, 1].set_xlabel("log(count+1) - media_fila")

    fig.suptitle("Distribución de cada matriz justo ANTES de su StandardScaler\n"
                 "(la entrada real de cada panel de pca_transform_steps_pca_by_stage.png)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out_path = Path(args.output)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Wrote {out_path}")
    if out_path.suffix != ".png":
        png_path = out_path.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight")
        print(f"Wrote {png_path}")

    # --- figure 1b: distribution of the same 4 matrices, AFTER StandardScaler
    # -- i.e. what TruncatedSVD actually receives in each pca_by_stage.png
    # panel. Same 2x2 layout/order as figure 1, so the "before" and "after"
    # figures read as a matched pair. ---
    fig1b, axes1b = plt.subplots(2, 2, figsize=(12, 9))

    hist_panel(axes1b[0, 0], raw_scaled_sample, raw_scaled_stats,
               "1. Raw counts (después de StandardScaler)", BLUE, mark_zero=True)
    axes1b[0, 0].set_xlabel("z-score")

    hist_panel(axes1b[0, 1], rel_scaled_sample, rel_scaled_stats,
               "2. Abundancia relativa (count/Total_prots)\n(después de StandardScaler)", BLUE, mark_zero=True)
    axes1b[0, 1].set_xlabel("z-score")

    hist_panel(axes1b[1, 0], log_scaled_sample, log_scaled_stats,
               "3. log(count+1), sin centrar por fila\n(después de StandardScaler)", BLUE, mark_zero=True)
    axes1b[1, 0].set_xlabel("z-score")

    hist_panel(axes1b[1, 1], clr_scaled_sample, clr_scaled_stats,
               "4. CLR (log + centrado por fila)\n(después de StandardScaler)  [pipeline real]", BLUE, mark_zero=True)
    axes1b[1, 1].set_xlabel("z-score")

    fig1b.suptitle("Distribución de cada matriz justo DESPUÉS de su StandardScaler\n"
                   "(lo que realmente entra a TruncatedSVD en cada panel de pca_transform_steps_pca_by_stage.png)",
                   fontsize=13)
    fig1b.tight_layout(rect=[0, 0, 1, 0.93])

    after_out = out_path.with_name(out_path.stem + "_after_scaler" + out_path.suffix)
    fig1b.savefig(after_out, bbox_inches="tight")
    print(f"Wrote {after_out}")
    if after_out.suffix != ".png":
        after_png = after_out.with_suffix(".png")
        fig1b.savefig(after_png, bbox_inches="tight")
        print(f"Wrote {after_png}")

    # --- figure 2: PCA scatter per stage, colored by taxon ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(13, 11))
    panel_axes = [axes2[0, 0], axes2[0, 1], axes2[1, 0], axes2[1, 1]]
    panel_titles = {
        "1. Raw counts": "1. Raw counts + StandardScaler",
        "2. Relative abundance": "2. Abundancia relativa (count/Total_prots) + StandardScaler",
        "3. log(count+1)": "3. log(count+1), sin centrar por fila + StandardScaler",
        "4. CLR": "4. CLR (log + centrado por fila) + StandardScaler  [pipeline real]",
    }
    silhouette_by_stage = {}
    for ax, (label, (pca_df, explained)) in zip(panel_axes, pca_results.items()):
        sil = scatter_panel(ax, pca_df, color_map, panel_titles[label], explained,
                             ellipses=not args.no_ellipses, ellipse_std=args.ellipse_std)
        silhouette_by_stage[label] = sil

    print("\nSilhouette score (taxón) por etapa -- rango [-1, 1], > 0 = taxones mejor "
          "separados que confundidos, calculado directamente sobre los puntos PC1/PC2:")
    for label, sil in silhouette_by_stage.items():
        sil_str = f"{sil:.3f}" if sil is not None else "n/a (<2 taxones con >=2 especies)"
        print(f"  {label:<24s} {sil_str}")

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
