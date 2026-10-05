#!/usr/bin/env python3
"""
dark_proteome_incompat_vs_size.py — Does a bigger dark proteome mean a worse
FANTASIA annotation?

Correlates, per species, the share of GO annotations that are incompatible
with a GO taxon constraint (output of dark_proteome_taxon_check.py) with the
size of the dark proteome (% of the proteome without homology GO, from
dark_proteome_matrices.py stats). Four annotation sets are checked: dark
(only_fantasia), both (FANTASIA on homology-supported proteins), homology
(AHRD, whole proteome) and FANTASIA-all (dark + both pooled). Spearman rho
overall and within each taxonomy group (to rule out Simpson's paradox), plus
medians by dark-share bin and a four-panel scatter.
"""

VERSION = "v0.1.0"

import argparse
import getpass
import json
import os
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL_DARK, COL_BOTH, COL_GREY, COL_AMBER = "#E8604C", "#4C9BE8", "#888888", "#F5A623"
GROUP_COLORS = {"Fungi": "#4C9BE8", "angiosperms": "#2E9E5B", "Protists": "#F5A623",
                "chlorophyta": "#8E5BD1", "other": "#888888"}
SETS = [("dark", "pct_viol_any_annotations_dark", "dark proteome (FANTASIA)"),
        ("both", "pct_viol_any_annotations_both", "both (FANTASIA)"),
        ("homology", "pct_viol_any_annotations_homology", "whole proteome (homology)"),
        ("fantasia_all", "pct_viol_fantasia_all", "whole proteome (FANTASIA, dark + both)")]
X = "Pct_only_fantasia_of_proteome"
BINS = [0, 20, 30, 40, 50, 60, 100]

_LOG_FH = None


def _log(msg: str) -> None:
    ts   = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, file=sys.stderr)
    if _LOG_FH is not None:
        print(line, file=_LOG_FH, flush=True)


def _banner(title: str) -> None:
    bar = "─" * (len(title) + 4)
    _log(f"┌{bar}┐")
    _log(f"│  {title}  │")
    _log(f"└{bar}┘")


def _dated_log_path(logs_dir: Path, base_name: str) -> Path:
    date_str = datetime.now().strftime("%Y%m%d")
    candidate = logs_dir / f"{base_name}_{date_str}.log"
    if not candidate.exists():
        return candidate
    n = 2
    while (logs_dir / f"{base_name}_{date_str}_{n}.log").exists():
        n += 1
    return logs_dir / f"{base_name}_{date_str}_{n}.log"


def _validate_inputs(pairs: list) -> None:
    ok = True
    for flag, path in pairs:
        if not path.exists():
            print(f"ERROR: {flag} not found: {path}", file=sys.stderr)
            ok = False
    if not ok:
        sys.exit(1)


def load(species_tsv: Path, stats_tsv: Path) -> pd.DataFrame:
    sp = pd.read_csv(species_tsv, sep="\t")
    sp = sp[sp["has_lineage"]].set_index("Species")
    st = pd.read_csv(stats_tsv, sep="\t").set_index("Species")
    d = sp.join(st[[X, "N_only_fantasia", "N_proteome", "Median_length_only_fantasia"]], how="inner")
    d["pct_viol_fantasia_all"] = 100 * (d["viol_any_annotations_dark"] + d["viol_any_annotations_both"]) \
        / (d["annotations_dark"] + d["annotations_both"])
    return d


def correlations(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("all", d)] + [(g, d[d["tax_group"] == g]) for g in sorted(d["tax_group"].dropna().unique())]
    for g, sub in groups:
        if len(sub) < 10:
            continue
        for key, col, _ in SETS + [("dark_terms", "pct_viol_terms_dark", "")]:
            rho, p = spearmanr(sub[X], sub[col], nan_policy="omit")
            rows.append({"group": g, "n_species": len(sub), "set": key, "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows)


def binned(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["dark_share_bin"] = pd.cut(d[X], BINS)
    rows = []
    for g, sub in [("all", d)] + [(g, d[d["tax_group"] == g]) for g in sorted(d["tax_group"].dropna().unique())]:
        for b, bsub in sub.groupby("dark_share_bin", observed=True):
            if len(bsub) < 3:
                continue
            r = {"group": g, "dark_share_bin": str(b), "n_species": len(bsub)}
            for key, col, _ in SETS + [("dark_terms", "pct_viol_terms_dark", "")]:
                r[f"median_pct_incompat_{key}"] = bsub[col].median()
            rows.append(r)
    return pd.DataFrame(rows)


def plot(d: pd.DataFrame, corr: pd.DataFrame, out_path: Path, plot_formats: list) -> None:
    matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                                "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharex=True)
    groups = [g for g in GROUP_COLORS if g in set(d["tax_group"])] + \
             [g for g in sorted(d["tax_group"].dropna().unique()) if g not in GROUP_COLORS]
    for ax, (key, col, title) in zip(axes.ravel(), SETS):
        for g in groups:
            sub = d[d["tax_group"] == g]
            ax.scatter(sub[X], sub[col], s=7, alpha=0.45, linewidths=0,
                       color=GROUP_COLORS.get(g, COL_GREY), label=f"{g} (n={len(sub)})")
        # running median over dark-share bins (all species)
        bins = np.arange(0, 101, 5)
        centers, meds = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (d[X] >= lo) & (d[X] < hi)
            if m.sum() >= 10:
                centers.append((lo + hi) / 2)
                meds.append(d.loc[m, col].median())
        ax.plot(centers, meds, color="black", lw=1.8, label="median (5-point bins)")
        txt = []
        for g in ["all"] + groups:
            r = corr[(corr["group"] == g) & (corr["set"] == key)]
            if len(r):
                txt.append(f"{g}: ρ = {r['spearman_rho'].iloc[0]:+.2f}")
        ax.text(0.02, 0.97, "\n".join(txt), transform=ax.transAxes, va="top", fontsize=8.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("% annotations incompatible with a taxon constraint")
        ax.grid(lw=0.4, alpha=0.5)
    for ax in axes[1]:
        ax.set_xlabel("dark proteome share of the proteome (% proteins without homology GO)")
    axes[0][0].legend(fontsize=8, frameon=False, loc="lower right", markerscale=2.5)
    fig.suptitle("Taxon-constraint incompatibilities vs dark-proteome size (Spearman ρ, per species)",
                 fontsize=12)
    fig.tight_layout()
    for fmt in plot_formats:
        fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    global _LOG_FH
    ap = argparse.ArgumentParser(description="Correlate taxon-constraint incompatibilities with dark-proteome size.")
    ap.add_argument("--species", type=Path, required=True,
                    help="mod01_taxon_violations_species_*.tsv from dark_proteome_taxon_check.py (dark-proteome mode)")
    ap.add_argument("--stats", type=Path, required=True, help="mod02_dark_proteome_stats_*_clean.tsv")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--format", default="pdf", help="Plot format(s), comma-separated (default: pdf)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args()
    t_start = time.monotonic()
    args.species, args.stats = args.species.resolve(), args.stats.resolve()
    _validate_inputs([("--species", args.species), ("--stats", args.stats)])

    run_dir = Path(args.output)
    results, logs_dir = run_dir / "results", run_dir / "logs"
    for dd in (results, logs_dir):
        dd.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name
    plot_formats = [f.strip().lstrip(".") for f in args.format.split(",")]

    _LOG_FH = open(_dated_log_path(logs_dir, "Run_DarkProteomeIncompatVsSize"), "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  DarkProteomeIncompatVsSize {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nUser      : {getpass.getuser()}\n"
                  f"Server    : {platform.node()}\nOS        : {platform.system()} {platform.release()} "
                  f"({platform.machine()})\nDirectory : {os.getcwd()}\nCommand   : {' '.join(sys.argv)}\n{sep}\n\n")
    _LOG_FH.flush()

    _banner(f"DarkProteomeIncompatVsSize {VERSION}")
    d = load(args.species, args.stats)
    _log(f"  {len(d)} species with lineage and stats")
    corr = correlations(d)
    corr.to_csv(results / f"mod01_incompat_vs_dark_share_correlations_{prefix}.tsv", sep="\t", index=False,
                float_format="%.4g")
    b = binned(d)
    b.to_csv(results / f"mod01_incompat_vs_dark_share_binned_{prefix}.tsv", sep="\t", index=False, float_format="%.4g")
    plot(d, corr, results / f"mod01_incompat_vs_dark_share_{prefix}.pdf", plot_formats)
    for _, r in corr[corr["group"] == "all"].iterrows():
        _log(f"  all species — {r['set']:<13} Spearman rho = {r['spearman_rho']:+.2f}")
    for g in sorted(set(corr["group"]) - {"all"}):
        sub = corr[corr["group"] == g]
        _log(f"  {g:<13} (n={int(sub['n_species'].iloc[0]):4d}) " +
             "  ".join(f"{r['set']} {r['spearman_rho']:+.2f}" for _, r in sub.iterrows()))

    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024
    summary = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION,
               "input_species": str(args.species), "input_stats": str(args.stats), "n_species": len(d),
               "spearman_all": {r["set"]: round(float(r["spearman_rho"]), 3)
                                for _, r in corr[corr["group"] == "all"].iterrows()},
               "resource_usage": {"wall_clock_s": round(elapsed_s, 1), "peak_mem_mb": round(peak_mem_mb, 1)}}
    with open(results / f"{prefix}.run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    _banner("Done")
    _log(f"  Results in {results}/")
    _LOG_FH.close()


if __name__ == "__main__":
    main()
