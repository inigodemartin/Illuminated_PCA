#!/usr/bin/env python3
"""
dark_proteome_enrichment.py — What is FANTASIA annotating in the dark proteome?

Compares, species by species, the GO profile of the *dark proteome*
(proteins annotated by FANTASIA but without any homology GO — the
`fantasia_only` matrix) against the proteins annotated by both tools
(`fantasia_both`, same GO source, so the comparison is not biased by tool
vocabulary). Both matrices come from dark_proteome_matrices.py (+ the
*_clean.tsv filtering of filter_dark_proteome_results.py).

Modules
  1. Term-level paired enrichment: for every GO term, the fraction of
     proteins carrying it in the dark group vs. the both group of the SAME
     species; per-species log2 fold change (pseudocount 0.5), median across
     species, fraction of species where the term is more frequent in the
     dark group, paired Wilcoxon signed-rank test across species and
     Benjamini–Hochberg FDR. Species are the replicates, so a term is
     "enriched in the dark proteome" only if it is so consistently across
     the 2.6k species, not because of a few huge proteomes.
  2. GO-slim profile (a `subset:` of go-basic.obo, default goslim_pir —
     the only slim that covers ~97 % of these annotations; goslim_generic
     has no cytoplasm / protein binding / response-to-stress categories
     and leaves ~36 % unmapped): each GO term is mapped to all its slim
     ancestors (is_a + part_of) and the counts are summed per slim term.
     This is an ANNOTATION-level profile
     (a protein with two terms under the same slim category counts twice);
     it is normalised by the total number of GO instances in the group.
     Same paired test as Module 1.
  3. Protein characteristics of the dark proteome from the mod02 stats
     table (length, short-protein fraction, GO richness, IC, namespace
     balance, FANTASIA/homology agreement), per species, summarised
     overall and per taxonomic group.

All plots are paired at the species level: red = dark proteome
(only_fantasia), blue = both.
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
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_IC_PATH  = Path(__file__).parent.parent / "data" / "All_GOs_ic.tsv"
DEFAULT_OBO_PATH = Path(__file__).parent.parent / "data" / "go-basic_2025.obo"

NAMESPACE_SHORT = {"biological_process": "BP", "cellular_component": "CC",
                   "molecular_function": "MF"}
GO_ROOTS = {"GO:0008150", "GO:0003674", "GO:0005575"}

COL_DARK = "#E8604C"   # dark proteome (only_fantasia)
COL_BOTH = "#4C9BE8"   # both
COL_GREY = "#888888"
COL_AMBER = "#F5A623"

# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------
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


def _checkpoint(path: Path, label: str, force: bool) -> bool:
    if not force and path.exists() and path.stat().st_size > 0:
        _log(f"  [checkpoint] {label} — {path.name} already exists, skipping")
        return True
    return False


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


def _set_plot_style() -> None:
    matplotlib.rcParams.update({
        "font.size":        11,
        "axes.titlesize":   12,
        "axes.labelsize":   11,
        "figure.dpi":       150,
        "savefig.dpi":      150,
        "figure.facecolor": "white",
    })


def _save(fig, out_path: Path, plot_formats: list) -> None:
    for fmt in plot_formats:
        fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_counts_matrix(path: Path):
    """Stream a Species x GO integer TSV into an int32 array (pandas' parser
    OOMs on 2.7k x 24k in a 3 GB WSL)."""
    rows, species = [], []
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        go_ids = header[1:]
        for line in fh:
            sp, rest = line.rstrip("\n").split("\t", 1)
            species.append(sp)
            rows.append(np.fromstring(rest, dtype=np.int32, sep="\t"))
    mat = np.vstack(rows)
    del rows
    if mat.shape[1] != len(go_ids):
        raise ValueError(f"{path.name}: header has {len(go_ids)} GO columns "
                         f"but rows have {mat.shape[1]}")
    return species, go_ids, mat


def align_to_columns(mat: np.ndarray, go_ids: list, union: list) -> np.ndarray:
    """Scatter the columns of `mat` into a zero int32 array over `union`."""
    pos = {g: i for i, g in enumerate(union)}
    idx = np.fromiter((pos[g] for g in go_ids), dtype=np.int64, count=len(go_ids))
    out = np.zeros((mat.shape[0], len(union)), dtype=np.int32)
    out[:, idx] = mat
    return out


def load_ic_table(ic_file: Path):
    """GO -> (namespace short, depth, IC, description). Headerless TSV:
    go_id, namespace, n_descendants, depth, ic, description."""
    info = {}
    with open(ic_file) as fh:
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6 or p[0] in info:
                continue
            try:
                ic = float(p[4])
                depth = int(p[3])
            except ValueError:
                continue
            info[p[0]] = (NAMESPACE_SHORT.get(p[1], ""), depth, ic, p[5])
    return info


def parse_obo(obo_file: Path, slim_subset: str = "goslim_pir"):
    """Minimal go-basic parser: name, namespace, parents (is_a + part_of),
    membership of `slim_subset`, obsolete flag, alt_id -> id."""
    name, ns, parents, slim, obsolete, alt = {}, {}, defaultdict(set), set(), set(), {}
    slim_tag = f"subset: {slim_subset}"
    cur = None
    with open(obo_file) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                cur = None
            elif line.startswith("[") and line.endswith("]"):
                cur = None          # [Typedef] etc.
            elif line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif cur is None:
                continue
            elif line.startswith("name:"):
                name[cur] = line[5:].strip()
            elif line.startswith("namespace:"):
                ns[cur] = NAMESPACE_SHORT.get(line[10:].strip(), "")
            elif line.startswith("is_a:"):
                parents[cur].add(line[5:].split("!")[0].strip())
            elif line.startswith("relationship: part_of"):
                parents[cur].add(line.split("part_of", 1)[1].split("!")[0].strip())
            elif line == slim_tag:
                slim.add(cur)
            elif line == "is_obsolete: true":
                obsolete.add(cur)
            elif line.startswith("alt_id:"):
                alt[line[7:].strip()] = cur
    return name, ns, parents, slim, obsolete, alt


def ancestors_closure(parents: dict):
    cache = {}

    def anc(g):
        if g in cache:
            return cache[g]
        acc = set()
        for p in parents.get(g, ()):
            acc.add(p)
            acc |= anc(p)
        cache[g] = acc
        return acc
    return anc


# ---------------------------------------------------------------------------
# Paired comparison core
# ---------------------------------------------------------------------------
def paired_compare(c_only: np.ndarray, c_both: np.ndarray,
                   n_only: np.ndarray, n_both: np.ndarray,
                   groups: dict, chunk: int = 2000) -> pd.DataFrame:
    """Per column: prevalence, mean frequency, per-species log2FC summary,
    paired Wilcoxon across species. `groups` = {label: boolean mask} for
    stratified medians. n_* are the per-species denominators."""
    n_sp, n_col = c_only.shape
    N_only = n_only[:, None].astype(np.float64)
    N_both = n_both[:, None].astype(np.float64)
    out = defaultdict(list)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)   # all-NaN medians, 0/0
    for start in range(0, n_col, chunk):
        sl = slice(start, min(start + chunk, n_col))
        a = c_only[:, sl].astype(np.float64)
        b = c_both[:, sl].astype(np.float64)
        f_a = a / N_only
        f_b = b / N_both
        present = (a > 0) | (b > 0)
        n_present = present.sum(axis=0)
        out["n_species_present"].append(n_present)
        out["n_species_dark"].append((a > 0).sum(axis=0))
        out["n_species_both"].append((b > 0).sum(axis=0))
        out["mean_pct_dark"].append(100 * f_a.mean(axis=0))
        out["mean_pct_both"].append(100 * f_b.mean(axis=0))
        pooled_a = a.sum(axis=0)
        pooled_b = b.sum(axis=0)
        out["pooled_log2OR"].append(
            np.log2((pooled_a + 0.5) / (N_only.sum() + 1))
            - np.log2((pooled_b + 0.5) / (N_both.sum() + 1)))
        lfc = (np.log2((a + 0.5) / (N_only + 1))
               - np.log2((b + 0.5) / (N_both + 1)))
        lfc_present = np.where(present, lfc, np.nan)
        with np.errstate(all="ignore"):
            out["median_log2FC"].append(np.nanmedian(lfc_present, axis=0))
            up = ((f_a > f_b) & present).sum(axis=0)
            out["frac_species_up"].append(
                np.where(n_present > 0, up / np.maximum(n_present, 1), np.nan))
            for label, mask in groups.items():
                sub = lfc_present[mask]
                out[f"median_log2FC_{label}"].append(np.nanmedian(sub, axis=0))
                np_sub = present[mask].sum(axis=0)
                up_sub = ((f_a[mask] > f_b[mask]) & present[mask]).sum(axis=0)
                out[f"frac_up_{label}"].append(
                    np.where(np_sub > 0, up_sub / np.maximum(np_sub, 1), np.nan))
        d = f_a - f_b
        with np.errstate(all="ignore"):
            res = stats.wilcoxon(d, axis=0, zero_method="wilcox")
        p = np.asarray(res.pvalue, dtype=np.float64)
        p[n_present < 2] = np.nan
        out["wilcoxon_p"].append(p)
        del a, b, f_a, f_b, lfc, lfc_present, d
    return pd.DataFrame({k: np.concatenate(v) for k, v in out.items()})


def add_fdr_and_direction(df: pd.DataFrame, min_species: int,
                          fdr_thr: float, min_lfc: float) -> pd.DataFrame:
    tested = (df["n_species_present"] >= min_species) & df["wilcoxon_p"].notna()
    df["tested"] = tested
    df["fdr"] = np.nan
    if tested.any():
        df.loc[tested, "fdr"] = multipletests(df.loc[tested, "wilcoxon_p"],
                                              method="fdr_bh")[1]
    direction = np.full(len(df), "not_tested", dtype=object)
    sig = tested & (df["fdr"] < fdr_thr)
    direction[tested.to_numpy()] = "ns"
    direction[(sig & (df["median_log2FC"] >= min_lfc)
               & (df["frac_species_up"] > 0.5)).to_numpy()] = "enriched_dark"
    direction[(sig & (df["median_log2FC"] <= -min_lfc)
               & (df["frac_species_up"] < 0.5)).to_numpy()] = "depleted_dark"
    df["direction"] = direction
    return df


# ---------------------------------------------------------------------------
# Module 1 — term-level enrichment
# ---------------------------------------------------------------------------
def run_term_enrichment(c_only, c_both, n_only, n_both, union, go_info, groups,
                        out_tsv: Path, min_species, fdr_thr, min_lfc, force):
    if _checkpoint(out_tsv, "term enrichment", force):
        return pd.read_csv(out_tsv, sep="\t")
    _log(f"  Paired comparison over {len(union)} GO terms x {c_only.shape[0]} species")
    df = paired_compare(c_only, c_both, n_only, n_both, groups)
    df.insert(0, "GO", union)
    df.insert(1, "description", [go_info.get(g, ("", 0, np.nan, ""))[3] for g in union])
    df.insert(2, "namespace",   [go_info.get(g, ("", 0, np.nan, ""))[0] for g in union])
    df.insert(3, "depth",       [go_info.get(g, ("", np.nan, np.nan, ""))[1] for g in union])
    df.insert(4, "IC",          [go_info.get(g, ("", 0, np.nan, ""))[2] for g in union])
    df = add_fdr_and_direction(df, min_species, fdr_thr, min_lfc)
    df = df.sort_values(["direction", "median_log2FC"],
                        ascending=[True, False]).reset_index(drop=True)
    df.to_csv(out_tsv, sep="\t", index=False, float_format="%.5g")
    _log(f"  Written {out_tsv.name}")
    return df


def plot_volcano(df: pd.DataFrame, out_path: Path, plot_formats: list,
                 fdr_thr: float, min_lfc: float) -> None:
    _set_plot_style()
    t = df[df["tested"]].copy()
    t["neglog_fdr"] = -np.log10(np.clip(t["fdr"].astype(float), 1e-300, 1))
    cap = np.nanpercentile(t["neglog_fdr"], 99.5)
    t["neglog_fdr"] = np.minimum(t["neglog_fdr"], cap)
    colors = t["direction"].map({"enriched_dark": COL_DARK, "depleted_dark": COL_BOTH}).fillna(COL_GREY)
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(t["median_log2FC"], t["neglog_fdr"], s=8, c=colors, alpha=0.6, linewidths=0)
    ax.axhline(-np.log10(fdr_thr), color=COL_GREY, lw=0.8, ls="--")
    for x in (-min_lfc, min_lfc):
        ax.axvline(x, color=COL_GREY, lw=0.8, ls="--")
    n_up = (t["direction"] == "enriched_dark").sum()
    n_dn = (t["direction"] == "depleted_dark").sum()
    ax.set_xlabel("median per-species log2( freq. dark proteome / freq. both )")
    ax.set_ylabel("−log10 FDR (paired Wilcoxon across species)")
    ax.set_title(f"GO terms: dark proteome vs both  —  {n_up} enriched (red), "
                 f"{n_dn} depleted (blue), {len(t)} tested")
    _save(fig, out_path, plot_formats)


def plot_top_terms(df: pd.DataFrame, out_path: Path, plot_formats: list,
                   n_top: int = 20, min_frac: float = 0.9) -> None:
    """The most abundant consistently enriched / depleted GO terms: bars =
    median per-species log2FC, label = mean % of proteins carrying the term."""
    _set_plot_style()
    up = df[(df["direction"] == "enriched_dark") & (df["frac_species_up"] >= min_frac)]
    up = up.sort_values("mean_pct_dark", ascending=False).head(n_top)
    dn = df[(df["direction"] == "depleted_dark") & (df["frac_species_up"] <= 1 - min_frac)]
    dn = dn.sort_values("mean_pct_both", ascending=False).head(n_top)
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.32 * max(len(up), len(dn)) + 1.5)))
    for ax, sub, col, title in ((axes[0], up, COL_DARK, "Enriched in the dark proteome"),
                                (axes[1], dn, COL_BOTH, "Depleted in the dark proteome")):
        sub = sub.iloc[::-1]
        y = np.arange(len(sub))
        ax.barh(y, sub["median_log2FC"], color=col, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([f"[{r['namespace']}] {str(r['description'])[:42]}" for _, r in sub.iterrows()],
                           fontsize=8)
        for yi, (_, r) in zip(y, sub.iterrows()):
            ax.text(r["median_log2FC"] + (0.05 if r["median_log2FC"] >= 0 else -0.05), yi,
                    f"{r['mean_pct_dark']:.2f}% vs {r['mean_pct_both']:.2f}%", va="center",
                    ha="left" if r["median_log2FC"] >= 0 else "right", fontsize=7, color="#333333")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("median per-species log2FC (dark / both)")
        ax.set_title(f"{title}\n(top {len(sub)} by abundance; same direction in ≥ {int(100 * min_frac)}% of species)",
                     fontsize=10)
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo - 0.6 * (hi - lo) if hi <= 0 else lo, hi + 0.6 * (hi - lo) if hi > 0 else hi)
    fig.tight_layout()
    _save(fig, out_path, plot_formats)


# ---------------------------------------------------------------------------
# Module 2 — GO slim profile
# ---------------------------------------------------------------------------
def build_slim_mapping(union: list, obo):
    """0/1 int32 matrix (n_go x n_slim) mapping each GO column to its slim
    ancestors (roots excluded). Returns (M, slim_ids, n_unmapped)."""
    name, ns, parents, slim, obsolete, alt = obo
    anc = ancestors_closure(parents)
    slim_ids = sorted(s for s in slim if s not in GO_ROOTS and s not in obsolete)
    pos = {s: i for i, s in enumerate(slim_ids)}
    M = np.zeros((len(union), len(slim_ids)), dtype=np.int32)
    n_unmapped = 0
    for i, g in enumerate(union):
        g2 = alt.get(g, g)
        targets = ({g2} | anc(g2)) & set(slim_ids)
        if not targets:
            n_unmapped += 1
            continue
        for s in targets:
            M[i, pos[s]] = 1
    return M, slim_ids, n_unmapped


def _blocked_matmul(counts: np.ndarray, M: np.ndarray, block: int = 256) -> np.ndarray:
    """counts (int32) @ M (0/1) in float64 row blocks: exact and BLAS-fast
    without a full float copy of the counts matrix."""
    M64 = M.astype(np.float64)
    out = np.empty((counts.shape[0], M.shape[1]), dtype=np.int64)
    for i in range(0, counts.shape[0], block):
        out[i:i + block] = np.rint(counts[i:i + block].astype(np.float64) @ M64).astype(np.int64)
    return out


def run_goslim_profile(c_only, c_both, union, obo, go_info, groups, slim_name: str,
                       out_tsv: Path, fdr_thr, min_lfc, force):
    if _checkpoint(out_tsv, "GO slim profile", force):
        return pd.read_csv(out_tsv, sep="\t")
    M, slim_ids, n_unmapped = build_slim_mapping(union, obo)
    tot_only = c_only.sum(axis=1).astype(np.int64)
    tot_both = c_both.sum(axis=1).astype(np.int64)
    mapped = M.any(axis=1)
    unm_only = 100 * (1 - c_only[:, mapped].sum() / max(tot_only.sum(), 1))
    unm_both = 100 * (1 - c_both[:, mapped].sum() / max(tot_both.sum(), 1))
    _log(f"  {slim_name}: {len(slim_ids)} slim terms, "
         f"{len(union) - n_unmapped}/{len(union)} GO columns mapped "
         f"({n_unmapped} without a slim ancestor)")
    _log(f"  Annotations without any slim category: dark {unm_only:.1f} %, both {unm_both:.1f} %")
    S_only = _blocked_matmul(c_only, M)
    S_both = _blocked_matmul(c_both, M)
    df = paired_compare(S_only, S_both, tot_only, tot_both, groups, chunk=len(slim_ids))
    name = obo[0]
    df.insert(0, "GO", slim_ids)
    df.insert(1, "description", [name.get(g, "") for g in slim_ids])
    df.insert(2, "namespace",   [obo[1].get(g, "") for g in slim_ids])
    df.insert(3, "n_go_terms_mapped", M.sum(axis=0))
    df = df.rename(columns={"mean_pct_dark": "mean_pct_annotations_dark",
                            "mean_pct_both": "mean_pct_annotations_both"})
    df = add_fdr_and_direction(df, min_species=2, fdr_thr=fdr_thr, min_lfc=min_lfc)
    df = df.sort_values(["namespace", "mean_pct_annotations_dark"],
                        ascending=[True, False]).reset_index(drop=True)
    df.to_csv(out_tsv, sep="\t", index=False, float_format="%.5g")
    _log(f"  Written {out_tsv.name}")
    return df


def plot_goslim(df: pd.DataFrame, out_path: Path, plot_formats: list,
                min_pct: float = 0.3) -> None:
    """Dumbbell: % of annotations per slim term, dark (red) vs both (blue)."""
    _set_plot_style()
    keep = df[(df[["mean_pct_annotations_dark", "mean_pct_annotations_both"]].max(axis=1) >= min_pct)]
    keep = keep.sort_values(["namespace", "mean_pct_annotations_dark"], ascending=[True, True])
    n = len(keep)
    fig, ax = plt.subplots(figsize=(9, max(4, 0.26 * n + 1.5)))
    y = np.arange(n)
    for yi, (_, r) in zip(y, keep.iterrows()):
        a, b = r["mean_pct_annotations_dark"], r["mean_pct_annotations_both"]
        ax.plot([b, a], [yi, yi], color=COL_GREY, lw=1.2, zorder=1)
        ax.scatter([b], [yi], color=COL_BOTH, s=28, zorder=2)
        ax.scatter([a], [yi], color=COL_DARK, s=28, zorder=3)
        if r["direction"] in ("enriched_dark", "depleted_dark"):
            ax.text(max(a, b) * 1.08, yi, f"{r['median_log2FC']:+.2f}", va="center",
                    fontsize=7, color=COL_DARK if r["direction"] == "enriched_dark" else COL_BOTH)
    labels = [f"[{r['namespace']}] {r['description'][:48]}" for _, r in keep.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("% of GO annotations in the group (mean over species, log scale)")
    ax.set_title("GO slim profile — dark proteome (red) vs both (blue)\n"
                 "number = median per-species log2FC where FDR < 0.05 and |log2FC| ≥ threshold",
                 fontsize=10)
    ax.grid(axis="x", lw=0.4, alpha=0.5)
    ax.set_ylim(-0.8, n - 0.2)
    _save(fig, out_path, plot_formats)


# ---------------------------------------------------------------------------
# Module 3 — protein characteristics
# ---------------------------------------------------------------------------
CHAR_PAIRS = [
    # (label, column dark, column both, unit)
    ("Median protein length",      "Median_length_only_fantasia",     "Median_length_both",     "aa"),
    ("Proteins < 100 aa",          "Pct_short_lt100aa_only_fantasia", "Pct_short_lt100aa_both", "%"),
    ("GO terms per protein (mean)", "GO_per_protein_mean_only_fantasia", "GO_per_protein_mean_both", "GO"),
    ("Mean IC of GO terms",        "Mean_IC_only_fantasia",           "Mean_IC_both",           "bits"),
    ("BP share of annotations",    "Pct_BP_only_fantasia",            "Pct_BP_both",            "%"),
    ("MF share of annotations",    "Pct_MF_only_fantasia",            "Pct_MF_both",            "%"),
    ("CC share of annotations",    "Pct_CC_only_fantasia",            "Pct_CC_both",            "%"),
]
CHAR_SINGLE = [
    ("Dark proteome share of proteome", "Pct_only_fantasia_of_proteome", "%"),
    ("Dark GO terms absent from both",  "Pct_GO_exclusive_only_fantasia", "%"),
    ("FANTASIA/homology Jaccard (both)", "Jaccard_fantasia_homology_mean_both", ""),
    ("Homology GO per protein (both)",  "Homology_GO_per_protein_mean_both", "GO"),
]


def run_characteristics(stats_df: pd.DataFrame, strata: dict, out_tsv: Path, force) -> pd.DataFrame:
    if _checkpoint(out_tsv, "characteristics", force):
        return pd.read_csv(out_tsv, sep="\t")
    rows = []
    for stratum, mask in strata.items():
        sub = stats_df[mask]
        for label, ca, cb, unit in CHAR_PAIRS:
            a, b = sub[ca].astype(float), sub[cb].astype(float)
            ok = a.notna() & b.notna()
            if ok.sum() < 3:
                continue
            d = a[ok] - b[ok]
            try:
                p = stats.wilcoxon(a[ok], b[ok]).pvalue
            except ValueError:
                p = np.nan
            rows.append({"stratum": stratum, "n_species": int(ok.sum()), "variable": label,
                         "unit": unit, "median_dark": a[ok].median(), "median_both": b[ok].median(),
                         "median_diff_dark_minus_both": d.median(),
                         "frac_species_dark_higher": float((d > 0).mean()),
                         "wilcoxon_p": p})
        for label, col, unit in CHAR_SINGLE:
            v = sub[col].astype(float).dropna()
            if len(v) < 3:
                continue
            rows.append({"stratum": stratum, "n_species": int(len(v)), "variable": label,
                         "unit": unit, "median_dark": v.median(), "median_both": np.nan,
                         "median_diff_dark_minus_both": np.nan,
                         "frac_species_dark_higher": np.nan, "wilcoxon_p": np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(out_tsv, sep="\t", index=False, float_format="%.5g")
    _log(f"  Written {out_tsv.name}")
    return df


def plot_characteristics(stats_df: pd.DataFrame, out_path: Path, plot_formats: list) -> None:
    _set_plot_style()
    pairs = CHAR_PAIRS[:6]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (label, ca, cb, unit) in zip(axes.ravel(), pairs):
        a, b = stats_df[ca].astype(float), stats_df[cb].astype(float)
        ok = a.notna() & b.notna()
        bp = ax.boxplot([a[ok], b[ok]], widths=0.55, patch_artist=True, showfliers=False,
                        medianprops={"color": "black"})
        for patch, col in zip(bp["boxes"], (COL_DARK, COL_BOTH)):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["dark proteome", "both"])
        try:
            p = stats.wilcoxon(a[ok], b[ok]).pvalue
            ptxt = "p < 1e-300" if p == 0 else f"p = {p:.1e}"
        except ValueError:
            ptxt = ""
        frac = float(((a[ok] - b[ok]) > 0).mean())
        ax.set_title(f"{label}\ndark higher in {100 * frac:.0f}% of species, {ptxt}", fontsize=9.5)
        ax.set_ylabel(unit)
    fig.suptitle(f"Dark proteome vs both — per-species paired values (n = {len(stats_df)} species)",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, out_path, plot_formats)


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="GO enrichment and protein characteristics of the dark proteome "
                    "(only_fantasia) vs proteins annotated by both tools.")
    ap.add_argument("--only", type=Path, required=True,
                    help="mod01_fantasia_only_counts_*_clean.tsv (dark proteome)")
    ap.add_argument("--both", type=Path, required=True,
                    help="mod01_fantasia_both_counts_*_clean.tsv (control, same GO source)")
    ap.add_argument("--stats", type=Path, required=True,
                    help="mod02_dark_proteome_stats_*_clean.tsv (per-species stats)")
    ap.add_argument("--taxonomy", type=Path, default=None,
                    help="Optional Species/Group TSV for stratified summaries "
                         "(e.g. species_taxonomy.tsv)")
    ap.add_argument("--ic", type=Path, default=DEFAULT_IC_PATH,
                    help=f"GO IC table with namespace/depth (default: {DEFAULT_IC_PATH})")
    ap.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH,
                    help=f"go-basic OBO for the goslim_generic mapping (default: {DEFAULT_OBO_PATH})")
    ap.add_argument("--slim", default="goslim_pir",
                    help="OBO subset used as GO slim in Module 2 (default: goslim_pir, "
                         "which covers ~97%% of the annotations; goslim_generic leaves "
                         "~36%% unmapped: no cytoplasm / protein binding / response to stress)")
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--min_species", type=int, default=50,
                    help="Test a GO term only if present (in either group) in at least "
                         "this many species (default: 50)")
    ap.add_argument("--min_group_species", type=int, default=30,
                    help="Taxonomy groups with fewer species are pooled into 'other' (default: 30)")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR threshold (default: 0.05)")
    ap.add_argument("--min_log2fc", type=float, default=1.0,
                    help="|median log2FC| threshold to call a term enriched/depleted (default: 1.0)")
    ap.add_argument("--min_log2fc_slim", type=float, default=0.5,
                    help="Same threshold for the GO slim profile (default: 0.5)")
    ap.add_argument("--skip_term_test", action="store_true",
                    help="Skip Module 1 — term-level paired enrichment")
    ap.add_argument("--skip_goslim", action="store_true",
                    help="Skip Module 2 — GO slim annotation profile")
    ap.add_argument("--skip_characteristics", action="store_true",
                    help="Skip Module 3 — protein characteristics from the stats table")
    ap.add_argument("--format", default="pdf",
                    help="Plot format(s): pdf, png, svg — comma-separated (default: pdf)")
    ap.add_argument("--force", action="store_true",
                    help="Rerun all steps from scratch even if outputs exist")
    ap.add_argument("--dry_run", action="store_true",
                    help="Validate inputs and print the steps that would run, then exit")
    ap.add_argument("--disable_co2_tracking", action="store_true",
                    help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    global _LOG_FH
    args = parse_args()
    t_start = time.monotonic()

    for name in ("only", "both", "stats", "ic", "obo"):
        setattr(args, name, getattr(args, name).resolve())
    if args.taxonomy is not None:
        args.taxonomy = args.taxonomy.resolve()
    pairs = [("--only", args.only), ("--both", args.both), ("--stats", args.stats),
             ("--ic", args.ic), ("--obo", args.obo)]
    if args.taxonomy is not None:
        pairs.append(("--taxonomy", args.taxonomy))
    _validate_inputs(pairs)

    run_dir  = Path(args.output)
    results  = run_dir / "results"
    workdir  = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name
    plot_formats = [f.strip().lstrip(".") for f in args.format.split(",")]

    log_path = _dated_log_path(logs_dir, "Run_DarkProteomeEnrichment")
    _LOG_FH  = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  DarkProteomeEnrichment {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    _banner(f"DarkProteomeEnrichment {VERSION}")
    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif results.exists() and any(results.iterdir()):
        _log("Existing results found — resuming from checkpoints (use --force to rerun)")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Dark matrix : {args.only}")
        _log(f"  Both matrix : {args.both}")
        _log(f"  Stats       : {args.stats}")
        _log(f"  Taxonomy    : {args.taxonomy}")
        _log(f"  Output      : {run_dir}/")
        _log("  Steps that would run:")
        if not args.skip_term_test:
            _log("    [1] Term-level paired enrichment  →  results/mod01_term_enrichment_*.tsv + volcano")
        if not args.skip_goslim:
            _log("    [2] GO slim annotation profile    →  results/mod02_goslim_profile_*.tsv + dumbbell")
        if not args.skip_characteristics:
            _log("    [3] Protein characteristics       →  results/mod03_dark_characteristics_*.tsv + boxplots")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir),
                                        output_file=f"{prefix}.emissions.csv",
                                        project_name="DarkProteomeEnrichment",
                                        log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped "
                 "(conda install -c conda-forge codecarbon)")

    # ---- shared inputs -----------------------------------------------------
    _banner("Loading inputs")
    stats_df = pd.read_csv(args.stats, sep="\t")
    stats_df = stats_df.set_index("Species")
    _log(f"  Stats table: {len(stats_df)} species")

    tax = None
    if args.taxonomy is not None:
        tax = pd.read_csv(args.taxonomy, sep="\t").drop_duplicates("Species").set_index("Species")["Group"]
        _log(f"  Taxonomy: {len(tax)} species, {tax.nunique()} groups")

    def make_strata(species: list) -> dict:
        """{label: bool mask over species}: all, viridi/non_viridi, taxonomy groups."""
        sp = pd.Index(species)
        strata = {"all": np.ones(len(sp), dtype=bool)}
        grp = stats_df.reindex(sp)["Group"]
        for g in sorted(grp.dropna().unique()):
            strata[g] = (grp == g).to_numpy()
        if tax is not None:
            tg = tax.reindex(sp).fillna("Unclassified")
            counts = tg.value_counts()
            small = counts[counts < args.min_group_species].index
            tg = tg.where(~tg.isin(small), "other")
            for g in sorted(tg.unique()):
                if g in strata:
                    continue
                strata[f"tax_{g}"] = (tg == g).to_numpy()
        return strata

    summary = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION,
               "input_only": str(args.only), "input_both": str(args.both),
               "input_stats": str(args.stats),
               "input_taxonomy": str(args.taxonomy) if args.taxonomy else None,
               "parameters": {"min_species": args.min_species, "fdr": args.fdr,
                              "min_log2fc": args.min_log2fc,
                              "min_log2fc_slim": args.min_log2fc_slim,
                              "min_group_species": args.min_group_species,
                              "slim": args.slim}}

    need_matrices = not (args.skip_term_test and args.skip_goslim)
    c_only = c_both = union = species = None
    if need_matrices:
        go_info = load_ic_table(args.ic)
        _log(f"  IC table: {len(go_info)} GO terms")
        _log(f"  Loading dark matrix: {args.only.name}")
        species, go_a, A = load_counts_matrix(args.only)
        _log(f"    {A.shape[0]} species x {A.shape[1]} GO")
        _log(f"  Loading both matrix: {args.both.name}")
        species_b, go_b, B = load_counts_matrix(args.both)
        _log(f"    {B.shape[0]} species x {B.shape[1]} GO")
        if species_b != species:
            if set(species_b) != set(species):
                print("ERROR: the two matrices do not contain the same species", file=sys.stderr)
                sys.exit(1)
            order = [species_b.index(s) for s in species]
            B = B[order]
            _log("    both matrix reordered to match the dark matrix species order")
        union = sorted(set(go_a) | set(go_b))
        _log(f"  Union of GO columns: {len(union)} "
             f"({len(set(go_a) - set(go_b))} only in dark, {len(set(go_b) - set(go_a))} only in both)")
        c_only = align_to_columns(A, go_a, union); del A
        c_both = align_to_columns(B, go_b, union); del B
        missing = [s for s in species if s not in stats_df.index]
        if missing:
            print(f"ERROR: {len(missing)} matrix species missing from the stats table, "
                  f"e.g. {missing[:3]}", file=sys.stderr)
            sys.exit(1)
        st = stats_df.reindex(species)
        n_only = st["N_only_fantasia"].to_numpy(dtype=np.int64)
        n_both = st["N_both"].to_numpy(dtype=np.int64)
        strata = make_strata(species)
        groups = {k: v for k, v in strata.items() if k != "all"}
        _log("  Strata: " + ", ".join(f"{k}={int(v.sum())}" for k, v in strata.items()))
        summary["n_species"] = len(species)
        summary["n_go_union"] = len(union)

    # ---- Module 1 ----------------------------------------------------------
    if not args.skip_term_test:
        _banner("Module 1 — term-level paired enrichment")
        out_tsv = results / f"mod01_term_enrichment_{prefix}.tsv"
        term_df = run_term_enrichment(c_only, c_both, n_only, n_both, union, go_info, groups,
                                      out_tsv, args.min_species, args.fdr, args.min_log2fc, args.force)
        vc = term_df["direction"].value_counts()
        _log("  " + ", ".join(f"{k}: {v}" for k, v in vc.items()))
        plot_volcano(term_df, results / f"mod01_volcano_{prefix}.pdf", plot_formats,
                     args.fdr, args.min_log2fc)
        plot_top_terms(term_df, results / f"mod01_top_terms_{prefix}.pdf", plot_formats)
        summary["module1_terms"] = {k: int(v) for k, v in vc.items()}
        dark_excl = term_df[(term_df["n_species_both"] == 0) & (term_df["n_species_dark"] > 0)]
        summary["module1_terms"]["never_in_both"] = int(len(dark_excl))
        _log(f"  GO terms never seen in the both group of any species: {len(dark_excl)}")

    # ---- Module 2 ----------------------------------------------------------
    if not args.skip_goslim:
        _banner("Module 2 — GO slim annotation profile")
        obo = parse_obo(args.obo, args.slim)
        _log(f"  OBO: {len(obo[0])} terms, {len(obo[3])} in {args.slim}, {len(obo[4])} obsolete")
        if not obo[3]:
            print(f"ERROR: no term carries 'subset: {args.slim}' in {args.obo}", file=sys.stderr)
            sys.exit(1)
        out_tsv = results / f"mod02_goslim_profile_{prefix}.tsv"
        slim_df = run_goslim_profile(c_only, c_both, union, obo, go_info, groups, args.slim,
                                     out_tsv, args.fdr, args.min_log2fc_slim, args.force)
        vc = slim_df["direction"].value_counts()
        _log("  " + ", ".join(f"{k}: {v}" for k, v in vc.items()))
        plot_goslim(slim_df, results / f"mod02_goslim_profile_{prefix}.pdf", plot_formats)
        summary["module2_slim"] = {k: int(v) for k, v in vc.items()}

    if need_matrices:
        del c_only, c_both

    # ---- Module 3 ----------------------------------------------------------
    if not args.skip_characteristics:
        _banner("Module 3 — protein characteristics of the dark proteome")
        sdf = stats_df.reset_index()
        strata3 = make_strata(sdf["Species"].tolist())
        out_tsv = results / f"mod03_dark_characteristics_{prefix}.tsv"
        char_df = run_characteristics(sdf, strata3, out_tsv, args.force)
        plot_characteristics(sdf, results / f"mod03_dark_characteristics_{prefix}.pdf", plot_formats)
        allrows = char_df[(char_df["stratum"] == "all")]
        for _, r in allrows.iterrows():
            if pd.notna(r["median_both"]):
                _log(f"  {r['variable']:<34} dark {r['median_dark']:>8.3g}  both {r['median_both']:>8.3g}  "
                     f"dark higher in {100 * r['frac_species_dark_higher']:.0f}% of species")
            else:
                _log(f"  {r['variable']:<34} median {r['median_dark']:.3g} {r['unit']}")
        summary["module3_all"] = {r["variable"]: {"median_dark": r["median_dark"],
                                                  "median_both": r["median_both"]}
                                  for _, r in allrows.iterrows()}

    # ---- wrap up -----------------------------------------------------------
    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass
    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin"
                   else ru.ru_maxrss / 1024)
    summary["resource_usage"] = {"wall_clock_s": round(elapsed_s, 1),
                                 "peak_mem_mb": round(peak_mem_mb, 1),
                                 "emissions_kg_CO2eq": emissions_kg}
    with open(results / f"{prefix}.run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: None if pd.isna(o) else float(o))
        fh.write("\n")
    _banner("Done")
    _log(f"  Wall clock: {elapsed_s:.1f} s, peak RSS: {peak_mem_mb:.0f} MB")
    _log(f"  Results in {results}/")
    if _LOG_FH is not None:
        _LOG_FH.close()


if __name__ == "__main__":
    main()
