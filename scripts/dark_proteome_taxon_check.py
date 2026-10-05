#!/usr/bin/env python3
"""
dark_proteome_taxon_check.py — How many GO terms in the dark proteome make no
sense for the organism that carries them?

Two complementary, species-level measures of "nonsense" annotations, computed
for the dark proteome (FANTASIA-only proteins), the control group (proteins
annotated by both tools, GO from FANTASIA) and, when given, the classical
homology annotation of the whole proteome (AHRD):

Module 1 — GO taxon-constraint violations (the official, ontology-defined
  measure). The GO Consortium asserts `only_in_taxon` / `never_in_taxon`
  constraints on ~1 400 terms (e.g. "granulosa cell proliferation" only in
  Metazoa; "cell wall mannoprotein biosynthesis" never in Metazoa). The
  constraints are inherited by every descendant through is_a and part_of.
  An annotation violates a constraint when the species' NCBI lineage does
  not contain the `only_in` taxon (or any member of a union taxon such as
  "Fungi or Bacteria") or does contain a `never_in` taxon. We report, per
  species and group, the % of GO annotations (and of distinct GO terms)
  that violate at least one constraint, split by constraint type, plus the
  offending terms and constraints ranked by how many annotations they cost.

Module 2 — Clade-unsupported terms (data-driven, softer). A GO term is
  "unsupported in clade G" when homology (AHRD) never assigns it to ANY
  species of G in the whole dataset (G = the taxonomy groups with ≥
  --min_group_species species). The % of dark / both annotations on such
  terms per species is an upper bound of novelty-or-noise: genuinely new
  functions would also land here, so read it next to Module 1.

Counts are annotation-level (a protein with two violating terms counts
twice) because the input matrices are Species x GO protein counts.
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dark_proteome_enrichment import (load_counts_matrix, align_to_columns, load_ic_table,
                                      parse_obo, ancestors_closure)

HERE = Path(__file__).parent.parent
DEFAULT_IC_PATH          = HERE / "data" / "All_GOs_ic.tsv"
DEFAULT_OBO_PATH         = HERE / "data" / "go-basic_2025.obo"
DEFAULT_CONSTRAINTS_PATH = HERE / "data" / "go_taxon_constraints.tsv"
DEFAULT_UNIONS_PATH      = HERE / "data" / "go_taxon_unions.tsv"
DEFAULT_LINEAGES_PATH    = HERE / "data" / "species_lineage_taxids.tsv"

COL_DARK = "#E8604C"
COL_BOTH = "#4C9BE8"
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
# Constraints, lineages, violation matrix
# ---------------------------------------------------------------------------
def load_constraints(path: Path):
    """GO -> set of (kind, taxon_id_string); taxon names for labels."""
    df = pd.read_csv(path, sep="\t", comment="#")
    cons, names = defaultdict(set), {}
    for r in df.itertuples(index=False):
        cons[r.GO].add((r.constraint, r.taxon))
        names[r.taxon] = r.taxon_name if isinstance(r.taxon_name, str) else r.taxon
    return cons, names


def load_unions(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t", comment="#")
    return {r.union: {int(t) for t in str(r.member_taxids).split(",") if t}
            for r in df.itertuples(index=False)}


def load_lineages(path: Path) -> dict:
    df = pd.read_csv(path, sep="\t", comment="#", dtype=str).drop_duplicates("Species")
    return {r.Species: {int(t) for t in str(r.lineage_taxids).split(",") if t}
            for r in df.itertuples(index=False) if isinstance(r.lineage_taxids, str)}


def taxon_members(taxon: str, unions: dict) -> set:
    if taxon.startswith("NCBITaxon_Union:"):
        return unions.get(taxon, set())
    return {int(taxon.split(":")[1])}


def effective_constraints(union_cols: list, cons: dict, obo) -> list:
    """Per GO column: set of inherited (kind, taxon) constraints — its own
    plus those of every is_a / part_of ancestor."""
    name, ns, parents, slim, obsolete, alt = obo
    anc = ancestors_closure(parents)
    constrained = set(cons)
    out = []
    for g in union_cols:
        g2 = alt.get(g, g)
        hits = ({g2} | anc(g2)) & constrained
        eff = set()
        for h in hits:
            eff |= cons[h]
        out.append(frozenset(eff))
    return out


def violation_matrices(eff: list, species: list, lineages: dict, unions: dict):
    """Boolean species x GO matrices: violates an only_in constraint,
    violates a never_in constraint. Species without lineage -> all False
    (and reported separately)."""
    n_sp, n_go = len(species), len(eff)
    lin = [lineages.get(s) for s in species]
    has_lin = np.array([l is not None for l in lin])
    pair_vec = {}

    def vec(pair):
        if pair in pair_vec:
            return pair_vec[pair]
        kind, taxon = pair
        members = taxon_members(taxon, unions)
        v = np.zeros(n_sp, dtype=bool)
        for i, l in enumerate(lin):
            if l is None:
                continue
            inside = bool(l & members)
            v[i] = (not inside) if kind == "only_in" else inside
        pair_vec[pair] = v
        return v

    V_only = np.zeros((n_sp, n_go), dtype=bool)
    V_never = np.zeros((n_sp, n_go), dtype=bool)
    sig_cache = {}
    for j, sig in enumerate(eff):
        if not sig:
            continue
        if sig not in sig_cache:
            vo = np.zeros(n_sp, dtype=bool)
            vn = np.zeros(n_sp, dtype=bool)
            for pair in sig:
                if pair[0] == "only_in":
                    vo |= vec(pair)
                else:
                    vn |= vec(pair)
            sig_cache[sig] = (vo, vn)
        vo, vn = sig_cache[sig]
        V_only[:, j] = vo
        V_never[:, j] = vn
    return V_only, V_never, has_lin


def masked_row_sums(counts: np.ndarray, mask: np.ndarray, chunk: int = 2000):
    """(counts * mask).sum(axis=1) and ((counts > 0) & mask).sum(axis=1)
    without materialising a full copy."""
    ann = np.zeros(counts.shape[0], dtype=np.int64)
    terms = np.zeros(counts.shape[0], dtype=np.int64)
    for s in range(0, counts.shape[1], chunk):
        c = counts[:, s:s + chunk]
        m = mask[:, s:s + chunk]
        ann += (c * m).sum(axis=1, dtype=np.int64)
        terms += ((c > 0) & m).sum(axis=1)
    return ann, terms


# ---------------------------------------------------------------------------
# Module 1 — taxon-constraint violations
# ---------------------------------------------------------------------------
def run_taxon_check(matrices: dict, union_cols: list, species: list, eff: list,
                    lineages: dict, unions: dict, taxon_names: dict, go_info: dict,
                    meta: pd.DataFrame, results: Path, prefix: str, force: bool):
    out_species = results / f"mod01_taxon_violations_species_{prefix}.tsv"
    out_terms   = results / f"mod01_taxon_violations_terms_{prefix}.tsv"
    out_cons    = results / f"mod01_taxon_violations_constraints_{prefix}.tsv"
    out_groups  = results / f"mod01_taxon_violations_groups_{prefix}.tsv"
    if all(_checkpoint(p, "taxon check", force) for p in (out_species, out_terms, out_cons, out_groups)):
        return (pd.read_csv(out_species, sep="\t"), pd.read_csv(out_terms, sep="\t"),
                pd.read_csv(out_groups, sep="\t"))

    V_only, V_never, has_lin = violation_matrices(eff, species, lineages, unions)
    V_any = V_only | V_never
    n_constrained = sum(1 for e in eff if e)
    _log(f"  {n_constrained}/{len(eff)} GO columns inherit at least one taxon constraint")
    _log(f"  {int(has_lin.sum())}/{len(species)} species with NCBI lineage; "
         f"{int(V_any.any(axis=0).sum())} GO columns violated in at least one species")

    sp = meta.copy()
    sp["has_lineage"] = has_lin
    for label, counts in matrices.items():
        total_ann = counts.sum(axis=1, dtype=np.int64)
        total_terms = (counts > 0).sum(axis=1)
        sp[f"annotations_{label}"] = total_ann
        sp[f"terms_{label}"] = total_terms
        for kind, V in (("any", V_any), ("only_in", V_only), ("never_in", V_never)):
            ann, terms = masked_row_sums(counts, V)
            sp[f"viol_{kind}_annotations_{label}"] = ann
            with np.errstate(all="ignore"):
                sp[f"pct_viol_{kind}_annotations_{label}"] = np.where(
                    total_ann > 0, 100 * ann / np.maximum(total_ann, 1), np.nan)
            if kind == "any":
                sp[f"viol_terms_{label}"] = terms
                with np.errstate(all="ignore"):
                    sp[f"pct_viol_terms_{label}"] = np.where(
                        total_terms > 0, 100 * terms / np.maximum(total_terms, 1), np.nan)
    for c in sp.columns:
        if c.startswith("pct_viol") or c.startswith("viol_"):
            sp.loc[~has_lin, c] = np.nan
    sp.to_csv(out_species, sep="\t", index=False, float_format="%.4f")
    _log(f"  Written {out_species.name}")

    # --- per GO term: how many violating annotations does it cost ------------
    rows = []
    cons_tot = defaultdict(lambda: defaultdict(int))
    for j, g in enumerate(union_cols):
        if not eff[j] or not V_any[:, j].any():
            continue
        vmask = V_any[:, j]
        r = {"GO": g, "description": go_info.get(g, ("", 0, np.nan, ""))[3],
             "namespace": go_info.get(g, ("", 0, np.nan, ""))[0],
             "IC": go_info.get(g, ("", 0, np.nan, ""))[2],
             "constraints": "; ".join(sorted(f"{k} {taxon_names.get(t, t)}" for k, t in eff[j])),
             "n_species_violating_possible": int(vmask.sum())}
        for label, counts in matrices.items():
            col = counts[:, j]
            r[f"viol_annotations_{label}"] = int(col[vmask].sum())
            r[f"n_species_violating_{label}"] = int((col[vmask] > 0).sum())
            r[f"annotations_{label}_total"] = int(col.sum())
        rows.append(r)
        # attribute to the constraint pairs actually violated
        for kind, taxon in eff[j]:
            members = taxon_members(taxon, unions)
            for i in np.flatnonzero(vmask):
                l = lineages.get(species[i])
                inside = bool(l & members) if l else False
                if (kind == "only_in" and not inside) or (kind == "never_in" and inside):
                    for label, counts in matrices.items():
                        cons_tot[(kind, taxon)][label] += int(counts[i, j])
    terms_df = pd.DataFrame(rows)
    if len(terms_df):
        terms_df = terms_df.sort_values(f"viol_annotations_{list(matrices)[0]}", ascending=False)
    terms_df.to_csv(out_terms, sep="\t", index=False, float_format="%.4g")
    _log(f"  Written {out_terms.name} ({len(terms_df)} violating GO terms)")

    cons_rows = []
    for (kind, taxon), d in cons_tot.items():
        r = {"constraint": kind, "taxon": taxon, "taxon_name": taxon_names.get(taxon, taxon)}
        r.update({f"viol_annotations_{label}": d.get(label, 0) for label in matrices})
        cons_rows.append(r)
    cons_df = pd.DataFrame(cons_rows)
    if len(cons_df):
        cons_df = cons_df.sort_values(f"viol_annotations_{list(matrices)[0]}", ascending=False)
    cons_df.to_csv(out_cons, sep="\t", index=False)
    _log(f"  Written {out_cons.name}")

    # --- per group summary -----------------------------------------------------
    grp_rows = []
    labels = list(matrices)
    for grp, sub in [("all", sp[sp["has_lineage"]])] + [
            (g, sp[(sp["tax_group"] == g) & sp["has_lineage"]]) for g in sorted(sp["tax_group"].dropna().unique())]:
        if len(sub) < 3:
            continue
        r = {"group": grp, "n_species": len(sub)}
        for label in labels:
            r[f"median_pct_viol_annotations_{label}"] = sub[f"pct_viol_any_annotations_{label}"].median()
            r[f"median_pct_viol_terms_{label}"] = sub[f"pct_viol_terms_{label}"].median()
            r[f"median_pct_viol_only_in_{label}"] = sub[f"pct_viol_only_in_annotations_{label}"].median()
            r[f"median_pct_viol_never_in_{label}"] = sub[f"pct_viol_never_in_annotations_{label}"].median()
        if "dark" in labels and "both" in labels:
            a = sub["pct_viol_any_annotations_dark"]
            b = sub["pct_viol_any_annotations_both"]
            ok = a.notna() & b.notna()
            r["frac_species_dark_higher"] = float(((a[ok] - b[ok]) > 0).mean())
            r["median_ratio_dark_over_both"] = float(((a[ok] + 1e-9) / (b[ok] + 1e-9)).median())
            try:
                r["wilcoxon_p_dark_vs_both"] = stats.wilcoxon(a[ok], b[ok]).pvalue
            except ValueError:
                r["wilcoxon_p_dark_vs_both"] = np.nan
        grp_rows.append(r)
    groups_df = pd.DataFrame(grp_rows)
    groups_df.to_csv(out_groups, sep="\t", index=False, float_format="%.4g")
    _log(f"  Written {out_groups.name}")
    return sp, terms_df, groups_df


def plot_violations(sp: pd.DataFrame, labels: list, out_path: Path, plot_formats: list,
                    col_prefix: str = "pct_viol_any_annotations",
                    ylabel: str = "% of GO annotations violating a GO taxon constraint",
                    title: str = "GO taxon-constraint violations per species") -> None:
    _set_plot_style()
    colors = {"dark": COL_DARK, "both": COL_BOTH, "homology": COL_GREY}
    sub = sp[sp["has_lineage"]] if "has_lineage" in sp else sp
    groups = ["all"] + sorted(g for g in sub["tax_group"].dropna().unique()
                              if (sub["tax_group"] == g).sum() >= 3)
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(groups) + 2), 5.5))
    width = 0.8 / len(labels)
    for k, label in enumerate(labels):
        data, pos = [], []
        for gi, g in enumerate(groups):
            s = sub if g == "all" else sub[sub["tax_group"] == g]
            v = s[f"{col_prefix}_{label}"].dropna()
            data.append(v)
            pos.append(gi + (k - (len(labels) - 1) / 2) * width)
        bp = ax.boxplot(data, positions=pos, widths=width * 0.9, patch_artist=True,
                        showfliers=False, medianprops={"color": "black"})
        for patch in bp["boxes"]:
            patch.set_facecolor(colors.get(label, COL_AMBER))
            patch.set_alpha(0.8)
        ax.plot([], [], color=colors.get(label, COL_AMBER), lw=8, alpha=0.8,
                label={"dark": "dark proteome (FANTASIA)", "both": "both (FANTASIA)",
                       "homology": "whole proteome (homology)"}.get(label, label))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g}\n(n={len(sub) if g == 'all' else int((sub['tax_group'] == g).sum())})"
                        for g in groups], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    _save(fig, out_path, plot_formats)


def plot_top_violating_terms(terms_df: pd.DataFrame, labels: list, out_path: Path,
                             plot_formats: list, n_top: int = 25) -> None:
    _set_plot_style()
    if not len(terms_df):
        return
    top = terms_df.sort_values("viol_annotations_dark", ascending=False).head(n_top).iloc[::-1]
    colors = {"dark": COL_DARK, "both": COL_BOTH, "homology": COL_GREY}
    fig, ax = plt.subplots(figsize=(11, 0.34 * len(top) + 1.5))
    y = np.arange(len(top))
    h = 0.8 / len(labels)
    for k, label in enumerate(labels):
        ax.barh(y + (k - (len(labels) - 1) / 2) * h, top[f"viol_annotations_{label}"], height=h * 0.9,
                color=colors.get(label, COL_AMBER), alpha=0.85, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{r['namespace']}] {str(r['description'])[:38]}  | {str(r["constraints"])[:34]}"
                        for _, r in top.iterrows()], fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("violating annotations summed over all species (log scale)")
    ax.set_title(f"Top {len(top)} GO terms by taxon-constraint violations in the dark proteome\n"
                 "(after | : the inherited constraint that is violated)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, out_path, plot_formats)


# ---------------------------------------------------------------------------
# Module 2 — clade-unsupported terms
# ---------------------------------------------------------------------------
def run_clade_check(matrices: dict, union_cols: list, species: list, meta: pd.DataFrame,
                    go_info: dict, results: Path, prefix: str, min_group: int, force: bool):
    out_species = results / f"mod02_clade_unsupported_species_{prefix}.tsv"
    out_terms   = results / f"mod02_clade_unsupported_terms_{prefix}.tsv"
    if _checkpoint(out_species, "clade check", force) and out_terms.exists():
        return pd.read_csv(out_species, sep="\t"), pd.read_csv(out_terms, sep="\t")
    hom = matrices["homology"]
    sp = meta.copy()
    groups = [g for g, n in sp["tax_group"].value_counts().items()
              if n >= min_group and g not in ("other", "Unclassified")]
    _log(f"  Clades evaluated (≥ {min_group} species, excluding other/Unclassified): "
         + ", ".join(f"{g}={int((sp['tax_group'] == g).sum())}" for g in groups))
    for label in ("dark", "both"):
        sp[f"pct_clade_unsupported_annotations_{label}"] = np.nan
        sp[f"pct_clade_unsupported_terms_{label}"] = np.nan
    term_rows = []
    for g in groups:
        mask = (sp["tax_group"] == g).to_numpy()
        hom_prev = (hom[mask] > 0).sum(axis=0)
        unsupported = hom_prev == 0
        for label in ("dark", "both"):
            counts = matrices[label][mask]
            tot = counts.sum(axis=1, dtype=np.int64)
            tot_t = (counts > 0).sum(axis=1)
            ann = counts[:, unsupported].sum(axis=1, dtype=np.int64)
            terms = (counts[:, unsupported] > 0).sum(axis=1)
            with np.errstate(all="ignore"):
                sp.loc[mask, f"pct_clade_unsupported_annotations_{label}"] = np.where(tot > 0, 100 * ann / np.maximum(tot, 1), np.nan)
                sp.loc[mask, f"pct_clade_unsupported_terms_{label}"] = np.where(tot_t > 0, 100 * terms / np.maximum(tot_t, 1), np.nan)
        dark_g = matrices["dark"][mask]
        both_g = matrices["both"][mask]
        d_sum = dark_g[:, unsupported].sum(axis=0, dtype=np.int64)
        b_sum = both_g[:, unsupported].sum(axis=0, dtype=np.int64)
        d_sp = (dark_g[:, unsupported] > 0).sum(axis=0)
        idx = np.flatnonzero(unsupported)
        for k in np.argsort(-d_sum)[:300]:
            if d_sum[k] == 0:
                continue
            j = idx[k]
            gid = union_cols[j]
            info = go_info.get(gid, ("", 0, np.nan, ""))
            term_rows.append({"clade": g, "GO": gid, "description": info[3], "namespace": info[0],
                              "IC": info[2], "dark_annotations": int(d_sum[k]),
                              "dark_species": int(d_sp[k]), "both_annotations": int(b_sum[k]),
                              "n_species_in_clade": int(mask.sum())})
        _log(f"    {g}: {int(unsupported.sum())} GO columns never assigned by homology in the clade; "
             f"median % dark annotations on them "
             f"{sp.loc[mask, 'pct_clade_unsupported_annotations_dark'].median():.2f} "
             f"(both {sp.loc[mask, 'pct_clade_unsupported_annotations_both'].median():.2f})")
    sp.to_csv(out_species, sep="\t", index=False, float_format="%.4f")
    terms_df = pd.DataFrame(term_rows)
    terms_df.to_csv(out_terms, sep="\t", index=False, float_format="%.4g")
    _log(f"  Written {out_species.name}, {out_terms.name}")
    return sp, terms_df


# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Quantify GO annotations that make no sense for the organism "
                    "(taxon-constraint violations, clade-unsupported terms) in the dark "
                    "proteome vs both vs homology.")
    ap.add_argument("--only", type=Path, required=True, help="mod01_fantasia_only_counts_*_clean.tsv")
    ap.add_argument("--both", type=Path, required=True, help="mod01_fantasia_both_counts_*_clean.tsv")
    ap.add_argument("--homology", type=Path, default=None,
                    help="mod01_homology_all_counts_*_clean.tsv (reference; required for Module 2)")
    ap.add_argument("--stats", type=Path, required=True, help="mod02_dark_proteome_stats_*_clean.tsv")
    ap.add_argument("--taxonomy", type=Path, default=None, help="Species/Group TSV (species_taxonomy.tsv)")
    ap.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS_PATH,
                    help=f"GO taxon constraints TSV (default: {DEFAULT_CONSTRAINTS_PATH})")
    ap.add_argument("--unions", type=Path, default=DEFAULT_UNIONS_PATH,
                    help=f"Union taxa TSV (default: {DEFAULT_UNIONS_PATH})")
    ap.add_argument("--lineages", type=Path, default=DEFAULT_LINEAGES_PATH,
                    help=f"Species NCBI lineage taxids TSV (default: {DEFAULT_LINEAGES_PATH})")
    ap.add_argument("--ic", type=Path, default=DEFAULT_IC_PATH)
    ap.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--min_group_species", type=int, default=30,
                    help="Taxonomy groups below this size are pooled as 'other' (default: 30)")
    ap.add_argument("--skip_taxon_check", action="store_true", help="Skip Module 1 — taxon-constraint violations")
    ap.add_argument("--skip_clade_check", action="store_true", help="Skip Module 2 — clade-unsupported terms")
    ap.add_argument("--format", default="pdf", help="Plot format(s), comma-separated (default: pdf)")
    ap.add_argument("--force", action="store_true", help="Rerun all steps even if outputs exist")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs, print steps, exit")
    ap.add_argument("--disable_co2_tracking", action="store_true",
                    help="Disable carbon footprint tracking even if codecarbon is installed")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    global _LOG_FH
    args = parse_args()
    t_start = time.monotonic()
    for name in ("only", "both", "homology", "stats", "taxonomy", "constraints", "unions", "lineages", "ic", "obo"):
        v = getattr(args, name)
        if v is not None:
            setattr(args, name, v.resolve())
    pairs = [(f"--{n}", getattr(args, n)) for n in
             ("only", "both", "homology", "stats", "taxonomy", "constraints", "unions", "lineages", "ic", "obo")
             if getattr(args, n) is not None]
    _validate_inputs(pairs)
    if not args.skip_clade_check and args.homology is None:
        print("ERROR: Module 2 needs --homology (or pass --skip_clade_check)", file=sys.stderr)
        sys.exit(1)

    run_dir  = Path(args.output)
    results  = run_dir / "results"
    workdir  = run_dir / "workdir"
    logs_dir = run_dir / "logs"
    for d in (results, workdir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name
    plot_formats = [f.strip().lstrip(".") for f in args.format.split(",")]

    log_path = _dated_log_path(logs_dir, "Run_DarkProteomeTaxonCheck")
    _LOG_FH  = open(log_path, "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  DarkProteomeTaxonCheck {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    _LOG_FH.write(f"User      : {getpass.getuser()}\n")
    _LOG_FH.write(f"Server    : {platform.node()}\n")
    _LOG_FH.write(f"OS        : {platform.system()} {platform.release()} ({platform.machine()})\n")
    _LOG_FH.write(f"Directory : {os.getcwd()}\n")
    _LOG_FH.write(f"Command   : {' '.join(sys.argv)}\n")
    _LOG_FH.write(f"{sep}\n\n")
    _LOG_FH.flush()

    _banner(f"DarkProteomeTaxonCheck {VERSION}")
    if args.force:
        _log("--force set: all steps will rerun regardless of existing outputs")
    elif results.exists() and any(results.iterdir()):
        _log("Existing results found — resuming from checkpoints (use --force to rerun)")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Dark matrix     : {args.only}")
        _log(f"  Both matrix     : {args.both}")
        _log(f"  Homology matrix : {args.homology}")
        _log(f"  Constraints     : {args.constraints}")
        _log(f"  Lineages        : {args.lineages}")
        _log(f"  Output          : {run_dir}/")
        _log("  Steps that would run:")
        if not args.skip_taxon_check:
            _log("    [1] Taxon-constraint violations  →  results/mod01_taxon_violations_*")
        if not args.skip_clade_check:
            _log("    [2] Clade-unsupported terms      →  results/mod02_clade_unsupported_*")
        _log("  Exiting (--dry_run).")
        sys.exit(0)

    _tracker = None
    if args.disable_co2_tracking:
        _log("  Carbon footprint tracking disabled (--disable_co2_tracking)")
    else:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="DarkProteomeTaxonCheck", log_level="warning")
            _tracker.start()
            _log("  codecarbon tracker started")
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped "
                 "(conda install -c conda-forge codecarbon)")

    # ---- inputs ------------------------------------------------------------
    _banner("Loading inputs")
    go_info = load_ic_table(args.ic)
    cons, taxon_names = load_constraints(args.constraints)
    unions = load_unions(args.unions)
    lineages = load_lineages(args.lineages)
    _log(f"  {sum(len(v) for v in cons.values())} asserted constraints on {len(cons)} GO terms, "
         f"{len(unions)} union taxa, {len(lineages)} species lineages")

    paths = {"dark": args.only, "both": args.both}
    if args.homology is not None:
        paths["homology"] = args.homology
    raw, species = {}, None
    for label, p in paths.items():
        _log(f"  Loading {label} matrix: {p.name}")
        sp_l, go_l, m = load_counts_matrix(p)
        _log(f"    {m.shape[0]} species x {m.shape[1]} GO")
        if species is None:
            species = sp_l
        elif sp_l != species:
            if set(sp_l) != set(species):
                print(f"ERROR: {label} matrix species differ from the dark matrix", file=sys.stderr)
                sys.exit(1)
            m = m[[sp_l.index(s) for s in species]]
        raw[label] = (go_l, m)
    union_cols = sorted(set().union(*(set(g) for g, _ in raw.values())))
    _log(f"  Union of GO columns: {len(union_cols)}")
    matrices = {}
    for label, (go_l, m) in raw.items():
        matrices[label] = align_to_columns(m, go_l, union_cols)
    del raw

    stats_df = pd.read_csv(args.stats, sep="\t").set_index("Species")
    meta = pd.DataFrame({"Species": species})
    meta["Group"] = stats_df.reindex(species)["Group"].to_numpy()
    if args.taxonomy is not None:
        tax = pd.read_csv(args.taxonomy, sep="\t").drop_duplicates("Species").set_index("Species")["Group"]
        tg = tax.reindex(species).fillna("Unclassified")
        small = tg.value_counts()
        small = small[small < args.min_group_species].index
        meta["tax_group"] = tg.where(~tg.isin(small), "other").to_numpy()
    else:
        meta["tax_group"] = meta["Group"]
    _log("  Groups: " + ", ".join(f"{g}={n}" for g, n in meta["tax_group"].value_counts().items()))

    obo = parse_obo(args.obo)
    eff = effective_constraints(union_cols, cons, obo)

    summary = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION,
               "inputs": {k: str(v) for k, v in paths.items()},
               "n_species": len(species), "n_go_union": len(union_cols),
               "n_go_with_inherited_constraint": sum(1 for e in eff if e),
               "parameters": {"min_group_species": args.min_group_species}}
    labels = list(matrices)

    if not args.skip_taxon_check:
        _banner("Module 1 — GO taxon-constraint violations")
        sp, terms_df, groups_df = run_taxon_check(matrices, union_cols, species, eff, lineages, unions,
                                                  taxon_names, go_info, meta, results, prefix, args.force)
        plot_violations(sp, labels, results / f"mod01_taxon_violations_{prefix}.pdf", plot_formats)
        plot_violations(sp, labels, results / f"mod01_taxon_violations_terms_{prefix}.pdf", plot_formats,
                        col_prefix="pct_viol_terms",
                        ylabel="% of distinct GO terms violating a GO taxon constraint",
                        title="GO taxon-constraint violations per species (distinct terms)")
        plot_top_violating_terms(terms_df, labels, results / f"mod01_top_violating_terms_{prefix}.pdf", plot_formats)
        for _, r in groups_df.iterrows():
            _log(f"  {r['group']:<14} n={int(r['n_species']):4d}  median % violating annotations: "
                 + "  ".join(f"{l} {r[f'median_pct_viol_annotations_{l}']:.2f}" for l in labels)
                 + (f"  | dark>both in {100 * r['frac_species_dark_higher']:.0f}% species"
                    if "frac_species_dark_higher" in r else ""))
        summary["module1_groups"] = groups_df.to_dict(orient="records")

    if not args.skip_clade_check:
        _banner("Module 2 — clade-unsupported terms (never assigned by homology in the clade)")
        sp2, terms2 = run_clade_check(matrices, union_cols, species, meta, go_info, results, prefix,
                                      args.min_group_species, args.force)
        sp2["has_lineage"] = True
        plot_violations(sp2[sp2["pct_clade_unsupported_annotations_dark"].notna()], ["dark", "both"],
                        results / f"mod02_clade_unsupported_{prefix}.pdf", plot_formats,
                        col_prefix="pct_clade_unsupported_annotations",
                        ylabel="% of GO annotations on terms homology never assigns in the clade",
                        title="Clade-unsupported annotations per species")
        summary["module2_median_pct_dark"] = float(sp2["pct_clade_unsupported_annotations_dark"].median())
        summary["module2_median_pct_both"] = float(sp2["pct_clade_unsupported_annotations_both"].median())

    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass
    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = (ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024)
    summary["resource_usage"] = {"wall_clock_s": round(elapsed_s, 1), "peak_mem_mb": round(peak_mem_mb, 1),
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
