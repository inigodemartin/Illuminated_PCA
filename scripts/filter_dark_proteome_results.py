#!/usr/bin/env python3
"""
Remove broken species from a dark_proteome_matrices.py results directory.

A species is "broken" when its FANTASIA and AHRD files were evidently not
produced from the same proteome, or when one of them is truncated. Flags,
from the mod02 statistics table:

  mismatch   N_fantasia_ids_not_in_proteome / N_fantasia > --max_mismatch_pct
             (FANTASIA protein IDs absent from the AHRD table: different
             proteome versions; 1-4 stray IDs out of tens of thousands are
             tolerated)
  tiny_ahrd  N_proteome  < --min_proteins   (AHRD table truncated/empty)
  tiny_fan   N_fantasia  < --min_proteins   (FANTASIA output truncated)

Writes, next to the originals (never overwritten), a *_clean.tsv version of
the three mod01 count matrices, the mod02 stats and the mod03 taxons, with
the flagged species dropped and GO columns that become all-zero removed, plus
an excluded_species_*.tsv listing every removed species with its reasons.
"""

VERSION = "v0.1.0"

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def filter_matrix(src: Path, dst: Path, bad: set):
    """Stream a wide Species x GO TSV twice (no pandas: 2.7k x 30k wide
    tables exhaust memory when parsed as a DataFrame). Pass 1 finds the GO
    columns that stay non-zero once the bad species are dropped; pass 2
    writes the kept rows restricted to those columns."""
    with open(src) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        n_go = len(header) - 1
        keep_col = np.zeros(n_go, dtype=bool)
        n_in = n_out = 0
        for line in fh:
            n_in += 1
            sp, rest = line.rstrip("\n").split("\t", 1)
            if sp in bad:
                continue
            n_out += 1
            keep_col |= np.fromstring(rest, dtype=np.int64, sep="\t") != 0
    idx = np.flatnonzero(keep_col)
    with open(src) as fh, open(dst, "w") as out:
        fh.readline()
        out.write("\t".join([header[0]] + [header[i + 1] for i in idx]) + "\n")
        for line in fh:
            sp, rest = line.rstrip("\n").split("\t", 1)
            if sp in bad:
                continue
            vals = np.fromstring(rest, dtype=np.int64, sep="\t")[idx]
            out.write(sp + "\t" + "\t".join(map(str, vals.tolist())) + "\n")
    return n_in, n_out, n_go, len(idx)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, required=True,
                    help="results/ directory of a dark_proteome_matrices.py run")
    ap.add_argument("--max_mismatch_pct", type=float, default=1.0,
                    help="Flag species with more than this %% of FANTASIA IDs absent from AHRD (default: 1.0)")
    ap.add_argument("--min_proteins", type=int, default=500,
                    help="Flag species with fewer AHRD rows or FANTASIA proteins than this (default: 500)")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    args = parse_args()
    res = args.results.resolve()
    stats_files = sorted(res.glob("mod02_dark_proteome_stats_*.tsv"))
    stats_files = [p for p in stats_files if not p.stem.endswith("_clean")]
    if len(stats_files) != 1:
        print(f"ERROR: expected one mod02_dark_proteome_stats_*.tsv in {res}, found {len(stats_files)}", file=sys.stderr)
        sys.exit(1)
    stats_path = stats_files[0]
    prefix = stats_path.stem[len("mod02_dark_proteome_stats_"):]

    stats = pd.read_csv(stats_path, sep="\t")
    mism_pct = 100.0 * stats["N_fantasia_ids_not_in_proteome"] / stats["N_fantasia"].replace(0, pd.NA)
    reasons = pd.DataFrame({
        "mismatch": mism_pct.fillna(0) > args.max_mismatch_pct,
        "tiny_ahrd": stats["N_proteome"] < args.min_proteins,
        "tiny_fan": stats["N_fantasia"] < args.min_proteins,
    })
    flagged = reasons.any(axis=1)
    excluded = stats.loc[flagged, ["Species", "Group", "N_proteome", "N_fantasia",
                                   "N_fantasia_ids_not_in_proteome", "Pct_only_fantasia_of_proteome"]].copy()
    excluded["Pct_ids_mismatch"] = mism_pct[flagged].fillna(0).round(2).values
    excluded["Reasons"] = reasons[flagged].apply(lambda r: ";".join(c for c in reasons.columns if r[c]), axis=1).values
    excl_path = res / f"excluded_species_{prefix}.tsv"
    excluded.to_csv(excl_path, sep="\t", index=False, float_format="%.2f")
    bad = set(excluded["Species"])
    print(f"{len(bad)} species flagged ({reasons[flagged].sum().to_dict()}) -> {excl_path.name}", file=sys.stderr)
    for _, r in excluded.iterrows():
        print(f"  {r['Species']:32s} {r['Group']:11s} {r['Reasons']}", file=sys.stderr)

    stats[~flagged].to_csv(res / f"mod02_dark_proteome_stats_{prefix}_clean.tsv", sep="\t", index=False, float_format="%.4f")
    print(f"mod02 stats: {len(stats)} -> {(~flagged).sum()} species", file=sys.stderr)

    tax_path = res / f"mod03_taxons_{prefix}.tsv"
    if tax_path.exists():
        tax = pd.read_csv(tax_path, sep="\t")
        tax[~tax["Species"].isin(bad)].to_csv(res / f"mod03_taxons_{prefix}_clean.tsv", sep="\t", index=False)

    for key in ("fantasia_only", "fantasia_both", "homology_all"):
        p = res / f"mod01_{key}_counts_{prefix}.tsv"
        if not p.exists():
            print(f"  [WARN] {p.name} not found, skipping", file=sys.stderr)
            continue
        n_in, n_out, n_go, n_go_out = filter_matrix(p, res / f"mod01_{key}_counts_{prefix}_clean.tsv", bad)
        print(f"mod01 {key}: {n_in}x{n_go} -> {n_out}x{n_go_out}", file=sys.stderr)

if __name__ == "__main__":
    main()
