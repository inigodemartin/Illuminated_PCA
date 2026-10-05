#!/usr/bin/env python3
"""
build_go_taxon_constraints.py — Extract the GO taxon constraints and the NCBI
lineages our species need into three small TSVs (committed in data/), so
that dark_proteome_taxon_check.py does not depend on the 35 MB go-edit.obo
or the 230 MB NCBI taxdump.

Sources (download manually, URLs below):
  go-edit.obo         https://raw.githubusercontent.com/geneontology/go-ontology/master/src/ontology/go-edit.obo
                      Constraints are encoded as `is_a: onlyin:<taxid>` /
                      `is_a: neverin:<taxid>` lines inside each [Term]
                      (idspaces onlyin / neverin). They are NOT in go.obo or
                      go-basic.obo, only in go-edit.obo and go-plus.owl.
  go-taxon-groupings.obo
                      https://raw.githubusercontent.com/geneontology/go-ontology/master/src/ontology/imports/go-taxon-groupings.obo
                      Defines NCBITaxon_Union:* terms (e.g. "Fungi or Bacteria")
                      through `union_of: NCBITaxon:<taxid>` lines.
  taxdump             https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz
                      (nodes.dmp, names.dmp, merged.dmp).

Outputs:
  data/go_taxon_constraints.tsv   GO, constraint (only_in|never_in), taxon id, taxon name
  data/go_taxon_unions.tsv        union id, name, member NCBI taxids (comma-separated)
  data/species_lineage_taxids.tsv Species, TaxID, full ancestor taxid path (comma-separated,
                                  species first, root `1` excluded) for every species of
                                  data/species_lineage.tsv that has a TaxID
"""

VERSION = "v0.2.0"

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent.parent


def parse_constraints(go_edit: Path):
    rows, cur = [], None
    with open(go_edit) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                cur = None
            elif line.startswith("[") and line.endswith("]"):
                cur = None
            elif line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif cur and line.startswith("is_a: onlyin:"):
                rows.append((cur, "only_in", line.split("onlyin:", 1)[1].split()[0]))
            elif cur and line.startswith("is_a: neverin:"):
                rows.append((cur, "never_in", line.split("neverin:", 1)[1].split()[0]))
    return rows


def parse_unions(groupings: Path):
    unions, cur, name = {}, None, {}
    with open(groupings) as fh:
        for raw in fh:
            line = raw.strip()
            if line == "[Term]":
                cur = None
            elif line.startswith("id: NCBITaxon_Union:"):
                cur = line[4:].strip()
                unions[cur] = set()
            elif line.startswith("id: "):
                cur = None
            elif cur and line.startswith("name:"):
                name[cur] = line[5:].strip()
            elif cur and line.startswith("union_of: NCBITaxon:"):
                unions[cur].add(int(line.split("NCBITaxon:", 1)[1].split()[0]))
    return unions, name


def load_taxdump(taxdump_dir: Path):
    ids, pars = [], []
    with open(taxdump_dir / "nodes.dmp") as fh:
        for line in fh:
            a, b = line.split("\t|\t", 2)[:2]
            ids.append(int(a))
            pars.append(int(b))
    ids, pars = np.array(ids), np.array(pars)
    parent = np.zeros(ids.max() + 1, dtype=np.int64)
    parent[ids] = pars
    merged = {}
    with open(taxdump_dir / "merged.dmp") as fh:
        for line in fh:
            a, b = line.split("\t|\t")[:2]
            merged[int(a)] = int(b.split("\t")[0])
    sci = {}
    with open(taxdump_dir / "names.dmp") as fh:
        for line in fh:
            p = line.split("\t|\t")
            if len(p) >= 4 and p[3].startswith("scientific name"):
                sci[int(p[0])] = p[1]
    return parent, merged, sci


def lineage(taxid: int, parent, merged):
    t = int(taxid)
    if t >= len(parent) or parent[t] == 0:
        t = merged.get(t, t)
    out = []
    while 0 < t < len(parent) and t != 1 and len(out) < 200:
        out.append(t)
        t = int(parent[t])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--go_edit", type=Path, required=True, help="go-edit.obo")
    ap.add_argument("--groupings", type=Path, required=True, help="go-taxon-groupings.obo")
    ap.add_argument("--taxdump_dir", type=Path, required=True,
                    help="Directory with extracted nodes.dmp, names.dmp, merged.dmp")
    ap.add_argument("--species_lineage", type=Path, default=HERE / "data" / "species_lineage.tsv",
                    help="Species/TaxID table (default: data/species_lineage.tsv)")
    ap.add_argument("--group_fallback", action="append", default=[],
                    metavar="GROUP=TAXID",
                    help="Species of this Group without a TaxID get this group-level taxid "
                         "(e.g. Asgard=1935183, Promethearchaeati = Asgard archaea); repeatable")
    ap.add_argument("--outdir", type=Path, default=HERE / "data")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = ap.parse_args()

    rows = parse_constraints(args.go_edit)
    unions, uname = parse_unions(args.groupings)
    print(f"{len(rows)} asserted constraints on {len({r[0] for r in rows})} GO terms; "
          f"{len(unions)} union taxa", file=sys.stderr)

    parent, merged, sci = load_taxdump(args.taxdump_dir)
    print(f"taxdump: {len(sci)} scientific names", file=sys.stderr)

    def taxon_id(tax: str) -> str:
        # go-edit writes `onlyin:Union_0000004`; groupings call it NCBITaxon_Union:0000004
        return "NCBITaxon_Union:" + tax[6:] if tax.startswith("Union_") else "NCBITaxon:" + tax

    def taxon_label(tax: str) -> str:
        if tax.startswith("Union_"):
            return uname.get(taxon_id(tax), tax)
        return sci.get(int(tax), "")

    stamp = datetime.now().strftime("%Y-%m-%d")
    out = args.outdir / "go_taxon_constraints.tsv"
    with open(out, "w") as fh:
        fh.write(f"# built {stamp} by build_go_taxon_constraints.py from go-edit.obo "
                 f"(asserted only_in / never_in taxon constraints; propagate to descendants yourself)\n")
        fh.write("GO\tconstraint\ttaxon\ttaxon_name\n")
        for go, kind, tax in sorted(set(rows)):
            fh.write(f"{go}\t{kind}\t{taxon_id(tax)}\t{taxon_label(tax)}\n")
    print(f"written {out}", file=sys.stderr)

    out = args.outdir / "go_taxon_unions.tsv"
    with open(out, "w") as fh:
        fh.write(f"# built {stamp} from go-taxon-groupings.obo\n")
        fh.write("union\tname\tmember_taxids\n")
        for u in sorted(unions):
            fh.write(f"{u}\t{uname.get(u, '')}\t{','.join(str(t) for t in sorted(unions[u]))}\n")
    print(f"written {out}", file=sys.stderr)

    fallback = {}
    for item in args.group_fallback:
        g, t = item.split("=", 1)
        fallback[g] = int(t)
    sp = pd.read_csv(args.species_lineage, sep="\t").drop_duplicates("Species")
    has_tid = sp["TaxID"].notna()
    fb_mask = ~has_tid & sp["Group"].isin(fallback)
    sp = sp[has_tid | fb_mask].copy()
    sp["TaxID"] = np.where(sp["TaxID"].notna(), sp["TaxID"],
                           sp["Group"].map(fallback)).astype(float)
    out = args.outdir / "species_lineage_taxids.tsv"
    n_missing = 0
    with open(out, "w") as fh:
        fh.write(f"# built {stamp} from NCBI taxdump; lineage = species taxid first, root 1 excluded\n")
        if fallback:
            fh.write("# group-level fallback taxid for species without a TaxID: "
                     + ", ".join(f"{g}={t} ({sci.get(t, '?')}, {int(((sp['Group'] == g) & fb_mask).sum())} species)"
                                 for g, t in fallback.items()) + "\n")
        fh.write("Species\tTaxID\tlineage_taxids\n")
        for _, r in sp.iterrows():
            lin = lineage(int(r["TaxID"]), parent, merged)
            if not lin:
                n_missing += 1
            fh.write(f"{r['Species']}\t{int(r['TaxID'])}\t{','.join(map(str, lin))}\n")
    print(f"written {out}: {len(sp)} species ({int(fb_mask.sum())} via group fallback), "
          f"{n_missing} without lineage", file=sys.stderr)


if __name__ == "__main__":
    main()
