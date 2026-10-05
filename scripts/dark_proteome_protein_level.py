#!/usr/bin/env python3
"""
dark_proteome_protein_level.py — Protein by protein: which proteins carry GO
terms incompatible with the organism, and how does that relate to length?

Runs on the server (same inputs as dark_proteome_matrices.py: the status
TSVs with FANTASIA / AHRD paths, proteome .pep files found next to them) or
on one species given explicitly (for local tests). For every protein it
records its group (dark = FANTASIA GO but no homology GO; both; only_homology;
unannotated), its length, the number of FANTASIA GO terms and how many of
them are incompatible with a GO taxon constraint for that species' NCBI
lineage (only_in / never_in, inherited through is_a + part_of; see
dark_proteome_taxon_check.py), and the same for the homology (AHRD) GO terms
when present.

Outputs
  workdir/per_species/{species}.proteins.tsv.gz   one row per protein (kept for inspection)
  workdir/per_species/{species}.json              per-species checkpoint
  results/mod01_protein_level_species_*.tsv       per species: % proteins with >= 1 incompatible
                                                  term (dark / both / homology), median lengths,
                                                  Spearman(length, share of incompatible terms)
  results/mod02_protein_level_length_bins_*.tsv   pooled over species and per Group: by length
                                                  bin, % proteins with >= 1 incompatible term
  results/mod02_protein_level_length_bins_*.pdf   the plot of that table
"""

VERSION = "v0.1.0"

import argparse
import getpass
import gzip
import json
import os
import platform
import resource
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from dark_proteome_matrices import (parse_fantasia, parse_ahrd, parse_fasta_lengths,
                                    find_proteomes, load_manifests, DEFAULT_PROTEOME_GLOB)
from dark_proteome_enrichment import parse_obo, ancestors_closure
from dark_proteome_taxon_check import (load_constraints, load_unions, load_lineages, taxon_members,
                                       DEFAULT_CONSTRAINTS_PATH, DEFAULT_UNIONS_PATH,
                                       DEFAULT_LINEAGES_PATH, DEFAULT_OBO_PATH)

COL_DARK, COL_BOTH, COL_GREY, COL_AMBER = "#E8604C", "#4C9BE8", "#888888", "#F5A623"
LENGTH_BINS = [0, 100, 200, 300, 500, 1000, 10 ** 9]
LENGTH_LABELS = ["<100", "100-199", "200-299", "300-499", "500-999", ">=1000"]
GROUPS = ("dark", "both", "only_homology", "unannotated")

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


# ---------------------------------------------------------------------------
# Constraint evaluation
# ---------------------------------------------------------------------------
class ConstraintChecker:
    """Inherited taxon constraints per GO (cached) and, per species lineage,
    which GO terms are incompatible (cached per GO within the species)."""

    def __init__(self, cons: dict, unions: dict, obo):
        self.cons = cons
        self.unions = unions
        self.name, _, self.parents, _, _, self.alt = obo
        self.anc = ancestors_closure(self.parents)
        self.constrained = set(cons)
        self._eff = {}

    def effective(self, go: str) -> frozenset:
        if go in self._eff:
            return self._eff[go]
        g2 = self.alt.get(go, go)
        eff = set()
        for h in ({g2} | self.anc(g2)) & self.constrained:
            eff |= self.cons[h]
        self._eff[go] = frozenset(eff)
        return self._eff[go]

    def species_checker(self, lineage: set):
        cache = {}
        pair_cache = {}

        def pair_violated(pair):
            if pair not in pair_cache:
                kind, taxon = pair
                inside = bool(lineage & taxon_members(taxon, self.unions))
                pair_cache[pair] = (not inside) if kind == "only_in" else inside
            return pair_cache[pair]

        def incompatible(go: str) -> bool:
            if go not in cache:
                cache[go] = any(pair_violated(p) for p in self.effective(go))
            return cache[go]
        return incompatible


# ---------------------------------------------------------------------------
# Per-species work
# ---------------------------------------------------------------------------
_W = {}


def _init_worker(checker, lineages, proteome_glob, per_species_dir, force):
    _W.update(checker=checker, lineages=lineages, proteome_glob=proteome_glob,
              per_species_dir=per_species_dir, force=force)


def _bin_index(length):
    if length is None:
        return None
    for i in range(len(LENGTH_BINS) - 1):
        if LENGTH_BINS[i] <= length < LENGTH_BINS[i + 1]:
            return i
    return len(LENGTH_BINS) - 2


def process_species(entry: dict) -> dict:
    species = entry["species"]
    out_json = _W["per_species_dir"] / f"{species}.json"
    out_prot = _W["per_species_dir"] / f"{species}.proteins.tsv.gz"
    if not _W["force"] and out_json.exists() and out_json.stat().st_size > 0:
        return {"species": species, "status": "checkpoint"}
    lineage = _W["lineages"].get(species)
    if lineage is None:
        return {"species": species, "status": "no_lineage"}
    try:
        fan = parse_fantasia(Path(entry["fantasia_file"]))
        proteome, hom = parse_ahrd(Path(entry["homology_file"]))
        lengths = {}
        if entry.get("pep_files"):
            for pf in entry["pep_files"]:
                parse_fasta_lengths(Path(pf), lengths)
        else:
            for pf in find_proteomes(Path(entry["fantasia_file"]), _W["proteome_glob"]):
                parse_fasta_lengths(pf, lengths)
        incompatible = _W["checker"].species_checker(lineage)

        # per-protein rows + accumulators
        bins = {g: np.zeros((len(LENGTH_LABELS), 4), dtype=np.int64) for g in ("dark", "both", "homology")}
        per_group = {g: {"n": 0, "n_incompat": 0, "len_incompat": [], "len_clean": [],
                         "len_all": [], "frac_incompat": []} for g in ("dark", "both", "homology")}
        n_group = Counter()
        with gzip.open(out_prot, "wt") as fh:
            fh.write("species\tprotein\tgroup\tlength\tn_go_fantasia\tn_incompat_fantasia\t"
                     "n_go_homology\tn_incompat_homology\tincompat_fantasia_terms\tincompat_homology_terms\n")
            for pid in proteome:
                in_fan, in_hom = pid in fan, pid in hom
                group = ("both" if in_fan and in_hom else "dark" if in_fan
                         else "only_homology" if in_hom else "unannotated")
                n_group[group] += 1
                L = lengths.get(pid)
                fan_gos = fan.get(pid, [])
                hom_gos = hom.get(pid, [])
                bad_f = [g for g in fan_gos if incompatible(g)]
                bad_h = [g for g in hom_gos if incompatible(g)]
                fh.write(f"{species}\t{pid}\t{group}\t{'' if L is None else L}\t{len(fan_gos)}\t{len(bad_f)}\t"
                         f"{len(hom_gos)}\t{len(bad_h)}\t{';'.join(bad_f)}\t{';'.join(bad_h)}\n")
                targets = []
                if group in ("dark", "both"):
                    targets.append((group, fan_gos, bad_f))
                if in_hom:
                    targets.append(("homology", hom_gos, bad_h))
                for g, gos, bad in targets:
                    acc = per_group[g]
                    acc["n"] += 1
                    has = len(bad) > 0
                    acc["n_incompat"] += has
                    if L is not None:
                        acc["len_all"].append(L)
                        (acc["len_incompat"] if has else acc["len_clean"]).append(L)
                        acc["frac_incompat"].append(len(bad) / len(gos))
                        b = _bin_index(L)
                        bins[g][b, 0] += 1
                        bins[g][b, 1] += has
                        bins[g][b, 2] += len(gos)
                        bins[g][b, 3] += len(bad)
        row = {"Species": species, "Group": entry["group"], "N_proteome": len(proteome),
               "N_with_length": sum(1 for p in proteome if p in lengths)}
        for g in GROUPS:
            row[f"N_{g}"] = n_group[g]
        for g in ("dark", "both", "homology"):
            acc = per_group[g]
            row[f"N_proteins_{g}"] = acc["n"]
            row[f"N_with_incompat_{g}"] = acc["n_incompat"]
            row[f"Pct_with_incompat_{g}"] = 100 * acc["n_incompat"] / acc["n"] if acc["n"] else np.nan
            row[f"Median_length_incompat_{g}"] = float(np.median(acc["len_incompat"])) if acc["len_incompat"] else np.nan
            row[f"Median_length_clean_{g}"] = float(np.median(acc["len_clean"])) if acc["len_clean"] else np.nan
            if len(acc["len_all"]) >= 10:
                rho, p = spearmanr(acc["len_all"], acc["frac_incompat"])
                row[f"Spearman_length_vs_frac_incompat_{g}"] = rho
            else:
                row[f"Spearman_length_vs_frac_incompat_{g}"] = np.nan
        with open(out_json, "w") as fh:
            json.dump({"row": row, "bins": {g: b.tolist() for g, b in bins.items()}}, fh)
        return {"species": species, "status": "ok"}
    except Exception as exc:   # one broken species must not kill the run
        if out_prot.exists():
            out_prot.unlink()
        return {"species": species, "status": f"error: {type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
def plot_bins(table: pd.DataFrame, out_path: Path, plot_formats: list) -> None:
    matplotlib.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                                "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white"})
    strata = ["all"] + sorted(s for s in table["stratum"].unique() if s != "all")
    metrics = [("pct_proteins_with_incompat", "% proteins with >= 1 incompatible GO term"),
               ("pct_incompat_annotations", "% of the proteins' GO annotations that are incompatible")]
    fig, axes = plt.subplots(2, len(strata), figsize=(5.2 * len(strata), 8.6), sharey="row", squeeze=False)
    colors = {"dark": COL_DARK, "both": COL_BOTH, "homology": COL_GREY}
    labels = {"dark": "dark proteome (FANTASIA)", "both": "both (FANTASIA)", "homology": "homology (AHRD)"}
    x = np.arange(len(LENGTH_LABELS))
    w = 0.27
    for row, (metric, ylabel) in enumerate(metrics):
        for ax, s in zip(axes[row], strata):
            sub = table[table["stratum"] == s]
            for k, g in enumerate(("dark", "both", "homology")):
                ss = sub[sub["set"] == g].set_index("length_bin").reindex(LENGTH_LABELS)
                vals = ss[metric].fillna(0).to_numpy()
                ax.bar(x + (k - 1) * w, vals, w, color=colors[g], alpha=0.85, label=labels[g])
                for xi, v, n in zip(x + (k - 1) * w, vals, ss["n_proteins"].fillna(0)):
                    if n:
                        ax.text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=6.5, color="#333333")
            ax.set_xticks(x)
            ax.set_xticklabels(LENGTH_LABELS)
            ax.grid(axis="y", lw=0.4, alpha=0.5)
            if row == 0:
                n_sp = int(sub["n_species"].max()) if len(sub) else 0
                ax.set_title(f"{s} (n = {n_sp} species)", fontsize=11)
            else:
                ax.set_xlabel("protein length (aa)")
        axes[row][0].set_ylabel(ylabel)
    axes[0][0].legend(fontsize=8, frameon=False)
    fig.suptitle("Taxon-incompatible annotations by protein length (pooled over species)", fontsize=12)
    fig.tight_layout()
    for fmt in plot_formats:
        fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Protein-level taxon-constraint incompatibilities vs length.")
    ap.add_argument("--manifest", type=Path, nargs="+", default=None,
                    help="Status TSV(s) as for dark_proteome_matrices.py (server mode)")
    ap.add_argument("--species", default=None, help="Single-species mode: species directory name")
    ap.add_argument("--fantasia", type=Path, default=None, help="Single-species mode: *_GOs_merged.tsv")
    ap.add_argument("--ahrd", type=Path, default=None, help="Single-species mode: *.funct_ahrd.tsv")
    ap.add_argument("--pep", type=Path, nargs="+", default=None, help="Single-species mode: proteome FASTA(s)")
    ap.add_argument("--group", default="unknown", help="Single-species mode: Group label (default: unknown)")
    ap.add_argument("--constraints", type=Path, default=DEFAULT_CONSTRAINTS_PATH)
    ap.add_argument("--unions", type=Path, default=DEFAULT_UNIONS_PATH)
    ap.add_argument("--lineages", type=Path, default=DEFAULT_LINEAGES_PATH)
    ap.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    ap.add_argument("--proteome_glob", default=DEFAULT_PROTEOME_GLOB)
    ap.add_argument("--output", required=True, help="Output directory")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--format", default="pdf", help="Plot format(s), comma-separated (default: pdf)")
    ap.add_argument("--force", action="store_true", help="Recompute every species even if its checkpoint exists")
    ap.add_argument("--dry_run", action="store_true", help="Validate inputs, print steps, exit")
    ap.add_argument("--disable_co2_tracking", action="store_true")
    ap.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    return ap.parse_args()


def main():
    global _LOG_FH
    args = parse_args()
    t_start = time.monotonic()
    single = args.species is not None
    if single and not (args.fantasia and args.ahrd and args.pep):
        print("ERROR: single-species mode needs --species --fantasia --ahrd --pep", file=sys.stderr)
        sys.exit(1)
    if not single and not args.manifest:
        print("ERROR: give --manifest (server mode) or --species/--fantasia/--ahrd/--pep", file=sys.stderr)
        sys.exit(1)
    pairs = [("--constraints", args.constraints), ("--unions", args.unions),
             ("--lineages", args.lineages), ("--obo", args.obo)]
    if single:
        pairs += [("--fantasia", args.fantasia), ("--ahrd", args.ahrd)] + [("--pep", p) for p in args.pep]
    else:
        pairs += [("--manifest", m) for m in args.manifest]
    _validate_inputs(pairs)

    run_dir = Path(args.output)
    results, workdir, logs_dir = run_dir / "results", run_dir / "workdir", run_dir / "logs"
    per_species_dir = workdir / "per_species"
    for d in (results, per_species_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    prefix = run_dir.name
    plot_formats = [f.strip().lstrip(".") for f in args.format.split(",")]

    _LOG_FH = open(_dated_log_path(logs_dir, "Run_DarkProteomeProteinLevel"), "w")
    sep = "=" * 62
    _LOG_FH.write(f"{sep}\n  DarkProteomeProteinLevel {VERSION}  —  Run Log\n{sep}\n")
    _LOG_FH.write(f"Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nUser      : {getpass.getuser()}\n"
                  f"Server    : {platform.node()}\nOS        : {platform.system()} {platform.release()} "
                  f"({platform.machine()})\nDirectory : {os.getcwd()}\nCommand   : {' '.join(sys.argv)}\n{sep}\n\n")
    _LOG_FH.flush()
    _banner(f"DarkProteomeProteinLevel {VERSION}")
    if args.force:
        _log("--force set: all species will be recomputed")
    elif any(per_species_dir.iterdir()):
        _log("Existing per-species checkpoints found — resuming (use --force to recompute)")

    if single:
        entries = [{"species": args.species, "group": args.group, "fantasia_file": str(args.fantasia.resolve()),
                    "homology_file": str(args.ahrd.resolve()), "pep_files": [str(p.resolve()) for p in args.pep]}]
    else:
        entries = load_manifests(args.manifest)
    _log(f"  {len(entries)} species to process")

    if args.dry_run:
        _banner("Dry run — no steps will be executed")
        _log(f"  Species     : {len(entries)}")
        _log(f"  Constraints : {args.constraints}")
        _log(f"  Lineages    : {args.lineages}")
        _log(f"  Output      : {run_dir}/")
        _log("  Steps: per-protein tables → per-species summary → length-bin table + plot")
        sys.exit(0)

    _tracker = None
    if not args.disable_co2_tracking:
        try:
            from codecarbon import EmissionsTracker
            _tracker = EmissionsTracker(output_dir=str(logs_dir), output_file=f"{prefix}.emissions.csv",
                                        project_name="DarkProteomeProteinLevel", log_level="warning")
            _tracker.start()
        except ImportError:
            _log("  codecarbon not installed — carbon tracking skipped")

    _banner("Loading constraints, unions, lineages, ontology")
    cons, _ = load_constraints(args.constraints)
    unions = load_unions(args.unions)
    lineages = load_lineages(args.lineages)
    obo = parse_obo(args.obo)
    checker = ConstraintChecker(cons, unions, obo)
    missing = [e["species"] for e in entries if e["species"] not in lineages]
    _log(f"  {len(cons)} constrained GO terms, {len(lineages)} lineages; "
         f"{len(missing)} species without lineage will be skipped" + (f": {missing[:5]}" if missing else ""))

    _banner("Per-species processing")
    status = Counter()
    errors = []
    with Pool(args.threads, initializer=_init_worker,
              initargs=(checker, lineages, args.proteome_glob, per_species_dir, args.force)) as pool:
        for i, res in enumerate(pool.imap_unordered(process_species, entries, chunksize=1), 1):
            st = res["status"].split(":")[0]
            status[st] += 1
            if st == "error":
                errors.append(res)
                _log(f"  [{i}/{len(entries)}] {res['species']}: {res['status']}")
            elif i % 100 == 0 or i == len(entries):
                _log(f"  [{i}/{len(entries)}] {dict(status)}")
    _log("  " + ", ".join(f"{k}: {v}" for k, v in status.items()))

    _banner("Aggregating")
    rows, bins = [], defaultdict(lambda: np.zeros((len(LENGTH_LABELS), 4), dtype=np.int64))
    n_species_stratum = Counter()
    for e in entries:
        p = per_species_dir / f"{e['species']}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        rows.append(d["row"])
        for stratum in ("all", d["row"]["Group"]):
            n_species_stratum[stratum] += 1
            for g, b in d["bins"].items():
                bins[(stratum, g)] += np.array(b)
    sp = pd.DataFrame(rows)
    sp.to_csv(results / f"mod01_protein_level_species_{prefix}.tsv", sep="\t", index=False, float_format="%.4g")
    brows = []
    for (stratum, g), b in bins.items():
        for i, lab in enumerate(LENGTH_LABELS):
            brows.append({"stratum": stratum, "set": g, "length_bin": lab, "n_species": n_species_stratum[stratum],
                          "n_proteins": int(b[i, 0]), "n_with_incompat": int(b[i, 1]),
                          "pct_proteins_with_incompat": 100 * b[i, 1] / b[i, 0] if b[i, 0] else np.nan,
                          "n_annotations": int(b[i, 2]), "n_incompat_annotations": int(b[i, 3]),
                          "go_per_protein": b[i, 2] / b[i, 0] if b[i, 0] else np.nan,
                          "pct_incompat_annotations": 100 * b[i, 3] / b[i, 2] if b[i, 2] else np.nan})
    bt = pd.DataFrame(brows)
    bt.to_csv(results / f"mod02_protein_level_length_bins_{prefix}.tsv", sep="\t", index=False, float_format="%.4g")
    plot_bins(bt, results / f"mod02_protein_level_length_bins_{prefix}.pdf", plot_formats)
    for g in ("dark", "both", "homology"):
        sub = bt[(bt["stratum"] == "all") & (bt["set"] == g)]
        _log(f"  {g:<9} by length — % proteins with >= 1 incompatible term / % incompatible annotations / GO per protein: " +
             "  ".join(f"{r['length_bin']} {r['pct_proteins_with_incompat']:.1f}/{r['pct_incompat_annotations']:.1f}/{r['go_per_protein']:.1f}"
                       for _, r in sub.iterrows()))
    if len(sp):
        for g in ("dark", "both", "homology"):
            _log(f"  {g:<9} median over species: {sp[f'Pct_with_incompat_{g}'].median():.2f} % proteins with >= 1 "
                 f"incompatible term; median length incompat / clean "
                 f"{sp[f'Median_length_incompat_{g}'].median():.0f} / {sp[f'Median_length_clean_{g}'].median():.0f} aa; "
                 f"median Spearman(length, share incompat) {sp[f'Spearman_length_vs_frac_incompat_{g}'].median():+.2f}")

    emissions_kg = None
    if _tracker is not None:
        try:
            emissions_kg = _tracker.stop()
        except Exception:
            pass
    elapsed_s = time.monotonic() - t_start
    ru = resource.getrusage(resource.RUSAGE_SELF)
    peak_mem_mb = ru.ru_maxrss / (1024 * 1024) if platform.system() == "Darwin" else ru.ru_maxrss / 1024
    summary = {"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "version": VERSION,
               "n_species_requested": len(entries), "n_species_done": len(sp), "status": dict(status),
               "errors": errors[:50], "parameters": {"threads": args.threads, "proteome_glob": args.proteome_glob},
               "resource_usage": {"wall_clock_s": round(elapsed_s, 1), "peak_mem_mb": round(peak_mem_mb, 1),
                                  "emissions_kg_CO2eq": emissions_kg}}
    with open(results / f"{prefix}.run_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: None if pd.isna(o) else float(o))
        fh.write("\n")
    _banner("Done")
    _log(f"  Wall clock: {elapsed_s:.1f} s, peak RSS: {peak_mem_mb:.0f} MB — results in {results}/")
    _LOG_FH.close()


if __name__ == "__main__":
    main()
