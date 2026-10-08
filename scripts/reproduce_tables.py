#!/usr/bin/env python3
"""Recompute the tables of the paper from the released run files.

No API key and no network access are needed. The script reads
  results/runs/<dataset>/*.jsonl(.gz)   one file per system run (answers + full agent traces)
  results/review/review_pooled.csv      pooled, blind manual review (the corrected labels)
and writes one CSV per table to results/tables/.

Run names:
  <config>              first run with the main model (Gemini), e.g. full_no_memory
  <config>@r2, @r3      repeated runs with the main model
  <config>@<model>      runs with another model, e.g. full_no_memory@qwen3-4b-instruct-2507-q4_K_M

Tables (numbers as in the paper):
  table2_first_runs_original / _corrected   every first run against the LLM-only run (Table 2)
  table3_within_run_corrected / _no_dates   repairs and breaks inside each verification run (Table 3)
  table3_within_run_original                the same against the original labels
  table3_llm_only_spread                    first answers of every main-model run = independent LLM-only runs
  table4_checks_<run>                       what each check did (Table 4)
  table5_label_review                       labels the review could not accept, with dates apart (Table 5)
  table6_popularity                         first runs by popularity bucket (Table 6)
  nil_check_vs_rule                         the agent's NIL check against the fixed NIL rule
  by_script                                 first runs by script of the mention
  gold_vs_agent_vs_search                   do the original labels side with search? (original labels)

Usage:
  python scripts/reproduce_tables.py
  python scripts/reproduce_tables.py --runs results/runs/elner --review results/review/review_pooled.csv
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import elagent as E  # noqa: E402

REPEAT_TAG = re.compile(r"^@r\d+$")
VERIFICATION = ("full", "grounded", "critic", "llm_plus_nil_rule")           # runs with an initial and a final answer
LLM_ONLY_FROM = ("llm_no_correction", "full_no_memory", "full_plus_role", "grounded_only", "critic_only")
CHECK_RUNS = ("full_no_memory", "grounded_only", "full_plus_role", "critic_only")
YEAR = re.compile(r"^\s*[0-9٠-٩]{4}\s*$")                       # Western or Arabic-Indic digits


def find_runs(run_dir):
    """{run name: path} for every run file in a folder (dev and memory-building runs are left out)."""
    runs = {}
    for f in sorted(os.listdir(run_dir)):
        for ext in (".jsonl.gz", ".jsonl"):
            if f.endswith(ext) and not f.startswith("dev_"):
                runs[f[: -len(ext)]] = os.path.join(run_dir, f)
                break
    return runs


def split_tag(name):
    """'full_no_memory@r2' -> ('full_no_memory', '@r2'); 'llm_no_correction' -> ('llm_no_correction', '')."""
    return (name.split("@", 1)[0], "@" + name.split("@", 1)[1]) if "@" in name else (name, "")


def main_model_runs(runs):
    """First runs and repeated runs of the main model (no tag, or a repeat tag such as @r2)."""
    return {k: v for k, v in runs.items() if split_tag(k)[1] == "" or REPEAT_TAG.match(split_tag(k)[1])}


def date_ids(records):
    """Ids of date mentions: ELNER-DZ type DATE/TIME, or a mention that is a four-digit year."""
    return {str(r["id"]) for r in records
            if str(r.get("type") or "").upper() in ("DATE", "TIME") or YEAR.match(str(r.get("mention") or ""))}


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def within_run(runs, acceptable=None, exclude=()):
    """One row per verification run: n, initial and final accuracy, repairs, breaks and the exact sign test."""
    rows = {}
    for name, path in runs.items():
        if not split_tag(name)[0].startswith(VERIFICATION):
            continue
        t = _quiet(E.paired_table, {name: path}, acceptable=acceptable, exclude=exclude, n_boot=0)
        rows[name] = t.loc[name, ["n", "initial_accuracy", "accuracy", "fixed", "broken", "sign_test_p",
                                  "avg_llm_calls"]]
    return pd.DataFrame(rows).T


def label_review(review_csv, dates):
    """Table 5: labels the review could not accept, with the date mentions in a group of their own."""
    acc, unclear, _ = E.pooled_gold(review_csv)
    df = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    gold = {str(r["id"]): json.loads(r["sources"])["gold"] for _, r in df.iterrows()}
    groups = {"dates labeled NIL": [], "other NIL labels": [], "other disagreements": [], "audit of agreements": []}
    for _, r in df.iterrows():
        i = str(r["id"])
        if r["group"] == "audit":
            groups["audit of agreements"].append(i)
        elif gold[i] == "NIL":
            groups["dates labeled NIL" if i in dates else "other NIL labels"].append(i)
        else:
            groups["other disagreements"].append(i)
    out = {}
    for g, ids in groups.items():
        judged = [i for i in ids if i in acc]
        wrong = sum(gold[i] not in acc[i] for i in judged)
        lo, hi = E._wilson(wrong, len(judged))
        out[g] = {"rows": len(ids), "judged": len(judged), "wrong": wrong,
                  "wrong_rate": round(wrong / len(judged), 4) if judged else None,
                  "ci95_low": None if lo is None else round(lo, 4), "ci95_high": None if hi is None else round(hi, 4)}
    nondate_wrong = sum(out[g]["wrong"] for g in ("other NIL labels", "other disagreements", "audit of agreements"))
    nondate_judged = sum(out[g]["judged"] for g in ("other NIL labels", "other disagreements", "audit of agreements"))
    out["test set without dates (lower bound)"] = {"rows": None, "judged": nondate_judged, "wrong": nondate_wrong,
                                                   "wrong_rate": None, "ci95_low": None, "ci95_high": None}
    t = pd.DataFrame(out).T
    for c in ("rows", "judged", "wrong"):
        t[c] = t[c].astype("Int64")
    t.attrs["unclear"] = len(unclear)
    return t


def reproduce(runs_dir, review_csv=None, agreement_csv=None, ref="llm_no_correction", out_dir=None,
              n_boot=2000, verbose=True):
    """Compute all tables; returns {name: DataFrame} and saves them as CSV in out_dir.
    agreement_csv is optional (a second annotator's review file, for Cohen's kappa)."""
    runs = find_runs(runs_dir)
    if not runs:
        raise SystemExit(f"No run files found in {runs_dir}")
    first = {k: v for k, v in runs.items() if "@" not in k}
    main = main_model_runs(runs)
    others = sorted({split_tag(k)[1] for k in runs} - {""} - {split_tag(k)[1] for k in main})
    ref_recs = E.load_rows(first.get(ref) or next(iter(first.values())))
    dates = date_ids(ref_recs)
    tables = {}

    # ---- original labels
    tables["table2_first_runs_original"] = _quiet(E.paired_table, first, ref=ref, n_boot=n_boot)
    tables["table3_within_run_original"] = within_run(runs)
    if "full_no_memory" in first and "baseline_search_top1" in first:
        tables["gold_vs_agent_vs_search"] = E.gold_source_check(first["full_no_memory"],
                                                                first["baseline_search_top1"]).to_frame()

    # ---- corrected labels
    if review_csv and os.path.exists(review_csv):
        acc, unclear, summ = E.pooled_gold(review_csv)
        no_dates = set(unclear) | dates
        tables["table2_first_runs_corrected"] = _quiet(E.paired_table, first, ref=ref, acceptable=acc,
                                                       exclude=unclear, n_boot=n_boot)
        tables["table2_first_runs_corrected_no_dates"] = _quiet(E.paired_table, first, ref=ref, acceptable=acc,
                                                                exclude=no_dates, n_boot=n_boot)
        tables["table3_within_run_corrected"] = within_run(runs, acc, unclear)
        tables["table3_within_run_no_dates"] = within_run(runs, acc, no_dates)
        spread = {k: v for k, v in main.items() if split_tag(k)[0] in LLM_ONLY_FROM}
        s = _quiet(E.paired_table, spread, acceptable=acc, exclude=unclear, n_boot=0)[["n", "initial_accuracy"]]
        ia = s["initial_accuracy"].astype(float)
        s.loc["(mean)"] = [None, round(ia.mean(), 4)]
        s.loc["(sd)"] = [None, round(ia.std(), 4)]
        tables["table3_llm_only_spread"] = s
        for name in [k for k in runs if split_tag(k)[0] in CHECK_RUNS]:
            tables[f"table4_checks_{name}"] = E.fix_attribution(E.load_rows(runs[name]), acceptable=acc,
                                                                exclude=unclear)
        tables["table5_label_review"] = label_review(review_csv, dates)
        tables["gold_quality"] = pd.DataFrame(summ).T
        tables["table6_popularity"] = E.paired_by(first, "category", ref=ref, acceptable=acc, exclude=unclear)
        tables["by_script"] = E.by_script(first, ref=ref, acceptable=acc, exclude=unclear)
        if "full_no_memory" in first:
            _, nil = E.nil_rule_analysis(E.load_rows(first["full_no_memory"]), acceptable=acc, exclude=unclear)
            tables["nil_check_vs_rule"] = pd.Series(nil).to_frame("value")
        for tag in others:                                   # other models: their own within-run table
            sub = {k: v for k, v in runs.items() if split_tag(k)[1] == tag}
            tables[f"other_model{tag}"] = _quiet(E.paired_table, sub, acceptable=acc, exclude=unclear, n_boot=0)
        if agreement_csv and os.path.exists(agreement_csv):
            tables["inter_annotator_agreement"] = pd.Series(E.agreement(review_csv, agreement_csv)).to_frame("value")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(os.path.join(out_dir, re.sub(r"[^\w.-]", "_", name.replace("@", "_at_")) + ".csv"))
    if verbose:
        with pd.option_context("display.width", 200, "display.max_columns", 30):
            for name, df in tables.items():
                print(f"\n=== {name} ===")
                print(df.to_string())
    return tables


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=os.path.join(ROOT, "results", "runs", "elner"))
    ap.add_argument("--review", default=os.path.join(ROOT, "results", "review", "review_pooled.csv"))
    ap.add_argument("--agreement", default=None, help="optional second-annotator review file (Cohen's kappa)")
    ap.add_argument("--ref", default="llm_no_correction", help="reference run for the paired comparison")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables"))
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for the 95%% CI")
    args = ap.parse_args(argv)
    reproduce(args.runs, args.review, args.agreement, args.ref, args.out, args.n_boot)


if __name__ == "__main__":
    main()
