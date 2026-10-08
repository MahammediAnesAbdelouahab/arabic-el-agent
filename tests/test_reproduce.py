"""The reproduction script must rebuild every table from run files alone (no network, no API key)."""
import gzip
import json
import os
import subprocess
import sys

import elagent as E

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _run(gold, answers, init=None, types=None, checks=None):
    """Synthetic run records: answers[i] is the final answer, init[i] the initial one (default: the same)."""
    out = []
    for i, (g, a) in enumerate(zip(gold, answers)):
        ini = (init or {}).get(i, a)
        out.append({"id": str(i), "mention": "2015" if (types or {}).get(i) == "DATE" else "m", "text": f"t{i}",
                    "type": (types or {}).get(i, "LOC"), "gold": g, "qid": a, "initial_qid": ini,
                    "category": "tail" if i % 2 else "head", "llm_calls": 1, "candidates": [a, ini],
                    "trace": [{"step": "verify", "issues": [{"check": (checks or {}).get(i, "NIL")}]}] if ini != a else []})
    return out


def _write(tmp_path):
    gold = ["NIL", "Q1", "Q2", "NIL"] + ["Q9"] * 6
    types = {3: "DATE"}
    runs = {
        "baseline_search_top1": _run(gold, ["Q90", "Q1", "Q3", "Q2015"] + ["Q9"] * 6, types=types),
        "llm_no_correction": _run(gold, ["Q90", "NIL", "Q2", "Q2015"] + ["Q9"] * 6, types=types),
        "full_no_memory": _run(gold, ["Q90", "Q1", "Q2", "Q77"] + ["Q9"] * 6, init={1: "NIL", 3: "Q2015"},
                               types=types, checks={3: "COHERENCE"}),
        "full_no_memory@r2": _run(gold, ["Q90", "Q1", "Q2", "Q2015"] + ["Q9"] * 6, init={1: "NIL"}, types=types),
        "critic_only": _run(gold, ["NIL", "NIL", "Q2", "Q2015"] + ["Q9"] * 6, init={0: "Q90"}, types=types,
                            checks={0: "CRITIC"}),
        "full_no_memory@qwen-test": _run(gold, ["Q90", "Q1", "Q2", "Q2015"] + ["Q9"] * 6, init={1: "NIL"}, types=types),
    }
    rd = tmp_path / "results" / "runs" / "elner"
    rd.mkdir(parents=True)
    for name, recs in runs.items():
        with gzip.open(rd / f"{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    rev = tmp_path / "results" / "review"
    rev.mkdir(parents=True)
    path = str(rev / "review_pooled.csv")
    df = E.export_pooled_review(runs, path, None, audit=2, include_initial=True)
    df["status"] = "done"
    accepted = {"0": "Q90", "3": "Q2015"}                     # the NIL label of 0 and the year label of 3 are wrong
    df["accepted"] = [accepted.get(i, json.loads(s)["gold"]) for i, s in zip(df["id"], df["sources"])]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return rd, path


def test_reproduce_tables(tmp_path):
    import reproduce_tables as R
    rd, review = _write(tmp_path)
    out = tmp_path / "tables"
    t = R.reproduce(str(rd), review, out_dir=str(out), n_boot=50, verbose=False)
    c = t["table2_first_runs_corrected"]
    assert c.loc["llm_no_correction", "accuracy"] == 0.9                 # misses mention 1 only
    assert c.loc["full_no_memory", "fixed"] == 1 and c.loc["full_no_memory", "broken"] == 1
    assert t["table2_first_runs_original"].loc["full_no_memory", "accuracy"] == 0.8
    w = t["table3_within_run_no_dates"]
    assert w.loc["full_no_memory", "fixed"] == 1 and w.loc["full_no_memory", "broken"] == 0   # the year is left out
    assert "full_no_memory@r2" in w.index and "critic_only" in w.index
    lab = t["table5_label_review"]
    assert lab.loc["dates labeled NIL", "wrong"] == 1 and lab.loc["other NIL labels", "wrong"] == 1
    assert t["table4_checks_full_no_memory"].loc["COHERENCE", "broken"] == 1
    assert "other_model@qwen-test" in t and "table6_popularity" in t
    assert set(t["table3_llm_only_spread"].index) >= {"llm_no_correction", "full_no_memory", "(mean)", "(sd)"}
    assert (out / "table2_first_runs_corrected.csv").exists()


def test_cli(tmp_path):
    rd, review = _write(tmp_path)
    res = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "reproduce_tables.py"), "--runs", str(rd),
                          "--review", review, "--out", str(tmp_path / "t"), "--n-boot", "20"],
                         capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "=== table2_first_runs_corrected ===" in res.stdout
