# Evidence, Not Doubt: Self-Verification in LLM-Based Arabic Entity Linking to Wikidata

<!-- DOI_BADGE -->
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23239230.svg)](https://doi.org/10.5281/zenodo.23239230)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MahammediAnesAbdelouahab/arabic-el-agent/blob/main/notebooks/reproduce_and_run.ipynb)
[![Code license: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Data license: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](data/LICENSE)

An LLM agent that links Arabic entity mentions to Wikidata and checks its answers, mostly against Wikidata, before accepting them; with every comparison system of the paper (an ungrounded LLM critic, a fixed NIL rule, retrieval baselines), all run traces, the review tools, and a blind-reviewed, corrected ELNER-DZ test sample.

## Findings
1. **Evidence, not doubt.** In the same correction loop, the checks computed from Wikidata repaired answers and rarely broke any, whereas a general LLM critic (`critic_only`), which also fired far more often, broke many correct answers.
2. **A rule instead of a loop.** Nearly all repairs came from the NIL check, and a fixed rule acting on its trigger (`llm_plus_nil_rule`) did as well without extra model calls.
3. **Labels matter more than verification.** ELNER-DZ leaves dates unlinked, and the blind pooled review found other wrong labels in our test sample; both effects are larger than the gain from verification.

## Method
For every mention the agent
1. **retrieves candidates:** Wikidata search, Arabic Wikipedia titles, redirects and disambiguation pages, clitic-aware query variants, and English/French search for mentions written in Latin script;
2. **lets an LLM decide:** a QID, NIL, or a new search;
3. **checks the answer:** candidate membership, item kind, entity type (P31/P279*), label match and a challenge to NIL answers, all computed from Wikidata, plus an LLM coherence check over the links of a text;
4. **corrects itself:** problems go back to the LLM together with their evidence, for at most three rounds.

`critic_only` replaces the checks by a general LLM critic that judges any answer from the text and its own knowledge; `grounded_only` keeps the Wikidata checks only; `llm_plus_nil_rule` applies the NIL check's trigger as a fixed rule to the LLM-only run.

## Main results
Main model: `gemini-3.8-flash` (thinking level `low`); second model: Qwen3-4B-Instruct-2507 (qwen3:4b-instruct-2507-q4_K_M, served locally with Ollama). Wikidata accessed 2026-10-03 to 2026-10-05.

**First run of every system, corrected labels** (`table2_first_runs_corrected`; `wins_vs_ref` / `losses_vs_ref` against `llm_no_correction`, exact McNemar test):

| | n | accuracy | initial_accuracy | fixed | broken | sign_test_p | wins_vs_ref | losses_vs_ref | p_vs_ref | diff_vs_ref_95CI | avg_llm_calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_exact_popularity | 838 | 0.858 | 0.858 | 0 | 0 | 1 | 29 | 88 | 0 | [-0.095, -0.047] | 0 |
| baseline_popularity | 838 | 0.2804 | 0.2804 | 0 | 0 | 1 | 11 | 554 | 0 | [-0.680, -0.613] | 0 |
| baseline_search_top1 | 838 | 0.9117 | 0.9117 | 0 | 0 | 1 | 31 | 45 | 0.1354 | [-0.037, +0.004] | 0 |
| critic_only | 838 | 0.8532 | 0.9308 | 8 | 73 | 0 | 9 | 72 | 0 | [-0.097, -0.055] | 2.6026 |
| full_no_memory | 838 | 0.9368 | 0.9296 | 8 | 2 | 0.1094 | 12 | 5 | 0.1435 | [-0.001, +0.018] | 1.3302 |
| full_plus_role | 838 | 0.9344 | 0.932 | 6 | 4 | 0.7539 | 12 | 7 | 0.3593 | [-0.004, +0.017] | 1.5115 |
| grounded_only | 838 | 0.9296 | 0.9236 | 6 | 1 | 0.125 | 7 | 6 | 1 | [-0.007, +0.011] | 1.2411 |
| llm_no_correction | 838 | 0.9284 | 0.9284 | 0 | 0 | 1 | 0 | 0 | 1 | [+0.000, +0.000] | 1.1993 |
| llm_plus_nil_rule | 838 | 0.9403 | 0.9284 | 13 | 3 | 0.0213 | 13 | 3 | 0.0213 | [+0.004, +0.021] | 1.1993 |


**Repairs and breaks inside each verification run** (exact sign test), on all scored mentions and without the date mentions that ELNER-DZ does not link:

| | n | initial_accuracy | accuracy | fixed | broken | sign_test_p |
|---|---|---|---|---|---|---|
| critic_only | 838 | 0.9308 | 0.8532 | 8 | 73 | 0 |
| full_no_memory | 838 | 0.9296 | 0.9368 | 8 | 2 | 0.1094 |
| full_no_memory@qwen3-4b-instruct-2507-q4_K_M | 838 | 0.8962 | 0.9045 | 11 | 4 | 0.1185 |
| full_no_memory@r2 | 838 | 0.9284 | 0.9344 | 7 | 2 | 0.1797 |
| full_plus_role | 838 | 0.932 | 0.9344 | 6 | 4 | 0.7539 |
| grounded_only | 838 | 0.9236 | 0.9296 | 6 | 1 | 0.125 |
| llm_plus_nil_rule | 838 | 0.9284 | 0.9403 | 13 | 3 | 0.0213 |


| | n | initial_accuracy | accuracy | fixed | broken | sign_test_p |
|---|---|---|---|---|---|---|
| critic_only | 787 | 0.9288 | 0.8653 | 8 | 58 | 0 |
| full_no_memory | 787 | 0.9276 | 0.9377 | 8 | 0 | 0.0078 |
| full_no_memory@qwen3-4b-instruct-2507-q4_K_M | 787 | 0.8983 | 0.9034 | 8 | 4 | 0.3877 |
| full_no_memory@r2 | 787 | 0.9263 | 0.9352 | 7 | 0 | 0.0156 |
| full_plus_role | 787 | 0.9301 | 0.9327 | 6 | 4 | 0.7539 |
| grounded_only | 787 | 0.9212 | 0.9276 | 6 | 1 | 0.125 |
| llm_plus_nil_rule | 787 | 0.9263 | 0.939 | 13 | 3 | 0.0213 |


Reproduce every table, with no API key:
```bash
pip install -r requirements.txt
python scripts/reproduce_tables.py      # writes results/tables/*.csv
```

## About the data
- **Sample.** The test sample comes from the first 580,916 records (28%) of the ELNER-DZ version 2 data file, the part our downloaded copy contained. That part is almost entirely in Arabic script, whereas 57% of the mentions in the full release are in Latin script, so the results concern Arabic-script text.
- **Text.** ELNER-DZ sentences are short and formulaic: the same dialectal templates recur with different entity names, so mentions are mostly Wikidata labels in generic contexts.
- **Dates.** ELNER-DZ labels every date mention NIL. The corrected labels link years to their Wikidata items; the tables are also given without the date mentions.
- **Review.** The corrected labels rest on the verdicts of one annotator, who was blind to the source of each answer.

**Labels the review could not accept** (`table5_label_review`, 95% Wilson intervals):

| | rows | judged | wrong | wrong_rate | ci95_low | ci95_high |
|---|---|---|---|---|---|---|
| dates labeled NIL | 51 | 51 | 51 | 1 | 0.93 | 1 |
| other NIL labels | 4 | 3 | 2 | 0.6667 | 0.2077 | 0.9385 |
| other disagreements | 651 | 640 | 39 | 0.0609 | 0.0449 | 0.0822 |
| audit of agreements | 40 | 40 | 0 | 0 | 0 | 0.0876 |
| test set without dates (lower bound) |  | 683 | 41 |  |  |  |


## Repository
| Path | Content |
|---|---|
| `elagent.py` | the agent, the comparison systems, evaluation, statistics, review and release utilities |
| `notebooks/reproduce_and_run.ipynb` | reproduce the tables, run the agent, use your own data, review tools |
| `notebooks/revision_experiments.ipynb` | the analyses and runs added during the revision (critic, NIL rule, repeated runs, Qwen, review) |
| `notebooks/extra_numbers.ipynb` | corpus profile, dates and every repair and break |
| `scripts/reproduce_tables.py` | recompute every table from the released runs |
| `results/runs/` | every complete system run, with the full agent traces (`.jsonl.gz`) |
| `results/review/` | the pooled blind review (and the earlier non-blind pass if present) |
| `results/revision/` | the revision reports |
| `data/elner_dz_test_corrected.jsonl` | corrected ELNER-DZ test sample (see [data/README.md](data/README.md)) |
| `data/challenge_set.jsonl` | 59 hand-verified hard mentions |
| `cache/wikidata_cache.json.gz` | Wikidata/Wikipedia responses used in the experiments |
| `tests/` | unit tests (`pytest`) |

## Data license
`data/elner_dz_test_corrected.jsonl` is derived from ELNER-DZ (Bouguettoucha, H. H., & Djouablia, I. (2025). ELNER-DZ: A dataset for named entity recognition and entity linking in Algerian Arabic dialect (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15798592), licensed under CC BY 4.0.

| label_status | mentions | meaning |
|---|---|---|
| `confirmed` | 588 | a reviewer judged the original gold label correct |
| `alternatives_acceptable` | 14 | the original label is correct, but another QID is equally acceptable |
| `corrected` | 92 | the original label was judged wrong; gold_corrected holds the right QID(s), or [] if none is known |
| `audited_confirmed` | 40 | random audit of a mention where every system agreed with the gold: judged correct |
| `unreviewed_agreement` | 104 | every system agreed with the original label and the mention was not sampled for audit |
| `not_reviewed` | 0 | a system disagreed with the gold but the mention was not reviewed |
| `unclear` | 12 | the reviewer could not decide; excluded from all scores |

## Citation
Archived on Zenodo: [10.5281/zenodo.23239230](https://doi.org/10.5281/zenodo.23239230).

Please cite this repository using [`CITATION.cff`](CITATION.cff) (GitHub: “Cite this repository”).


## License
- **Code:** MIT ([LICENSE](LICENSE)).
- **Data in `data/` and `results/`:** CC BY 4.0 ([data/LICENSE](data/LICENSE)), with attribution to ELNER-DZ.
