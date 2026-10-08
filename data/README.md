# Corrected ELNER-DZ test sample

**Source:** Bouguettoucha, H. H., & Djouablia, I. (2025). ELNER-DZ: A dataset for named entity recognition and entity linking in Algerian Arabic dialect (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15798592. CC BY 4.0.
**License:** CC BY 4.0 (see [LICENSE](LICENSE)).

## Construction
1. **Sampling:** 1061 mentions from 1,000 sentences sampled uniformly at random (reservoir sampling, fixed seed) from the first 580,916 records of the ELNER-DZ version 2 data file, the part our downloaded copy contained (28% of the release, almost entirely in Arabic script). They were split by sentence into dev (20%) and test (80%) with `split_rows(seed=13)`. This file contains the **850 test mentions**.
2. **Systems:** every system in `results/runs/elner/` was run on the test mentions. Main model: `gemini-3.8-flash`; second model: Qwen3-4B-Instruct-2507 (qwen3:4b-instruct-2507-q4_K_M, served locally with Ollama); Wikidata accessed 2026-10-03 to 2026-10-05.
3. **Pooled review:** every mention on which any answer of any run, final or initial, disagreed with the ELNER-DZ label was reviewed, plus a random audit sample of 40 mentions on which every answer agreed.
4. **Blind review:** the reviewer saw the distinct answers in random order, with their sources hidden, and marked every acceptable one, or the mention as unclear. The corrected labels rest on the verdicts of one annotator, who was blind to the source of each answer.
5. **Dates:** ELNER-DZ labels every date mention NIL. The review accepted the Wikidata item of the year (for example Q2002 for 2015). To follow the ELNER-DZ convention instead, leave out the mentions of type `DATE`.
6. **First pass:** an earlier non-blind pass over the agent's disagreements is kept in `results/review/review_first_pass_nonblind.csv` for transparency. Its verdicts were reused only when every option shown in the blind review had been visible.

## Label statuses
| label_status | mentions | meaning |
|---|---|---|
| `confirmed` | 588 | a reviewer judged the original gold label correct |
| `alternatives_acceptable` | 14 | the original label is correct, but another QID is equally acceptable |
| `corrected` | 92 | the original label was judged wrong; gold_corrected holds the right QID(s), or [] if none is known |
| `audited_confirmed` | 40 | random audit of a mention where every system agreed with the gold: judged correct |
| `unreviewed_agreement` | 104 | every system agreed with the original label and the mention was not sampled for audit |
| `not_reviewed` | 0 | a system disagreed with the gold but the mention was not reviewed |
| `unclear` | 12 | the reviewer could not decide; excluded from all scores |

## Fields
| Field | Description |
|---|---|
| `id` | mention id: `<ELNER-DZ sentence id>-<entity index>` |
| `doc_id` | ELNER-DZ sentence id |
| `text` | sentence text (absent if the release has no texts; see `restore_texts`) |
| `mention` | entity mention |
| `start, end` | character offsets of the mention in `text` |
| `type` | ELNER-DZ entity type |
| `popularity` | `head` / `torso` / `tail` by Wikidata sitelinks of the original label, or `nil` |
| `sitelinks` | number of Wikimedia sitelinks of the original label |
| `gold_original` | ELNER-DZ label (QID or NIL) |
| `gold_corrected` | list of acceptable QIDs after review (`NIL` = not in Wikidata; `[]` = none known; null = unclear) |
| `label_status` | see the table below |
| `annotator` | who judged the mention (`A`, `first_pass`, ...) |
| `text_sha1` | SHA-1 of the sentence text, to check alignment with ELNER-DZ |

## Known limitations
- **Coverage:** the sample represents the Arabic-script first part of the ELNER-DZ file, not the whole corpus.
- **Formulaic text:** ELNER-DZ sentences are short templates around entity names, which makes the task easier for label-based retrieval than natural text would.
- **One annotator:** the corrected labels rest on one annotator's verdicts.
- **Unreviewed agreements:** of the mentions on which every answer agreed with the original label, only a random audit sample was reviewed.
- **Wikidata changes over time:** the cache snapshot in `cache/` fixes the state used here.
