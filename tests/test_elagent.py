"""Offline tests: fake Wikidata/Wikipedia/SPARQL HTTP layer + scripted LLM."""
import json, os, re
import pytest
import elagent as E


# ---------------- fake Wikidata world (raw API JSON format) ----------------
def _item(v):
    return {"mainsnak": {"datavalue": {"type": "wikibase-entityid", "value": {"id": v}}}, "rank": "normal"}


def ent(qid, ar=None, en=None, desc_en=None, aliases_ar=(), p31=(), facts=None, sitelinks=10, times=None):
    claims = {"P31": [_item(v) for v in p31]}
    for p, vs in (facts or {}).items():
        claims[p] = [_item(v) for v in vs]
    for p, t in (times or {}).items():
        claims[p] = [{"mainsnak": {"datavalue": {"type": "time", "value": {"time": t}}}, "rank": "normal"}]
    labels = {}
    if ar: labels["ar"] = {"language": "ar", "value": ar}
    if en: labels["en"] = {"language": "en", "value": en}
    return {"id": qid, "labels": labels,
            "descriptions": {"en": {"language": "en", "value": desc_en}} if desc_en else {},
            "aliases": {"ar": [{"language": "ar", "value": a} for a in aliases_ar]},
            "claims": claims, "sitelinks": {f"w{i}": {} for i in range(sitelinks)}}


ENTS = {e["id"]: e for e in [
    ent("Q3783", "نهر الأمازون", "Amazon River", "river in South America", ["الأمازون", "أمازون"], ["Q4022"], {"P17": ["Q155"]}, 200),
    ent("Q3884", "أمازون", "Amazon", "American technology company", [], ["Q4830453"], {"P17": ["Q30"]}, 150),
    ent("Q1", "أمازون (توضيح)", "Amazon (disambiguation)", None, [], ["Q4167410"], {}, 30),
    ent("Q90", "باريس", "Paris", "capital of France", [], ["Q515"], {"P17": ["Q142"]}, 300),
    ent("Q47899", "باريس هيلتون", "Paris Hilton", "American media personality", [], ["Q5"], {}, 90),
    ent("Q41421", "مايكل جوردن", "Michael Jordan", "American basketball player", ["مايكل جوردان"], ["Q5"],
        {"P106": ["Q3665646"]}, 120, {"P569": "+1963-02-17T00:00:00Z"}),
    ent("Q810", "الأردن", "Jordan", "country in Western Asia", [], ["Q6256"], {}, 250),
    ent("Q40059", "نهر الأردن", "Jordan River", "river in the Middle East", [], ["Q4022"], {}, 80),
    ent("Q1492", "برشلونة", "Barcelona", "city in Spain", [], ["Q515"], {"P17": ["Q29"]}, 220),
    ent("Q7156", "نادي برشلونة", "FC Barcelona", "football club", ["برشلونة"], ["Q476028"], {}, 180),
    ent("Q8682", "ريال مدريد", "Real Madrid CF", "football club", [], ["Q476028"], {}, 170),
    ent("Q262", "الجزائر", "Algeria", "country in North Africa", [], ["Q6256"], {}, 260),
    ent("Q181903", "منتخب الجزائر لكرة القدم", "Algeria national football team", "men's national association football team",
        ["محاربو الصحراء"], ["Q6979593"], {}, 90),
]}
LABELS = {"Q6979593": "national association football team", "Q4022": "river", "Q4830453": "business", "Q4167410": "Wikimedia disambiguation page", "Q515": "city",
          "Q5": "human", "Q6256": "country", "Q476028": "association football club", "Q30": "United States",
          "Q142": "France", "Q155": "Brazil", "Q29": "Spain", "Q3665646": "basketball player"}
SUBCLASS = {"Q6979593": ["Q43229"], "Q4022": ["Q618123"], "Q515": ["Q486972"], "Q476028": ["Q847017"], "Q847017": ["Q43229"],
            "Q4830453": ["Q43229"], "Q6256": ["Q56061"], "Q4167410": []}
WP_TITLES = {"باريس": {"qid": "Q90"}, "أمازون": {"disambiguation": True}, "الجزائر": {"qid": "Q262"}}
WP_DISAMBIG = {"أمازون": ["Q3884", "Q3783", "Q1"]}


class FakeWD(E.WikidataClient):
    def __init__(self, cache_path=None):
        super().__init__("test/0.1 (contact: test@example.org)", cache_path, pause=0)
        self.log = []

    def _isa(self, q, roots):
        seen, stack = set(), list((ENTS.get(q) and [c["mainsnak"]["datavalue"]["value"]["id"] for c in ENTS[q]["claims"]["P31"]]) or [])
        while stack:
            c = stack.pop()
            if c in roots: return True
            if c in seen: continue
            seen.add(c); stack += SUBCLASS.get(c, [])
        return False

    def _get_json(self, url, params, headers=None):
        self.n_requests += 1
        self.log.append((url, dict(params)))
        if url == self.API and params["action"] == "wbsearchentities":
            q, lang = E.normalize_ar(params["search"]), params["language"]
            hits = []
            for e in ENTS.values():
                names = ([e["labels"].get("ar", {}).get("value")] + [a["value"] for a in e["aliases"].get("ar", [])]) if lang == "ar" \
                    else [e["labels"].get("en", {}).get("value")]
                if any(n and E.normalize_ar(n).startswith(q) for n in names):
                    hits.append(e)
            hits.sort(key=lambda e: -len(e["sitelinks"]))
            return {"search": [{"id": e["id"]} for e in hits[: int(params["limit"])]]}
        if url == self.API and params["action"] == "wbgetentities":
            ids = params["ids"].split("|")
            if params["props"] == "labels":
                return {"entities": {q: {"id": q, "labels": {"en": {"value": LABELS.get(q) or ENTS.get(q, {}).get("labels", {}).get("en", {}).get("value", q)}}} for q in ids}}
            return {"entities": {q: (ENTS[q] if q in ENTS else {"id": q, "missing": ""}) for q in ids}}
        if url == self.SPARQL:
            qid = re.search(r"wd:(Q\d+) wdt:P31", params["query"]).group(1)
            roots = re.findall(r"wd:(Q\d+)", params["query"].split("}")[0])
            return {"boolean": self._isa(qid, set(roots))}
        if "wikipedia.org" in url:
            if "titles" in params and params.get("generator") == "links":
                return {"query": {"pages": [{"pageprops": {"wikibase_item": q}} for q in WP_DISAMBIG.get(params["titles"], [])]}}
            if "titles" in params:
                t = WP_TITLES.get(params["titles"])
                if not t: return {"query": {"pages": [{"missing": True}]}}
                pp = {"disambiguation": ""} if t.get("disambiguation") else {"wikibase_item": t["qid"]}
                return {"query": {"pages": [{"pageprops": pp}]}}
            return {"query": {"pages": []}}
        raise AssertionError(f"unexpected request {url} {params}")


class FakeLLM(E.LLM):
    """Rules: list of (predicate(user_prompt) -> bool, response dict | str). First match wins."""
    def __init__(self, rules):
        self.rules, self.prompts = rules, []
        self.backend, self.model, self.temperature, self.thinking_level = "fake", "fake", None, None
        self.min_interval, self.calls, self.seconds, self._last = 0, 0, 0.0, 0.0
        self._no_temp = self._no_thinking = self._no_json_mode = False

    def _raw(self, system, user):
        self.prompts.append(user)
        for pred, resp in self.rules:
            if pred(user):
                return resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
        return json.dumps({"action": "nil", "confidence": 0.3, "reason": "no rule"})


def target(m): return lambda u: f"Target mention: «{m}»" in u
def first(m): return lambda u: target(m)(u) and "Your previous answer" not in u and "Give up to 3" not in u
def retry(m, check): return lambda u: target(m)(u) and f"[{check}]" in u
EXPAND = lambda u: "Give up to 3 search queries" in u
COHER = lambda u: "Entity links:" in u


def make(rules, tmp_path, **cfg):
    wd = FakeWD(str(tmp_path / "cache.json"))
    llm = FakeLLM(rules)
    mem = E.ErrorMemory(str(tmp_path / "memory.json"))
    c = E.Config(project_dir=str(tmp_path), expand_queries=cfg.pop("expand_queries", False), **cfg)
    return E.Agent(c, wd, llm, mem), wd, llm, mem


# ---------------- tests ----------------
def test_variants():
    p, f = E.search_variants("للجزائر")
    assert p == ["للجزائر", "الجزائر"]
    assert "القاهرة" in E.search_variants("بالقاهرة")[0]
    assert "برشلونة" in E.search_variants("وبرشلونة")[0]
    assert E.search_variants("لمصر")[1] == ["مصر"]
    assert E.search_variants("بغداد")[0] == ["بغداد"]          # risky stripping only in fallback
    assert E.normalize_ar("إِسْلامِيّة") == "اسلاميه"
    assert E.tokens("الجزائر") == ["جزاير"] and E.tokens("الجزائر") == E.tokens("جزائر")


def test_parse_json():
    assert E.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert E.parse_json('Sure! {"a": 2} hope this helps') == {"a": 2}
    assert E.parse_json("nope") is None


def test_compact_entity_and_disambiguation_filter(tmp_path):
    agent, wd, *_ = make([], tmp_path)
    cands = agent.candidates(E.Mention("أعلنت أمازون", "أمازون"))
    ids = [c["qid"] for c in cands]
    assert "Q1" not in ids                                   # disambiguation item filtered out
    assert {"Q3884", "Q3783"} <= set(ids)
    mj = wd.get_entities(["Q41421"])["Q41421"]
    assert mj["facts"]["P569"] == ["1963-02-17"] and mj["aliases_ar"] == ["مايكل جوردان"]


def test_type_check_corrects_and_records_memory(tmp_path):
    rules = [(retry("أمازون", "TYPE"), {"action": "select", "qid": "Q3884", "confidence": 0.9, "reason": "company"}),
             (first("أمازون"), {"action": "select", "qid": "Q3783", "confidence": 0.8, "reason": "river"})]
    agent, wd, llm, mem = make(rules, tmp_path)
    r = agent.link(E.Mention("أعلنت أمازون عن مركز بيانات جديد", "أمازون", "ORG"))
    assert (r["initial_qid"], r["qid"], r["corrected"], r["flagged"]) == ("Q3783", "Q3884", True, False)
    assert len(mem.cases) == 1 and mem.cases[0]["right"] == "Q3884"
    assert "Q3783" in mem.hints("أمازون")
    assert os.path.exists(tmp_path / "memory.json")


def test_confirm_keeps_metonymy(tmp_path):
    rules = [(retry("الأردن", "TYPE"), {"action": "confirm", "reason": "country name used for national team"}),
             (first("الأردن"), {"action": "select", "qid": "Q810", "confidence": 0.9})]
    agent, *_ = make(rules, tmp_path)
    r = agent.link(E.Mention("فاز الأردن على العراق", "الأردن", "ORG"))
    assert (r["qid"], r["corrected"], r["flagged"]) == ("Q810", False, False)
    assert any(t["step"] == "confirm" for t in r["trace"])


def test_invented_qid_is_hard_failure(tmp_path):
    rules = [(retry("باريس", "CANDIDATE"), {"action": "confirm"}),          # confirm not allowed -> invalid
             (first("باريس"), {"action": "select", "qid": "Q999999", "confidence": 0.9})]
    agent, _, llm, _ = make(rules, tmp_path, max_rounds=2)
    r = agent.link(E.Mention("تعد باريس مدينة جميلة", "باريس", "LOC"))
    assert r["qid"] == "NIL" and r["flagged"]                 # never returns an invented QID
    assert all("confirm" not in p.split("Allowed actions:")[1] for p in llm.prompts if "[CANDIDATE]" in p)


def test_invented_qid_repaired(tmp_path):
    rules = [(retry("باريس", "CANDIDATE"), {"action": "select", "qid": "Q90", "confidence": 0.9}),
             (first("باريس"), {"action": "select", "qid": "Q999999", "confidence": 0.9})]
    agent, *_ = make(rules, tmp_path)
    r = agent.link(E.Mention("تعد باريس مدينة جميلة", "باريس", "LOC"))
    assert (r["initial_qid"], r["qid"], r["flagged"]) == ("Q999999", "Q90", False)


def test_nil_is_challenged(tmp_path):
    rules = [(retry("باريس هيلتون", "NIL"), {"action": "select", "qid": "Q47899", "confidence": 0.9}),
             (first("باريس هيلتون"), {"action": "nil", "confidence": 0.6})]
    agent, *_ = make(rules, tmp_path)
    r = agent.link(E.Mention("زارت باريس هيلتون دبي", "باريس هيلتون", "PER"))
    assert (r["initial_qid"], r["qid"]) == ("NIL", "Q47899")


def test_expansion_and_search_action(tmp_path):
    rules = [(EXPAND, {"queries": [{"q": "Michael Jordan", "lang": "en"}]}),
             (lambda u: target("مايكل جوردون")(u) and "Q41421" in u, {"action": "select", "qid": "Q41421", "confidence": 0.9}),
             (first("مايكل جوردون"), {"action": "nil"})]
    agent, *_ = make(rules, tmp_path, expand_queries=True)
    r = agent.link(E.Mention("يعتبر مايكل جوردون أفضل لاعب", "مايكل جوردون", "PER"))
    assert r["qid"] == "Q41421" and r["trace"][0]["step"] == "expand"

    rules2 = [(lambda u: target("جوردون")(u) and "Q41421" in u, {"action": "select", "qid": "Q41421", "confidence": 0.9}),
              (first("جوردون"), {"action": "search", "query": "Michael Jordan", "lang": "en"})]
    agent2, *_ = make(rules2, tmp_path / "b")
    r2 = agent2.link(E.Mention("سجل جوردون 40 نقطة", "جوردون", "PER"))
    assert r2["qid"] == "Q41421" and any(t["step"] == "search" for t in r2["trace"])


def test_coherence_fixes_document(tmp_path):
    rules = [(COHER, {"inconsistent": [{"index": 0, "reason": "a football match: Barcelona is the club"}]}),
             (retry("برشلونة", "COHERENCE"), {"action": "select", "qid": "Q7156", "confidence": 0.9}),
             (first("برشلونة"), {"action": "select", "qid": "Q1492", "confidence": 0.9}),
             (first("ريال مدريد"), {"action": "select", "qid": "Q8682", "confidence": 0.9})]
    agent, *_ = make(rules, tmp_path)
    t = "فاز برشلونة على ريال مدريد في الكلاسيكو"
    res = agent.link_document([E.Mention(t, "برشلونة"), E.Mention(t, "ريال مدريد")])
    assert [r["qid"] for r in res] == ["Q7156", "Q8682"]
    assert res[0]["initial_qid"] == "Q1492" and res[0]["corrected"]


def test_no_correction_baseline_keeps_initial(tmp_path):
    rules = [(first("أمازون"), {"action": "select", "qid": "Q3783", "confidence": 0.8})]
    agent, _, llm, _ = make(rules, tmp_path, max_rounds=0)
    r = agent.link(E.Mention("أعلنت أمازون", "أمازون", "ORG"))
    assert r["qid"] == "Q3783" and llm.calls == 1


def test_prefix_variant_retrieval(tmp_path):
    agent, *_ = make([], tmp_path)
    assert "Q262" in [c["qid"] for c in agent.candidates(E.Mention("عاد للجزائر", "للجزائر"))]


def test_baselines(tmp_path):
    agent, *_ = make([], tmp_path, baseline="search_top1")
    assert agent.link(E.Mention("أعلنت أمازون", "أمازون"))["qid"] == "Q3783"   # most sitelinks first
    agent, *_ = make([], tmp_path / "b", baseline="popularity")
    assert agent.link(E.Mention("تعد باريس", "باريس"))["qid"] == "Q90"


def test_evaluate_metrics_and_resume(tmp_path):
    rules = [(retry("أمازون", "TYPE"), {"action": "select", "qid": "Q3884", "confidence": 0.9}),
             (first("أمازون"), {"action": "select", "qid": "Q3783", "confidence": 0.8}),
             (first("باريس"), {"action": "select", "qid": "Q90", "confidence": 0.95}),
             (first("سمير بن عيسى"), {"action": "nil", "confidence": 0.8})]
    agent, wd, llm, _ = make(rules, tmp_path)
    rows = [{"id": 1, "text": "أعلنت أمازون عن مركز بيانات", "mention": "أمازون", "type": "ORG", "gold": "Q3884"},
            {"id": 2, "text": "تعد باريس مدينة جميلة", "mention": "باريس", "type": "LOC", "gold": "http://www.wikidata.org/entity/Q90"},
            {"id": 3, "text": "قال المهندس سمير بن عيسى", "mention": "سمير بن عيسى", "type": "PER", "gold": None}]
    m, recs = E.evaluate(agent, rows, "t", str(tmp_path / "runs"), progress=False)
    assert m["n"] == 3 and m["accuracy"] == 1.0 and m["initial_accuracy"] == round(2 / 3, 4)
    assert m["fixed (wrong→right)"] == 1 and m["broken (right→wrong)"] == 0
    assert m["nil_precision"] == 1.0 and m["nil_recall"] == 1.0 and m["candidate_recall"] == 1.0
    calls = llm.calls
    m2, _ = E.evaluate(agent, rows, "t", str(tmp_path / "runs"), progress=False)   # resume: nothing re-run
    assert llm.calls == calls and m2 == m
    assert os.path.exists(tmp_path / "cache.json")


def test_ablation_table(tmp_path):
    rules = [(retry("أمازون", "TYPE"), {"action": "select", "qid": "Q3884", "confidence": 0.9}),
             (first("أمازون"), {"action": "select", "qid": "Q3783", "confidence": 0.8})]
    agent, wd, llm, mem = make(rules, tmp_path)
    rows = [{"id": 1, "text": "أعلنت أمازون عن مركز بيانات", "mention": "أمازون", "type": "ORG", "gold": "Q3884"}]
    df = E.run_ablations(rows, agent.cfg, wd, llm,
                         names=["baseline_search_top1", "llm_no_correction", "full_no_memory", "minus_type_check"],
                         out_dir=str(tmp_path / "runs"))
    acc = df["accuracy"].to_dict()
    assert acc == {"baseline_search_top1": 0.0, "llm_no_correction": 0.0, "full_no_memory": 1.0, "minus_type_check": 0.0}
    assert os.path.exists(tmp_path / "runs" / "ablations.csv")


def test_memory_distill_and_frozen(tmp_path):
    mem = E.ErrorMemory(str(tmp_path / "m.json"))
    mem.add_case("أمازون", "ORG", "Q3783", "Q3884", [E._issue("TYPE", "river vs ORG")], "أعلنت أمازون")
    llm = FakeLLM([(lambda u: "verified corrections" in u, {"rules": ["Check the entity type against the context."]})])
    assert mem.distill(llm) == ["Check the entity type against the context."]
    frozen = E.ErrorMemory(str(tmp_path / "m.json"), frozen=True)
    frozen.add_case("x", None, "Q1", "Q2", [], "")
    assert len(frozen.cases) == 1 and frozen.rules


def test_llm_retry_flags(tmp_path):
    class Flaky(FakeLLM):
        def _raw(self, system, user):
            if not self._no_temp:
                raise ValueError("temperature is not supported for this model")
            return "not json at all" if self.calls == 0 and not getattr(self, "_once", False) and not setattr(self, "_once", True) else '{"ok": true}'
    f = Flaky([])
    f.temperature = 0.0
    assert f.json("s", "u") == {"ok": True} and f._no_temp


def test_errors_df_and_flatten(tmp_path):
    rows = E.flatten_docs([{"id": "d1", "text": "فاز برشلونة على ريال مدريد",
                            "entities": [{"mention": "برشلونة", "qid": "Q7156", "type": "ORG"},
                                         {"mention": "ريال مدريد", "qid": "Q8682", "type": "ORG"}]}])
    assert [r["id"] for r in rows] == ["d1-0", "d1-1"] and rows[0]["doc_id"] == "d1"
    recs = [{"id": "1", "mention": "x", "gold": "Q1", "qid": "Q2", "initial_qid": "Q2", "candidates": ["Q1", "Q2"],
             "trace": [{"step": "verify", "issues": [{"check": "TYPE"}]}]},
            {"id": "2", "mention": "y", "gold": "Q3", "qid": "Q3"}]
    df = E.errors_dataframe(recs, str(tmp_path / "err.csv"))
    assert len(df) == 1 and df.iloc[0]["checks_fired"] == "TYPE" and bool(df.iloc[0]["gold_in_candidates"])


def test_split_rows():
    rows = [{"text": f"t{i // 2}", "mention": "m", "gold": "NIL"} for i in range(20)]
    dev, test = E.split_rows(rows, 0.2)
    assert len(dev) + len(test) == 20 and len(dev) == 4
    assert not ({r["text"] for r in dev} & {r["text"] for r in test})
    assert E.split_rows(rows, 0.2) == (dev, test)


def test_metrics_by_category(tmp_path):
    rules = [(retry("أمازون", "TYPE"), {"action": "select", "qid": "Q3884", "confidence": 0.9}),
             (first("أمازون"), {"action": "select", "qid": "Q3783", "confidence": 0.8}),
             (first("باريس"), {"action": "select", "qid": "Q90", "confidence": 0.95})]
    agent, *_ = make(rules, tmp_path)
    rows = [{"id": 1, "text": "أعلنت أمازون عن مركز", "mention": "أمازون", "type": "ORG", "gold": "Q3884", "category": "ambiguous"},
            {"id": 2, "text": "تعد باريس مدينة جميلة", "mention": "باريس", "type": "LOC", "gold": "Q90", "category": "easy"}]
    _, recs = E.evaluate(agent, rows, "cat", str(tmp_path / "runs"), progress=False)
    df = E.metrics_by(recs)
    assert list(df.index) == ["ambiguous", "easy", "ALL"]
    assert df.loc["ambiguous", "fixed (wrong→right)"] == 1 and df.loc["easy", "accuracy"] == 1.0


ROLE = lambda u: "metonym for a different kind of entity" in u


def test_role_check_fixes_metonymy(tmp_path):
    rules = [(ROLE, {"fits": False, "better_referent": "منتخب الجزائر لكرة القدم", "lang": "ar", "reason": "a country cannot win a final"}),
             (lambda u: retry("الجزائر", "ROLE")(u) and "Q181903" in u, {"action": "select", "qid": "Q181903", "confidence": 0.9}),
             (retry("الجزائر", "TYPE"), {"action": "confirm", "reason": "metonymy: the country name denotes its team"}),
             (first("الجزائر"), {"action": "select", "qid": "Q262", "confidence": 0.9})]
    agent, _, llm, _ = make(rules, tmp_path, verify_role=True)
    m = E.Mention("فازت الجزائر على السنغال في النهائي", "الجزائر", "GPE")
    r = agent.link(m)
    assert (r["initial_qid"], r["qid"], r["flagged"]) == ("Q262", "Q181903", False)
    assert any(t["step"] == "search" and t.get("by") == "verifier" and "Q181903" in t["added"] for t in r["trace"])
    n_role_prompts = sum(ROLE(p) for p in llm.prompts)
    assert n_role_prompts == 1                                 # cached, not re-asked; team is not place-like

    agent2, _, llm2, _ = make(rules, tmp_path / "off")          # flag off: stays on the country
    assert agent2.link(m)["qid"] == "Q262" and not any(ROLE(p) for p in llm2.prompts)


def test_role_check_keeps_places(tmp_path):
    rules = [(ROLE, {"fits": True, "reason": "diplomatic statement"}),
             (first("الجزائر"), {"action": "select", "qid": "Q262", "confidence": 0.9})]
    agent, *_ = make(rules, tmp_path, verify_role=True)
    r = agent.link(E.Mention("وقّعت الجزائر اتفاقية تعاون مع تونس", "الجزائر", "GPE"))
    assert r["qid"] == "Q262" and not r["corrected"] and r["rounds"] == 1


# ---------------- dataset tools ----------------
def test_detect_and_convert_nested_flat_bio(tmp_path):
    nested = [{"sent_id": "s1", "sentence": "خويا يخدم في سوناطراك فحاسي مسعود",
               "entities": [{"text": "سوناطراك", "wikidata": "Q1090717", "label": "ORG", "start": 13, "end": 21},
                            {"text": "حاسي مسعود", "wikidata": None, "label": "LOC", "start": 23, "end": 33}]},
              {"sent_id": "s2", "sentence": "الوفاق ربح على الشبيبة",
               "entities": [{"text": "الوفاق", "wikidata": "http://www.wikidata.org/entity/Q1059484", "label": "ORG", "start": 0, "end": 6}]}]
    sch = E.detect_schema(nested)
    assert sch == {"layout": "nested", "text": "sentence", "entities": "entities", "mention": "text", "qid": "wikidata",
                   "type": "label", "start": "start", "end": "end", "id": "sent_id"}
    rows = E.convert_records(nested, sch, "elner")
    assert [(r["id"], r["mention"], r["gold"], r["type"]) for r in rows] == [
        ("s1-0", "سوناطراك", "Q1090717", "ORG"), ("s1-1", "حاسي مسعود", "NIL", "LOC"), ("s2-0", "الوفاق", "Q1059484", "ORG")]

    flat = [{"context": "فاز سينر ببطولة أستراليا المفتوحة", "span": "سينر", "qid": "Q54812588", "ner": "PER"}] * 3
    sch = E.detect_schema(flat)
    assert (sch["layout"], sch["text"], sch["mention"], sch["qid"], sch["type"]) == ("flat", "context", "span", "qid", "PER" and "ner")
    assert E.convert_records(flat, sch)[0]["gold"] == "Q54812588"

    bio = [{"id": 7, "tokens": ["Messi", "marka", "m3a", "PSG"], "tags": ["B-PER", "O", "O", "B-ORG"],
            "qids": ["Q615", None, None, "Q483020"]}]
    sch = E.detect_schema(bio)
    assert sch["layout"] == "bio"
    rows = E.convert_records(bio, sch)
    assert [(r["mention"], r["gold"], r["type"], r["text"][r["start"]:r["end"]]) for r in rows] == [
        ("Messi", "Q615", "PER", "Messi"), ("PSG", "Q483020", "ORG", "PSG")]


def test_iter_records_formats(tmp_path):
    data = [{"t": "نص", "q": "Q1"}, {"t": "نص آخر", "q": "Q2"}]
    (tmp_path / "a.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"meta": 1, "data": data}, ensure_ascii=False), encoding="utf-8")
    (tmp_path / "c.json").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data), encoding="utf-8")
    (tmp_path / "d.jsonl").write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in data), encoding="utf-8")
    for f in "a.json b.json c.json d.jsonl".split():
        assert list(E.iter_records(str(tmp_path / f))) == data, f
    assert len(list(E.iter_records(str(tmp_path / "a.json"), limit=1))) == 1


def test_sampling_popularity_and_ner_types(tmp_path):
    rows = [{"id": f"{d}-{i}", "doc_id": d, "text": "تعد باريس مدينة", "mention": "باريس", "start": 4, "end": 9, "gold": g}
            for d, g in [("a", "Q90"), ("b", "Q47899"), ("c", "NIL")] for i in range(3)]
    s = E.sample_by_doc(rows, 4, max_per_doc=2)
    assert len(s) == 4 and len({r["doc_id"] for r in s}) == 2
    wd = FakeWD()
    E.add_popularity(rows, wd, head=100, tail=95)
    assert {r["gold"]: r["category"] for r in rows} == {"Q90": "head", "Q47899": "tail", "NIL": "nil"}
    ner = lambda t: [{"entity_group": "LOC", "start": 4, "end": 9}]
    E.assign_ner_types(rows, ner)
    assert rows[0]["type"] == "LOC"


WIKI_HTML = ('<div class="mw-parser-output"><p>زار الوفد <a href="/wiki/X" title="باريس">العاصمة الفرنسية</a> '
             'والتقى <a href="/wiki/Y" title="باريس هيلتون">باريس هيلتون</a><sup class="reference">[1]</sup> عام '
             '<a href="/wiki/1990" title="1990">1990</a> قرب <a class="new" href="/w/index.php?title=Z&amp;redlink=1" '
             'title="Z (الصفحة غير موجودة)">Z</a> و<a href="/wiki/C" title="تصنيف:مدن">مدن</a> '
             'ثم زار <a href="/wiki/R" title="مدينة الأنوار">مدينة النور</a> مرة أخرى في رحلة طويلة.</p>'
             '<table><tr><td><p><a href="/wiki/X" title="باريس">باريس</a></p></td></tr></table></div>')


class WikiWD(FakeWD):
    def _get_json(self, url, params, headers=None):
        if "wikipedia.org" in url and params.get("generator") == "random":
            return {"query": {"pages": [{"title": "مقالة تجريبية", "length": 5000}, {"title": "قصيرة", "length": 100}]}}
        if "wikipedia.org" in url and params.get("action") == "parse":
            return {"parse": {"title": params["page"], "text": WIKI_HTML}}
        if "wikipedia.org" in url and params.get("titles") and "generator" not in params and "|" in params["titles"] or \
                ("wikipedia.org" in url and params.get("titles") in ("باريس هيلتون", "مدينة الأنوار")):
            ts = params["titles"].split("|")
            m = {"باريس": "Q90", "باريس هيلتون": "Q47899"}
            return {"query": {"redirects": [{"from": "مدينة الأنوار", "to": "باريس"}] if "مدينة الأنوار" in ts else [],
                              "pages": [{"title": t, "pageprops": {"wikibase_item": m[t]}} for t in m]}}
        return super()._get_json(url, params, headers)


def test_build_wiki_link_set(tmp_path):
    wd = WikiWD()
    rows = E.build_wiki_link_set(wd, n_mentions=10, max_per_article=5, log=lambda *a: None)
    got = {(r["mention"], r["gold"], r["surface"]) for r in rows}
    assert got == {("العاصمة الفرنسية", "Q90", "alias"), ("باريس هيلتون", "Q47899", "title")}   # Q90 used once per article
    for r in rows:
        assert r["text"][r["start"]:r["end"]] == r["mention"] and "[1]" not in r["text"]
    paras = E._wp_paragraph_links(wd, "x")
    assert len(paras) == 1 and [l[0] for l in paras[0][1]] == ["العاصمة الفرنسية", "باريس هيلتون", "مدينة النور"]
    p = E.save_rows(rows, str(tmp_path / "d" / "w.jsonl"))
    assert E.load_rows(p) == rows


def test_evaluate_passes_extra_fields(tmp_path):
    rules = [(first("باريس"), {"action": "select", "qid": "Q90", "confidence": 0.95})]
    agent, *_ = make(rules, tmp_path)
    rows = [{"id": 1, "text": "تعد باريس مدينة", "mention": "باريس", "gold": "Q90", "surface": "title", "sitelinks": 300}]
    _, recs = E.evaluate(agent, rows, "x", str(tmp_path / "runs"), progress=False)
    assert recs[0]["surface"] == "title" and recs[0]["sitelinks"] == 300
    assert list(E.metrics_by(recs, "surface").index) == ["title", "ALL"]


def test_reservoir_sample():
    s = E.reservoir_sample(iter(range(10000)), 50)
    assert len(s) == 50 and len(set(s)) == 50 and max(s) > 5000
    assert E.reservoir_sample(range(3), 10) == [0, 1, 2]
    assert E.reservoir_sample(range(1000), 20) == E.reservoir_sample(range(1000), 20)


# ---------------- tolerant JSON reading (non-standard dataset dumps) ----------------
RECS = [{"sent_id": f"e{i}", "sentence": f"جملة رقم {i} فيها كيان مثل باريس", "entities": [{"text": "باريس", "qid": "Q90"}]}
        for i in range(6)]


def _dump(path, text, bom=False):
    path.write_bytes((b"\xef\xbb\xbf" if bom else b"") + text.encode("utf-8"))
    return str(path)


@pytest.mark.parametrize("stream", [False, True])
def test_iter_records_nonstandard_json(tmp_path, stream):
    mb = 0 if stream else 50
    pretty = json.dumps(RECS, ensure_ascii=False, indent=2)
    cases = {
        "pretty_array": pretty,
        "concat_arrays": json.dumps(RECS[:3], ensure_ascii=False, indent=2) + "\n" + json.dumps(RECS[3:], ensure_ascii=False, indent=2),
        "jsonl_in_json": "\n".join(json.dumps(r, ensure_ascii=False) for r in RECS),
        "concat_pretty_objects": "\n".join(json.dumps(r, ensure_ascii=False, indent=2) for r in RECS),
        "wrapper": json.dumps({"version": 1, "data": RECS}, ensure_ascii=False, indent=2),
        "dict_of_dicts": json.dumps({r["sent_id"]: r for r in RECS}, ensure_ascii=False, indent=2),
    }
    for name, text in cases.items():
        for bom in (False, True):
            p = _dump(tmp_path / f"{name}{int(bom)}.json", text, bom)
            got = list(E.iter_records(p, stream_over_mb=mb))
            assert got == RECS, (name, bom, stream, len(got))
    assert len(list(E.iter_records(_dump(tmp_path / "x.json", pretty), limit=2, stream_over_mb=mb))) == 2


def test_iter_records_truncated_python_and_errors(tmp_path, capsys):
    pretty = json.dumps(RECS, ensure_ascii=False, indent=2)
    got = list(E.iter_records(_dump(tmp_path / "t.json", pretty[:-40])))          # cut inside the last record
    assert got == RECS[:5] and "truncated" in capsys.readouterr().out
    py = str(RECS).replace("'", "'")                                                  # Python repr: single quotes
    assert list(E.iter_records(_dump(tmp_path / "py.json", py))) == RECS
    bad = pretty.replace('"sentence": "جملة رقم 2', '"sentence": جملة رقم 2', 1)   # broken in the middle
    with pytest.raises(ValueError) as ei:
        list(E.iter_records(_dump(tmp_path / "bad.json", bad)))
    msg = str(ei.value)
    assert "line" in msg and "2 records were read" in msg and "جملة رقم 2" in msg


def test_iter_records_sniffs_real_format(tmp_path):
    import gzip, zipfile
    pretty = json.dumps(RECS, ensure_ascii=False, indent=2).encode("utf-8")
    (tmp_path / "a.json.gz").write_bytes(gzip.compress(pretty))
    assert list(E.iter_records(str(tmp_path / "a.json.gz"))) == RECS
    (tmp_path / "b.json").write_bytes(gzip.compress(pretty))                    # gzip hiding behind .json
    assert list(E.iter_records(str(tmp_path / "b.json"), stream_over_mb=50)) == RECS
    with zipfile.ZipFile(tmp_path / "c.json", "w") as z:
        z.writestr("inner.json", pretty)
    with pytest.raises(ValueError, match="ZIP archive"):
        list(E.iter_records(str(tmp_path / "c.json")))
    (tmp_path / "d.json").write_text("\r\n<!DOCTYPE html><html><body>Log in</body></html>", encoding="utf-8")
    with pytest.raises(ValueError, match="HTML web page"):
        list(E.iter_records(str(tmp_path / "d.json")))
    (tmp_path / "e.json").write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        list(E.iter_records(str(tmp_path / "e.json")))
    conll = "# sent 1\nMessi\tB-PER\tQ615\nmarka\tO\t_\nm3a\tO\t_\nPSG\tB-ORG\tQ483020\n\nخويا\tO\tO\nسوناطراك\tB-ORG\tQ1090717\n"
    (tmp_path / "f.txt").write_text(conll, encoding="utf-8")
    recs = list(E.iter_records(str(tmp_path / "f.txt")))
    assert recs[0] == {"tokens": ["Messi", "marka", "m3a", "PSG"], "tags": ["B-PER", "O", "O", "B-ORG"],
                       "qids": ["Q615", None, None, "Q483020"]}
    rows = E.convert_records(recs, E.detect_schema(recs))
    assert [(r["mention"], r["gold"]) for r in rows] == [("Messi", "Q615"), ("PSG", "Q483020"), ("سوناطراك", "Q1090717")]
    (tmp_path / "g.conll.gz").write_bytes(gzip.compress(conll.encode("utf-8")))
    assert list(E.iter_records(str(tmp_path / "g.conll.gz"))) == recs


def test_elner_dz_real_format(tmp_path):
    recs = [{"id": 1, "text": "meme dangereux kima blida fiparout",
             "entities": [{"entity": "blida", "label": "LOC", "start": 20, "end": 25, "wikidata_id": "Q216990"}]},
            {"id": 2, "text": "rani fi dar lyoum", "entities": []},                      # empty list used to crash
            {"id": 3, "text": "Messi marka but zwin m3a PSG lbare7",
             "entities": [{"entity": "Messi", "label": "PER", "start": 0, "end": 5, "wikidata_id": "Q615"},
                          {"entity": "PSG", "label": "ORG", "start": 24, "end": 27, "wikidata_id": None}]}]
    sch = E.detect_schema(recs)
    assert sch == {"layout": "nested", "text": "text", "entities": "entities", "mention": "entity", "qid": "wikidata_id",
                   "type": "label", "start": "start", "end": "end", "id": "id"}
    rows = E.convert_records(recs, sch)
    assert [(r["id"], r["mention"], r["gold"], r["type"]) for r in rows] == [
        ("1-0", "blida", "Q216990", "LOC"), ("3-0", "Messi", "Q615", "PER"), ("3-1", "PSG", "NIL", "ORG")]


def test_latin_mentions_search_en_fr(tmp_path):
    agent, wd, *_ = make([], tmp_path)
    cands = agent.candidates(E.Mention("Messi marka m3a PSG", "PSG"))
    langs = {p["language"] for u, p in wd.log if p.get("action") == "wbsearchentities"}
    assert {"en", "fr"} <= langs
    agent2, wd2, *_ = make([], tmp_path / "b")
    agent2.candidates(E.Mention("تعد باريس مدينة", "باريس"))
    assert {p["language"] for u, p in wd2.log if p.get("action") == "wbsearchentities"} == {"ar"}
    assert E.label_match("barcelona", {"label_ar": "برشلونة", "label_fr": "Barcelone", "label_en": "Barcelona"})


def test_errors_dataframe_labels(tmp_path):
    wd = FakeWD()
    recs = [{"id": "1", "mention": "باريس", "gold": "Q47899", "qid": "Q90", "initial_qid": "Q90", "candidates": ["Q90", "Q47899"],
             "trace": [{"step": "verify", "issues": [{"check": "TYPE"}]}], "text": "تعد باريس مدينة", "category": "tail"},
            {"id": "3", "mention": "x", "gold": "Q90", "qid": "Q90"}]
    df = E.errors_dataframe(recs, str(tmp_path / "err.csv"), wd)
    assert list(df["id"]) == ["1"] and df.loc[0, "gold_label"] == "باريس هيلتون" and df.loc[0, "agent_desc"] == "capital of France"
    assert df.loc[0, "checks_fired"] == "TYPE" and df.loc[0, "category"] == "tail"


def _run(rows):
    return [{"id": str(i), "mention": "m", "text": f"t{i}", "gold": g, "qid": a, "initial_qid": ini or a}
            for i, g, a, ini in rows]


def test_export_review_groups_audit_and_merge(tmp_path):
    agent = _run([(1, "NIL", "Q90", None), (2, "Q1", "Q90", None), (3, "Q90", "Q1", None), (4, "Q90", "NIL", None),
                  (5, "Q90", "Q3884", None)] + [(i, "Q90", "Q90", None) for i in range(10, 30)])
    search = _run([(1, "NIL", "Q5", None), (2, "Q1", "Q90", None), (3, "Q90", "Q90", None), (4, "Q90", "Q7", None),
                   (5, "Q90", "Q8", None)] + [(i, "Q90", "Q90", None) for i in range(10, 30)])
    path = str(tmp_path / "rev.csv")
    df = E.export_review(agent, path, FakeWD(), search=search, audit=5)
    groups = dict(zip(df["id"], df["group"]))
    assert [groups[i] for i in "12345"] == list(E.REVIEW_GROUPS[:5])
    assert sum(g.startswith("9") for g in df["group"]) == 5 and df.loc[0, "id"] == "1"
    assert df.set_index("id").loc["2", "agent_label"] == "باريس"
    df.loc[df["id"] == "2", "verdict"] = "gold_wrong"                    # old verdict name is mapped
    df.to_csv(path, index=False, encoding="utf-8-sig")
    df2 = E.export_review(agent, path, FakeWD(), search=search, audit=5)
    assert df2.set_index("id").loc["2", "verdict"] == "agent"


def test_rescore_with_corrected_gold(tmp_path):
    agent = _run([(1, "NIL", "Q90", None), (2, "Q1", "Q90", "Q1"), (3, "Q2", "Q2", "NIL"), (4, "Q3", "Q3", None), (5, "Q4", "Q9", None)])
    search = _run([(1, "NIL", "Q90", None), (2, "Q1", "Q1", None), (3, "Q2", "Q2", None), (4, "Q3", "Q8", None), (5, "Q4", "Q4", None)])
    path = str(tmp_path / "rev.csv")
    df = E.export_review(agent, path, None, search=search, audit=2, seed=1).set_index("id")
    df.loc["1", "verdict"] = "agent"                                   # gold NIL was wrong: entity exists
    df.loc["2", "verdict"] = "gold+agent"                              # both acceptable
    df.loc["5", "verdict"] = "none"
    df.loc["5", "correct_qid"] = "Q4"                                  # typed: equals gold -> gold right, both wrong
    for i in df.index:
        if df.loc[i, "group"].startswith("9"):
            df.loc[i, "verdict"] = "none" if i == "3" else "gold"     # audit finds 1 wrong agreement (if sampled)
    df.reset_index().to_csv(path, index=False, encoding="utf-8-sig")
    acc, unclear, summ = E.reviewed_gold(path)
    assert acc["1"] == {"Q90"} and acc["2"] == {"Q1", "Q90"} and acc["5"] == {"Q4"} and not unclear
    assert summ["reviewed"] == 3 and summ["gold wrong"] == 1 and summ["both acceptable"] == 1 and summ["gold right"] == 1
    t = E.rescore({"search": search, "agent": agent}, path, ref="search", n_boot=100)
    audited_wrong = "3" in df.index and df.loc["3", "group"].startswith("9")
    exp_agent = (1 + 1 + (0 if audited_wrong else 1) + 1 + 0) / 5          # ids 1,2,(3),4 right; 5 wrong
    assert t.loc["agent", "accuracy"] == round(exp_agent, 4)
    assert t.loc["agent", "broken"] == 0 and t.loc["agent", "fixed"] == (0 if audited_wrong else 1)
    assert t.loc["search", "accuracy"] == round((1 + 1 + (0 if audited_wrong else 1) + 0 + 1) / 5, 4)


def test_review_widget_saves_verdicts(tmp_path):
    agent = _run([(i, "Q47899", "Q90", None) for i in range(3)])
    path = str(tmp_path / "r.csv")
    E.export_review(agent, path, FakeWD(), audit=0)
    box = E.review(path)

    def buttons(b):
        return {w.description: w for row in b.children[1:] for w in row.children if hasattr(w, "description")}, \
               [w for row in b.children[1:] for w in row.children if not hasattr(w, "on_click")][0]
    btn, qbox = buttons(box)
    assert btn["search ✓"].disabled                                     # no search column
    btn["agent ✓"].click()
    btn["skip"].click()
    qbox.value = "q123"
    btn["none ✓"].click()
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert df["verdict"].tolist() == ["agent", "", "none"] and df["correct_qid"].tolist() == ["", "", "Q123"]
    btn2, _ = buttons(E.review(path))                                    # resumes with the skipped row only
    btn2["gold ✓"].click()
    assert pd.read_csv(path, dtype=str, keep_default_na=False)["verdict"].tolist() == ["agent", "gold", "none"]


# ---------------- fair comparison & analysis ----------------
def _rec(i, gold, qid, init=None, checks=(), cat=None):
    return {"id": str(i), "mention": "m", "gold": gold, "qid": qid, "initial_qid": init or qid, "candidates": [gold, qid],
            "trace": [{"step": "verify", "issues": [{"check": c} for c in checks]}] if checks else [], "category": cat}


def test_mcnemar_and_sign_test():
    assert E.mcnemar_p(0, 0) == 1.0 and E.mcnemar_p(5, 5) == 1.0
    assert E.mcnemar_p(10, 0) == round(2 / 2 ** 10, 6)
    m = E.compute_metrics([_rec(1, "Q1", "Q1", "Q2", ["TYPE"]), _rec(2, "Q1", "Q1"), {"id": "3", "qid": "ERROR", "gold": "Q1"}])
    assert m["n"] == 2 and m["errors (excluded)"] == 1 and m["fixed (wrong→right)"] == 1 and m["sign_test_p (fixed vs broken)"] == 1.0


def test_paired_table_uses_common_mentions():
    a = [_rec(i, "Q1", "Q1" if i < 8 else "Q2") for i in range(10)]                            # 8/10
    b = [_rec(i, "Q1", "Q1") for i in range(10)]                                                  # 10/10
    b[9] = {"id": "9", "qid": "ERROR", "gold": "Q1"}                                              # b failed on 9
    df = E.paired_table({"a": a, "b": b}, ref="a", n_boot=200)
    assert df.loc["a", "n"] == 9 and df.loc["b", "n"] == 9                                        # same 9 mentions
    stale = E.paired_table({"a": a, "b": b, "old": a[:3]}, ref="a", n_boot=50)                   # partial run is skipped
    assert list(stale.index) == ["a", "b"] and stale.loc["a", "n"] == 9
    assert df.loc["b", "wins_vs_ref"] == 1 and df.loc["b", "losses_vs_ref"] == 0
    assert df.loc["a", "p_vs_ref"] == 1.0 and df.attrs["ref"] == "a"


def test_fix_attribution():
    recs = [_rec(1, "Q1", "Q1", "Q2", ["TYPE", "LABEL"]), _rec(2, "Q1", "Q3", "Q1", ["NIL"]),
            _rec(3, "Q1", "Q2", "Q2", ["TYPE"]), _rec(4, "Q1", "Q1", "Q1", ["COHERENCE"]), _rec(5, "Q1", "Q1")]
    df = E.fix_attribution(recs)
    assert df.loc["TYPE", "fired"] == 2 and df.loc["TYPE", "fixed"] == 1 and df.loc["TYPE", "still_wrong"] == 1
    assert df.loc["LABEL", "fired_first"] == 0 and df.loc["NIL", "broken"] == 1 and df.loc["COHERENCE", "kept_right"] == 1
    assert df.loc["TYPE", "fix_rate"] == 0.5


def test_gold_source_check_and_review_columns(tmp_path):
    agent = [_rec(1, "Q1", "Q1"), _rec(2, "Q1", "Q2"), _rec(3, "Q1", "Q1"), _rec(4, "Q1", "Q3")]
    search = [_rec(1, "Q1", "Q1"), _rec(2, "Q1", "Q1"), _rec(3, "Q1", "Q5"), _rec(4, "Q1", "Q3")]
    s = E.gold_source_check(agent, search)
    assert s.to_dict() == {"all three agree": 1, "gold = agent ≠ search": 1, "gold = search ≠ agent": 1,
                           "agent = search ≠ gold": 1, "all differ": 0}
    df = E.errors_dataframe(agent, None, None, others={"search_top1": search})
    assert list(df["search_top1"]) == ["Q1", "Q3"]


def test_evaluate_retries_errors_and_stops_on_fatal(tmp_path):
    rules = [(first("باريس"), {"action": "select", "qid": "Q90", "confidence": 0.95})]
    agent, _, llm, _ = make(rules, tmp_path)
    rows = [{"id": 1, "text": "تعد باريس مدينة", "mention": "باريس", "gold": "Q90"},
            {"id": 2, "text": "زار باريس أمس", "mention": "باريس", "gold": "Q90"}]
    out = tmp_path / "runs"
    out.mkdir()
    (out / "r.jsonl").write_text(json.dumps({"id": "1", "qid": "ERROR", "gold": "Q90", "reason": "boom"}) + "\n", encoding="utf-8")
    m, recs = E.evaluate(agent, rows, "r", str(out), progress=False)
    assert m["n"] == 2 and m["errors (excluded)"] == 0 and m["accuracy"] == 1.0           # failed row was retried

    class Quota(FakeLLM):
        def _raw(self, system, user):
            raise Exception("429 RESOURCE_EXHAUSTED quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier")
    agent.llm = Quota([])
    with pytest.raises(E.FatalLLMError, match="Daily quota"):
        E.evaluate(agent, rows, "r2", str(out), progress=False)


# ---------------- pooled, blind review ----------------
def test_pooled_review_carry_blind_rescore_and_kappa(tmp_path):
    # ids: 1 gold NIL (agent links Q90, nocorr links Q90); 2 agent=gold, nocorr differs (unreviewed before!);
    #      3 agent≠gold (old review: agent right); 4..13 everyone agrees
    agent = _run([(1, "NIL", "Q90", None), (2, "Q5", "Q5", None), (3, "Q1", "Q2", "Q1")] + [(i, "Q9", "Q9", None) for i in range(4, 14)])
    nocor = _run([(1, "NIL", "Q90", None), (2, "Q5", "Q6", None), (3, "Q1", "Q1", None)] + [(i, "Q9", "Q9", None) for i in range(4, 14)])
    search = _run([(1, "NIL", "Q90", None), (2, "Q5", "Q5", None), (3, "Q1", "Q2", None)] + [(i, "Q9", "Q9", None) for i in range(4, 14)])
    old_csv = str(tmp_path / "old.csv")
    old = E.export_review(agent, old_csv, None, search=search, audit=0).set_index("id")
    old.loc["1", "verdict"], old.loc["3", "verdict"] = "agent", "agent"
    old.reset_index().to_csv(old_csv, index=False, encoding="utf-8-sig")

    runs = {"search": search, "llm_no_correction": nocor, "full": agent}
    path = str(tmp_path / "pooled.csv")
    df = E.export_pooled_review(runs, path, FakeWD(), audit=3, carry_from=old_csv)
    d = df.set_index("id")
    assert set(d.index) >= {"1", "2", "3"} and len(d) == 6 and (d["group"] == "audit").sum() == 3
    assert d.loc["1", "status"] == "done" and d.loc["1", "accepted"] == "Q90" and d.loc["1", "carried"] == "yes"
    assert d.loc["3", "accepted"] == "Q2"
    assert d.loc["2", "status"] == "" and set(d.loc["2", "options"].split("|")) == {"Q5", "Q6"}   # pooled: new row
    assert d.loc["1", "group"] == "gold NIL"

    box = E.review_blind(path, "A")
    h = box._elagent
    import pandas as pd
    while True:
        cur = pd.read_csv(path, dtype=str, keep_default_na=False)
        left = cur[cur["status"] == ""]
        if not len(left):
            break
        r = left.iloc[0]
        opts = r["options"].split("|")
        visible = [b for b in h["boxes"] if b.layout.display != "none"]
        assert len(visible) == len(opts) and "gold" not in "".join(b.description for b in visible).lower()
        want = "Q6" if r["id"] == "2" else json.loads(r["sources"])["gold"]           # id 2: gold Q5 wrong, Q6 right
        visible[opts.index(want)].value = True
        h["btns"]["save ✓"].click()
    acc, unclear, summ = E.pooled_gold(path)
    assert acc["2"] == {"Q6"} and acc["1"] == {"Q90"} and summ["disagreement"]["gold wrong"] == 2
    assert summ["audit"]["gold wrong"] == 0 and summ["gold NIL"]["gold wrong rate"] == 1.0
    t = E.rescore_pooled(runs, path, ref="llm_no_correction", n_boot=50)
    # nocorr: right on 2 (Q6, unreviewed before pooling) but wrong on 3; full: the reverse
    assert t.loc["llm_no_correction", "accuracy"] == round(12 / 13, 4)
    assert t.loc["full", "accuracy"] == round(12 / 13, 4)
    assert (t.loc["full", "wins_vs_ref"], t.loc["full", "losses_vs_ref"]) == (1, 1)

    # re-export keeps the new verdicts
    E.export_pooled_review(runs, path, FakeWD(), audit=3, carry_from=old_csv)
    assert E.pooled_gold(path)[0]["2"] == {"Q6"}

    # second annotator + kappa
    b_path = str(tmp_path / "b.csv")
    E.agreement_sample(path, b_path, n=10)
    bdf = pd.read_csv(b_path, dtype=str, keep_default_na=False)
    bdf["status"] = "done"
    bdf["accepted"] = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("id").loc[bdf["id"], "accepted"].values
    bdf.loc[bdf["id"] == "2", "accepted"] = "Q5"                    # disagree on one row
    bdf.to_csv(b_path, index=False, encoding="utf-8-sig")
    ag = E.agreement(path, b_path)
    assert ag["rows"] == 6 and ag["exact agreement"] == round(5 / 6, 4) and 0 < ag["cohen_kappa"] < 1


def test_wilson():
    lo, hi = E._wilson(70, 850)
    assert 0.06 < lo < 0.0824 < hi < 0.11
    assert E._wilson(0, 40)[0] == 0.0 and 0.08 < E._wilson(0, 40)[1] < 0.1


def test_pooled_review_adding_a_model_reopens_only_new_answers(tmp_path):
    a = _run([(1, "Q1", "Q2", None)] + [(i, "Q9", "Q9", None) for i in range(2, 30)])
    b = _run([(1, "Q1", "Q2", None)] + [(i, "Q9", "Q9", None) for i in range(2, 30)])
    path = str(tmp_path / "p.csv")
    E.export_pooled_review({"a": a, "b": b}, path, None, audit=5)
    import pandas as pd
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df["status"], df["accepted"] = "done", df["options"].str.split("|").str[0]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    audit_before = set(df.loc[df["group"] == "audit", "id"])
    # a new model disagrees on mention 1 (new answer Q3) and on mention 2 (was maybe audited)
    c = _run([(1, "Q1", "Q3", None), (2, "Q9", "Q7", None)] + [(i, "Q9", "Q9", None) for i in range(3, 30)])
    df2 = E.export_pooled_review({"a": a, "b": b, "a@new": c}, path, None, audit=5).set_index("id")
    assert df2.loc["1", "status"] == "" and set(df2.loc["1", "options"].split("|")) == {"Q1", "Q2", "Q3"}
    assert df2.loc["2", "status"] == "" and df2.loc["2", "group"] == "disagreement"
    kept = set(df2.index[df2["group"] == "audit"])
    assert len(kept) == 5 and len(kept & audit_before) >= 4                       # audit sample stays stable
    assert all(df2.loc[i, "status"] == "done" for i in kept & audit_before)


def test_run_ablations_tag(tmp_path):
    rules = [(first("باريس"), {"action": "select", "qid": "Q90", "confidence": 0.95})]
    agent, wd, llm, _ = make(rules, tmp_path)
    rows = [{"id": 1, "text": "تعد باريس مدينة", "mention": "باريس", "gold": "Q90"}]
    out = str(tmp_path / "runs")
    E.run_ablations(rows, agent.cfg, wd, llm, names=["llm_no_correction"], out_dir=out)
    df = E.run_ablations(rows, agent.cfg, wd, llm, names=["llm_no_correction"], out_dir=out, tag="@m2")
    assert list(df.index) == ["llm_no_correction@m2"]
    assert os.path.exists(os.path.join(out, "llm_no_correction@m2.jsonl")) and os.path.exists(os.path.join(out, "ablations@m2.csv"))


# ---------------- release helpers ----------------
def test_gzip_cache_and_rows(tmp_path):
    c = E.JsonCache(str(tmp_path / "c.json.gz"))
    c.set("k", {"a": "باريس"})
    c.dirty = 1
    c.save()
    assert E.JsonCache(str(tmp_path / "c.json.gz")).get("k") == {"a": "باريس"}
    import gzip
    with gzip.open(tmp_path / "r.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps({"id": "1", "qid": "Q1", "gold": "Q1"}) + "\n")
    assert E.load_rows(str(tmp_path / "r.jsonl.gz")) == [{"id": "1", "qid": "Q1", "gold": "Q1"}]


def test_corrected_dataset_paired_by_and_restore(tmp_path):
    rows = [{"id": str(i), "doc_id": f"d{i}", "text": f"نص رقم {i} فيه باريس", "mention": "باريس", "start": 8, "end": 13,
             "type": "LOC", "category": "tail" if i % 2 else "head", "gold": g}
            for i, g in enumerate(["NIL", "Q5", "Q1", "Q9", "Q9", "Q9"])]
    agent = _run([(0, "NIL", "Q90", None), (1, "Q5", "Q5", None), (2, "Q1", "Q2", None), (3, "Q9", "Q9", None),
                  (4, "Q9", "Q9", None), (5, "Q9", "Q9", None)])
    nocor = _run([(0, "NIL", "Q90", None), (1, "Q5", "Q6", None), (2, "Q1", "Q1", None), (3, "Q9", "Q9", None),
                  (4, "Q9", "Q9", None), (5, "Q9", "Q9", None)])
    for recs in (agent, nocor):
        for r in recs:
            r["category"] = "tail" if int(r["id"]) % 2 else "head"
    path = str(tmp_path / "p.csv")
    df = E.export_pooled_review({"llm_no_correction": nocor, "full": agent}, path, None, audit=1)
    want = {"0": "Q90", "1": "Q5|Q6", "2": "unclear"}
    for k in df.index:
        i = df.loc[k, "id"]
        if i in want and want[i] == "unclear":
            df.loc[k, "status"] = "unclear"
        elif i in want:
            df.loc[k, "status"], df.loc[k, "accepted"] = "done", want[i]
        else:
            df.loc[k, "status"], df.loc[k, "accepted"] = "done", "Q9"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    audited = [i for i in df["id"] if df.set_index("id").loc[i, "group"] == "audit"][0]
    data = E.corrected_dataset(rows, path, include_text=False)
    by = {r["id"]: r for r in data}
    assert by["0"]["label_status"] == "corrected" and by["0"]["gold_corrected"] == ["Q90"]
    assert by["1"]["label_status"] == "alternatives_acceptable" and by["1"]["gold_corrected"] == ["Q5", "Q6"]
    assert by["2"]["label_status"] == "unclear" and by["2"]["gold_corrected"] is None
    assert by[audited]["label_status"] == "audited_confirmed"
    assert sum(r["label_status"] == "unreviewed_agreement" for r in data) == 2 and "text" not in by["0"]
    acc, unclear = E.corrected_labels(data)
    assert unclear == {"2"} and acc["1"] == {"Q5", "Q6"}
    t = E.paired_table({"llm_no_correction": nocor, "full": agent}, ref="llm_no_correction", acceptable=acc, exclude=unclear, n_boot=20)
    assert t.loc["full", "accuracy"] == 1.0 and t.loc["llm_no_correction", "accuracy"] == 1.0
    pb = E.paired_by({"llm_no_correction": nocor, "full": agent}, "category", ref="llm_no_correction", acceptable=acc, exclude=unclear)
    assert set(pb.index.get_level_values(0)) == {"head", "tail"} and pb.loc[("tail", "full"), "n"] == 3
    src = tmp_path / "elner.json"
    src.write_text(json.dumps([{"id": f"d{i}", "text": r["text"], "entities": [{"entity": "باريس", "wikidata_id": "Q90"}]}
                               for i, r in enumerate(rows)], ensure_ascii=False), encoding="utf-8")
    E.restore_texts(data, str(src))
    assert by["3"]["text"] == rows[3]["text"]


def test_jsonlines_records_with_list_fields_are_not_unwrapped(tmp_path):
    recs = [{"id": str(i), "qid": "Q1", "trace": [{"step": "verify", "issues": []}]} for i in range(3)]
    text = "\n".join(json.dumps(r) for r in recs)
    (tmp_path / "a.json").write_text(text, encoding="utf-8")
    import gzip
    (tmp_path / "b.json.gz").write_bytes(gzip.compress(text.encode("utf-8")))
    assert list(E.iter_records(str(tmp_path / "a.json"))) == recs
    assert list(E.iter_records(str(tmp_path / "b.json.gz"))) == recs                 # streamed (gzip)
    assert list(E.iter_records(str(tmp_path / "a.json"), stream_over_mb=0)) == recs   # streamed (plain)
    wrapper = {"meta": {"v": 1}, "data": recs}
    (tmp_path / "w.json").write_text(json.dumps(wrapper), encoding="utf-8")
    assert list(E.iter_records(str(tmp_path / "w.json"))) == recs
    assert list(E.iter_records(str(tmp_path / "w.json"), stream_over_mb=0)) == recs


# ---------------- revision analyses ----------------
CRITIC = lambda u: "Judge from the text and your general knowledge" in u


def test_parse_json_strips_think_blocks():
    assert E.parse_json('<think>maybe {"x": 0}?</think>\n{"action": "nil"}') == {"action": "nil"}


def test_critic_only_reviews_every_answer(tmp_path):
    rules = [(lambda u: CRITIC(u) and "Q90" in u, {"correct": False, "reason": "the text is about a person"}),
             (lambda u: CRITIC(u) and "Q47899" in u, {"correct": True, "reason": "fine"}),
             (lambda u: CRITIC(u) and "linked to nothing" in u, {"correct": True, "reason": "a private person"}),
             (retry("باريس", "CRITIC"), {"action": "select", "qid": "Q47899", "confidence": 0.9}),
             (first("باريس"), {"action": "select", "qid": "Q90", "confidence": 0.9}),
             (first("سمير بن عيسى"), {"action": "nil", "confidence": 0.8})]
    agent, _, llm, _ = make(rules, tmp_path, **E.ABLATIONS["critic_only"])
    r = agent.link(E.Mention("غنّت باريس في الحفل", "باريس", "PER"))
    assert (r["initial_qid"], r["qid"], r["flagged"]) == ("Q90", "Q47899", False)
    checks = [i["check"] for t in r["trace"] if t.get("step") == "verify" for i in t["issues"]]
    assert checks == ["CRITIC"]                                           # no Wikidata checks in this variant
    r2 = agent.link(E.Mention("قال المهندس سمير بن عيسى", "سمير بن عيسى", "PER"))
    assert r2["qid"] == "NIL" and sum(CRITIC(p) and "linked to nothing" in p for p in llm.prompts) == 1
    agent2, _, llm2, _ = make(rules, tmp_path / "off")                  # default config: no critic calls
    agent2.link(E.Mention("غنّت باريس في الحفل", "باريس", "PER"))
    assert not any(CRITIC(p) for p in llm2.prompts)
    assert any("A reviewer who read the text raised these doubts" in p for p in llm.prompts)


def test_exact_then_popular_baseline(tmp_path):
    exact = E.compact_entity(ent("Q500", "البليدة", "Blida", None, [], ["Q515"], {}, 10))
    popular = E.compact_entity(ent("Q501", "ولاية البليدة", "Blida Province", None, [], ["Q515"], {}, 100))
    for kind, want in (("exact_popularity", "Q500"), ("popularity", "Q501")):
        agent, *_ = make([], tmp_path / kind, baseline=kind)
        agent.candidates = lambda m, extra_queries=(): [popular, exact]
        assert agent.link(E.Mention("زرت البليدة", "البليدة"))["qid"] == want


def _traced(i, gold, final, init, nil_msg=None):
    tr = [{"step": "verify", "issues": [{"check": "NIL", "msg": nil_msg, "hard": False}]}] if nil_msg else []
    return {"id": str(i), "mention": "m", "gold": gold, "qid": final, "initial_qid": init, "trace": tr}


def test_nil_rule_analysis():
    recs = [_traced(1, "Q5", "NIL", "NIL", "You answered NIL, but ...: Q5 (a), Q6 (b)."),
            _traced(2, "Q7", "Q7", "NIL", "You answered NIL, but ...: Q7 (c)."),
            _traced(3, "NIL", "NIL", "NIL", "You answered NIL, but ...: Q8 (d)."),
            _traced(4, "Q9", "Q9", "Q9")]
    df, s = E.nil_rule_analysis(recs)
    assert s["NIL check fired"] == 3 and s["agent kept NIL"] == 2
    assert s["  kept NIL, NIL right and rule wrong"] == 1 and s["  kept NIL, rule right and agent wrong"] == 1
    assert s["  switched to the rule's candidate"] == 1 and (s["agent right (of fired)"], s["rule right (of fired)"]) == (2, 2)
    _, s2 = E.nil_rule_analysis(recs, acceptable={"3": {"Q8"}})           # review: mention 3 is Q8, not NIL
    assert s2["rule right, agent wrong"] == 2 and s2["agent right, rule wrong"] == 0


def test_apply_nil_rule(tmp_path):
    wd = FakeWD(str(tmp_path / "c.json"))
    recs = [{"id": "1", "mention": "باريس هيلتون", "type": "PER", "gold": "Q47899", "qid": "NIL", "initial_qid": "NIL",
             "candidates": ["Q90", "Q47899"]},
            {"id": "2", "mention": "باريس", "type": "PER", "gold": "NIL", "qid": "NIL", "initial_qid": "NIL",
             "candidates": ["Q90"]},                                    # exact name, but a city is not a PER
            {"id": "3", "mention": "باريس", "type": "LOC", "gold": "Q90", "qid": "Q90", "initial_qid": "Q90",
             "candidates": ["Q90"]}]
    out = E.apply_nil_rule(recs, wd)
    assert [r["qid"] for r in out] == ["Q47899", "NIL", "Q90"] and out[0]["initial_qid"] == "NIL"
    m = E.compute_metrics(out)
    assert m["accuracy"] == 1.0 and m["fixed (wrong→right)"] == 1


def test_unseen_initial_and_include_initial(tmp_path):
    # mention 1: initial Q3 (never shown), final Q2 accepted -> a repair that rests on an unjudged initial answer
    agent = _run([(1, "Q1", "Q2", "Q3"), (2, "Q4", "Q4", "Q5")] + [(i, "Q9", "Q9", None) for i in range(3, 20)])
    nocor = _run([(1, "Q1", "Q1", None), (2, "Q4", "Q4", None)] + [(i, "Q9", "Q9", None) for i in range(3, 20)])
    runs = {"llm_no_correction": nocor, "full_no_memory": agent}
    path = str(tmp_path / "p.csv")
    import pandas as pd
    df = E.export_pooled_review(runs, path, None, audit=3)
    df["status"], df["accepted"] = "done", ["Q2" if i == "1" else json.loads(s)["gold"] for i, s in zip(df["id"], df["sources"])]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    u = E.unseen_initial(runs, path)
    assert u.loc["full_no_memory", "fixed"] == 2                          # 1 (reviewed) and 2 (not reviewed)
    assert u.loc["full_no_memory", "fixed_unseen_initial"] == 1 and u.loc["full_no_memory", "fixed_not_reviewed"] == 1
    df2 = E.export_pooled_review(runs, path, None, audit=3, include_initial=True).set_index("id")
    assert df2.loc["1", "status"] == "" and "Q3" in df2.loc["1", "options"].split("|")   # reopened, now judged
    assert df2.loc["2", "status"] == "" and set(df2.loc["2", "options"].split("|")) == {"Q4", "Q5"}
    assert "full_no_memory#initial" in json.loads(df2.loc["1", "sources"])


def test_by_script_and_multi_run_summary():
    def run(ok, tag_ok=None):
        return [{"id": str(i), "mention": "blida" if i < 4 else "البليدة", "gold": "Q1", "qid": "Q1" if i in ok else "Q2",
                 "initial_qid": "Q1" if i in (tag_ok if tag_ok is not None else ok) else "Q2", "llm_calls": 1}
                for i in range(10)]
    runs = {"llm_no_correction": run({0, 1, 4, 5, 6}), "full_no_memory": run({0, 1, 2, 4, 5, 6, 7}, {0, 1, 4, 5, 6}),
            "llm_no_correction@r2": run({0, 4, 5, 6, 7}), "full_no_memory@r2": run({0, 1, 4, 5, 6, 7, 8}, {0, 4, 5, 6, 7})}
    t = E.by_script({k: v for k, v in runs.items() if "@" not in k}, ref="llm_no_correction")
    assert t.loc[("latin", "full_no_memory"), "n"] == 4 and t.loc[("latin", "full_no_memory"), "fixed"] == 1
    per, agg = E.multi_run_summary(runs, ["llm_no_correction", "full_no_memory"], ["", "@r2"])
    assert len(per) == 4 and agg.loc["full_no_memory", "runs"] == 2
    assert agg.loc["full_no_memory", "acc_mean"] == 0.7 and agg.loc["full_no_memory", "fixed_total"] == 4
    assert agg.loc["full_no_memory", "diff_vs_ref_mean"] == 0.2 and per["diff_vs_ref"].notna().all()
    assert list(per.loc[per.system == "full_no_memory", "diff_vs_ref"].round(4)) == [0.2, 0.2]
    assert E.mention_script("PSG") == "latin" and E.mention_script("الأهلي") == "arabic" and E.mention_script("CR بلوزداد") == "mixed"


def test_nil_label_timing(tmp_path):
    class DatedWD(FakeWD):
        def _get_json(self, url, params, headers=None):
            if params.get("prop") == "revisions":
                ts = {"Q90": "2012-10-29T00:00:00Z", "Q777": "2026-01-15T00:00:00Z"}.get(params["titles"])
                return {"query": {"pages": [{"title": params["titles"], "revisions": [{"timestamp": ts}] if ts else []}]}}
            return super()._get_json(url, params, headers)
    import pandas as pd
    path = str(tmp_path / "r.csv")
    pd.DataFrame([
        {"id": "1", "mention": "a", "status": "done", "accepted": "Q90", "sources": json.dumps({"gold": "NIL"})},
        {"id": "2", "mention": "b", "status": "done", "accepted": "Q777", "sources": json.dumps({"gold": "NIL"})},
        {"id": "3", "mention": "c", "status": "done", "accepted": "NIL", "sources": json.dumps({"gold": "NIL"})},
        {"id": "4", "mention": "d", "status": "done", "accepted": "Q90", "sources": json.dumps({"gold": "Q5"})},
    ]).to_csv(path, index=False)
    df, counts = E.nil_label_timing(path, DatedWD(str(tmp_path / "c.json")))
    assert counts == {"item existed before cutoff": 1, "item created after cutoff": 1}
    assert df.set_index("id").loc["2", "earliest"] == "2026-01-15"


def test_corpus_profile(tmp_path):
    recs = [{"id": 1, "text": "meme dangereux kima blida", "entities": [{"start": 20, "end": 25, "label": "LOC", "wikidata_id": "Q1"}]},
            {"id": 2, "text": "زرت البليدة و وهران", "entities": [{"start": 4, "end": 11, "label": "LOC", "wikidata_id": "Q1"},
                                                                {"start": 14, "end": 19, "label": "LOC", "wikidata_id": "NIL"}]},
            {"id": 3, "text": "لا شيء هنا", "entities": []}]
    schema = E.detect_schema(recs[:2])
    p = E.corpus_profile(recs, schema)
    assert p["sentences"] == 3 and p["mentions"] == 3 and p["share of mentions labeled NIL"] == round(1 / 3, 4)
    assert p["share of mentions in Latin script"] == round(1 / 3, 4) and p["mentions per sentence with mentions"] == 1.5
    rows = E.convert_records(recs, schema)
    q = E.corpus_profile(rows)
    assert q["mentions"] == 3 and q["sentences"] == 2 and q["type LOC"] == 1.0


def test_fix_attribution_with_corrected_labels():
    recs = [_rec(1, "NIL", "Q5", "NIL", ["NIL"]), _rec(2, "Q1", "Q1", "Q2", ["TYPE"])]
    orig = E.fix_attribution(recs)
    assert orig.loc["NIL", "broken"] == 1 and orig.loc["TYPE", "fixed"] == 1
    corr = E.fix_attribution(recs, acceptable={"1": {"Q5"}})            # review: the NIL label was wrong
    assert corr.loc["NIL", "fixed"] == 1 and corr.loc["NIL", "broken"] == 0
