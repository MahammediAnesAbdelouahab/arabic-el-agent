"""
elagent: an LLM agent that links Arabic and Arabizi entity mentions to Wikidata and verifies its own answers.

Pipeline for every mention:
  1. candidate retrieval: Wikidata search, Arabic Wikipedia titles / redirects / disambiguation pages,
     clitic-aware query variants, and English/French search for Latin-script (Arabizi) mentions;
  2. LLM decision: a QID from the candidate list, NIL, or a new search query;
  3. verification against Wikidata: candidate membership, entity type (P31/P279*), label match,
     NIL challenge, document-level coherence (and an optional role/metonymy critic);
  4. correction loop: failed checks are sent back to the LLM with the evidence.
Also includes evaluation, paired statistics, pooled blind review, release utilities and the analyses
added for the revision (general LLM critic, fixed NIL rule, script breakdown, repeated runs, QID dates).

License: MIT
"""
# %% [config]
ELAGENT_VERSION = "1.1.1"
import os, re, json, time
from dataclasses import dataclass, replace, asdict
from collections import OrderedDict
from typing import Any, Optional

import requests


@dataclass
class Config:
    # Where everything is stored (cache, runs, error memory). On Colab: a folder in your Drive.
    project_dir: str = "/content/drive/MyDrive/arabic_el_agent"

    # ---- LLM backend ----
    llm_backend: str = "gemini"            # "gemini" | "anthropic" | "openai" (any OpenAI-compatible server)
    llm_model: str = "gemini-3.8-flash"     # e.g. "gemini-3.1-flash-lite", "claude-sonnet-5-5", "gpt-..."
    llm_base_url: Optional[str] = None      # only for OpenAI-compatible servers (Groq, vLLM, Ollama, ...)
    temperature: Optional[float] = None     # None = provider default (recommended for Gemini 3.x)
    gemini_thinking_level: Optional[str] = "low"   # keeps latency/cost down; None = model default
    min_llm_interval: float = 0.0           # seconds between LLM calls (raise it for free-tier rate limits)

    # ---- Candidate retrieval ----
    max_candidates: int = 10
    search_limit: int = 7
    use_wikipedia: bool = True              # Arabic Wikipedia titles, redirects, disambiguation pages, search
    expand_queries: bool = True             # ask the LLM for English / full-name queries when retrieval looks weak
    min_candidates: int = 3

    # ---- Self-correction ----
    max_rounds: int = 3                     # 0 = no verification / correction (plain LLM baseline)
    verify_type: bool = True                # NER type vs. Wikidata P31/P279* (SPARQL)
    verify_label: bool = True               # mention vs. Arabic labels/aliases (only when confidence is low)
    verify_nil: bool = True                 # challenge NIL when an exact, type-compatible candidate exists
    verify_coherence: bool = True           # document-level consistency between all links
    verify_role: bool = False               # LLM critic for metonymy: is a PLACE really the referent here?
                                            # (country -> national team, city -> club). Off by default: an ablation.
    verify_critic: bool = False             # general LLM critic (DeepEL-style self-validation) that reviews EVERY
                                            # answer from the text alone; no Wikidata check behind its verdict.
    label_conf_threshold: float = 0.7
    max_searches: int = 2                   # how many extra searches the agent may request per mention

    # ---- Error memory ----
    use_memory: bool = True
    memory_frozen: bool = False             # True = read-only (use this on the TEST set)

    # ---- Baselines (no LLM) ----
    baseline: Optional[str] = None          # None | "search_top1" | "popularity"

    # ---- Misc ----
    context_chars: int = 350
    user_agent: str = "ArabicELAgent/0.1 (research prototype; contact: YOUR_EMAIL)"
    verbose: bool = False


# %% [arabic]
_DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
_PUNCT = re.compile(r"[\"'«»“”()\[\]{}<>،,.:;؛!?؟\-_/\\|]")
_EDGE = re.compile(r"^[\"'«»“”(\[\s]+|[\"'«»“”)\]\s.,،:;؛!?؟]+$")


def strip_diacritics(s: str) -> str:
    return _DIAC.sub("", s or "")


def normalize_ar(s: str) -> str:
    """Aggressive normalization used for MATCHING only (never for querying)."""
    s = strip_diacritics(s)
    s = re.sub("[إأآٱ]", "ا", s)
    s = s.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def tokens(s: str) -> list:
    """Normalized tokens with the definite article al- removed (so al-Jaza'ir ~ Jaza'ir)."""
    return [t[2:] if t.startswith("ال") and len(t) > 3 else t for t in normalize_ar(s).split()]


def search_variants(mention: str):
    """Query variants for an Arabic mention with attached clitics.

    Returns (primary, fallback). Primary variants are safe (conjunctions, bi-al-/li-al-);
    fallback variants strip a single bi-/li-/ka- prefix and are only searched if retrieval is weak,
    because they can also damage real names (Baghdad -> ghdad).
    """
    m = _EDGE.sub("", strip_diacritics(mention).strip()).strip()
    primary, fallback = [], []

    def add(lst, x):
        x = x.strip()
        if len(x) >= 2 and x not in primary and x not in fallback:
            lst.append(x)

    add(primary, m)
    words = m.split()
    if not words:
        return primary, fallback
    first, rest = words[0], words[1:]
    forms = [first]
    if len(first) > 3 and first[0] in "وف":                       # conjunction wa-/fa-: wa-Barshaluna -> Barshaluna
        forms.append(first[1:])
    for w in list(forms):
        if w.startswith("لل") and len(w) > 4:                      # li- + article al-: lil-Jaza'ir -> al-Jaza'ir
            forms.append("ال" + w[2:])
        elif len(w) > 4 and w[0] in "بكف" and w[1:3] == "ال":     # bi-/ka-/fa- + article al-: bil-Qahira -> al-Qahira
            forms.append(w[1:])
    for w in forms[1:]:
        add(primary, " ".join([w] + rest))
    for w in forms:
        if len(w) > 3 and w[0] in "بلك" and not w.startswith(("ال", "لل")):
            add(fallback, " ".join([w[1:]] + rest))                # single bi-/li-/ka- prefix: li-Misr -> Misr (risky: fallback only)
    return primary, fallback


def all_variants(mention: str) -> list:
    p, f = search_variants(mention)
    return p + f


# %% [wikidata]
NER_TYPES = {"PER": "PER", "PERS": "PER", "PERSON": "PER",
             "LOC": "LOC", "LOCATION": "LOC", "GPE": "GPE",
             "ORG": "ORG", "ORGANIZATION": "ORG", "ORGANISATION": "ORG",
             "FAC": "FAC", "FACILITY": "FAC", "EVENT": "EVENT", "EVT": "EVENT"}


def norm_type(t):
    if not t:
        return None
    t = re.sub(r"^[BI]-", "", str(t).strip().upper())
    return NER_TYPES.get(t, t)


_LOC_ROOTS = ["Q2221906", "Q618123", "Q56061", "Q486972", "Q6256", "Q17334923", "Q1048835"]
TYPE_ROOTS = {
    "PER": ["Q5", "Q95074", "Q15632617"],                 # human, fictional character, fictional human
    "LOC": _LOC_ROOTS,                                     # geographic location/feature, admin. entity, settlement, country, location, political territory
    "GPE": _LOC_ROOTS + ["Q7275"],                         # + state
    "ORG": ["Q43229", "Q4830453", "Q7278", "Q847017"],     # organization, business, political party, sports club
    "FAC": ["Q13226383", "Q41176", "Q811979"],             # facility, building, architectural structure
    "EVENT": ["Q1656682", "Q1190554"],                     # event, occurrence
}
BAD_CLASSES = {"Q4167410": "Wikimedia disambiguation page", "Q22808320": "Wikimedia human name disambiguation page",
               "Q4167836": "Wikimedia category", "Q11266439": "Wikimedia template", "Q13406463": "Wikimedia list article"}
FACT_PROPS = {"P17": "country", "P131": "located in", "P106": "occupation", "P27": "citizenship",
              "P569": "born", "P570": "died", "P571": "inception", "P159": "headquarters",
              "P641": "sport", "P118": "league", "P452": "industry", "P39": "position held", "P279": "subclass of"}


class JsonCache:
    """Tiny persistent cache (one JSON file). Saved atomically; survives Colab restarts when on Drive."""

    def __init__(self, path=None):
        self.path, self.data, self.dirty = path, {}, 0
        if path and os.path.exists(path):
            try:
                opener = __import__("gzip").open if path.endswith(".gz") else open
                with opener(path, "rt", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def __contains__(self, k):
        return k in self.data

    def get(self, k, default=None):
        return self.data.get(k, default)

    def set(self, k, v):
        self.data[k] = v
        self.dirty += 1
        if self.dirty >= 200:
            self.save()

    def save(self):
        if not self.path or not self.dirty:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        opener = __import__("gzip").open if self.path.endswith(".gz") else open
        with opener(tmp, "wt", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False)
        os.replace(tmp, self.path)
        self.dirty = 0


def _claim_values(claims, pid, k=3):
    out = []
    for c in claims.get(pid, []):
        if c.get("rank") == "deprecated":
            continue
        dv = (c.get("mainsnak") or {}).get("datavalue")
        if not dv:
            continue
        v, t = dv.get("value"), dv.get("type")
        if t == "wikibase-entityid":
            out.append(v.get("id"))
        elif t == "time":
            out.append((v.get("time") or "")[1:11].lstrip("0") or v.get("time"))
        elif t == "monolingualtext":
            out.append(v.get("text"))
        elif t in ("string", "quantity"):
            out.append(v if t == "string" else v.get("amount"))
        if len(out) >= k:
            break
    return [x for x in out if x]


def compact_entity(e):
    lab, des, al = e.get("labels", {}), e.get("descriptions", {}), e.get("aliases", {})
    claims, sl = e.get("claims", {}), e.get("sitelinks", {})
    facts = {}
    for pid in FACT_PROPS:
        vals = _claim_values(claims, pid, 3)
        if vals:
            facts[pid] = vals
    return {
        "qid": e["id"],
        "label_ar": lab.get("ar", {}).get("value"), "label_en": lab.get("en", {}).get("value"),
        "label_fr": lab.get("fr", {}).get("value"),
        "desc_ar": des.get("ar", {}).get("value"), "desc_en": des.get("en", {}).get("value"),
        "aliases_ar": [a["value"] for a in al.get("ar", [])][:10],
        "aliases_en": [a["value"] for a in al.get("en", [])][:6],
        "aliases_fr": [a["value"] for a in al.get("fr", [])][:6],
        "p31": _claim_values(claims, "P31", 5),
        "facts": facts,
        "sitelinks": len(sl),
        "arwiki": sl.get("arwiki", {}).get("title"),
    }


class WikidataClient:
    API = "https://www.wikidata.org/w/api.php"
    SPARQL = "https://query.wikidata.org/sparql"
    WP_API = "https://{lang}.wikipedia.org/w/api.php"

    def __init__(self, user_agent, cache_path=None, pause=0.05):
        if "YOUR_EMAIL" in user_agent or "example.com" in user_agent:
            print("⚠️  Put your real contact e-mail in Config.user_agent (Wikimedia's API policy asks for it).")
        self.http = requests.Session()
        self.http.headers.update({"User-Agent": user_agent})
        self.cache = JsonCache(cache_path)
        self.pause = pause
        self.n_requests = 0

    # ---- HTTP ----
    def _get_json(self, url, params, headers=None, tries=8):
        """GET with patient retries: Wikimedia answers 429 (slow down), 5xx or 'maxlag' (replication lag) when
        busy, sometimes for minutes. Waits 5, 10, 20, 40, 60, 60... s (or Retry-After), about 5 min in total."""
        last = None
        for attempt in range(tries):
            wait = min(60, 5 * 2 ** attempt)
            try:
                r = self.http.get(url, params=params, headers=headers, timeout=60)
                self.n_requests += 1
                if r.status_code == 429 or r.status_code >= 500:
                    last = f"HTTP {r.status_code}"
                    ra = r.headers.get("Retry-After", "")
                    time.sleep(min(120, float(ra)) if ra.isdigit() else wait)
                    continue
                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and (data.get("error") or {}).get("code") == "maxlag":
                    last = "maxlag (Wikidata replication lag)"
                    time.sleep(wait)
                    continue
                time.sleep(self.pause)
                return data
            except (requests.RequestException, ValueError) as e:
                last = repr(e)[:300]
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"Wikimedia did not answer after {tries} tries ({last}): {url} {params}. "
                           "The server is busy: wait a few minutes and rerun the cell (finished work is kept).")

    def _cached(self, key, fn):
        if key in self.cache:
            return self.cache.get(key)
        v = fn()
        self.cache.set(key, v)
        return v

    # ---- Candidate sources ----
    def search(self, query, lang="ar", limit=7):
        def fn():
            d = self._get_json(self.API, {"action": "wbsearchentities", "search": query, "language": lang,
                                          "uselang": lang, "type": "item", "limit": limit,
                                          "format": "json", "maxlag": 5})
            return [x["id"] for x in d.get("search", [])]
        return self._cached(f"wbsearch|{lang}|{limit}|{query}", fn)

    def wikipedia_title(self, title, lang="ar"):
        """Exact page title (redirects followed) -> {'qid': .., 'disambiguation': bool} or None."""
        def fn():
            d = self._get_json(self.WP_API.format(lang=lang), {
                "action": "query", "titles": title, "redirects": 1, "prop": "pageprops",
                "ppprop": "wikibase_item|disambiguation", "format": "json", "formatversion": 2})
            for p in d.get("query", {}).get("pages", []):
                pp = p.get("pageprops") or {}
                if "disambiguation" in pp:
                    return {"qid": None, "disambiguation": True}
                if pp.get("wikibase_item"):
                    return {"qid": pp["wikibase_item"], "disambiguation": False}
            return None
        return self._cached(f"wptitle|{lang}|{title}", fn)

    def wikipedia_disambig_links(self, title, lang="ar", limit=30):
        """QIDs of articles linked from a disambiguation page (great recall for ambiguous Arabic names)."""
        def fn():
            d = self._get_json(self.WP_API.format(lang=lang), {
                "action": "query", "titles": title, "redirects": 1, "generator": "links", "gplnamespace": 0,
                "gpllimit": limit, "prop": "pageprops", "ppprop": "wikibase_item|disambiguation",
                "format": "json", "formatversion": 2})
            out = []
            for p in d.get("query", {}).get("pages", []):
                pp = p.get("pageprops") or {}
                if pp.get("wikibase_item") and "disambiguation" not in pp:
                    out.append(pp["wikibase_item"])
            return out
        return self._cached(f"wpdis|{lang}|{limit}|{title}", fn)

    def wikipedia_search(self, query, lang="ar", limit=5):
        def fn():
            d = self._get_json(self.WP_API.format(lang=lang), {
                "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": limit,
                "gsrnamespace": 0, "prop": "pageprops", "ppprop": "wikibase_item|disambiguation",
                "format": "json", "formatversion": 2})
            pages = sorted(d.get("query", {}).get("pages", []), key=lambda p: p.get("index", 99))
            return [p["pageprops"]["wikibase_item"] for p in pages
                    if (p.get("pageprops") or {}).get("wikibase_item") and "disambiguation" not in p["pageprops"]]
        return self._cached(f"wpsearch|{lang}|{limit}|{query}", fn)

    # ---- Entity data ----
    def get_entities(self, qids):
        qids = [q for q in dict.fromkeys(qids) if q]
        missing = [q for q in qids if f"ent2|{q}" not in self.cache]
        for i in range(0, len(missing), 20):
            chunk = missing[i:i + 20]
            d = self._get_json(self.API, {"action": "wbgetentities", "ids": "|".join(chunk),
                                          "props": "labels|descriptions|aliases|claims|sitelinks",
                                          "languages": "ar|en|fr", "format": "json", "maxlag": 5})
            for k, e in d.get("entities", {}).items():
                if not e or "missing" in e or "id" not in e:
                    continue
                ce = compact_entity(e)
                src = (e.get("redirects") or {}).get("from", k)
                self.cache.set(f"ent2|{src}", ce)
                self.cache.set(f"ent2|{e['id']}", ce)
                self.cache.set(f"lab|{e['id']}", ce["label_en"] or ce["label_ar"] or e["id"])
            for q in chunk:
                if f"ent2|{q}" not in self.cache:
                    self.cache.set(f"ent2|{q}", None)
        return {q: self.cache.get(f"ent2|{q}") for q in qids if self.cache.get(f"ent2|{q}")}

    def get_labels(self, qids):
        qids = [q for q in dict.fromkeys(qids) if isinstance(q, str) and re.fullmatch(r"Q\d+", q)]
        need = [q for q in qids if f"lab|{q}" not in self.cache]
        for i in range(0, len(need), 50):
            chunk = need[i:i + 50]
            d = self._get_json(self.API, {"action": "wbgetentities", "ids": "|".join(chunk), "props": "labels",
                                          "languages": "en|ar", "format": "json", "maxlag": 5})
            for k, e in d.get("entities", {}).items():
                labs = (e or {}).get("labels", {})
                self.cache.set(f"lab|{k}", labs.get("en", {}).get("value") or labs.get("ar", {}).get("value") or k)
            for q in chunk:
                if f"lab|{q}" not in self.cache:
                    self.cache.set(f"lab|{q}", q)
        return {q: self.cache.get(f"lab|{q}", q) for q in qids}

    def is_instance_of(self, qid, roots, p31=None):
        """True / False / None (unknown, e.g. SPARQL timeout). Uses P31/P279* via SPARQL ASK."""
        if p31 and set(p31) & set(roots):
            return True

        def fn():
            vals = " ".join(f"wd:{r}" for r in roots)
            q = f"ASK {{ VALUES ?root {{ {vals} }} wd:{qid} wdt:P31/wdt:P279* ?root . }}"
            d = self._get_json(self.SPARQL, {"query": q, "format": "json"},
                               headers={"Accept": "application/sparql-results+json"})
            return bool(d.get("boolean"))
        try:
            return self._cached(f"isa|{qid}|{','.join(sorted(roots))}", fn)
        except Exception:
            return None

    def created(self, qid):
        """Timestamp of the first revision of an item ('2013-02-21T05:00:00Z'), i.e. when the item was created.
        Redirects are not followed: a merged QID keeps the creation date of its own page."""
        def fn():
            d = self._get_json(self.API, {"action": "query", "prop": "revisions", "titles": qid,
                                          "rvlimit": 1, "rvdir": "newer", "rvprop": "timestamp",
                                          "format": "json", "formatversion": 2})
            for p in d.get("query", {}).get("pages", []):
                revs = p.get("revisions") or []
                if revs:
                    return revs[0].get("timestamp")
            return None
        return self._cached(f"created|{qid}", fn)


# %% [llm]
def parse_json(txt):
    if not txt:
        return None
    t = re.sub(r"<think>.*?</think>", "", txt, flags=re.S)          # reasoning models (e.g. Qwen3) may emit this
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t.strip())
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


class FatalLLMError(RuntimeError):
    """Errors that retrying cannot fix (bad key, unknown model, daily quota). Stops an evaluation run."""


class LLM:
    """One interface over Gemini / Claude / any OpenAI-compatible endpoint. Always asks for JSON."""

    FATAL = ("api key", "api_key", "401", "403", "permission", "unauthorized", "not found", "404", "invalid model")

    def __init__(self, backend, model, api_key=None, base_url=None, temperature=None,
                 thinking_level=None, min_interval=0.0, system_suffix=""):
        self.backend, self.model, self.temperature = backend, model, temperature
        self.thinking_level, self.min_interval = thinking_level, min_interval
        self.system_suffix = system_suffix      # e.g. " /no_think" to switch off Qwen3's thinking mode
        self.calls, self.seconds, self._last = 0, 0.0, 0.0
        self._no_temp = self._no_thinking = self._no_json_mode = False
        if backend == "gemini":
            from google import genai
            self._client = genai.Client(api_key=api_key)
        elif backend == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif backend == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            raise ValueError(f"Unknown backend {backend}")

    def _raw(self, system, user):
        system = system + self.system_suffix
        temp = None if self._no_temp else self.temperature
        if self.backend == "gemini":
            from google.genai import types
            kw = dict(system_instruction=system, response_mime_type="application/json")
            if temp is not None:
                kw["temperature"] = temp
            if self.thinking_level and not self._no_thinking:
                kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
            r = self._client.models.generate_content(model=self.model, contents=user,
                                                     config=types.GenerateContentConfig(**kw))
            return r.text or ""
        if self.backend == "anthropic":
            kw = dict(model=self.model, max_tokens=1024, system=system,
                      messages=[{"role": "user", "content": user}])
            if temp is not None:
                kw["temperature"] = temp
            r = self._client.messages.create(**kw)
            return "".join(getattr(b, "text", "") for b in r.content)
        kw = dict(model=self.model, messages=[{"role": "system", "content": system},
                                              {"role": "user", "content": user}])
        if temp is not None:
            kw["temperature"] = temp
        if not self._no_json_mode:
            kw["response_format"] = {"type": "json_object"}
        r = self._client.chat.completions.create(**kw)
        return r.choices[0].message.content or ""

    def complete(self, system, user):
        last = None
        for attempt in range(5):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            t0 = time.time()
            try:
                out = self._raw(system, user)
                self.calls += 1
                self.seconds += time.time() - t0
                self._last = time.time()
                return out
            except Exception as e:
                self._last = time.time()
                last, msg = e, str(e).lower()
                if "temperature" in msg and not self._no_temp:
                    self._no_temp = True
                    continue
                if "thinking" in msg and not self._no_thinking:
                    self._no_thinking = True
                    continue
                if "response_format" in msg and not self._no_json_mode:
                    self._no_json_mode = True
                    continue
                if any(s in msg for s in self.FATAL):
                    raise FatalLLMError(str(e)[:500]) from e
                if "perday" in msg.replace(" ", "").replace("_", "") or "daily" in msg:
                    raise FatalLLMError("Daily quota exhausted for this model/key. Resume tomorrow (results are "
                                        "kept on Drive) or use a paid key. Original error: " + str(e)[:300]) from e
                time.sleep(min(60, 5 * 2 ** attempt))   # rate limit / transient error
        raise RuntimeError(f"LLM call failed: {last}")

    def json(self, system, user):
        obj = parse_json(self.complete(system, user))
        if not isinstance(obj, dict):
            obj = parse_json(self.complete(system, user + "\n\nReturn ONLY one valid JSON object."))
        return obj if isinstance(obj, dict) else {}


# %% [memory]
class ErrorMemory:
    """Error book: verified self-corrections become hints (same mention) and distilled rules (general)."""

    def __init__(self, path=None, frozen=False):
        self.path, self.frozen, self.cases, self.rules = path, frozen, [], []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            self.cases, self.rules = d.get("cases", []), d.get("rules", [])

    def save(self):
        if not self.path or self.frozen:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"cases": self.cases, "rules": self.rules}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def add_case(self, mention, ntype, wrong, right, issues, context):
        if self.frozen:
            return
        self.cases.append({"key": " ".join(tokens(mention)), "mention": mention, "type": ntype,
                           "wrong": wrong, "right": right, "checks": sorted({i["check"] for i in issues}),
                           "why": "; ".join(i["msg"] for i in issues)[:300], "context": context[:200]})
        self.save()

    def hints(self, mention, k=3, n_rules=8):
        key = " ".join(tokens(mention))
        same = [c for c in self.cases if c["key"] == key][-k:]
        lines = [f"- In «{c['context']}», «{c['mention']}» was first linked to {c['wrong']} but the verifier "
                 f"rejected it ({', '.join(c['checks'])}); the verified link was {c['right']}." for c in same]
        lines += [f"- {r}" for r in self.rules[:n_rules]]
        return "\n".join(lines)

    def distill(self, llm, max_rules=10):
        """Turn accumulated cases into short general rules (run after a dev-set pass, then freeze)."""
        if not self.cases:
            return []
        sample = "\n".join(f"- mention «{c['mention']}» ({c['type']}): {c['wrong']} -> {c['right']}; checks={c['checks']}; {c['why']}"
                           for c in self.cases[-80:])
        d = llm.json(SYSTEM_PROMPT,
                     "These are verified corrections an Arabic entity-linking agent made to its own mistakes:\n"
                     f"{sample}\n\nWrite at most {max_rules} short, general, reusable rules that would have prevented "
                     'these mistakes. Do not mention specific QIDs. JSON: {"rules": ["..."]}')
        self.rules = [str(r) for r in d.get("rules", [])][:max_rules]
        self.save()
        return self.rules


# %% [agent]
SYSTEM_PROMPT = (
    "You are an entity-linking agent for Arabic text (Modern Standard Arabic, dialects, and Arabizi, i.e. "
    "dialectal Arabic written in Latin letters and digits). "
    "Link the target mention to the single Wikidata item it refers to in this context, or answer NIL if it is "
    "not in Wikidata. Base decisions on the candidate evidence (labels, descriptions, types, facts) and the "
    "context. Never invent QIDs: only use QIDs from the candidate list. Respond with one JSON object only."
)


@dataclass
class Mention:
    text: str
    mention: str
    type: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    id: Any = None


def _issue(check, msg, hard=False, suggest=None):
    d = {"check": check, "msg": msg, "hard": hard}
    if suggest:
        d["suggest"] = suggest          # {"q": query, "lang": "ar"|"en"}: searched automatically before re-deciding
    return d


def label_match(mention, c, exact=False, unknown=None):
    """Does the mention match an Arabic label/alias of candidate c? `unknown` = answer when c has no Arabic names."""
    names = [x for x in [c.get("label_ar")] + list(c.get("aliases_ar", [])) if x]
    if re.search("[A-Za-z]", mention):                      # Arabizi / code-switching
        names += [x for x in [c.get("label_en"), c.get("label_fr")] + list(c.get("aliases_en", []))
                  + list(c.get("aliases_fr", [])) if x]
    if not names:
        return (not exact) if unknown is None else unknown   # unverifiable: don't penalise a selection
    mts = [tokens(v) for v in all_variants(mention)]
    for n in names:
        nt = tokens(n)
        for mt in mts:
            if mt and (mt == nt or (not exact and set(mt) <= set(nt))):
                return True
    return False


class Agent:
    def __init__(self, cfg, wd, llm=None, memory=None, log=print):
        self.cfg, self.wd, self.llm, self.memory, self.log = cfg, wd, llm, memory, log
        self._role_cache = {}

    # ---------- helpers ----------
    def _say(self, *a):
        if self.cfg.verbose:
            self.log(*a)

    def _context(self, m):
        def as_int(x):
            try:
                return None if x is None or x != x else int(x)   # x != x catches NaN from pandas
            except (TypeError, ValueError):
                return None
        t, s, e = m.text, as_int(m.start), as_int(m.end)
        if s is None or e is None or t[s:e] != m.mention:
            s = t.find(m.mention)
            e = s + len(m.mention)
        if s < 0:
            return t[: 2 * self.cfg.context_chars]
        a, b = max(0, s - self.cfg.context_chars), min(len(t), e + self.cfg.context_chars)
        return ("…" if a else "") + t[a:s] + "⟦" + t[s:e] + "⟧" + t[e:b] + ("…" if b < len(t) else "")

    def _enrich(self, qids):
        ents = self.wd.get_entities(qids)
        out = []
        for q in qids:
            c = ents.get(q)
            if c and not (set(c["p31"]) & set(BAD_CLASSES)) and c["qid"] not in {x["qid"] for x in out}:
                out.append(c)
        return out

    def candidates(self, m, extra_queries=()):
        cfg, wd = self.cfg, self.wd
        order = []

        def add(qids):
            for q in qids or []:
                if q and q not in order:
                    order.append(q)

        primary, fallback = search_variants(m.mention)
        disamb = []
        for variants in (primary, fallback):
            if variants is fallback and len(order) >= cfg.min_candidates:
                break
            for v in variants:
                if cfg.use_wikipedia:
                    r = wd.wikipedia_title(v)
                    if r and r.get("qid"):
                        add([r["qid"]])
                    elif r and r.get("disambiguation"):
                        disamb += wd.wikipedia_disambig_links(v)
                add(wd.search(v, "ar", cfg.search_limit))
        if re.search("[A-Za-z]", m.mention):                  # Arabizi / Latin script
            for lang in ("en", "fr"):
                add(wd.search(_EDGE.sub("", m.mention.strip()), lang, cfg.search_limit))
        if cfg.use_wikipedia and primary:
            add(wd.wikipedia_search(primary[0], "ar", 5))
        for q in extra_queries:
            add(wd.search(q["q"], q.get("lang", "en"), cfg.search_limit))
        cands = self._enrich(order)[: cfg.max_candidates]
        if disamb and len(cands) < cfg.max_candidates:
            extra = [c for c in self._enrich(disamb) if c["qid"] not in {x["qid"] for x in cands}]
            extra.sort(key=lambda c: -c["sitelinks"])
            cands += extra[: cfg.max_candidates - len(cands)]
        return cands

    def _labels(self, cands):
        ids = []
        for c in cands:
            ids += c["p31"][:3]
            for vals in c["facts"].values():
                ids += vals
        return self.wd.get_labels(ids)

    def _fmt(self, i, c, labels):
        types = ", ".join(labels.get(t, t) for t in c["p31"][:3]) or "-"
        facts = "; ".join(f"{FACT_PROPS[p]}={', '.join(str(labels.get(v, v)) for v in vals)}"
                          for p, vals in c["facts"].items())
        names = " / ".join(dict.fromkeys(x for x in [c["label_ar"], c["label_en"], c.get("label_fr")] if x)) or "-"
        al = ", ".join(c["aliases_ar"][:4]) or "-"
        desc = " / ".join(x for x in [c["desc_ar"], c["desc_en"]] if x) or "-"
        return (f"[{i}] {c['qid']} | {names} | aliases: {al} | desc: {desc} | type: {types}"
                + (f" | {facts}" if facts else "") + f" | wiki-links: {c['sitelinks']}")

    @staticmethod
    def _norm_decision(d):
        a = str(d.get("action", "")).strip().lower()
        q = str(d.get("qid") or "").strip().upper()
        q = q if re.fullmatch(r"Q\d+", q) else ""
        if not a and q:
            a = "select"
        if a == "select" and not q:
            a = "invalid"
        if a not in ("select", "nil", "search", "confirm", "invalid"):
            a = "invalid"
        try:
            conf = max(0.0, min(1.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        return {"action": a, "qid": q if a == "select" else "", "query": str(d.get("query") or ""),
                "lang": "en" if str(d.get("lang", "")).lower().startswith("en") else "ar",
                "confidence": conf, "reason": str(d.get("reason", ""))[:300]}

    # ---------- LLM steps ----------
    def _expand(self, ctx, m):
        d = self.llm.json(SYSTEM_PROMPT,
                          f"Context: {ctx}\nTarget mention: «{m.mention}»\n"
                          "Give up to 3 search queries likely to find this entity in Wikidata: its standard English "
                          "name, its full name in Arabic, or common spelling variants.\n"
                          'JSON: {"queries": [{"q": "...", "lang": "en" or "ar"}]}')
        qs = []
        for q in d.get("queries", [])[:3]:
            if isinstance(q, dict) and str(q.get("q", "")).strip():
                qs.append({"q": str(q["q"]).strip(), "lang": "en" if str(q.get("lang", "en")).startswith("en") else "ar"})
        return qs

    def _decide(self, ctx, m, ntype, cands, labels, hints="", prev=None, issues=None,
                searches_left=0, allow_confirm=True):
        L = [f"Context: {ctx}", f"Target mention: «{m.mention}»" + (f"   (NER type: {ntype})" if ntype else ""),
             "Candidates:"]
        L += [self._fmt(i + 1, c, labels) for i, c in enumerate(cands)] or ["(no candidates found)"]
        if hints:
            L += ["", "Lessons from earlier verified corrections (apply only if relevant):", hints]
        if prev:
            what = f"select {prev['qid']}" if prev["action"] == "select" else prev["action"]
            only_critic = bool(issues) and all(x["check"] == "CRITIC" for x in issues)
            L += ["", f"Your previous answer: {what} (reason: {prev.get('reason', '')})",
                  "A reviewer who read the text raised these doubts:" if only_critic
                  else "An automatic verifier that checks Wikidata found these problems:"]
            L += [f"- [{x['check']}] {x['msg']}" for x in issues]
            L += ["Re-decide carefully. Do not repeat an answer that was rejected for a hard problem."]
        acts = ['"select" with a "qid" from the candidate list',
                '"nil" if the entity is not among the candidates and probably not in Wikidata']
        if searches_left > 0:
            acts.append('"search" with a new "query" and "lang" ("ar" or "en") if a better query could find it')
        if prev and allow_confirm:
            acts.append('"confirm" to keep your previous answer, only if the reported problems are not real (explain why)')
        L += ["", "Allowed actions: " + "; ".join(acts) + ".",
              'Answer JSON only: {"action": "...", "qid": "...", "query": "...", "lang": "...", '
              '"confidence": 0.0-1.0, "reason": "one short sentence"}']
        return self._norm_decision(self.llm.json(SYSTEM_PROMPT, "\n".join(L)))

    def _add_search(self, st, query, lang, by):
        """Search Wikidata and append new candidates (at most 5) to the agent state."""
        new_ids = self.wd.search(query, lang, self.cfg.search_limit)
        have = {c["qid"] for c in st["cands"]}
        new = [c for c in self._enrich(new_ids) if c["qid"] not in have][:5]
        st["cands"] = st["cands"] + new
        st["labels"] = self._labels(st["cands"])
        st["trace"].append({"step": "search", "query": query, "lang": lang, "by": by, "added": [c["qid"] for c in new]})

    def _role_check(self, m, c):
        """LLM critic (evidence = Wikidata type + description): does a PLACE really fill this role in the sentence?"""
        ctx = self._context(m)
        key = f"{ctx}|{c['qid']}"
        if key in self._role_cache:
            return self._role_cache[key]
        types = ", ".join(self.wd.get_labels(c["p31"][:3]).values()) or "-"
        d = self.llm.json(SYSTEM_PROMPT,
                          f"Context: {ctx}\n"
                          f"The mention «{m.mention}» was linked to the place {c['qid']}: "
                          f"{c['label_ar'] or ''} / {c['label_en'] or ''} — {c['desc_en'] or c['desc_ar'] or ''} (type: {types}).\n"
                          "Does the mention refer to this place itself (its territory, or the country/city acting "
                          "politically, diplomatically, economically or demographically), or is it a metonym for a "
                          "different kind of entity such as a national sports team, a sports club, a company or an institution?\n"
                          "Places are correct for geographic, political, diplomatic, demographic and historical statements. "
                          "Answer fits=false only when the sentence clearly describes something a place cannot do "
                          "(e.g. winning or losing a match, scoring, qualifying for a tournament, being relegated).\n"
                          'JSON: {"fits": true or false, "better_referent": "full name of the real referent if fits is false", '
                          '"lang": "ar" or "en", "reason": "one short sentence"}')
        fits = d.get("fits")
        if isinstance(fits, str):
            fits = fits.strip().lower() not in ("false", "no", "0")
        res = None
        if fits is False:
            better = str(d.get("better_referent") or "").strip()
            lang = "en" if str(d.get("lang", "ar")).lower().startswith("en") else "ar"
            res = _issue("ROLE", f"In this sentence «{m.mention}» does not seem to denote the place {c['qid']} itself"
                         + (f" but rather: {better}" if better else "") + f". {d.get('reason', '')}".rstrip(),
                         suggest={"q": better, "lang": lang} if better else None)
        self._role_cache[key] = res
        return res

    def _critic_check(self, m, d, c=None):
        """General LLM critic in the style of DeepEL's self-validation: it reviews ANY answer (a QID or NIL) from
        the text and its own knowledge. It sees only the chosen item's label and description, never the other
        candidates, and it names no alternative: on 'wrong' the agent re-decides among its own candidates."""
        ctx = self._context(m)
        key = f"critic|{ctx}|{d['qid'] if d['action'] == 'select' else 'NIL'}"
        if key in self._role_cache:
            return self._role_cache[key]
        if d["action"] == "select" and c is not None:
            what = (f"The mention «{m.mention}» was linked to the Wikidata item {c['qid']}: "
                    f"{c.get('label_ar') or ''} / {c.get('label_en') or ''} — {c.get('desc_en') or c.get('desc_ar') or ''}.\n"
                    "Is this the entity the mention refers to in this text?")
        else:
            what = (f"The mention «{m.mention}» was linked to nothing (NIL): the linker judged that it refers to "
                    "nothing that has a Wikidata item. Is NIL right? Answer false if the mention clearly names a "
                    "notable person, place, organization, work or event that Wikidata very likely covers.")
        r = self.llm.json(SYSTEM_PROMPT,
                          f"Context: {ctx}\n{what}\nJudge from the text and your general knowledge.\n"
                          'JSON: {"correct": true or false, "reason": "one short sentence"}')
        ok = r.get("correct")
        if isinstance(ok, str):
            ok = ok.strip().lower() not in ("false", "no", "0")
        res = None
        if ok is False:
            res = _issue("CRITIC", f"A reviewer reading the text doubts this answer: {str(r.get('reason', ''))[:300]}")
        self._role_cache[key] = res
        return res

    def _run_searches(self, d, st, decide_kwargs):
        """Execute 'search' actions: add candidates and ask again. Returns the next non-search decision."""
        while d["action"] == "search":
            if st["searches_left"] <= 0 or not d["query"]:
                decide_kwargs = {**decide_kwargs, "searches_left": 0}
                d = self._decide(st["ctx"], st["m"], st["ntype"], st["cands"], st["labels"], **decide_kwargs)
                if d["action"] == "search":
                    d = {**d, "action": "nil"}
                break
            st["searches_left"] -= 1
            self._add_search(st, d["query"], d["lang"], by="agent")
            decide_kwargs = {**decide_kwargs, "searches_left": st["searches_left"]}
            d = self._decide(st["ctx"], st["m"], st["ntype"], st["cands"], st["labels"], **decide_kwargs)
        return d

    # ---------- verification ----------
    def verify(self, m, ntype, d, cands):
        cfg, by_id, issues = self.cfg, {c["qid"]: c for c in cands}, []
        if d["action"] == "invalid":
            return [_issue("FORMAT", "The answer was not a valid action/QID.", hard=True)]
        if d["action"] == "select":
            c = by_id.get(d["qid"])
            if c is None:
                return [_issue("CANDIDATE", f"{d['qid']} is not in the candidate list (it may be invented).", hard=True)]
            bad = [b for b in c["p31"] if b in BAD_CLASSES]
            if bad:
                issues.append(_issue("NOT_ENTITY", f"{c['qid']} is a {BAD_CLASSES[bad[0]]}, not a real-world entity.", hard=True))
            if cfg.verify_type and ntype in TYPE_ROOTS:
                ok = self.wd.is_instance_of(c["qid"], TYPE_ROOTS[ntype], c["p31"])
                if ok is False:
                    tl = ", ".join(self.wd.get_labels(c["p31"][:3]).values()) or "unknown"
                    issues.append(_issue("TYPE", f"{c['qid']} is an instance of [{tl}], which is not compatible with "
                                         f"the NER type {ntype}. Metonymy can be legitimate (e.g. a country name used "
                                         f"for its national team) — if so, confirm and explain."))
            if cfg.verify_label and not label_match(m.mention, c):
                if d["confidence"] < cfg.label_conf_threshold or issues:
                    issues.append(_issue("LABEL", f"«{m.mention}» does not match any Arabic label/alias of {c['qid']} "
                                         f"({c['label_ar'] or c['label_en']})."))
            if cfg.verify_role and not any(i["hard"] for i in issues) \
                    and self.wd.is_instance_of(c["qid"], TYPE_ROOTS["GPE"], c["p31"]) is True:
                role = self._role_check(m, c)
                if role:
                    issues.append(role)
            if cfg.verify_critic and not any(i["hard"] for i in issues):
                crit = self._critic_check(m, d, c)
                if crit:
                    issues.append(crit)
        elif d["action"] == "nil":
            if cfg.verify_nil:
                strong = [c for c in cands if label_match(m.mention, c, exact=True)]
                if ntype in TYPE_ROOTS:
                    strong = [c for c in strong if self.wd.is_instance_of(c["qid"], TYPE_ROOTS[ntype], c["p31"]) is not False]
                if strong:
                    lst = ", ".join(f"{c['qid']} ({c['label_ar'] or c['label_en']})" for c in strong[:3])
                    issues.append(_issue("NIL", f"You answered NIL, but these candidates match the mention exactly"
                                         f"{' and are compatible with the NER type' if ntype in TYPE_ROOTS else ''}: {lst}."))
            if cfg.verify_critic:
                crit = self._critic_check(m, d)
                if crit:
                    issues.append(crit)
        return issues

    # ---------- main loop ----------
    def _baseline(self, m):
        if self.cfg.baseline == "search_top1":
            ids = []
            for v in all_variants(m.mention):
                ids = self.wd.search(v, "ar", self.cfg.search_limit)
                if ids:
                    break
            q = ids[0] if ids else "NIL"
            cands = ids
        else:                                                  # the agent's own retrieval (no LLM query expansion)
            cs = self.candidates(m)
            pool = cs
            if self.cfg.baseline == "exact_popularity":        # exact name matches first, then the most sitelinks
                pool = [c for c in cs if label_match(m.mention, c, exact=True)] or cs
            q = max(pool, key=lambda c: c["sitelinks"])["qid"] if pool else "NIL"
            cands = [c["qid"] for c in cs]
        return {"qid": q, "initial_qid": q, "candidates": cands, "rounds": 0, "corrected": False,
                "flagged": False, "llm_calls": 0, "trace": []}

    def link(self, m, _resume=None):
        """Link one mention. Returns a dict with the final QID (or 'NIL'), the initial QID, and a full trace."""
        if self.cfg.baseline:
            return self._baseline(m)
        cfg, t0, calls0 = self.cfg, time.time(), self.llm.calls
        ntype = norm_type(m.type)
        ctx = self._context(m)
        hints = self.memory.hints(m.mention) if (self.memory and cfg.use_memory) else ""

        if _resume is None:
            trace = []
            cands = self.candidates(m)
            weak = len(cands) < cfg.min_candidates or not any(label_match(m.mention, c, unknown=False) for c in cands)
            if cfg.expand_queries and weak:
                qs = self._expand(ctx, m)
                trace.append({"step": "expand", "queries": qs})
                if qs:
                    cands = self.candidates(m, qs)
            st = {"m": m, "ctx": ctx, "ntype": ntype, "cands": cands, "labels": self._labels(cands),
                  "searches_left": cfg.max_searches, "trace": trace}
            d = self._decide(ctx, m, ntype, st["cands"], st["labels"], hints=hints, searches_left=cfg.max_searches)
            d = self._run_searches(d, st, {"hints": hints})
            initial, extra = dict(d), []
        else:
            prev, extra = _resume
            cands = self._enrich(prev["candidates"])
            st = {"m": m, "ctx": ctx, "ntype": ntype, "cands": cands, "labels": self._labels(cands),
                  "searches_left": 1, "trace": list(prev["trace"])}
            d, initial = dict(prev["decision"]), dict(prev["initial"])
            calls0 -= prev.get("llm_calls", 0)

        issues, rounds = [], 0
        if cfg.max_rounds > 0:
            for rounds in range(1, cfg.max_rounds + 1):
                issues = self.verify(m, ntype, d, st["cands"]) + (extra if rounds == 1 else [])
                st["trace"].append({"step": "verify", "decision": d, "issues": issues})
                if not issues:
                    break
                hard = any(x["hard"] for x in issues)
                for x in issues:                               # verifier-suggested referents are searched first
                    if x.get("suggest") and x["suggest"].get("q"):
                        self._add_search(st, x["suggest"]["q"], x["suggest"].get("lang", "ar"), by="verifier")
                kw = {"hints": hints, "prev": d, "issues": issues, "allow_confirm": not hard}
                new = self._decide(ctx, m, ntype, st["cands"], st["labels"], searches_left=st["searches_left"], **kw)
                new = self._run_searches(new, st, kw)
                if new["action"] == "confirm" and hard:          # not allowed for hard problems
                    st["trace"].append({"step": "confirm_rejected"})
                    new = dict(d)
                elif new["action"] == "confirm":
                    st["trace"].append({"step": "confirm", "reason": new["reason"]})
                    issues = []
                    break
                d = new
            else:
                issues = self.verify(m, ntype, d, st["cands"])
                st["trace"].append({"step": "verify", "decision": d, "issues": issues})
            if any(x["hard"] for x in issues):                 # could not repair -> abstain
                d = {**d, "action": "nil", "qid": ""}

        final = d["qid"] if d["action"] == "select" else "NIL"
        init_q = initial["qid"] if initial["action"] == "select" else "NIL"
        corrected = final != init_q
        if corrected and not issues and self.memory and cfg.use_memory and not cfg.memory_frozen:
            first = next((t["issues"] for t in st["trace"] if t.get("step") == "verify" and t["issues"]), extra)
            self.memory.add_case(m.mention, ntype, init_q, final, first, ctx)
        by_id = {c["qid"]: c for c in st["cands"]}
        return {"qid": final, "label": (by_id.get(final) or {}).get("label_ar") or (by_id.get(final) or {}).get("label_en"),
                "initial_qid": init_q, "corrected": corrected, "flagged": bool(issues), "rounds": rounds,
                "confidence": d.get("confidence"), "reason": d.get("reason"),
                "candidates": [c["qid"] for c in st["cands"]], "llm_calls": self.llm.calls - calls0,
                "seconds": round(time.time() - t0, 2), "trace": st["trace"], "decision": d, "initial": initial}

    def _coherence(self, text, mentions, results):
        L = []
        for i, (m, r) in enumerate(zip(mentions, results)):
            if r["qid"] == "NIL":
                continue
            c = (self.wd.get_entities([r["qid"]]) or {}).get(r["qid"]) or {}
            desc = c.get("desc_en") or c.get("desc_ar") or ""
            L.append(f"[{i}] «{m.mention}» ({norm_type(m.type) or '?'}) -> {r['qid']} "
                     f"{c.get('label_ar') or ''} / {c.get('label_en') or ''}: {desc}")
        d = self.llm.json(SYSTEM_PROMPT,
                          f"Text: {text[:3000]}\n\nEntity links:\n" + "\n".join(L) +
                          "\n\nWhich links are inconsistent with the text or with each other (wrong referent)? "
                          'JSON: {"inconsistent": [{"index": <number>, "reason": "..."}]} — empty list if all are fine.')
        out = {}
        for x in d.get("inconsistent", []) or []:
            try:
                i = int(x.get("index"))
            except (TypeError, ValueError, AttributeError):
                continue
            if 0 <= i < len(results) and results[i]["qid"] != "NIL":
                out[i] = str(x.get("reason", "inconsistent with the other entities in the text"))
        return out

    def link_document(self, mentions):
        """Link all mentions of one text, then run a document-level coherence check."""
        results = [self.link(m) for m in mentions]
        cfg = self.cfg
        if cfg.baseline or cfg.max_rounds == 0 or not cfg.verify_coherence:
            return results
        if sum(r["qid"] != "NIL" for r in results) >= 2:
            calls0 = self.llm.calls
            flags = self._coherence(mentions[0].text, mentions, results)
            for r in results:
                r["llm_calls"] += (self.llm.calls - calls0) / max(1, len(results))
            for i, reason in flags.items():
                iss = [_issue("COHERENCE", f"Inconsistent with the rest of the text: {reason}")]
                results[i] = self.link(mentions[i], _resume=(results[i], iss))
        return results


def show(r):
    """Readable trace of one linking result."""
    print(f"initial: {r['initial_qid']}  ->  final: {r['qid']} {r.get('label') or ''}"
          f"   (corrected={r['corrected']}, flagged={r['flagged']}, LLM calls={r['llm_calls']:.0f})")
    print("candidates:", ", ".join(r["candidates"]))
    for t in r["trace"]:
        if t["step"] == "verify":
            d = t["decision"]
            what = d["qid"] if d["action"] == "select" else d["action"].upper()
            if t["issues"]:
                for i in t["issues"]:
                    print(f"  ✗ {what}: [{i['check']}] {i['msg']}")
            else:
                print(f"  ✓ {what} passed verification")
        elif t["step"] == "search":
            print(f"  🔎 search by {t.get('by', 'agent')}: '{t['query']}' ({t['lang']}) -> added {t['added']}")
        elif t["step"] == "expand":
            print(f"  ➕ expanded queries: {[q['q'] for q in t['queries']]}")
        elif t["step"] == "confirm":
            print(f"  ✓ confirmed despite warnings: {t['reason']}")


# %% [eval]
def norm_gold(g):
    if g is None:
        return "NIL"
    s = str(g).strip()
    m = re.search(r"Q\d+", s)
    return m.group(0) if m else "NIL"


def load_rows(path):
    """Load .jsonl / .json / .csv / .tsv (optionally .gz-compressed JSON/JSONL) into a list of dicts."""
    if path.endswith((".jsonl.gz", ".ndjson.gz")):
        import gzip
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if path.endswith(".gz"):
        return list(iter_records(path))
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, list) else d.get("data", [])
    import pandas as pd
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    return df.where(df.notna(), None).to_dict("records")


def _div(a, b):
    return round(a / b, 4) if b else None


def mcnemar_p(b, c):
    """Exact two-sided McNemar / sign test p-value for b vs c discordant pairs."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return round(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), 6)


def compute_metrics(recs):
    n_err = sum(r["qid"] == "ERROR" for r in recs)
    recs = [r for r in recs if r["qid"] != "ERROR"]
    n = len(recs)
    right = lambda r: r["qid"] == r["gold"]
    nonnil = [r for r in recs if r["gold"] != "NIL"]
    in_c = [r for r in nonnil if r["gold"] in (r.get("candidates") or [])]
    pred_nil = [r for r in recs if r["qid"] == "NIL"]
    gold_nil = [r for r in recs if r["gold"] == "NIL"]
    init_ok = lambda r: r.get("initial_qid") == r["gold"]
    unfl = [r for r in recs if not r.get("flagged")]
    fl = [r for r in recs if r.get("flagged")]
    return {
        "n": n,
        "accuracy": _div(sum(map(right, recs)), n),
        "initial_accuracy": _div(sum(map(init_ok, recs)), n),
        "fixed (wrong→right)": sum(1 for r in recs if not init_ok(r) and right(r)),
        "broken (right→wrong)": sum(1 for r in recs if init_ok(r) and not right(r)),
        "sign_test_p (fixed vs broken)": mcnemar_p(sum(1 for r in recs if not init_ok(r) and right(r)),
                                                  sum(1 for r in recs if init_ok(r) and not right(r))),
        "acc_non_nil": _div(sum(map(right, nonnil)), len(nonnil)),
        "candidate_recall": _div(len(in_c), len(nonnil)),
        "acc_when_gold_in_candidates": _div(sum(map(right, in_c)), len(in_c)),
        "nil_precision": _div(sum(r["gold"] == "NIL" for r in pred_nil), len(pred_nil)),
        "nil_recall": _div(sum(r["qid"] == "NIL" for r in gold_nil), len(gold_nil)),
        "flagged_rate": _div(len(fl), n),
        "acc_unflagged": _div(sum(map(right, unfl)), len(unfl)),
        "acc_flagged": _div(sum(map(right, fl)), len(fl)),
        "avg_llm_calls": _div(sum(r.get("llm_calls") or 0 for r in recs), n),
        "avg_seconds": _div(sum(r.get("seconds") or 0 for r in recs), n),
        "errors (excluded)": n_err,
    }


KEEP = ("qid", "label", "initial_qid", "corrected", "flagged", "rounds", "confidence", "reason",
        "candidates", "llm_calls", "seconds", "trace")


def evaluate(agent, rows, run_name, out_dir, limit=None, keys=None, progress=True):
    """Run the agent on gold rows and save one JSON line per mention (resumable after a Colab disconnect).

    rows: dicts with text, mention, gold (QID or NIL) and optionally type, start, end, id, doc_id.
    keys: rename map if your columns are named differently, e.g. {"text": "sentence", "gold": "qid"}.
    """
    k = {"text": "text", "mention": "mention", "gold": "gold", "type": "type", "start": "start",
         "end": "end", "id": "id", "doc_id": "doc_id", **(keys or {})}
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{run_name}.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            lines = [l if l.endswith("\n") else l + "\n" for l in f if l.strip()]
        keep = [l for l in lines if json.loads(l).get("qid") != "ERROR"]
        if len(keep) < len(lines):                             # failed mentions are retried
            print(f"{run_name}: retrying {len(lines) - len(keep)} mentions that failed last time")
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(keep)
        done = {str(json.loads(l)["id"]) for l in keep}
    rows = rows[:limit] if limit else rows
    groups = OrderedDict()
    for i, r in enumerate(rows):
        rid = str(r.get(k["id"]) if r.get(k["id"]) is not None else i)
        groups.setdefault(str(r.get(k["doc_id"]) or r[k["text"]]), []).append((rid, r))
    todo = [g for g in groups.values() if not all(rid in done for rid, _ in g)]
    it = todo
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(todo, desc=run_name)
        except ImportError:
            pass
    with open(path, "a", encoding="utf-8") as f:
        for items in it:
            ms = [Mention(text=r[k["text"]], mention=r[k["mention"]], type=r.get(k["type"]),
                          start=r.get(k["start"]), end=r.get(k["end"]), id=rid) for rid, r in items]
            try:
                res = agent.link_document(ms)
            except FatalLLMError as e:
                agent.wd.cache.save()
                raise FatalLLMError(f"Run '{run_name}' stopped (results so far are saved; rerun to resume): {e}") from None
            except Exception as e:
                res = [{"qid": "ERROR", "reason": repr(e)[:300]} for _ in ms]
            for (rid, r), out in zip(items, res):
                if rid in done:
                    continue
                rec = {"id": rid, "text": r[k["text"]], "mention": r[k["mention"]], "type": r.get(k["type"]),
                       "category": r.get("category"), "gold": norm_gold(r.get(k["gold"])),
                       **{x: out.get(x) for x in KEEP}}
                for kk, v in r.items():                        # pass through extra annotations (surface, sitelinks, ...)
                    if kk not in rec and kk not in k.values() and isinstance(v, (str, int, float, bool)):
                        rec[kk] = v
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                f.flush()
    agent.wd.cache.save()
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]
    errs = [r for r in recs if r["qid"] == "ERROR"]
    if errs:
        from collections import Counter
        print(f"⚠️  {run_name}: {len(errs)} mentions failed (excluded from the metrics). Rerun the same cell to "
              "retry them. Most common errors:")
        for msg, c in Counter((r.get("reason") or "")[:160] for r in errs).most_common(3):
            print(f"   {c}× {msg}")
    return compute_metrics(recs), recs


def metrics_by(recs, key="category"):
    """compute_metrics for each group (e.g. difficulty category), plus an 'ALL' row."""
    import pandas as pd
    groups = OrderedDict()
    for r in recs:
        groups.setdefault(r.get(key) or "-", []).append(r)
    table = {g: compute_metrics(rs) for g, rs in groups.items()}
    table["ALL"] = compute_metrics(recs)
    return pd.DataFrame(table).T


ABLATIONS = {
    "baseline_search_top1": dict(baseline="search_top1", use_memory=False),
    "baseline_popularity": dict(baseline="popularity", use_memory=False),
    "llm_no_correction": dict(max_rounds=0, use_memory=False),
    "full_no_memory": dict(use_memory=False),
    "minus_type_check": dict(verify_type=False, use_memory=False),
    "minus_nil_check": dict(verify_nil=False, use_memory=False),
    "minus_coherence": dict(verify_coherence=False, use_memory=False),
    "full_plus_role": dict(verify_role=True, use_memory=False),
    "full_with_memory": dict(use_memory=True, memory_frozen=True),
    # ---- added for the revision ----
    "baseline_exact_popularity": dict(baseline="exact_popularity", use_memory=False),
    "grounded_only": dict(verify_coherence=False, use_memory=False),          # Wikidata checks only (no LLM judge)
    "critic_only": dict(verify_type=False, verify_label=False, verify_nil=False, verify_coherence=False,
                        verify_critic=True, use_memory=False),               # general LLM critic, same loop and budget
}


def run_ablations(rows, base_cfg, wd, llm, names=None, memory_path=None, out_dir=None, limit=None, keys=None, tag=""):
    """Run several configurations on the same rows and return a comparison table (also saved as CSV).
    tag: suffix for the run files, e.g. "@gemini-3.8-flash" when replicating with another model
    (without it, the existing files of the first model would be resumed and nothing would run)."""
    import pandas as pd
    out_dir = out_dir or os.path.join(base_cfg.project_dir, "runs")
    table = {}
    for name in names or list(ABLATIONS):
        cfg = replace(base_cfg, **ABLATIONS[name])
        mem = ErrorMemory(memory_path, frozen=True) if (cfg.use_memory and memory_path) else None
        if cfg.use_memory and mem is None:
            print(f"skip {name}: needs memory_path (build the memory on a dev set first)")
            continue
        agent = Agent(cfg, wd, llm, mem)
        table[name + tag], _ = evaluate(agent, rows, name + tag, out_dir, limit=limit, keys=keys)
    df = pd.DataFrame(table).T
    df.to_csv(os.path.join(out_dir, f"ablations{tag}.csv"))
    return df


def _as_records(v):
    return load_rows(v) if isinstance(v, str) else v


def _load_runs(runs, min_coverage=0.9):
    """{name: path|records} -> {name: {id: record}} without ERROR rows. Runs that answered far fewer mentions
    than the largest one (an old partial run left on Drive, for example) are skipped with a message."""
    data = {name: {str(r["id"]): r for r in _as_records(v) if r["qid"] != "ERROR"} for name, v in runs.items()}
    if data:
        top = max(len(d) for d in data.values())
        for name in [n for n, d in data.items() if len(d) < min_coverage * top]:
            print(f"skipping '{name}': {len(data[name])} answered mentions vs {top} in the largest run (partial or old run)")
            del data[name]
    return data


def paired_table(runs, ref=None, n_boot=2000, seed=13, acceptable=None, exclude=(), min_coverage=0.9):
    """Fair comparison of several runs on the mentions answered by EVERY run.
    runs: {name: run-file path or records}. For each run vs `ref`: wins/losses on the same mentions,
    exact McNemar p-value and a bootstrap 95% CI of the accuracy difference; plus fixed/broken inside the run.
    acceptable: {id: set of correct QIDs} from a manual review (see rescore); other mentions use the gold label."""
    import pandas as pd, random
    data = _load_runs(runs, min_coverage)
    ids = sorted(set.intersection(*(set(d) for d in data.values())) - {str(x) for x in exclude}) if data else []
    union = set().union(*(set(d) for d in data.values())) if data else set()
    if len(union) > len(ids):
        print(f"compared on {len(ids)} mentions answered by every run ({len(union) - len(ids)} excluded)")
    ref = ref if ref in data else next(iter(data), None)
    acceptable = acceptable or {}

    def good(r, q):
        acc = acceptable.get(str(r["id"]))
        return (q in acc) if acc is not None else (q == r["gold"])

    ok = {n: [good(d[i], d[i]["qid"]) for i in ids] for n, d in data.items()}
    rng = random.Random(seed)
    idx = [[rng.randrange(len(ids)) for _ in ids] for _ in range(n_boot)] if ids else []
    rows = {}
    for name, d in data.items():
        init = [good(d[i], d[i].get("initial_qid")) for i in ids]
        fixed = sum(1 for a_, b_ in zip(init, ok[name]) if not a_ and b_)
        broken = sum(1 for a_, b_ in zip(init, ok[name]) if a_ and not b_)
        wins = sum(a_ and not b_ for a_, b_ in zip(ok[name], ok[ref]))
        losses = sum(b_ and not a_ for a_, b_ in zip(ok[name], ok[ref]))
        diff = [int(a_) - int(b_) for a_, b_ in zip(ok[name], ok[ref])]
        boots = sorted(sum(diff[j] for j in s_) / len(ids) for s_ in idx)
        ci = f"[{boots[int(0.025 * n_boot)]:+.3f}, {boots[int(0.975 * n_boot) - 1]:+.3f}]" if boots else ""
        rows[name] = {"n": len(ids), "accuracy": _div(sum(ok[name]), len(ids)),
                      "initial_accuracy": _div(sum(init), len(ids)), "fixed": fixed, "broken": broken,
                      "sign_test_p": mcnemar_p(fixed, broken), "wins_vs_ref": wins, "losses_vs_ref": losses,
                      "p_vs_ref": mcnemar_p(wins, losses), "diff_vs_ref_95CI": ci,
                      "avg_llm_calls": _div(sum(d[i].get("llm_calls") or 0 for i in ids), len(ids))}
    df = pd.DataFrame(rows).T
    df.attrs["ref"], df.attrs["ids"] = ref, ids
    return df


def fix_attribution(recs, acceptable=None, exclude=()):
    """Which verifier checks actually helped? For every check: how often it fired, and what happened to those
    mentions (fixed / broken / changed / still wrong). Uses the traces of ONE run, so it costs no extra API calls.
    acceptable/exclude: corrected labels from the review (pooled_gold); default = the original labels."""
    import pandas as pd
    stats = {}
    ex = {str(x) for x in exclude}
    for r in _as_records(recs):
        if r["qid"] == "ERROR" or str(r["id"]) in ex:
            continue
        fired = []
        for t in r.get("trace") or []:
            if t.get("step") == "verify":
                for i in t["issues"]:
                    if i["check"] not in fired:
                        fired.append(i["check"])
        acc = (acceptable or {}).get(str(r["id"]))
        init_ok = (r.get("initial_qid") in acc) if acc is not None else r.get("initial_qid") == r["gold"]
        fin_ok = (r["qid"] in acc) if acc is not None else r["qid"] == r["gold"]
        outcome = ("fixed" if fin_ok else "still_wrong") if not init_ok else ("kept_right" if fin_ok else "broken")
        for k, chk in enumerate(fired):
            s_ = stats.setdefault(chk, {"fired": 0, "fired_first": 0, "changed_answer": 0, "fixed": 0,
                                        "broken": 0, "kept_right": 0, "still_wrong": 0})
            s_["fired"] += 1
            s_["fired_first"] += k == 0
            s_["changed_answer"] += r["qid"] != r.get("initial_qid")
            s_[outcome] += 1
    df = pd.DataFrame(stats).T
    if len(df):
        df["fix_rate"] = (df["fixed"] / df["fired"]).round(3)
        df["break_rate"] = (df["broken"] / df["fired"]).round(3)
        df = df.sort_values("fired", ascending=False)
    return df


def gold_source_check(agent_recs, search_recs):
    """Agreement pattern between the gold labels, the agent and the plain search baseline.
    If the gold sides with 'search top-1' in most agent/search disagreements, the gold was probably produced by
    a search-based linker: review those cases, the agent may be right."""
    import pandas as pd
    from collections import Counter
    srch = {str(r["id"]): r["qid"] for r in _as_records(search_recs) if r["qid"] != "ERROR"}
    pats = Counter()
    for r in _as_records(agent_recs):
        i = str(r["id"])
        if r["qid"] == "ERROR" or i not in srch:
            continue
        g, a, b = r["gold"], r["qid"], srch[i]
        pats["all three agree" if a == g == b else "gold = search ≠ agent" if g == b != a else
             "gold = agent ≠ search" if g == a != b else "agent = search ≠ gold" if a == b != g else "all differ"] += 1
    order = ["all three agree", "gold = agent ≠ search", "gold = search ≠ agent", "agent = search ≠ gold", "all differ"]
    return pd.Series({k: pats.get(k, 0) for k in order}, name="mentions")


VERDICTS = ("agent_wrong", "gold_wrong", "both_ok", "unclear")


def errors_dataframe(recs, path=None, wd=None, others=None):
    """Every disagreement between the agent and the gold label, ready for manual review.
    With `wd`, both entities get their label and description so you can judge without opening Wikidata.
    Fill two columns by hand (Excel / Google Sheets, then save back as CSV):
      verdict    : agent_wrong | gold_wrong | both_ok | unclear   (who is right?)
      error_type : clitic, spelling variant, missing Arabic label, dialect/Arabizi, metonymy, NIL confusion,
                   retrieval miss, ... (why did the wrong side fail?)"""
    import pandas as pd
    bad = [r for r in recs if r["qid"] != r["gold"] and r["qid"] != "ERROR"]
    others = {name: {str(x["id"]): x["qid"] for x in _as_records(v)} for name, v in (others or {}).items()}
    info = {}
    if wd is not None:
        info = wd.get_entities([q for r in bad for q in (r["gold"], r["qid"]) if q and q != "NIL"])

    def lab(q):
        c = info.get(q) or {}
        return c.get("label_ar") or c.get("label_en") or c.get("label_fr") or ""

    def desc(q):
        c = info.get(q) or {}
        return c.get("desc_en") or c.get("desc_ar") or ""

    rows = []
    for r in bad:
        issues = [i["check"] for t in (r.get("trace") or []) if t.get("step") == "verify" for i in t["issues"]]
        rows.append({"id": r["id"], "text": r.get("text"), "mention": r["mention"], "type": r.get("type"),
                     "gold": r["gold"], "gold_label": lab(r["gold"]), "gold_desc": desc(r["gold"]),
                     "agent": r["qid"], "agent_label": lab(r["qid"]), "agent_desc": desc(r["qid"]),
                     "initial": r.get("initial_qid"), "flagged": r.get("flagged"),
                     "gold_in_candidates": r["gold"] in (r.get("candidates") or []),
                     "checks_fired": ",".join(issues), "agent_reason": r.get("reason"),
                     **{name: o.get(str(r["id"]), "") for name, o in others.items()},
                     "category": r.get("category"), "verdict": "", "error_type": ""})
    df = pd.DataFrame(rows)
    if path:
        df.to_csv(path, index=False, encoding="utf-8-sig")       # utf-8-sig: opens correctly in Excel
    return df


REVIEW_CHOICES = ("gold", "agent", "search", "gold+agent", "none", "unclear")
_OLD_VERDICTS = {"agent_wrong": "gold", "gold_wrong": "agent", "both_ok": "gold+agent"}
REVIEW_GROUPS = ("1 gold NIL, agent linked", "2 agent = search ≠ gold", "3 gold = search ≠ agent",
                 "4 agent NIL, gold linked", "5 other disagreement", "9 audit: gold = agent")


def export_review(run, path, wd=None, search=None, audit=40, seed=13):
    """Review sheet for one run: every agent/gold disagreement (most informative groups first) plus a random
    sample of `audit` agreements, to estimate how often the gold is wrong even when the agent agrees with it.
    If `path` exists, verdicts already given are kept (safe to call again after extending a run)."""
    import pandas as pd, random
    recs = [r for r in _as_records(run) if r["qid"] != "ERROR"]
    srch = {str(r["id"]): r["qid"] for r in _as_records(search) if r["qid"] != "ERROR"} if search is not None else {}
    dis = [r for r in recs if r["qid"] != r["gold"]]
    agree = [r for r in recs if r["qid"] == r["gold"]]
    aud = random.Random(seed).sample(agree, min(audit, len(agree)))

    def group(r):
        g, a_, b_ = r["gold"], r["qid"], srch.get(str(r["id"]))
        if a_ == g:
            return REVIEW_GROUPS[5]
        if g == "NIL":
            return REVIEW_GROUPS[0]
        if b_ and a_ == b_:
            return REVIEW_GROUPS[1]
        if b_ and g == b_:
            return REVIEW_GROUPS[2]
        return REVIEW_GROUPS[3] if a_ == "NIL" else REVIEW_GROUPS[4]

    info = wd.get_entities([q for r in dis + aud for q in (r["gold"], r["qid"], srch.get(str(r["id"])))
                            if q and q != "NIL"]) if wd is not None else {}

    def lab(q):
        c = info.get(q) or {}
        return c.get("label_ar") or c.get("label_en") or c.get("label_fr") or ""

    def desc(q):
        c = info.get(q) or {}
        return c.get("desc_en") or c.get("desc_ar") or ""

    prev = {}
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        for _, o in old.iterrows():
            v = o.get("verdict", "")
            prev[str(o["id"])] = (_OLD_VERDICTS.get(v, v), o.get("correct_qid", ""), o.get("error_type", ""))
    rows = []
    for r in dis + aud:
        i, b_ = str(r["id"]), srch.get(str(r["id"]), "")
        issues = [x["check"] for t in (r.get("trace") or []) if t.get("step") == "verify" for x in t["issues"]]
        v, cq, et = prev.get(i, ("", "", ""))
        rows.append({"id": i, "group": group(r), "text": r.get("text"), "mention": r["mention"], "type": r.get("type"),
                     "category": r.get("category") or "", "gold": r["gold"], "gold_label": lab(r["gold"]),
                     "gold_desc": desc(r["gold"]), "agent": r["qid"], "agent_label": lab(r["qid"]),
                     "agent_desc": desc(r["qid"]), "search": b_, "search_label": lab(b_), "search_desc": desc(b_),
                     "initial": r.get("initial_qid"), "checks_fired": ",".join(issues),
                     "agent_reason": r.get("reason") or "", "verdict": v, "correct_qid": cq, "error_type": et})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["group", "id"], kind="stable").reset_index(drop=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    kept = sum(1 for x in rows if x["verdict"])
    print(f"{len(dis)} disagreements + {len(aud)} audited agreements -> {path}"
          + (f" ({kept} verdicts kept from before)" if kept else ""))
    if len(df):
        print(df["group"].value_counts().sort_index().to_string())
    return df


def review(csv_path):
    """Review one row at a time in Colab. Pick which answer is correct; it is saved to the CSV immediately,
    so you can stop and come back. Rows that already have a verdict are skipped."""
    import pandas as pd
    import ipywidgets as W
    from IPython.display import display, clear_output
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for col in ("verdict", "correct_qid", "search", "search_label", "search_desc", "group"):
        if col not in df:
            df[col] = ""
    todo = [i for i in range(len(df)) if not df.loc[i, "verdict"].strip()]
    out, st = W.Output(), {"k": 0}
    qbox = W.Text(placeholder="right QID or NIL (optional)", layout=W.Layout(width="220px"))
    btns = {}

    def link(q):
        return f"https://www.wikidata.org/wiki/{q}" if q and q != "NIL" else ""

    def show():
        with out:
            clear_output()
            if st["k"] >= len(todo):
                print(f"✅ Nothing left to review in {os.path.basename(csv_path)}")
                return
            r = df.loc[todo[st["k"]]]
            print(f"[{st['k'] + 1}/{len(todo)}]  {r['group']}\n\n{r['text']}\n")
            print(f"mention : «{r['mention']}»   type: {r['type']}   popularity: {r.get('category', '')}")
            print(f"GOLD    : {r['gold']:<11} {r['gold_label']} — {r['gold_desc']}   {link(r['gold'])}")
            print(f"AGENT   : {r['agent']:<11} {r['agent_label']} — {r['agent_desc']}   {link(r['agent'])}")
            other = bool(r["search"]) and r["search"] not in (r["gold"], r["agent"])
            if r["search"]:
                print(f"SEARCH  : {r['search']:<11} {r['search_label']} — {r['search_desc']}   {link(r['search'])}")
            print(f"agent's reason: {r['agent_reason']}")
            btns["search ✓"].disabled = not other
            btns["gold+agent ✓"].disabled = r["gold"] == r["agent"]

    def make(label, verdict):
        b = W.Button(description=label, layout=W.Layout(width="115px"))

        def click(_):
            if st["k"] >= len(todo):
                return
            i = todo[st["k"]]
            if verdict == "none":
                q = qbox.value.strip().upper()
                if q and not re.fullmatch(r"Q\d+|NIL", q):
                    with out:
                        print("⚠️  type a QID like Q12345, or NIL, or leave the box empty")
                    return
                df.loc[i, "correct_qid"] = q
            if verdict:
                df.loc[i, "verdict"] = verdict
                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            qbox.value = ""
            st["k"] += 1
            show()
        b.on_click(click)
        btns[label] = b
        return b

    row1 = W.HBox([make("gold ✓", "gold"), make("agent ✓", "agent"), make("search ✓", "search"),
                   make("gold+agent ✓", "gold+agent")])
    row2 = W.HBox([qbox, make("none ✓", "none"), make("unclear", "unclear"), make("skip", "")])
    box = W.VBox([W.HTML("<b>Which answer is correct?</b> (✓ = that one is right; <i>none</i>: all are wrong — "
                         "type the right QID first if you know it)"), row1, row2])
    display(box, out)
    show()
    return box


review_disagreements = review          # old name


def reviewed_gold(review_csv):
    """Manual verdicts -> ({id: set of acceptable QIDs}, {ids judged 'unclear'}, summary).
    Rows without a verdict keep their original gold label."""
    import pandas as pd
    rev = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    acceptable, unclear = {}, set()
    cnt = {"disagreements": 0, "reviewed": 0, "gold right": 0, "gold wrong": 0, "both acceptable": 0,
           "unclear": 0, "audit reviewed": 0, "audit wrong": 0}
    for _, r in rev.iterrows():
        i = str(r["id"])
        v = r.get("verdict", "")
        v = _OLD_VERDICTS.get(v, v).strip()
        is_audit = str(r.get("group", "")).startswith("9")
        cnt["disagreements"] += not is_audit
        if not v:
            continue
        if v == "unclear":
            unclear.add(i)
            cnt["unclear"] += not is_audit
            continue
        cq = (r.get("correct_qid") or "").strip().upper()
        acc = {"gold": {r["gold"]}, "agent": {r["agent"]}, "search": {r.get("search", "")},
               "gold+agent": {r["gold"], r["agent"]}, "none": {cq} if cq else set()}.get(v)
        if acc is None:
            print(f"unknown verdict '{v}' for id {i} (ignored)")
            continue
        acceptable[i] = acc
        if is_audit:
            cnt["audit reviewed"] += 1
            cnt["audit wrong"] += r["gold"] not in acc
        else:
            cnt["reviewed"] += 1
            cnt["gold right"] += r["gold"] in acc and len(acc) == 1
            cnt["both acceptable"] += r["gold"] in acc and len(acc) > 1
            cnt["gold wrong"] += r["gold"] not in acc
    summary = {**cnt, "gold wrong rate among reviewed disagreements": _div(cnt["gold wrong"], cnt["reviewed"]),
               "error rate among audited agreements": _div(cnt["audit wrong"], cnt["audit reviewed"])}
    return acceptable, unclear, summary


def rescore(runs, review_csv, ref=None, **kw):
    """Re-score every run against the manually corrected labels (same fair, paired comparison as paired_table).
    Mentions judged 'unclear' are left out of all runs; unreviewed mentions keep the original gold."""
    import pandas as pd
    acceptable, unclear, summary = reviewed_gold(review_csv)
    print(pd.Series(summary).to_string(), "\n")
    left = summary["disagreements"] - summary["reviewed"] - summary["unclear"]
    if left > 0:
        print(f"⚠️  {left} disagreements not reviewed yet (they keep the original gold label).\n")
    return paired_table(runs, ref=ref, acceptable=acceptable, exclude=unclear, **kw)


# ---------- Pooled, blind review (every system's disagreements; sources hidden) ----------
def _wilson(k, n, z=1.96):
    """95% Wilson interval for a proportion k/n, as (low, high)."""
    if not n:
        return (None, None)
    p_ = k / n
    d = 1 + z * z / n
    c = (p_ + z * z / (2 * n)) / d
    h = z * ((p_ * (1 - p_) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def _split(v):
    return [x for x in str(v or "").split("|") if x]


def _answers(rec, include_initial):
    out = [rec["qid"]]
    if include_initial and rec.get("initial_qid") not in (None, "", "ERROR"):
        out.append(rec["initial_qid"])
    return out


def export_pooled_review(runs, path, wd=None, audit=40, seed=13, carry_from=None, include_initial=False):
    """Pooled review sheet: every mention where ANY run disagrees with the gold, plus `audit` random mentions
    where all runs agree with it. In the reviewer the distinct answers are shown blind (A, B, C... in random
    order; nobody sees which one is the gold or which system produced it).
    carry_from: an earlier export_review CSV; its verdicts are reused when they cover every option shown here.
    include_initial: also pool and show the INITIAL answers (before verification) of every run, so that the
    fixed/broken counts under corrected labels rest on judged answers only. Rows whose options grow are
    reopened for review; all other verdicts in `path` are kept."""
    import pandas as pd, random
    data = _load_runs(runs)
    ids = sorted(set.intersection(*(set(d) for d in data.values()))) if data else []
    first = next(iter(data.values())) if data else {}
    pooled, agree = [], []
    for i in ids:
        g = first[i]["gold"]
        (pooled if any(a != g for d in data.values() for a in _answers(d[i], include_initial)) else agree).append(i)
    old = {}                                       # id -> (accepted set | "unclear", covered options)
    if carry_from and os.path.exists(carry_from):
        for _, o in pd.read_csv(carry_from, dtype=str, keep_default_na=False).iterrows():
            v = _OLD_VERDICTS.get(o.get("verdict", ""), o.get("verdict", "")).strip()
            if not v:
                continue
            cq = (o.get("correct_qid") or "").strip().upper()
            shown = {x for x in (o.get("gold"), o.get("agent"), o.get("search"), cq) if x}
            if v == "unclear":
                old[str(o["id"])] = ("unclear", shown)
                continue
            acc = {"gold": {o["gold"]}, "agent": {o["agent"]}, "search": {o.get("search", "")},
                   "gold+agent": {o["gold"], o["agent"]}, "none": {cq} if cq else set()}.get(v)
            if acc is not None:
                old[str(o["id"])] = (acc, shown)
    prev, prev_audit = {}, []
    if os.path.exists(path):
        for _, o in pd.read_csv(path, dtype=str, keep_default_na=False).iterrows():
            if o.get("group") == "audit":
                prev_audit.append(str(o["id"]))
            if o.get("status"):
                prev[str(o["id"])] = (o["status"], o.get("accepted", ""), o.get("annotator", ""), o.get("carried", ""),
                                      set(_split(o.get("options", ""))))
    agree_set = set(agree)
    keep_aud = [i for i in prev_audit if i in agree_set][:audit]          # keep the same audit rows when possible
    rest = [i for i in agree if i not in set(keep_aud)]
    aud = set(keep_aud) | set(random.Random(seed).sample(rest, max(0, min(audit - len(keep_aud), len(rest)))))

    rows_ids = pooled + sorted(aud)
    opts = {}
    for i in rows_ids:
        ans = [first[i]["gold"]] + [a for d in data.values() for a in _answers(d[i], include_initial)]
        o = list(dict.fromkeys(ans))
        random.Random(f"{seed}-{i}").shuffle(o)
        opts[i] = o
    info = wd.get_entities([q for o in opts.values() for q in o if q != "NIL"]) if wd is not None else {}
    out, carried, reopened = [], 0, 0
    for i in rows_ids:
        r = first[i]
        o = opts[i]
        status, accepted, annot, car = "", "", "", ""
        if i in prev and set(o) <= prev[i][4]:
            status, accepted, annot, car = prev[i][:4]
        elif i in prev:
            reopened += 1                                   # a new run added an answer nobody judged yet
        if status:
            pass
        elif i in old and set(o) <= old[i][1]:            # reuse only if every option was visible before
            acc, _ = old[i]
            if acc == "unclear":
                status = "unclear"
            else:
                status, accepted = "done", "|".join(sorted(acc))
            annot, car = "first_pass", "yes"
            carried += 1
        info_ = [[q, (info.get(q) or {}).get("label_ar") or (info.get(q) or {}).get("label_en")
                  or (info.get(q) or {}).get("label_fr") or "", (info.get(q) or {}).get("desc_en")
                  or (info.get(q) or {}).get("desc_ar") or ""] for q in o]
        out.append({"id": i, "group": "audit" if i in aud else ("gold NIL" if r["gold"] == "NIL" else "disagreement"),
                    "text": r.get("text"), "mention": r["mention"], "type": r.get("type") or "",
                    "category": r.get("category") or "", "options": "|".join(o),
                    "option_info": json.dumps(info_, ensure_ascii=False),
                    "sources": json.dumps({"gold": r["gold"], **{n: d[i]["qid"] for n, d in data.items()},
                                           **({f"{n}#initial": d[i].get("initial_qid") for n, d in data.items()}
                                              if include_initial else {})}),
                    "status": status, "accepted": accepted, "annotator": annot, "carried": car})
    df = pd.DataFrame(out)
    if len(df):
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)      # blind: no group order
    df.to_csv(path, index=False, encoding="utf-8-sig")
    todo = int((df["status"] == "").sum()) if len(df) else 0
    print(f"{len(pooled)} mentions where some run disagrees with the gold + {len(aud)} audited agreements -> {path}")
    print(f"{carried} verdicts reused from {os.path.basename(carry_from) if carry_from else '-'}, "
          f"{len(prev) - reopened} kept from this file, {reopened} reopened (new answers to judge), {todo} left to review")
    return df


def review_blind(csv_path, annotator="A"):
    """Blind reviewer: tick every answer that is correct in this context (sources are hidden). You can also
    type a QID (or NIL) that is not among the options. Saved immediately; rows already reviewed are skipped."""
    import pandas as pd
    import ipywidgets as W
    from IPython.display import display, clear_output
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    todo = [i for i in range(len(df)) if not df.loc[i, "status"].strip()]
    out, st = W.Output(), {"k": 0}
    boxes = [W.Checkbox(value=False, indent=False, layout=W.Layout(width="95%")) for _ in range(6)]
    qbox = W.Text(placeholder="other QID or NIL (optional)", layout=W.Layout(width="220px"))
    btns = {}

    def current():
        return df.loc[todo[st["k"]]] if st["k"] < len(todo) else None

    def show():
        for b in boxes:
            b.value, b.layout.display = False, "none"
        qbox.value = ""
        with out:
            clear_output()
            r = current()
            if r is None:
                print(f"✅ Nothing left to review in {os.path.basename(csv_path)}")
                return
            print(f"[{st['k'] + 1}/{len(todo)}]\n\n{r['text']}\n\nmention: «{r['mention']}»   type: {r['type']}\n")
        r = current()
        for b, (q, lab, desc) in zip(boxes, json.loads(r["option_info"])):
            url = f"  https://www.wikidata.org/wiki/{q}" if q != "NIL" else ""
            b.description = f"{q}  {lab} — {desc}{url}" if q != "NIL" else "NIL (not in Wikidata)"
            b.layout.display = ""

    def save(status, accepted):
        i = todo[st["k"]]
        df.loc[i, "status"], df.loc[i, "accepted"], df.loc[i, "annotator"] = status, "|".join(sorted(accepted)), annotator
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        st["k"] += 1
        show()

    def typed():
        q = qbox.value.strip().upper()
        if q and not re.fullmatch(r"Q\d+|NIL", q):
            with out:
                print("⚠️  type a QID like Q12345, or NIL")
            return None
        return {q} if q else set()

    def on_save(_):
        r = current()
        if r is None:
            return
        extra = typed()
        if extra is None:
            return
        chosen = {q for b, q in zip(boxes, _split(r["options"])) if b.value} | extra
        if not chosen:
            with out:
                print("⚠️  tick at least one option, type a QID, or use 'none correct'")
            return
        save("done", chosen)

    def on_none(_):
        if current() is not None:
            save("done", set())

    def on_unclear(_):
        if current() is not None:
            save("unclear", set())

    def on_skip(_):
        if current() is not None:
            st["k"] += 1
            show()

    for label, fn in (("save ✓", on_save), ("none correct", on_none), ("unclear", on_unclear), ("skip", on_skip)):
        b = W.Button(description=label, layout=W.Layout(width="115px"))
        b.on_click(fn)
        btns[label] = b
    box = W.VBox([W.HTML("<b>Tick every answer that is correct here</b> (more than one if both are acceptable)")]
                 + boxes + [W.HBox([qbox] + list(btns.values()))])
    display(out, box)
    show()
    box._elagent = {"boxes": boxes, "qbox": qbox, "btns": btns}     # handles for tests
    return box


def pooled_gold(csv_path):
    """Blind/pooled verdicts -> ({id: acceptable QIDs}, {unclear ids}, summary with gold error estimates)."""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    acceptable, unclear = {}, set()
    for _, r in df.iterrows():
        if r["status"] == "done":
            acceptable[str(r["id"])] = set(_split(r["accepted"]))
        elif r["status"] == "unclear":
            unclear.add(str(r["id"]))
    gold = {str(r["id"]): json.loads(r["sources"])["gold"] for _, r in df.iterrows()}
    summ = {}
    for grp in ("gold NIL", "disagreement", "audit"):
        g_ids = [str(i) for i in df.loc[df["group"] == grp, "id"]]
        rev = [i for i in g_ids if i in acceptable]
        wrong = sum(gold[i] not in acceptable[i] for i in rev)
        summ[grp] = {"rows": len(g_ids), "reviewed": len(rev), "gold wrong": wrong,
                     "gold wrong rate": _div(wrong, len(rev)), "95% CI": _wilson(wrong, len(rev))}
    return acceptable, unclear, summ


def rescore_pooled(runs, csv_path, ref=None, **kw):
    """Score every run against labels corrected by the pooled (and ideally blind) review."""
    import pandas as pd
    acceptable, unclear, summ = pooled_gold(csv_path)
    print(pd.DataFrame(summ).T.to_string(), "\n")
    left = sum(v["rows"] - v["reviewed"] for v in summ.values()) - len(unclear)
    if left > 0:
        print(f"⚠️  {left} pooled mentions not reviewed yet (they keep the original gold label).\n")
    return paired_table(runs, ref=ref, acceptable=acceptable, exclude=unclear, **kw)


def agreement_sample(csv_path, out_path, n=60, seed=7):
    """Copy n random reviewed rows with verdicts removed, for a SECOND annotator (review_blind(out_path, 'B'))."""
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    done = df[df["status"] == "done"]
    sample = done.sample(n=min(n, len(done)), random_state=seed).copy()
    sample[["status", "accepted", "annotator", "carried"]] = ""
    sample.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"{len(sample)} rows for the second annotator -> {out_path}")
    return sample


def agreement(csv_a, csv_b):
    """Inter-annotator agreement on rows reviewed in both files: exact agreement of the accepted sets and
    Cohen's kappa on the yes/no judgement of every option."""
    import pandas as pd
    a = pd.read_csv(csv_a, dtype=str, keep_default_na=False).set_index("id")
    b = pd.read_csv(csv_b, dtype=str, keep_default_na=False).set_index("id")
    ids = [i for i in b.index if i in a.index and a.loc[i, "status"] == "done" and b.loc[i, "status"] == "done"]
    ya, yb, exact = [], [], 0
    for i in ids:
        sa, sb = set(_split(a.loc[i, "accepted"])), set(_split(b.loc[i, "accepted"]))
        exact += sa == sb
        for q in set(_split(a.loc[i, "options"])) | sa | sb:
            ya.append(q in sa)
            yb.append(q in sb)
    n = len(ya)
    if not n:
        return {"rows": 0}
    po = sum(x == y for x, y in zip(ya, yb)) / n
    pa, pb = sum(ya) / n, sum(yb) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"rows": len(ids), "exact agreement": round(exact / len(ids), 4), "option judgements": n,
            "observed agreement": round(po, 4), "cohen_kappa": round(kappa, 4)}


def flatten_docs(docs, text_key="text", ents_key="entities", mention_key="mention", gold_key="qid",
                 type_key="type", start_key="start", end_key="end", id_key="id"):
    """For datasets that store several entities inside each sentence/document
    (e.g. {"text": ..., "entities": [{"mention":..., "qid":..., "type":...}, ...]}) -> one row per mention."""
    rows = []
    for di, d in enumerate(docs):
        did = d.get(id_key, di)
        for ei, e in enumerate(d.get(ents_key) or []):
            rows.append({"id": f"{did}-{ei}", "doc_id": str(did), "text": d[text_key], "mention": e.get(mention_key),
                         "gold": e.get(gold_key), "type": e.get(type_key),
                         "start": e.get(start_key), "end": e.get(end_key)})
    return rows


def split_rows(rows, dev_frac=0.2, seed=13, doc_key="doc_id", text_key="text"):
    """Split rows into dev/test by document (all mentions of one text stay together)."""
    import random
    groups = OrderedDict()
    for r in rows:
        groups.setdefault(str(r.get(doc_key) or r[text_key]), []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    n_dev = max(1, int(len(keys) * dev_frac))
    dev = [r for k in keys[:n_dev] for r in groups[k]]
    test = [r for k in keys[n_dev:] for r in groups[k]]
    return dev, test


# %% [datasets]
# ---------- Generic loader for third-party EL datasets (ELNER-DZ, ArabNED, ...) ----------
_QID_RE = re.compile(r"^(?:https?://(?:www\.)?wikidata\.org/(?:wiki|entity)/)?(Q\d+)$")
_NILISH = {"", "nil", "null", "none", "nan", "-", "o", "unk", "unknown"}


def _qidlike(v):
    return isinstance(v, str) and bool(_QID_RE.match(v.strip()))


def _unwrap(obj):
    """Top-level JSON value -> records. A dict that looks like a record is kept; a wrapper such as
    {"data": [...]} or {"s1": {...}, "s2": {...}} is opened."""
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict):
                yield x
    elif isinstance(obj, dict):
        is_record = any(isinstance(v, str) and len(v) > 20 for v in obj.values())
        lists = [v for v in obj.values() if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)]
        if is_record or not (lists or all(isinstance(v, dict) for v in obj.values())):
            yield obj
        elif lists:
            yield from max(lists, key=len)
        else:
            yield from obj.values()


def _iter_json_text(text, path=""):
    """Tolerant JSON reader: one array, several concatenated arrays/objects (pretty-printed or not),
    JSON-lines, a wrapper object, or Python-style literals with single quotes. Truncated files yield
    everything before the cut; other syntax errors raise with the position and surrounding text."""
    dec, i, n, count = json.JSONDecoder(), 0, len(text), 0

    def skip(i):
        while i < n and text[i] in " \t\r\n,":
            i += 1
        return i

    try:
        i = skip(0)
        while i < n:
            if text[i] == "[":
                i = skip(i + 1)
                while i < n and text[i] != "]":
                    obj, i = dec.raw_decode(text, i)
                    if isinstance(obj, dict):
                        count += 1
                        yield obj
                    else:
                        yield from _unwrap(obj)
                    i = skip(i)
                i = skip(i + 1)
            else:
                obj, i = dec.raw_decode(text, i)
                i = skip(i)
                single = count == 0 and i >= n                # one top-level value: may be a wrapper
                for r in (_unwrap(obj) if single or not isinstance(obj, dict) else [obj]):
                    count += 1
                    yield r
    except json.JSONDecodeError as e:
        if count and e.pos > 0.98 * n:
            print(f"⚠️  {os.path.basename(path)} looks truncated near the end; kept the {count} records before the cut.")
            return
        if count == 0 and "double quotes" in e.msg and n < 500 * 2**20:
            import ast
            try:                                               # Python repr instead of JSON (single quotes, None, True)
                yield from _unwrap(ast.literal_eval(text.strip()))
                return
            except Exception:
                pass
        ctx = text[max(0, e.pos - 120): e.pos + 120].replace("\n", "⏎")
        raise ValueError(f"{os.path.basename(path)}: invalid JSON at line {e.lineno}, column {e.colno} "
                         f"(char {e.pos} of {n}): {e.msg}. {count} records were read before it.\n"
                         f"Text around the error: …{ctx}…") from None


def _sniff(path):
    """What is really inside a data file (extensions lie): gzip, zip, json, html, conll, empty or unknown."""
    with open(path, "rb") as f:
        head = f.read(4096)
    if head[:2] == b"\x1f\x8b":
        return "gzip"
    if head[:4] == b"PK\x03\x04":
        return "zip"
    h = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    if not h:
        return "empty"
    if h[:1] in (b"[", b"{"):
        return "json"
    if h[:1] == b"<":
        return "html"
    if b"\t" in h or re.search(rb"(^|\s)[BIES]-[A-Za-z]", h):
        return "conll"
    return "unknown"


def describe_file(path, n=500):
    """Print what a data file looks like (size, real format, first characters) — paste this when asking for help."""
    kind = _sniff(path)
    opener = __import__("gzip").open if kind == "gzip" else open
    with opener(path, "rb") as f:
        head = f.read(n).decode("utf-8", "replace")
    print(f"{os.path.basename(path)} | {os.path.getsize(path) / 2**20:.1f} MB | real format: {kind}")
    print("first characters:", repr(head))


def _iter_conll(lines):
    """CoNLL-style text (one token per line, blank line between sentences) -> {'tokens', 'tags', 'qids'} records."""
    rows = []

    def flush():
        if not rows:
            return None
        width = max(len(r) for r in rows)
        rows_ = [r + [""] * (width - len(r)) for r in rows]
        cols = list(zip(*rows_))
        tag_c = next((j for j in range(1, width) if all(re.match(r"^(O|[BIES]-.+)$", x) for x in cols[j])), None)
        qid_c = next((j for j in range(1, width) if j != tag_c and any(_qidlike(x) for x in cols[j])), None)
        rec = {"tokens": list(cols[0]), "tags": list(cols[tag_c]) if tag_c is not None else ["O"] * len(rows_),
               "qids": [x if _qidlike(x) else None for x in cols[qid_c]] if qid_c is not None else [None] * len(rows_)}
        rows.clear()
        return rec

    for line in lines:
        line = line.rstrip("\r\n")
        if not line.strip():
            rec = flush()
            if rec:
                yield rec
            continue
        if line.startswith("#"):
            continue
        rows.append(line.split("\t") if "\t" in line else line.split())
    rec = flush()
    if rec:
        yield rec


def iter_records(path, limit=None, stream_over_mb=50):
    """Yield dict records from a data file whatever its real format: JSON (one array, concatenated arrays or
    objects, JSON-lines, wrappers, Python literals), CoNLL token lines, CSV/TSV — optionally gzip-compressed.
    JSON above `stream_over_mb` is streamed with ijson when installed. Clear errors for zip/html/empty files."""
    import gzip, io
    n = 0

    def emit(it):
        nonlocal n
        for r in it:
            if isinstance(r, dict):
                yield r
                n += 1
                if limit and n >= limit:
                    return

    name = os.path.basename(path)
    if path.endswith((".csv", ".tsv")):
        import pandas as pd
        for chunk in pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",", chunksize=20000):
            yield from emit(chunk.where(chunk.notna(), None).to_dict("records"))
            if limit and n >= limit:
                return
        return
    kind = _sniff(path)
    gz = kind == "gzip"
    if gz:
        with gzip.open(path, "rb") as f:
            head = f.read(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
        kind = "json" if head[:1] in (b"[", b"{") else ("html" if head[:1] == b"<" else "conll")
    if kind == "zip":
        raise ValueError(f"{name} is a ZIP archive, not a data file. Unzip it first, e.g. "
                         f"!unzip -o '{path}' -d '{os.path.dirname(path)}'  (the 11ب cell does this automatically for .zip files).")
    if kind == "html":
        raise ValueError(f"{name} is an HTML web page, not data. A download command probably saved Zenodo's "
                         "login/permission page. Download the file from the Zenodo page in a browser where you are "
                         "logged in (and have been granted access), then upload it to Drive.")
    if kind == "empty":
        raise ValueError(f"{name} is empty (0 useful bytes). Re-download it.")
    opener = (lambda mode: gzip.open(path, mode)) if gz else (lambda mode: open(path, mode))
    if kind == "conll":
        with opener("rb") as f:
            yield from emit(_iter_conll(io.TextIOWrapper(f, encoding="utf-8-sig")))
        return
    if kind == "unknown":
        describe_file(path)
        raise ValueError(f"{name}: unrecognised format (see the first characters above).")
    if (gz or os.path.getsize(path) > stream_over_mb * 2**20):
        try:
            import ijson
        except ImportError:
            ijson = None
            print("Large file: `pip install ijson` would stream it instead of loading it all into memory.")
        if ijson is not None:
            with opener("rb") as f:
                head = f.read(4096).lstrip(b"\xef\xbb\xbf \t\r\n")
            yield from emit(_ijson_records(path, head.startswith(b"["), gz))
            return
    with opener("rb") as f:
        text = f.read().decode("utf-8-sig")
    yield from emit(_iter_json_text(text, path))


def _ijson_records(path, is_array, gz=False):
    import ijson, gzip
    with (gzip.open(path, "rb") if gz else open(path, "rb")) as f:
        if f.read(3) != b"\xef\xbb\xbf":
            f.seek(0)
        try:
            it = ijson.items(f, "item" if is_array else "", multiple_values=True, use_float=True)
            if is_array:
                for obj in it:
                    yield from ([obj] if isinstance(obj, dict) else _unwrap(obj))
            else:
                first = next(it, None)
                second = next(it, None)
                if second is None:                        # a single top-level value: may be a wrapper
                    if first is not None:
                        yield from _unwrap(first)
                else:                                     # JSON-lines: every dict is a record
                    for obj in (first, second):
                        yield from ([obj] if isinstance(obj, dict) else _unwrap(obj))
                    for obj in it:
                        yield from ([obj] if isinstance(obj, dict) else _unwrap(obj))
        except ijson.common.IncompleteJSONError as e:
            print(f"⚠️  {os.path.basename(path)}: stopped streaming ({e}); the records read so far are kept. "
                  "If this happens early, set stream_over_mb higher to get a precise error message.")


def detect_schema(records):
    """Guess field names from a sample of records. Handles three layouts:
    'nested' (text + list of entity dicts), 'flat' (one mention per record), 'bio' (tokens + BIO tags + per-token QIDs)."""
    recs = [r for r in records if isinstance(r, dict)][:300]
    if not recs:
        raise ValueError("No dict records found.")
    keys = list(dict.fromkeys(k for r in recs for k in r))

    def frac(pred, vals):
        vals = [v for v in vals if v is not None]
        return sum(1 for v in vals if pred(v)) / len(vals) if vals else 0

    # --- BIO token layout
    list_str = [k for k in keys if frac(lambda v: isinstance(v, list) and v and all(isinstance(x, str) for x in v),
                                        [r.get(k) for r in recs]) > 0.8]
    tag_k = next((k for k in list_str if frac(lambda v: all(re.match(r"^(O|[BIES]-.+)$", x) for x in v), [r.get(k) for r in recs]) > 0.8), None)
    if tag_k:
        qid_k = next((k for k in keys if k != tag_k and frac(lambda v: isinstance(v, list) and any(_qidlike(x) for x in v if isinstance(x, str)),
                                                               [r.get(k) for r in recs]) > 0.3), None)
        tok_k = next((k for k in list_str if k not in (tag_k, qid_k)), None)
        if tok_k and qid_k:
            return {"layout": "bio", "tokens": tok_k, "tags": tag_k, "qids": qid_k,
                    "id": next((k for k in ("id", "sent_id", "doc_id") if k in keys), None)}

    # --- text field: longest string field present in most records
    strs = {k: [len(r[k]) for r in recs if isinstance(r.get(k), str)] for k in keys}
    strs = {k: v for k, v in strs.items() if len(v) >= 0.6 * len(recs)}
    if not strs:
        raise ValueError(f"Could not find a text field among {keys}")
    text_k = max(strs, key=lambda k: sum(strs[k]) / len(strs[k]))

    # --- nested entity list?
    ents_k = None
    for k in keys:
        vals = [r.get(k) for r in recs if isinstance(r.get(k), list) and r.get(k)]
        if vals and all(isinstance(e, dict) for v in vals for e in v) and \
                any(_qidlike(x) for v in vals for e in v for x in e.values() if isinstance(x, str)):
            ents_k = k
            break
    pairs = [(r[text_k], e) for r in recs if isinstance(r.get(text_k), str) for e in (r.get(ents_k) or [])] if ents_k \
        else [(r[text_k], r) for r in recs if isinstance(r.get(text_k), str)]
    ekeys = list(dict.fromkeys(k for _, e in pairs for k in e if k != text_k))
    score = {k: frac(lambda v: _qidlike(v) or str(v).strip().lower() in _NILISH, [e.get(k) for _, e in pairs])
             * frac(_qidlike, [e.get(k) for _, e in pairs]) ** 0.5 for k in ekeys}
    qid_k = max(score, key=score.get) if score and max(score.values()) > 0 else None
    if not qid_k:
        raise ValueError(f"Could not find a QID field among {ekeys}")
    sub = {k: sum(isinstance(e.get(k), str) and e[k].strip() != "" and e[k] in t for t, e in pairs) / len(pairs)
           for k in ekeys if k != qid_k}
    mention_k = max(sub, key=sub.get) if sub and max(sub.values()) > 0.5 else None
    small = {k for k in ekeys if k not in (qid_k, mention_k)
             and frac(lambda v: isinstance(v, str) and 0 < len(v) <= 12, [e.get(k) for _, e in pairs]) > 0.9
             and len({e.get(k) for _, e in pairs}) <= 60}
    type_k = next((k for k in small if any(w in k.lower() for w in ("type", "label", "tag", "class", "cat"))), None) \
        or (sorted(small)[0] if small else None)
    ints = [k for k in ekeys if frac(lambda v: isinstance(v, int) and not isinstance(v, bool), [e.get(k) for _, e in pairs]) > 0.9]
    start_k = next((k for k in ints if any(w in k.lower() for w in ("start", "begin", "offset"))), None)
    end_k = next((k for k in ints if "end" in k.lower()), None)
    id_k = next((k for k in ("id", "doc_id", "sent_id", "sentence_id") if k in keys and k != text_k), None)
    return {"layout": "nested" if ents_k else "flat", "text": text_k, "entities": ents_k, "mention": mention_k,
            "qid": qid_k, "type": type_k, "start": start_k, "end": end_k, "id": id_k}


def convert_records(records, schema, source=None):
    """Turn records into agent rows (one per mention) using a schema from detect_schema (edit it if a guess is wrong)."""
    rows = []
    for i, r in enumerate(records):
        did = str(r.get(schema.get("id")) if schema.get("id") and r.get(schema.get("id")) is not None else i)
        if schema["layout"] == "bio":
            toks, tags, qids = r.get(schema["tokens"]) or [], r.get(schema["tags"]) or [], r.get(schema["qids"]) or []
            text, offs = "", []
            for tok in toks:
                if text:
                    text += " "
                offs.append((len(text), len(text) + len(tok)))
                text += tok
            spans, cur = [], None
            for j, tag in enumerate(tags):
                if tag.startswith(("B-", "S-")) or (tag.startswith(("I-", "E-")) and (cur is None or cur[2] != tag[2:])):
                    if cur:
                        spans.append(cur)
                    cur = [j, j, tag[2:]]
                elif tag.startswith(("I-", "E-")):
                    cur[1] = j
                else:
                    if cur:
                        spans.append(cur)
                    cur = None
            if cur:
                spans.append(cur)
            for k, (a, b, typ) in enumerate(spans):
                q = next((x for x in qids[a:b + 1] if isinstance(x, str) and _qidlike(x)), None)
                s, e = offs[a][0], offs[b][1]
                rows.append({"id": f"{did}-{k}", "doc_id": did, "text": text, "mention": text[s:e], "type": typ,
                             "start": s, "end": e, "gold": norm_gold(q), "source": source})
            continue
        text = r.get(schema["text"])
        if not isinstance(text, str) or not text.strip():
            continue
        ents = (r.get(schema["entities"]) or []) if schema["layout"] == "nested" else [r]
        for k, e in enumerate(ents):
            s = e.get(schema["start"]) if schema.get("start") else None
            en = e.get(schema["end"]) if schema.get("end") else None
            mention = e.get(schema["mention"]) if schema.get("mention") else None
            if not mention and isinstance(s, int) and isinstance(en, int):
                mention = text[s:en]
            if not mention:
                continue
            rows.append({"id": f"{did}-{k}", "doc_id": did, "text": text, "mention": mention,
                         "type": e.get(schema["type"]) if schema.get("type") else None, "start": s, "end": en,
                         "gold": norm_gold(e.get(schema["qid"])), "source": source})
    return rows


def sample_by_doc(rows, n_mentions, seed=13, max_per_doc=5, linked_only=False):
    """Random sample of documents (all their mentions kept together, at most max_per_doc each)."""
    import random
    groups = OrderedDict()
    for r in rows:
        if linked_only and r["gold"] == "NIL":
            continue
        groups.setdefault(r.get("doc_id") or r["text"], []).append(r)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    out = []
    for k in keys:
        out += groups[k][:max_per_doc]
        if len(out) >= n_mentions:
            break
    return out[:n_mentions]


def add_popularity(rows, wd, head=50, tail=10, key="category"):
    """Bucket each gold entity by Wikidata sitelink count: head (>= head), torso, tail (< tail), or nil."""
    ents = wd.get_entities([r["gold"] for r in rows if r["gold"] != "NIL"])
    wd.cache.save()
    for r in rows:
        if r["gold"] == "NIL":
            r[key], r["sitelinks"] = "nil", None
            continue
        sl = (ents.get(r["gold"]) or {}).get("sitelinks", 0)
        r["sitelinks"] = sl
        r[key] = "head" if sl >= head else ("tail" if sl < tail else "torso")
    return rows


def assign_ner_types(rows, ner):
    """Give rows a realistic NER type (from any HF token-classification pipeline) instead of a gold-derived one."""
    cache = {}
    for r in rows:
        t = r["text"]
        if t not in cache:
            cache[t] = ner(t)
        s, e = r.get("start"), r.get("end")
        if not isinstance(s, int) or not isinstance(e, int):
            s = t.find(r["mention"])
            e = s + len(r["mention"])
        hit = [x for x in cache[t] if x["start"] < e and x["end"] > s]
        r["type"] = norm_type(max(hit, key=lambda x: min(e, x["end"]) - max(s, x["start"]))["entity_group"]) if hit else None
    return rows


# ---------- Silver data from Arabic Wikipedia links (human-made links = gold QIDs) ----------
NE_ROOTS = sorted({q for k in ("PER", "GPE", "ORG", "FAC", "EVENT") for q in TYPE_ROOTS[k]})


def _wp_random_titles(wd, n=50, lang="ar", min_bytes=4000):
    d = wd._get_json(wd.WP_API.format(lang=lang), {"action": "query", "generator": "random", "grnnamespace": 0,
                                                    "grnlimit": n, "prop": "info", "format": "json", "formatversion": 2})
    return [p["title"] for p in d.get("query", {}).get("pages", []) if p.get("length", 0) >= min_bytes and "redirect" not in p]


def _wp_paragraph_links(wd, title, lang="ar"):
    """[(paragraph_text, [(anchor, target_title, start, end), ...]), ...] from the rendered article."""
    from bs4 import BeautifulSoup, NavigableString, Comment
    d = wd._get_json(wd.WP_API.format(lang=lang), {"action": "parse", "page": title, "prop": "text", "redirects": 1,
                                                    "disableeditsection": 1, "disabletoc": 1, "format": "json", "formatversion": 2})
    html = (d.get("parse") or {}).get("text") or ""
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select("sup, style, script, span.mw-editsection, .noprint, .reference"):
        junk.decompose()
    out = []
    for p in soup.find_all("p"):
        if p.find_parent(["table", "figure", "li"]):
            continue
        text, spans = "", {}
        for node in p.descendants:
            if isinstance(node, NavigableString) and not isinstance(node, Comment):
                a = node.find_parent("a")
                start = len(text)
                text += str(node)
                if a is not None and p in a.parents:
                    sp = spans.setdefault(id(a), [a, start, len(text)])
                    sp[2] = len(text)
        links = []
        for a, s, e in spans.values():
            href, tgt = a.get("href", ""), a.get("title", "")
            if not href.startswith("/wiki/") or "redlink" in href or "new" in (a.get("class") or []) or not tgt or ":" in tgt:
                continue
            anchor = text[s:e]
            lead, trail = len(anchor) - len(anchor.lstrip()), len(anchor) - len(anchor.rstrip())
            s, e = s + lead, e - trail
            anchor = text[s:e]
            if len(anchor) < 2 or re.fullmatch(r"[\d\s٠-٩]+", anchor):
                continue
            links.append((anchor, tgt.split("#")[0], s, e))
        if len(text.strip()) > 40 and links:
            out.append((text, links))
    return out


def _wp_titles_to_qids(wd, titles, lang="ar"):
    res = {}
    titles = list(dict.fromkeys(titles))
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = wd._get_json(wd.WP_API.format(lang=lang), {"action": "query", "titles": "|".join(chunk), "redirects": 1,
                                                        "prop": "pageprops", "ppprop": "wikibase_item|disambiguation",
                                                        "format": "json", "formatversion": 2})
        q = d.get("query", {})
        fwd = {x["from"]: x["to"] for x in q.get("normalized", [])}
        red = {x["from"]: x["to"] for x in q.get("redirects", [])}
        pages = {p.get("title"): p for p in q.get("pages", [])}
        for t in chunk:
            final = red.get(fwd.get(t, t), fwd.get(t, t))
            pp = (pages.get(final) or {}).get("pageprops") or {}
            res[t] = None if "disambiguation" in pp else pp.get("wikibase_item")
    return res


def build_wiki_link_set(wd, n_mentions=600, lang="ar", max_per_article=4, min_bytes=4000, alias_frac=0.7,
                        named_entities_only=True, max_context=900, seed=13, log=print):
    """Sample mentions from random Arabic Wikipedia articles; the linked article's QID is the gold answer.
    'surface' = 'alias' when the link text differs from the article title (harder: partial names, nicknames,
    inflected forms), else 'title'. Save the result: random sampling is not reproducible, the file is."""
    import random
    rng = random.Random(seed)
    rows, seen_docs, tries, idle = [], set(), 0, 0
    while len(rows) < n_mentions and tries < 200 and idle < 10:
        tries += 1
        before = len(rows)
        for title in _wp_random_titles(wd, 50, lang, min_bytes):
            if title in seen_docs or len(rows) >= n_mentions:
                continue
            seen_docs.add(title)
            try:
                paras = _wp_paragraph_links(wd, title, lang)
            except Exception as e:
                log(f"skip {title}: {e}")
                continue
            cand = [(t, l) for t, ls in paras for l in ls]
            if not cand:
                continue
            qmap = _wp_titles_to_qids(wd, [l[1] for _, l in cand], lang)
            pool = []
            for t, (anchor, tgt, s, e) in cand:
                q = qmap.get(tgt)
                if not q:
                    continue
                surface = "title" if tokens(anchor) == tokens(re.sub(r"\s*\(.*?\)\s*$", "", tgt)) else "alias"
                pool.append((t, anchor, tgt, s, e, q, surface))
            rng.shuffle(pool)
            aliases = [x for x in pool if x[6] == "alias"]
            plain = [x for x in pool if x[6] != "alias"]
            ordered = []
            while aliases or plain:                             # prefer alias links with probability alias_frac
                src = aliases if (aliases and (not plain or rng.random() < alias_frac)) else plain
                ordered.append(src.pop(0))
            picked, used = [], set()
            for t, anchor, tgt, s, e, q, surface in ordered:
                if len(picked) >= max_per_article or q in used:
                    continue
                if named_entities_only and wd.is_instance_of(q, NE_ROOTS) is False:
                    continue
                used.add(q)
                if len(t) > max_context:                        # keep a window around the mention
                    a = max(0, min(s - max_context // 2, len(t) - max_context))
                    t, s, e = t[a:a + max_context], s - a, e - a
                picked.append({"id": f"{title}#{len(picked)}", "doc_id": title, "text": t, "mention": anchor,
                               "start": s, "end": e, "type": None, "gold": q, "target_title": tgt,
                               "surface": surface, "source": f"{lang}wiki"})
            rows += picked
        idle = idle + 1 if len(rows) == before else 0      # stop if random pages keep yielding nothing
        if len(rows) > before:
            log(f"{len(rows)}/{n_mentions} mentions from {len(seen_docs)} articles")
        wd.cache.save()
    return rows[:n_mentions]


def save_rows(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def reservoir_sample(iterable, k, seed=13):
    """Uniform random sample of k items from a stream of unknown length (one pass, O(k) memory)."""
    import random
    rng, out = random.Random(seed), []
    for i, x in enumerate(iterable):
        if i < k:
            out.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                out[j] = x
    return out


# %% [release]
# ---------- Helpers used to publish and reuse the corrected test set ----------
def paired_by(runs, key="category", ref=None, acceptable=None, exclude=()):
    """paired_table computed separately for every value of `key` (e.g. popularity bucket).
    Returns one row per (group, run) with n, accuracy, initial accuracy, fixed and broken."""
    import pandas as pd
    import contextlib, io
    data = {n: _as_records(v) for n, v in runs.items()}
    values = sorted({str(r.get(key)) for recs in data.values() for r in recs if r.get(key) not in (None, "")})
    parts = []
    for v in values:
        sub = {n: [r for r in recs if str(r.get(key)) == v] for n, recs in data.items()}
        with contextlib.redirect_stdout(io.StringIO()):
            t = paired_table(sub, ref=ref, acceptable=acceptable, exclude=exclude, n_boot=0)
        t = t[["n", "accuracy", "initial_accuracy", "fixed", "broken"]].copy()
        t.index = pd.MultiIndex.from_product([[v], t.index], names=[key, "run"])
        parts.append(t)
    return pd.concat(parts) if parts else pd.DataFrame()


LABEL_STATUSES = {
    "confirmed": "a reviewer judged the original gold label correct",
    "alternatives_acceptable": "the original label is correct, but another QID is equally acceptable",
    "corrected": "the original label was judged wrong; gold_corrected holds the right QID(s), or [] if none is known",
    "audited_confirmed": "random audit of a mention where every system agreed with the gold: judged correct",
    "unreviewed_agreement": "every system agreed with the original label and the mention was not sampled for audit",
    "not_reviewed": "a system disagreed with the gold but the mention was not reviewed",
    "unclear": "the reviewer could not decide; excluded from all scores",
}


def corrected_dataset(rows, pooled_csv, include_text=True):
    """Test rows + pooled review -> records with the original and the corrected gold label.
    gold_corrected is a list of acceptable QIDs ("NIL" = not in Wikidata); None for unclear mentions."""
    import hashlib
    import pandas as pd
    acceptable, unclear, _ = pooled_gold(pooled_csv)
    rev = pd.read_csv(pooled_csv, dtype=str, keep_default_na=False)
    group = dict(zip(rev["id"].astype(str), rev["group"]))
    annot = dict(zip(rev["id"].astype(str), rev.get("annotator", pd.Series([""] * len(rev)))))
    out = []
    for r in rows:
        i, g = str(r["id"]), norm_gold(r.get("gold"))
        if i in unclear:
            status, corr = "unclear", None
        elif i in acceptable:
            acc = acceptable[i]
            corr = sorted(acc)
            if g not in acc:
                status = "corrected"
            elif group.get(i) == "audit":
                status = "audited_confirmed"
            else:
                status = "confirmed" if len(acc) == 1 else "alternatives_acceptable"
        elif i in group:
            status, corr = "not_reviewed", [g]
        else:
            status, corr = "unreviewed_agreement", [g]
        text = r.get("text") or ""
        rec = {"id": i, "doc_id": str(r.get("doc_id") or ""), "mention": r.get("mention"),
               "start": r.get("start"), "end": r.get("end"), "type": r.get("type"),
               "popularity": r.get("category"), "sitelinks": r.get("sitelinks"),
               "gold_original": g, "gold_corrected": corr, "label_status": status,
               "annotator": annot.get(i, "") if i in acceptable or i in unclear else "",
               "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest()}
        if include_text:
            rec = {"id": i, "doc_id": rec["doc_id"], "text": text, **{k: v for k, v in rec.items() if k not in ("id", "doc_id")}}
        out.append(rec)
    return out


def corrected_labels(rows):
    """Records from corrected_dataset -> ({id: set of acceptable QIDs}, {unclear ids}) for paired_table/rescoring."""
    acc, unclear = {}, set()
    for r in rows:
        if r.get("gold_corrected") is None:
            unclear.add(str(r["id"]))
        else:
            acc[str(r["id"])] = set(r["gold_corrected"])
    return acc, unclear


def restore_texts(rows, data_path, schema=None):
    """Put the sentence text back into corrected-dataset rows published without text, using the original
    ELNER-DZ file (matched by sentence id and checked with the SHA-1 of the text)."""
    import hashlib
    schema = schema or detect_schema(list(iter_records(data_path, limit=300)))
    want = {str(r["doc_id"]) for r in rows}
    texts = {}
    for i, rec in enumerate(iter_records(data_path)):
        did = str(rec.get(schema.get("id")) if schema.get("id") and rec.get(schema.get("id")) is not None else i)
        if did in want:
            texts[did] = rec.get(schema["text"]) or ""
    ok = bad = 0
    for r in rows:
        t = texts.get(str(r["doc_id"]))
        if t is not None and hashlib.sha1(t.encode("utf-8")).hexdigest() == r.get("text_sha1"):
            r["text"] = t
            ok += 1
        else:
            bad += 1
    print(f"restored {ok} texts" + (f", {bad} not found or changed" if bad else ""))
    return rows



# ---------- Analyses added for the revision (no LLM calls unless stated) ----------
def _good(r, q, acceptable=None):
    """Is answer q correct for record r? Uses the reviewed acceptable set when there is one, else the gold label."""
    acc = (acceptable or {}).get(str(r["id"]))
    return (q in acc) if acc is not None else (q == r["gold"])


def nil_rule_analysis(recs, acceptable=None, exclude=()):
    """Mentions on which the NIL check fired, compared with a FIXED RULE: 'replace NIL by the first candidate whose
    name matches the mention exactly and whose type does not contradict the NER type' (the candidate the check
    showed first). Answers the reviewer's question: why a correction loop instead of the rule?
    Returns (one row per mention, summary dict)."""
    import pandas as pd
    ex = {str(x) for x in exclude}
    rows = []
    for r in _as_records(recs):
        if r["qid"] == "ERROR" or str(r["id"]) in ex:
            continue
        nil_issue = next((i for t in r.get("trace") or [] if t.get("step") == "verify"
                          for i in t["issues"] if i["check"] == "NIL"), None)
        if nil_issue is None:
            continue
        qids = re.findall(r"\b(Q\d+)\b", nil_issue.get("msg", ""))
        rule = qids[0] if qids else "NIL"
        rows.append({"id": str(r["id"]), "mention": r["mention"], "gold": r["gold"], "initial": r.get("initial_qid"),
                     "final": r["qid"], "rule": rule, "exact_matches": "|".join(qids),
                     "agent_kept_nil": r["qid"] == "NIL", "agent_right": _good(r, r["qid"], acceptable),
                     "rule_right": _good(r, rule, acceptable), "same_answer": r["qid"] == rule})
    df = pd.DataFrame(rows)
    if not len(df):
        return df, {}
    kept, changed = df[df.agent_kept_nil], df[~df.agent_kept_nil]
    a_only = int((df.agent_right & ~df.rule_right).sum())
    r_only = int((~df.agent_right & df.rule_right).sum())
    summ = {
        "NIL check fired": len(df),
        "agent kept NIL": len(kept),
        "  kept NIL, NIL right and rule wrong": int((kept.agent_right & ~kept.rule_right).sum()),
        "  kept NIL, rule right and agent wrong": int((~kept.agent_right & kept.rule_right).sum()),
        "  kept NIL, both wrong": int((~kept.agent_right & ~kept.rule_right).sum()),
        "  kept NIL, both right": int((kept.agent_right & kept.rule_right).sum()),
        "agent switched to an entity": len(changed),
        "  switched to the rule's candidate": int(changed.same_answer.sum()),
        "agent right (of fired)": int(df.agent_right.sum()),
        "rule right (of fired)": int(df.rule_right.sum()),
        "agent right, rule wrong": a_only,
        "rule right, agent wrong": r_only,
        "exact sign test p": mcnemar_p(a_only, r_only),
    }
    return df, summ


def apply_nil_rule(recs, wd):
    """'LLM + fixed NIL rule' as a run of its own, built from the LLM run without any LLM call: every NIL answer is
    replaced by the first candidate whose Arabic (for Latin-script mentions also English/French) label or alias
    matches the mention exactly and whose type does not contradict the NER type. initial_qid keeps the LLM's
    answer, so fixed/broken measure the rule itself. Uses the Wikidata cache (SPARQL only for unseen pairs)."""
    out = []
    for r in _as_records(recs):
        r2 = dict(r)
        if r["qid"] == "NIL" and r.get("candidates"):
            ents = wd.get_entities(r["candidates"])
            ntype = norm_type(r.get("type"))
            strong = [ents[q] for q in r["candidates"] if q in ents and label_match(r["mention"], ents[q], exact=True)]
            if ntype in TYPE_ROOTS:
                strong = [c for c in strong if wd.is_instance_of(c["qid"], TYPE_ROOTS[ntype], c["p31"]) is not False]
            if strong:
                r2["initial_qid"], r2["qid"] = "NIL", strong[0]["qid"]
                r2["trace"] = [{"step": "verify", "issues": [{"check": "NIL_RULE", "msg": strong[0]["qid"], "hard": False}]}]
        out.append(r2)
    wd.cache.save()
    return out


def mention_script(mention):
    """'latin' (Arabizi or code-switched names: the agent then also searches English/French), 'arabic' or 'mixed'."""
    lat, ar = bool(re.search("[A-Za-z]", mention or "")), bool(re.search("[؀-ۿ]", mention or ""))
    return "mixed" if lat and ar else ("latin" if lat else "arabic")


def by_script(runs, ref=None, acceptable=None, exclude=()):
    """paired_by over the script of the mention: n, accuracy, initial accuracy, fixed and broken per run."""
    data = {n: [dict(r, script=mention_script(r.get("mention"))) for r in _as_records(v)] for n, v in runs.items()}
    return paired_by(data, key="script", ref=ref, acceptable=acceptable, exclude=exclude)


def unseen_initial(runs, review_csv, acceptable=None, exclude=None):
    """How many repairs / breaks under the corrected labels rest on an INITIAL answer that the review never judged
    (reviewed mention, but the initial answer was not among the options shown). Such answers count as wrong.
    Also counts changes on mentions that were not reviewed at all (scored against the original label)."""
    import pandas as pd
    rev = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    shown = {str(i): set(_split(o)) for i, o in zip(rev["id"], rev["options"])}
    if acceptable is None:
        acceptable, unclear, _ = pooled_gold(review_csv)
    else:
        unclear = {str(x) for x in (exclude or ())}
    out = {}
    for name, recs in runs.items():
        c = dict(fixed=0, fixed_unseen_initial=0, fixed_not_reviewed=0, broken=0, broken_unseen_initial=0,
                 broken_not_reviewed=0)
        for r in _as_records(recs):
            i = str(r["id"])
            ini, fin = r.get("initial_qid"), r["qid"]
            if fin == "ERROR" or i in unclear or ini is None or ini == fin:
                continue
            ok_i, ok_f = _good(r, ini, acceptable), _good(r, fin, acceptable)
            if ok_i == ok_f:
                continue
            kind = "fixed" if ok_f else "broken"
            c[kind] += 1
            if i not in shown:
                c[kind + "_not_reviewed"] += 1
            elif ini not in shown[i]:
                c[kind + "_unseen_initial"] += 1
        out[name] = c
    return pd.DataFrame(out).T


def nil_label_timing(review_csv, wd, cutoff="2025-06-25"):
    """Original NIL labels judged wrong: when was the accepted Wikidata item created? An item created after
    `cutoff` (e.g. the dataset's publication date) did not exist when the data were labeled, so NIL was right
    then. Uses one Wikidata API request per QID (cached). Returns (one row per mention, counts)."""
    import pandas as pd
    df = pd.read_csv(review_csv, dtype=str, keep_default_na=False)
    rows = []
    for _, r in df.iterrows():
        if r["status"] != "done" or json.loads(r["sources"])["gold"] != "NIL":
            continue
        acc = set(_split(r["accepted"]))
        if "NIL" in acc:
            continue                                    # the NIL label was judged acceptable
        qids = sorted(q for q in acc if re.fullmatch(r"Q\d+", q))
        dates = {q: wd.created(q) for q in qids}
        first = min((d for d in dates.values() if d), default=None)
        status = ("no QID accepted" if not qids else "creation date unknown" if not first else
                  "item existed before cutoff" if first[:10] < cutoff else "item created after cutoff")
        rows.append({"id": r["id"], "mention": r["mention"], "accepted": "|".join(qids),
                     "created": "|".join(f"{q}:{(dates[q] or '?')[:10]}" for q in qids),
                     "earliest": (first or "")[:10], "status": status})
    wd.cache.save()
    out = pd.DataFrame(rows)
    return out, (out["status"].value_counts().to_dict() if len(out) else {})


def corpus_profile(records, schema=None, limit=None):
    """Profile of a corpus or a sample, to check that a sample looks like the corpus it came from.
    records: raw records (pass the schema from detect_schema) streamed from a file, or agent rows (schema=None).
    Mention-level shares (NIL, Latin script, entity types) are comparable between the two."""
    from collections import Counter
    st, types, docs = Counter(), Counter(), set()

    def add_rows(rows):
        st["mentions"] += len(rows)
        st["nil"] += sum(r["gold"] == "NIL" for r in rows)
        st["latin"] += sum(mention_script(r["mention"]) != "arabic" for r in rows)
        types.update(str(r.get("type")) for r in rows)

    if schema is None:
        rows = _as_records(records)
        add_rows(rows)
        texts = {}
        for r in rows:
            texts.setdefault(str(r.get("doc_id") or r["text"]), r["text"])
        st["sentences"] = len(texts)
        st["latin_sent"] = sum(not re.search("[؀-ۿ]", t) for t in texts.values())
        st["chars"] = sum(len(t) for t in texts.values())
    else:
        for k, rec in enumerate(records):
            if limit and k >= limit:
                break
            rows = convert_records([rec], schema)
            text = rows[0]["text"] if rows else rec.get(schema.get("text") or "", "")
            if not isinstance(text, str) or not text.strip():
                continue
            st["sentences"] += 1
            st["with_mentions"] += bool(rows)
            st["latin_sent"] += not re.search("[؀-ۿ]", text)
            st["chars"] += len(text)
            add_rows(rows)
    n, m = st["sentences"] or 1, st["mentions"] or 1
    prof = {"sentences": st["sentences"], "mentions": st["mentions"],
            "share of mentions labeled NIL": round(st["nil"] / m, 4),
            "share of mentions in Latin script": round(st["latin"] / m, 4),
            "share of sentences without Arabic script": round(st["latin_sent"] / n, 4),
            "mean sentence length (chars)": round(st["chars"] / n, 1)}
    if schema is not None:
        prof["mentions per sentence with mentions"] = round(st["mentions"] / (st["with_mentions"] or 1), 3)
    for t, c in types.most_common(10):
        prof[f"type {t}"] = round(c / m, 4)
    return prof


def multi_run_summary(run_files, names, tags, ref="llm_no_correction", acceptable=None, exclude=()):
    """Repeated runs: run files are named <config><tag>, e.g. 'full_no_memory' (first run, tag ''),
    'full_no_memory@r2', 'full_no_memory@r3'. Returns (one row per system and run, mean/sd/min/max per system).
    Runs repeat the same mentions, so per-run tests are kept separate rather than pooled."""
    import pandas as pd, contextlib, io
    per = []
    for tag in tags:
        sub = {n: run_files[n + tag] for n in names if (n + tag) in run_files}
        if not sub:
            continue
        r = ref if ref in sub else None
        with contextlib.redirect_stdout(io.StringIO()):
            t = paired_table(sub, ref=r, acceptable=acceptable, exclude=exclude, n_boot=0)
        ref_acc = float(t.loc[r, "accuracy"]) if r else None
        for n in t.index:
            row = {k: t.loc[n, k] for k in ("n", "accuracy", "initial_accuracy", "fixed", "broken", "sign_test_p",
                                             "wins_vs_ref", "losses_vs_ref", "p_vs_ref", "avg_llm_calls")}
            row.update(system=n.split("@")[0], run=tag or "(first)",
                       diff_vs_ref=(float(row["accuracy"]) - ref_acc) if ref_acc is not None else None)
            per.append(row)
    df = pd.DataFrame(per)
    if not len(df):
        return df, df
    for c in ("n", "accuracy", "initial_accuracy", "fixed", "broken", "wins_vs_ref", "losses_vs_ref",
              "avg_llm_calls", "diff_vs_ref"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby("system").agg(runs=("run", "count"), acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                                   acc_min=("accuracy", "min"), acc_max=("accuracy", "max"),
                                   diff_vs_ref_mean=("diff_vs_ref", "mean"), diff_vs_ref_sd=("diff_vs_ref", "std"),
                                   fixed_total=("fixed", "sum"), broken_total=("broken", "sum"),
                                   calls=("avg_llm_calls", "mean")).round(4)
    return df, agg
