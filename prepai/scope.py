"""Exam scopes (mid-sem / internal portions) as subsets of the full syllabus.
Portions text is matched to syllabus subtopics via module references (Module 1-2, Unit I to III),
topic names (whole topic), subtopic names / listed terms (specific subtopics), then fuzzy matching."""
import json, pathlib, re
from sklearn.feature_extraction.text import TfidfVectorizer

ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9, "x": 10}
MOD_REF = re.compile(r"\b(?:modules?|units?|chapters?|mods?)\s*((?:[0-9]+|[ivx]+)(?:\s*(?:,|and|&|-|\u2013|to)\s*(?:[0-9]+|[ivx]+))*)\b", re.I)

def _num(tok):
    tok = tok.lower().strip()
    return int(tok) if tok.isdigit() else ROMAN.get(tok)

def module_refs(text):
    mods = set()
    for m in MOD_REF.finditer(text):
        toks = re.split(r"\s*(,|and|&)\s*", m.group(1))
        for part in toks:
            if part in (",", "and", "&") or not part.strip():
                continue
            rng = re.split(r"\s*(?:-|\u2013|to)\s*", part)
            nums = [_num(x) for x in rng if _num(x)]
            if len(nums) == 2:
                mods.update(range(min(nums), max(nums) + 1))
            elif nums:
                mods.add(nums[0])
    return mods

def _items(text):
    out, buf, depth = [], "", 0
    for ch in text:
        depth += ch == "("; depth -= ch == ")"
        if ch in ",;\n" and depth == 0:
            out.append(buf); buf = ""
        else:
            buf += ch
    out.append(buf)
    return [re.sub(r"^[\s\-\u2022*\d.)]+", "", x).strip(" .:") for x in out if x.strip(" .:-\u2022*")]

def _contains(a, b):
    """True if phrase b appears in a as whole words (case-insensitive)."""
    return len(b) > 2 and re.search(rf"(?<![\w-]){re.escape(b.lower())}s?(?![\w-])", a.lower()) is not None

def match_scope(text, syl, fuzzy=0.55):
    """Returns dict(subtopics=set, matches=[(item, matched_to, how)], unmatched=[...])."""
    by_topic = {}
    for r in syl:
        by_topic.setdefault(r["topic"], []).append(r["subtopic"])
    sub_terms = {}
    for r in syl:
        m = re.search(r"\((.*?)\)", r["text"])
        sub_terms[r["subtopic"]] = [r["subtopic"]] + ([x.strip() for x in m.group(1).split(",") if x.strip()] if m else [])
    selected, matches, unmatched = set(), [], []

    mods = module_refs(text)
    for mnum in sorted(mods):
        subs = [r["subtopic"] for r in syl if r["module"] == mnum]
        if subs:
            selected.update(subs)
            matches.append((f"Module {mnum}", f"{subs and next(r['topic'] for r in syl if r['module'] == mnum)} ({len(subs)} subtopics)", "module number"))
        else:
            unmatched.append(f"Module {mnum} (not in syllabus)")

    names = list(sub_terms)
    tv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(names + list(by_topic))
    for item in _items(MOD_REF.sub(" ", text)):
        if len(item) < 3 or re.fullmatch(r"(?i)(mid[- ]?sem.*|internal.*|portions?.*|syllabus.*|exam.*|and|only)", item):
            continue
        subs = [s for s, terms in sub_terms.items() if any(_contains(item, t) for t in terms)]
        if subs:                                              # most specific: named subtopics / listed terms
            selected.update(subs); matches.append((item, ", ".join(subs), "subtopic name")); continue
        tops = [t for t in by_topic if _contains(item, t) or _contains(t, item)]
        if tops:                                              # whole topic
            for t in tops:
                selected.update(by_topic[t])
            matches.append((item, ", ".join(f"{t} (all {len(by_topic[t])})" for t in tops), "topic name")); continue
        sims = (tv.transform([item]) @ tv.transform(names).T).toarray().ravel()
        j = int(sims.argmax())
        if sims[j] >= fuzzy:
            selected.add(names[j]); matches.append((item, names[j], f"fuzzy ({sims[j]:.2f})"))
        else:
            unmatched.append(item)
    return dict(subtopics=selected, matches=matches, unmatched=unmatched)

class ScopeStore:
    def __init__(self, out):
        self.p = pathlib.Path(out) / "scopes.json"
    def all(self):
        return json.loads(self.p.read_text()) if self.p.exists() else {}
    def save(self, name, subtopics):
        d = self.all(); d[name] = sorted(subtopics); self.p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    def delete(self, name):
        d = self.all(); d.pop(name, None); self.p.write_text(json.dumps(d, indent=1), encoding="utf-8")
