"""Offline question generation (no LLM): sentence retrieval + cloze MCQs.
Answer terms  = syllabus terms, acronyms, RAKE-style phrases repeated in the notes.
Distractors   = same-topic terms (Hard) ... other-topic terms (Easy), same term shape."""
import re, pathlib, random, numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from .ingest import read_pages, chunk_material

def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]

class Retriever:
    """TF-IDF retrieval over study material (chunks for context, sentences for questions)."""
    def __init__(self, material_dir):
        p = pathlib.Path(material_dir) if material_dir else None
        files = sorted(x for x in p.iterdir() if x.is_file() and not x.name.startswith(".")) if p and p.exists() else []
        self.pages = []                                     # (file, location, text)
        for f in files:
            try:
                self.pages += [(f.name, loc, t) for loc, t in read_pages(f) if t.strip()]
            except Exception as e:
                print(f"[WARN] could not read {f.name}: {e}")
        self.texts = ["\n\n".join(t for fn, _, t in self.pages if fn == f.name) for f in files]
        self.chunks, self.chunk_src, self.src = [], [], {}
        for fn, loc, t in self.pages:
            for c in chunk_material(t):
                self.chunks.append(c); self.chunk_src.append(f"{fn}, {loc}")
            for sent in split_sentences(t):
                self.src.setdefault(sent, f"{fn}, {loc}")   # sentence -> where it came from
        self.sents = list(self.src)
        if self.chunks:
            self.v = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(self.chunks + self.sents)
            self.M, self.S = self.v.transform(self.chunks), self.v.transform(self.sents)
    def top_with_src(self, query, k=3):
        if not self.chunks:
            return []
        s = self._rank(self.M, query)
        return [(self.chunks[i], self.chunk_src[i]) for i in np.argsort(-s)[:k] if s[i] > 0]
    def _rank(self, X, query):
        return (X @ self.v.transform([query]).T).toarray().ravel()
    def top(self, query, k=3):
        if not self.chunks:
            return []
        s = self._rank(self.M, query)
        return [self.chunks[i] for i in np.argsort(-s)[:k] if s[i] > 0]
    topic_src = {}                                     # subtopic -> {"file, loc", ...} from the topic map

    def top_sentences(self, query, k=12, min_score=0.08, sub=None):
        if not self.chunks:
            return []
        s = self._rank(self.S, query)
        if sub and self.topic_src.get(sub):               # content-mapped material for this subtopic ranks higher
            mapped = self.topic_src[sub]
            s = s + np.array([0.15 if self.src.get(x) in mapped else 0.0 for x in self.sents])
        return [self.sents[i] for i in np.argsort(-s)[:k] if s[i] >= min_score]

STOP = set(ENGLISH_STOP_WORDS) | {"uses", "used", "using", "selects", "checks", "moves", "returns", "replaces",
    "occurs", "happens", "prevents", "requires", "include", "includes", "common", "like", "end", "spends", "finding",
    "choosing", "handling", "done", "granted", "satisfied", "additional", "information", "remaining", "shows", "reduces"}

def rake_phrases(texts, min_count=2):
    """Contiguous 2-3 word phrases between stopwords/punctuation, repeated >= min_count times."""
    cnt, form = Counter(), {}
    for t in texts:
        for sent in split_sentences(t):
            run = []
            for tok in re.findall(r"[A-Za-z][A-Za-z'-]*|[^\sA-Za-z]", sent) + ["."]:
                if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", tok) and tok.lower() not in STOP:
                    run.append(tok); continue
                if 2 <= len(run) <= 3:
                    k = " ".join(run).lower(); cnt[k] += 1; form.setdefault(k, " ".join(run))
                run = []
    return [form[k] for k, c in cnt.items() if c >= min_count]

FIG_REF = re.compile(r"\b(figure|fig\.|diagram|graph|plot|shown (?:above|below|here)|as shown|following (?:figure|table|diagram)|table \d|below|above)\b", re.I)

def _clean_sentence(s):
    """A cloze sentence must stand alone: no table rows, OCR'd image text, figure references or number dumps."""
    if "|" in s or "[Image]" in s or "Equation:" in s or FIG_REF.search(s):
        return False
    toks = s.split()
    return sum(bool(re.fullmatch(r"[\d.,=+\-()%]+", t)) for t in toks) / len(toks) < 0.25

GENERIC = {"process", "processes", "time", "system", "algorithm", "algorithms", "cpu", "general"}

class ClozeGenerator:
    def __init__(self, syl, retriever, seed=None):
        self.ret, self.rng = retriever, random.Random(seed)
        self.term_topic, self.sub_text = {}, {r["subtopic"]: f"{r.get('group', '')} {r['text']}" for r in syl}
        for r in syl:
            self.term_topic[r["subtopic"]] = r["topic"]
            m = re.search(r"\((.*?)\)", r["text"])
            for x in (m.group(1).split(",") if m else []):
                self.term_topic.setdefault(x.strip(), r["topic"])
        extra = set(rake_phrases(retriever.texts))
        for t in retriever.texts:
            extra.update(re.findall(r"\b[A-Z][A-Z0-9]{1,}(?:-[A-Z]+)?\b", t))
        labels = [f"{r['topic']} {r['subtopic']}" for r in syl]
        for term in extra - set(self.term_topic):          # assign topic via context sentences
            ctx = " ".join(s for s in retriever.sents if term.lower() in s.lower())
            if labels and ctx:
                sc = [(retriever.v.transform([ctx]) @ retriever.v.transform([l]).T).toarray()[0, 0] for l in labels]
                self.term_topic[term] = syl[int(np.argmax(sc))]["topic"]
        self.terms = sorted(t for t in self.term_topic if len(t) > 2 and t.lower() not in GENERIC)

    @staticmethod
    def _shape(t):
        return "acronym" if t.isupper() else "multi" if " " in t else "single"

    @staticmethod
    def _find(term, text):
        return re.search(rf"(?<![\w-]){re.escape(term)}(?![\w-])", text, re.I)

    def _distractors(self, ans, sentence, difficulty):
        topic, aw = self.term_topic.get(ans), set(ans.lower().split())
        ok = [t for t in self.terms if t != ans and not (set(t.lower().split()) & aw) and not self._find(t, sentence)]
        same_shape = [t for t in ok if self._shape(t) == self._shape(ans)]
        pool = same_shape if len(same_shape) >= 3 else ok
        near = [t for t in pool if self.term_topic.get(t) == topic]
        far = [t for t in pool if self.term_topic.get(t) != topic]
        self.rng.shuffle(near); self.rng.shuffle(far)
        k_near = {"Hard": 3, "Medium": 1, "Easy": 0}[difficulty]
        pick = near[:k_near] + far
        pick = (pick + near[k_near:])[:3]
        return pick if len(pick) == 3 else None

    def generate(self, topic, sub, difficulty="Medium", n=5):
        cands = []
        sub_terms = {sub.lower()} | {x.strip().lower() for x in re.findall(r"\((.*?)\)", self.sub_text.get(sub, ""))
                                     for x in x.split(",")}
        query = f"{topic} {sub} {self.sub_text.get(sub, '')}".replace("(", " ").replace(")", " ")
        for s in self.ret.top_sentences(query, sub=sub):
            if not (8 <= len(s.split()) <= 45) or not _clean_sentence(s):
                continue
            present = [t for t in self.terms if self._find(t, s) and len(t) / len(s) < 0.4]
            if not present:
                continue
            # prefer terms belonging to the asked topic, then the most specific (longest)
            present = [t for t in present if self.term_topic.get(t) == topic]   # must test the asked topic
            if not present:
                continue
            ans = max(present, key=lambda t: (t.lower() in sub_terms, len(t)))
            d = self._distractors(ans, s, difficulty)
            if not d:
                continue
            m = self._find(ans, s)
            blank = s[:m.start()] + "_____" + s[m.end():]
            opts = d + [ans]; self.rng.shuffle(opts)
            cands.append((ans.lower() in sub_terms, dict(question=f"Fill in the blank: {blank}", options=opts,
                           answer_index=opts.index(ans), explanation=f"From your notes: \"{s}\"",
                           source=self.ret.src.get(s, ""), generator="cloze")))
        exact = [q for e, q in cands if e]
        return (exact or [q for _, q in cands])[:n]   # exact-subtopic questions first; topic-level only as fallback
