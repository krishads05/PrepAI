"""Semantic PYQ -> syllabus mapping + question-type classification."""
import re, numpy as np
from sklearn.preprocessing import normalize

class Embedder:
    """mode: 'tfidf' | 'sbert' | 'hybrid' (w*sbert + (1-w)*tfidf cosine).
    Hybrid concatenates sqrt-weighted unit vectors, so a dot product = weighted sum of both cosines."""
    def __init__(self, model="all-MiniLM-L6-v2", mode="hybrid", w=0.5):
        self.mode, self.w, self.st, self.model = mode, w, None, model
        if mode in ("sbert", "hybrid"):
            try:
                from sentence_transformers import SentenceTransformer
                self.st = SentenceTransformer(model)
            except Exception as e:
                print(f"[WARN] Could not load '{model}' -> {type(e).__name__}: {e}\n[WARN] Falling back to TF-IDF.")
                self.mode = "tfidf"
        self.kind = self.mode
    def fit(self, corpus):
        if self.mode != "sbert":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.wv = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(corpus)
            self.cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit(corpus)
        return self
    def _tfidf(self, texts):
        from scipy.sparse import hstack
        return normalize(hstack([self.wv.transform(texts), 0.7 * self.cv.transform(texts)]).toarray())
    def encode(self, texts):
        texts = list(texts)
        if self.mode == "tfidf":
            return self._tfidf(texts)
        S = self.st.encode(texts, normalize_embeddings=True)
        if self.mode == "sbert":
            return S
        return np.hstack([np.sqrt(self.w) * S, np.sqrt(1 - self.w) * self._tfidf(texts)])

# thresholds per mode (cosine)
THRESH = {"sbert": 0.30, "tfidf": 0.10, "hybrid": 0.20}
MARGIN = {"sbert": 0.03, "tfidf": 0.05, "hybrid": 0.04}   # below -> low confidence

def subtopic_vectors(emb, syl, chunks, k=2, alpha=0.65, assigned=None, beta=0.5):
    """Label embedding blended with study material.
    assigned: {subtopic: [texts of slides/pages the topic map assigned to it]} - content-based grounding,
    so questions are matched against what the material actually says about each subtopic."""
    labels = [f"{r['topic']}: {r.get('group', '')}: {r['text']}" for r in syl]
    L = emb.encode(labels)
    out = []
    C = emb.encode(chunks) if chunks else None
    for i, r in enumerate(syl):
        texts = (assigned or {}).get(r["subtopic"], [])
        if texts:
            M = emb.encode([" ".join(t.split()[:200]) for t in texts[:25]])
            c = normalize(M.mean(0, keepdims=True))[0]
            v = beta * L[i] + (1 - beta) * c
        elif C is not None:
            S = L[i] @ C.T
            top = [j for j in np.argsort(-S)[:k] if S[j] > 0.05]
            v = alpha * L[i] + (1 - alpha) * C[top].mean(0) if top else L[i]
        else:
            v = L[i]
        out.append(v)
    return normalize(np.array(out))

STOPW = {"and", "of", "the", "for", "in", "a", "an", "to", "using", "from", "with", "on", "by"}
TAIL = {"algorithm", "algorithms", "model", "models", "method", "methods", "classifier"}

def acronyms(syl, texts=()):
    """{ACRONYM: long form} learned from syllabus names ('Support Vector Machines' -> SVM, KNN, CART...)
    and from 'Long Form (ACR)' / 'ACR (Long Form)' patterns in the study material."""
    out = {}
    names = [r["subtopic"] for r in syl] + [r.get("group", "") for r in syl] + [r["text"] for r in syl]
    for n in names:
        words = [w for w in re.findall(r"[A-Za-z]+", n) if w.lower() not in STOPW]
        for ws in (words, [w for w in words if w.lower() not in TAIL]):
            if 2 <= len(ws) <= 6:
                out.setdefault("".join(w[0] for w in ws).upper(), " ".join(ws))
    for t in texts:
        for long, ac in re.findall(r"((?:[A-Z][a-z]+[\s-]+){1,5}[A-Z][a-z]+)\s*\(([A-Z]{2,6})s?\)", t):
            out.setdefault(ac, long)
        for ac, long in re.findall(r"\b([A-Z]{2,6})\s*\(((?:[A-Z][a-z]+[\s-]+){1,5}[A-Z][a-z]+)\)", t):
            out.setdefault(ac, long)
    return {k: v for k, v in out.items() if len(k) >= 2}

def expand_acronyms(text, acr):
    """'Summarize SVMs and K-NN' -> 'Summarize SVMs (Support Vector Machines) and K-NN (K Nearest Neighbor)'"""
    def rep(m):
        key = m.group(1).replace("-", "").upper()
        return f"{m.group(0)} ({acr[key]})" if key in acr else m.group(0)
    return re.sub(r"\b([A-Z](?:-?[A-Z]){1,5})s?\b", rep, text)

def map_questions(emb, questions, syl, T, acr=None):
    Q = emb.encode([expand_acronyms(q["question"], acr) if acr else q["question"] for q in questions])
    S = Q @ T.T
    th = THRESH[emb.kind]
    for q, s in zip(questions, S):
        order = np.argsort(-s)
        b, b2 = order[0], order[1] if len(order) > 1 else order[0]
        q["sim"], q["margin"] = round(float(s[b]), 3), round(float(s[b] - s[b2]), 3)
        if s[b] >= th:
            q["topic"], q["subtopic"] = syl[b]["topic"], syl[b]["subtopic"]
        else:
            q["topic"], q["subtopic"] = "UNMAPPED", "UNMAPPED"
        q["low_confidence"] = bool(q["margin"] < MARGIN[emb.kind] or s[b] < th * 1.5)
        q["candidates"] = [syl[j]["subtopic"] for j in order[:4]]
    return questions

TYPE_RULES = [
    ("numerical", r"\b(calculate|compute|find|determine|how many|evaluate)\b.*\d|\d+\s*(processes|resources|ms|kb|mb|frames|blocks|instances)"),
    ("derivation", r"\b(derive|prove|show that)\b"),
    ("algorithm-tracing", r"\b(trace|simulate|apply|gantt|safe sequence|step[- ]by[- ]step|using .*algorithm)\b"),
    ("comparison", r"\b(compare|differentiate|distinguish|difference|versus|vs\.?)\b"),
    ("application", r"\b(design|suggest|propose|scenario|write (a|an) (program|algorithm|code)|implement)\b"),
    ("theory", r"\b(explain|define|describe|what (is|are)|discuss|list|state|why|briefly|short note|illustrate)\b"),
]

def classify_type(text):
    t = text.lower()
    types = [name for name, rx in TYPE_RULES if re.search(rx, t)]
    return types or ["theory"]

def llm_refine(questions, syl, model="claude-sonnet-4-5"):
    """Optional: re-map low-confidence questions with an LLM (needs ANTHROPIC_API_KEY)."""
    import os, json
    if not os.getenv("ANTHROPIC_API_KEY"):
        return questions
    import anthropic
    client = anthropic.Anthropic()
    topic_of = {r["subtopic"]: r["topic"] for r in syl}
    for q in questions:
        if not q.get("low_confidence"):
            continue
        prompt = (f"Exam question: {q['question']}\nCandidate syllabus subtopics: {json.dumps(list(topic_of))}\n"
                  "Which ONE subtopic does this question test? Reply with only the exact subtopic string, or NONE.")
        ans = client.messages.create(model=model, max_tokens=50,
                                     messages=[{"role": "user", "content": prompt}]).content[0].text.strip()
        if ans in topic_of:
            q["subtopic"], q["topic"], q["mapped_by"] = ans, topic_of[ans], "llm"
    return questions
