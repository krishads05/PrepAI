"""Answer Evaluator: grades free-text answers against the student's own notes.
Offline score = key-point coverage + key-term coverage + overall semantic similarity.
If an LLM backend (Ollama/Anthropic) is available, it also grades with a rubric; both are reported."""
import re, numpy as np
from . import llm
from .qgen import split_sentences, STOP

def _qwords(question):
    return {w.lower() for w in re.findall(r"[\w'-]+", question) if w.lower() not in STOP}

def _strip(text, qw):
    """Remove question words: restating the question earns no credit."""
    return " ".join(w for w in re.findall(r"[\w'-]+", text) if w.lower() not in qw)

class _Sim:
    """Sentence similarity: MiniLM if installed, else the retriever's TF-IDF space."""
    def __init__(self, retriever):
        self.ret = retriever
        try:
            from sentence_transformers import SentenceTransformer
            self.st, self.kind, self.cover_t = SentenceTransformer("all-MiniLM-L6-v2"), "sbert", 0.55
        except Exception:
            self.st, self.kind, self.cover_t = None, "tfidf", 0.25
    def enc(self, texts):
        if self.st is not None:
            return self.st.encode(list(texts), normalize_embeddings=True)
        from sklearn.preprocessing import normalize
        return normalize(self.ret.v.transform(list(texts)).toarray())

GENERIC_Q = {"explain", "describe", "discuss", "write", "short", "note", "notes", "define", "brief", "briefly",
             "example", "neat", "diagram", "problem", "problems", "solution", "using", "methods", "method", "different"}

class AnswerEvaluator:
    def __init__(self, retriever, terms=()):
        self.ret, self.terms, self.sim = retriever, list(terms), _Sim(retriever)
        self.pars = [split_sentences(p) for t in retriever.texts for p in re.split(r"\n\s*\n", t) if p.strip()]

    def reference(self, question, sub, window=4):
        """Sentence-window retrieval: best-matching sentences + following sentences in the same paragraph.
        A seed sentence must share >= 2 content words with the question (or score very high)."""
        stem = lambda w: w[:-1] if len(w) > 4 and w.endswith("s") else w
        qc = {stem(w) for w in _qwords(question) - GENERIC_Q}
        refs = []
        for seed in self.ret.top_sentences(f"{question} {sub}", k=3, min_score=0.12, sub=sub):
            overlap = qc & {stem(w.lower()) for w in re.findall(r"[\w'-]+", seed)}
            if len(overlap) < 2 and len(qc) >= 2:
                continue
            for par in self.pars:
                if seed in par:
                    i = par.index(seed)
                    refs += [x for x in par[i:i + window] if x not in refs]
        refs = refs[:6]
        qw = _qwords(question)
        terms = [t for t in self.terms if not all(w.lower() in qw for w in re.findall(r"[\w'-]+", t)) and any(re.search(rf"(?<![\w-]){re.escape(t)}(?![\w-])", r, re.I) for r in refs)]
        return refs, terms

    def grade(self, question, answer, topic, sub, max_marks=10):
        refs, terms = self.reference(question, sub)
        words = len(answer.split())
        if not refs:
            return dict(gradable=False, method="none",
                        feedback="Your study material doesn't cover this question, so it can't be graded automatically. Grade yourself.")
        if words < 5:
            return dict(gradable=True, method="offline", score=0.0, correct=False, covered=[], missed=refs,
                        missing_terms=terms, feedback="Answer too short to evaluate. Write at least a few sentences.")
        qw = _qwords(question)
        a_sents = [x for x in (_strip(t, qw) for t in (split_sentences(answer) or [answer])) if x.strip()] or [""]
        A = self.sim.enc(a_sents); R = self.sim.enc([_strip(r, qw) for r in refs])
        S = R @ A.T                                            # ref point x answer sentence
        hit = S.max(axis=1) >= self.sim.cover_t
        overall = float((self.sim.enc([_strip(answer, qw)]) @ self.sim.enc([_strip(" ".join(refs), qw)]).T).ravel()[0])
        low = answer.lower()
        term_hit = [t for t in terms if t.lower() in low]
        point_cov = hit.mean()
        term_cov = len(term_hit) / len(terms) if terms else point_cov
        sem = np.clip((overall - (0.2 if self.sim.kind == "sbert" else 0.05)) / 0.5, 0, 1)
        score = round(10 * (0.45 * point_cov + 0.30 * term_cov + 0.25 * sem), 1)
        res = dict(gradable=True, method=f"offline ({self.sim.kind})", score=score, correct=score >= 6,
                   covered=[r for r, h in zip(refs, hit) if h], missed=[r for r, h in zip(refs, hit) if not h],
                   missing_terms=[t for t in terms if t not in term_hit],
                   breakdown=dict(key_points=round(float(point_cov), 2), key_terms=round(float(term_cov), 2),
                                  similarity=round(float(sem), 2)))
        res["sources"] = {r: self.ret.src.get(r, "") for r in refs}
        res["feedback"] = self._feedback(res)
        if llm.available():
            try:
                res.update(self._llm_grade(question, answer, refs, res["score"]))
            except Exception as e:
                res["llm_error"] = type(e).__name__
        res["marks"] = round(res["score"] / 10 * max_marks, 1)
        return res

    @staticmethod
    def _feedback(r):
        if r["score"] >= 8:
            msg = "Strong answer that covers the key points."
        elif r["score"] >= 6:
            msg = "Good answer, but some points are missing."
        elif r["score"] >= 3:
            msg = "Partial answer. Several key points are missing."
        else:
            msg = "The answer misses most of the expected content."
        if r["missing_terms"]:
            msg += f" Try to mention: {', '.join(r['missing_terms'][:5])}."
        return msg

    def _llm_grade(self, question, answer, refs, offline_score):
        out = llm.ask_json(
            "You are an exam evaluator. Grade the student's answer out of 10 using ONLY the reference notes below as "
            "the source of truth - do not use outside knowledge, and do not penalise points the notes do not mention. "
            "Be fair to answers that use different wording.\n"
            f"QUESTION: {question}\nREFERENCE NOTES:\n- " + "\n- ".join(f"({self.ret.src.get(r, '')}) {r}" for r in refs) +
            f"\nSTUDENT ANSWER: {answer}\n"
            '{"score": n, "correct_points": ["..."], "missing_points": ["..."], "misconceptions": ["..."], '
            '"feedback": "2-3 sentences of specific, encouraging feedback"}', max_tokens=700)
        s = float(out.get("score", offline_score))
        return dict(method=f"LLM ({llm.backend()}) + offline", llm_score=s, offline_score=offline_score,
                    score=round(0.7 * s + 0.3 * offline_score, 1), correct=(0.7 * s + 0.3 * offline_score) >= 6,
                    feedback=out.get("feedback", ""), misconceptions=out.get("misconceptions", []),
                    llm_missing=out.get("missing_points", []))
