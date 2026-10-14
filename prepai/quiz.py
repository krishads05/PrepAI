"""Quiz Me: grounded MCQ generation (LLM, cached) + adaptive topic/difficulty selection.
Offline fallback: PYQ practice with self-grading."""
import json, pathlib, random, re, time, numpy as np, pandas as pd
from . import llm
from .qgen import Retriever, ClozeGenerator

def _grounded(quote, text):
    """Quote must be (almost) verbatim in the source text."""
    qw = re.findall(r"[a-z0-9]+", str(quote).lower()); tw = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(qw) >= 4 and sum(w in tw for w in qw) / len(qw) >= 0.85

def generate_mcqs(topic, sub, difficulty, context, pyqs, n=5):
    """context: [(chunk_text, source_label)]. Questions must be answerable ONLY from these excerpts."""
    if not context:
        return []
    ex = "\n".join(f"[S{i + 1}] ({src}) {txt}" for i, (txt, src) in enumerate(context))
    prompt = (f"Create {n} {difficulty}-difficulty multiple-choice questions on '{sub}' ({topic}).\n"
              "STRICT RULES: use ONLY facts stated in the excerpts below - no outside knowledge. "
              "Every question must be answerable from one excerpt. For each question give source_id (e.g. S2) "
              "and a supporting quote copied word-for-word from that excerpt (max 25 words). "
              "4 options, exactly one correct. If the excerpts support fewer questions, return fewer.\n\n"
              f"EXCERPTS:\n{ex}\n\nPAST EXAM QUESTIONS (style only, not facts):\n{chr(10).join(pyqs[:5])}\n\n"
              '{"questions": [{"question": "...", "options": ["A","B","C","D"], "answer_index": 0, '
              '"explanation": "...", "source_id": "S1", "quote": "exact words from the excerpt"}]}')
    out = []
    for q in llm.ask_json(prompt, max_tokens=2500).get("questions", []):
        if len(q.get("options", [])) != 4 or q.get("answer_index") not in range(4):
            continue
        m = re.match(r"S(\d+)", str(q.get("source_id", "")))
        i = int(m.group(1)) - 1 if m else -1
        if not (0 <= i < len(context)) or not _grounded(q.get("quote", ""), context[i][0]):
            continue                                        # not supported by the materials -> drop
        q.update(source=context[i][1], generator="llm",
                 explanation=f"{q.get('explanation', '')} Source text: \"{q['quote']}\"")
        out.append(q)
    return out

class QuizEngine:
    def __init__(self, table, qdf, out_dir, material_dir=None, syl=None, pyq_share=0.3):
        self.t, self.qdf, self.out, self.pyq_share = table, qdf, pathlib.Path(out_dir), pyq_share
        self.asked = set()
        self.bank_path = self.out / "question_bank_grounded.json"
        self.bank = json.loads(self.bank_path.read_text()) if self.bank_path.exists() else {}
        self.ret = Retriever(material_dir)
        self.cloze = ClozeGenerator(syl or [], self.ret) if self.ret.chunks else None

    scope = None                                               # set of subtopics, or None = full syllabus

    def pick_topic(self, mode, mastery, topic=None, recent=()):
        t = self.t[self.t.subtopic.isin(self.scope)].copy() if self.scope else self.t[self.t.papers > 0].copy()
        if t.empty:
            t = self.t[self.t.papers > 0].copy()
        if topic:
            t = t[t.topic.str.contains(topic, case=False) | t.subtopic.str.contains(topic, case=False)]
        t["m"] = t.subtopic.map(mastery).fillna(0.5)
        w = {"important": (t.importance / 100) ** 3,
             "weak": (1 - t.m) ** 2,
             "all": pd.Series(1.0, index=t.index)}.get(mode, (t.importance / 100) * (1.1 - t.m))  # mixed/adaptive
        w = (w + 0.05) * np.where(t.subtopic.isin(list(recent)[-2:]), 0.3, 1.0)               # avoid repeats; never zero
        r = t.loc[np.random.choice(t.index, p=(w / w.sum()).values)]
        return r.topic, r.subtopic, r.m

    def pick_difficulty(self, sub, fixed=None, attempts=None):
        """Adaptive: start at topic difficulty (capped at Medium), step up after a right answer,
        down after a wrong one on that subtopic."""
        if fixed and fixed != "Adaptive":
            return fixed
        levels = ["Easy", "Medium", "Hard"]
        base = self.t.set_index("subtopic")["difficulty"].get(sub, "Medium")
        i = min(levels.index(base), 1)
        if attempts is not None and len(attempts):
            g = attempts[attempts.subtopic == sub].sort_values("ts")
            if len(g):
                i += 1 if int(g.correct.iloc[-1]) == 1 else -1
        return levels[max(0, min(2, i))]

    def _take(self, pool):
        pool[:] = [q for q in pool if q["question"] not in self.asked]
        return pool.pop(random.randrange(len(pool))) if pool else None

    def _pyq(self, sub):
        rows = self.qdf[(self.qdf.subtopic == sub) & ~self.qdf.question.isin(self.asked)]
        if not len(rows):
            return None
        r = rows.sample(1).iloc[0]
        yr = f" {int(r.year)}" if pd.notna(r.get("year")) else ""
        img = r.get("image") if isinstance(r.get("image"), str) and pathlib.Path(r.get("image")).exists() else None
        return dict(kind="pyq", question=r.question, source=f"Past paper{yr} ({r.paper_id}, Q{r.qno}{r.part if isinstance(r.part, str) else ''})",
                    marks=None if pd.isna(r.get("marks")) else int(r.marks), image=img)

    def get_question(self, topic, sub, diff):
        """Order: PYQ-style practice (some of the time) -> grounded LLM MCQ -> offline cloze MCQ -> PYQ.
        Every question carries a 'source' citing the past paper or the material file + page/slide."""
        q = self._pyq(sub) if random.random() < self.pyq_share else None
        if q is None:
            key = f"{sub}|{diff}"
            if llm.available():
                if not self.bank.get(key):
                    try:
                        self.bank[key] = generate_mcqs(topic, sub, diff, self.ret.top_with_src(f"{topic} {sub}"),
                                                       self.qdf[self.qdf.subtopic == sub].question.tolist())
                        self.bank_path.write_text(json.dumps(self.bank, indent=1), encoding="utf-8")
                    except Exception as e:
                        print("[WARN] LLM generation failed, using offline generator:", type(e).__name__)
                qq = self._take(self.bank.get(key, []))
                q = dict(kind="mcq", **qq) if qq else None
            if q is None and self.cloze:
                qq = self._take(self.cloze.generate(topic, sub, diff, n=8))
                q = dict(kind="mcq", **qq) if qq else None
            if q is None:
                q = self._pyq(sub)
        if q is None and not getattr(self, "_retrying", False):
            # fresh questions for this topic are used up -> allow repeats (useful revision for small scopes)
            seen_here = {x for x in self.asked if x in set(self.qdf[self.qdf.subtopic == sub].question)} | \
                        {x for x in self.asked if sub.lower() in x.lower()}
            saved, self.asked, self._retrying = self.asked, set(), True
            try:
                q = self.get_question(topic, sub, diff)
            finally:
                self.asked, self._retrying = saved, False
            if q:
                q["repeat"] = True
        if q:
            self.asked.add(q["question"])
        return q

    def log(self, student, topic, sub, diff, correct, secs):
        p = self.out / "attempts.csv"
        row = pd.DataFrame([dict(student=student, ts=time.time(), topic=topic, subtopic=sub,
                                 difficulty={"Easy": 1, "Medium": 2, "Hard": 3}[diff],
                                 correct=int(correct), response_time=round(secs, 1))])
        row.to_csv(p, mode="a", header=not p.exists(), index=False)
