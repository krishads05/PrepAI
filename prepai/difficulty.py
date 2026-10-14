"""Topic difficulty = blend of PYQ evidence + (optional) LLM semantic rating."""
import numpy as np, pandas as pd
from . import llm

HARD_TYPES = {"numerical", "algorithm-tracing", "derivation", "application"}

def pyq_difficulty(qdf, pr):
    """0-1 from historical questions: share of procedural types, marks, type diversity."""
    rows = []
    for r in pr.itertuples():
        g = qdf[(qdf.topic == r.topic) & (qdf.subtopic == r.subtopic)]
        if len(g) == 0:
            rows.append(np.nan); continue
        types = g["types"].apply(lambda s: set(str(s).split(", ")))
        proc = types.apply(lambda t: bool(t & HARD_TYPES)).mean()
        marks = g["marks"].fillna(qdf["marks"].median()).mean() / qdf["marks"].max()
        rows.append(0.55 * proc + 0.30 * marks + 0.15 * min(r.n_types / 4, 1))
    return pd.Series(rows, index=pr.index)

MATH = r"\d|calculate|compute|formula|average|total|matrix|ratio|probability|equation|=|\+|%"
ALGO = r"algorithm|step|sequence|trace|queue|graph|table|matrix|order|iterate|gantt|simulate"

def text_difficulty(qdf, pr, retriever=None):
    """Offline proxy for the spec's semantic difficulty: math density + algorithmic cues
    in this subtopic's PYQs and study-material passages, rank-normalised across topics."""
    import re
    raw = []
    for r in pr.itertuples():
        txt = " ".join(qdf[(qdf.topic == r.topic) & (qdf.subtopic == r.subtopic)].question)
        if retriever is not None:
            txt += " " + " ".join(retriever.top(f"{r.topic} {r.subtopic}", k=2))
        w = max(len(txt.split()), 1)
        raw.append(len(re.findall(MATH, txt, re.I)) / w + len(re.findall(ALGO, txt, re.I)) / w if txt.strip() else np.nan)
    return pd.Series(raw, index=pr.index).rank(pct=True)

def llm_difficulty(pr):
    """LLM rates conceptual/mathematical/algorithmic/abstraction 1-5 for all subtopics in one call."""
    items = [f"{r.topic} > {r.subtopic}" for r in pr.itertuples()]
    prompt = ("Rate the difficulty of each undergraduate exam subtopic on four 1-5 scales: "
              "conceptual, mathematical, algorithmic, abstraction; also list key prerequisites (max 3).\n"
              f"Subtopics: {items}\n"
              'Return {"ratings": [{"subtopic": "<exact string>", "conceptual": n, "mathematical": n, '
              '"algorithmic": n, "abstraction": n, "prerequisites": ["..."]}]}')
    out = {d["subtopic"]: d for d in llm.ask_json(prompt, max_tokens=3000)["ratings"]}
    score, prereq = [], []
    for it in items:
        d = out.get(it, {})
        vals = [d.get(k) for k in ("conceptual", "mathematical", "algorithmic", "abstraction")]
        vals = [v for v in vals if isinstance(v, (int, float))]
        score.append((np.mean(vals) - 1) / 4 if vals else np.nan)
        prereq.append(", ".join(d.get("prerequisites", [])))
    return pd.Series(score, index=pr.index), pd.Series(prereq, index=pr.index)

def add_difficulty(qdf, pr, use_llm=True, material_dir=None):
    from .qgen import Retriever
    pr = pr.copy()
    pr["diff_pyq"] = pyq_difficulty(qdf, pr)
    pr["diff_text"] = text_difficulty(qdf, pr, Retriever(material_dir) if material_dir else None)
    pr["diff_llm"], pr["prerequisites"] = np.nan, ""
    if use_llm and llm.available():
        try:
            pr["diff_llm"], pr["prerequisites"] = llm_difficulty(pr)
        except Exception as e:
            print("[WARN] LLM difficulty failed:", e)
    both = pr[["diff_pyq", "diff_text", "diff_llm"]]
    pr["difficulty_score"] = (both.mean(axis=1)).fillna(0.5).round(2)  # average of available signals
    pr["difficulty"] = np.select([pr.difficulty_score >= 0.6, pr.difficulty_score >= 0.4], ["Hard", "Medium"], "Easy")
    return pr
