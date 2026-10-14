"""Evaluate PYQ->subtopic mapping against hand labels.
gold CSV columns: paper_id,qno,part,gold_subtopic  (label ~30-50 questions by hand for your subject)"""
import sys, pandas as pd
pred = pd.read_csv(sys.argv[1], dtype=str).fillna(""); gold = pd.read_csv(sys.argv[2], dtype=str).fillna("")
m = pred.merge(gold.drop(columns=["question"], errors="ignore"), on=["paper_id", "qno", "part"])
m["correct"] = [p in g.split("|") for p, g in zip(m.subtopic, m.gold_subtopic)]   # "|" = several acceptable answers
low = m.low_confidence == "True"
print(f"Mapping accuracy: {m.correct.mean():.1%} ({m.correct.sum()}/{len(m)})")
print(f"Errors caught by low-confidence flag: {(~m.correct & low).sum()}/{(~m.correct).sum()}")
print(f"Accuracy on confident items: {m[~low].correct.mean():.1%}  | flagged for review: {low.sum()}")
print(m[~m.correct][["question", "subtopic", "gold_subtopic"]].to_string(index=False))
