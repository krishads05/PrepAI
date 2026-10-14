"""Historical features, baseline priority score, recurring patterns."""
import numpy as np, pandas as pd

W = dict(freq=0.35, recency=0.25, marks=0.20, repetition=0.10, type_div=0.10)

def build_features(qdf, syl, half_life=2.0):
    years = sorted(qdf["year"].dropna().unique())
    n_papers = qdf["paper_id"].nunique()
    ymax = max(years)
    yw = {y: 0.5 ** ((ymax - y) / half_life) for y in years}
    tot_w = sum(yw.values())
    last2 = set(years[-2:])
    med_marks = qdf["marks"].median()
    tot_marks = qdf["marks"].fillna(med_marks).sum()
    rows = []
    keys = [(r["topic"], r["subtopic"], r["module"]) for r in syl]
    if (qdf["topic"] == "UNMAPPED").any():
        keys.append(("UNMAPPED", "UNMAPPED", -1))
    for topic, sub, mod in keys:
        g = qdf[(qdf.topic == topic) & (qdf.subtopic == sub)]
        yrs = set(g["year"].dropna())
        marks = g["marks"].fillna(med_marks)
        types = sorted({t for ts in g["types"] for t in ts})
        papers = g["paper_id"].nunique()
        rows.append(dict(
            module=mod, topic=topic, subtopic=sub,
            papers=papers, n_papers=n_papers, freq=papers / n_papers,
            occurrences=len(g),
            recency=sum(yw[y] for y in yrs) / tot_w,
            in_last2=bool(yrs & last2),
            avg_marks=marks.mean() if len(g) else 0.0,
            max_marks=marks.max() if len(g) else 0.0,
            marks_share=marks.sum() / tot_marks if len(g) else 0.0,
            repetition=len(g) / papers if papers else 0.0,
            n_types=len(types), question_types=", ".join(types),
            years=", ".join(str(int(y)) for y in sorted(yrs)),
        ))
    return pd.DataFrame(rows)

def _n(s):
    m = s.max()
    return s / m if m else s * 0

def priority(feat, weights=W):
    f = feat.copy()
    comp = pd.DataFrame(dict(
        freq=f["freq"], recency=f["recency"], marks=_n(f["avg_marks"]),
        repetition=_n(f["repetition"]), type_div=_n(f["n_types"])))
    f["importance"] = (100 * sum(comp[k] * w for k, w in weights.items())).round(1)
    for k, w in weights.items():
        f[f"contrib_{k}"] = (100 * comp[k] * w).round(1)
    f["trend"] = np.select([f.papers == 0, f.recency >= 0.6, f.recency >= 0.3], ["Not seen", "High", "Medium"], "Low")
    f["priority_label"] = np.select([f.importance >= 70, f.importance >= 50, f.importance >= 30], ["Very High", "High", "Medium"], "Low")
    return f.sort_values("importance", ascending=False).reset_index(drop=True)

def recurring_patterns(emb, qdf, distance=0.55):
    """Cluster differently worded PYQs within each subtopic into patterns."""
    from sklearn.cluster import AgglomerativeClustering
    out = {}
    for (topic, sub), g in qdf[qdf.topic != "UNMAPPED"].groupby(["topic", "subtopic"]):
        g = g.reset_index(drop=True)
        if len(g) >= 2:
            X = emb.encode(g["question"].tolist())
            lab = AgglomerativeClustering(n_clusters=None, distance_threshold=distance,
                                          metric="cosine", linkage="average").fit_predict(X)
        else:
            lab = np.zeros(len(g), int)
        pats = []
        for c in np.unique(lab):
            gg = g[lab == c]
            types = sorted({t for ts in gg["types"] for t in ts})
            rep = min(gg["question"], key=len)
            pats.append(dict(pattern=rep, types=types, count=len(gg),
                             years=sorted({int(y) for y in gg["year"].dropna()}),
                             examples=gg["question"].tolist()))
        pats.sort(key=lambda p: -p["count"])
        all_types = {t for p in pats for t in p["types"]}
        if all_types & {"numerical", "algorithm-tracing", "application"} and "theory" in all_types:
            rec = "Prepare both theory and numerical/application forms."
        elif all_types & {"numerical", "algorithm-tracing"}:
            rec = "Focus on solving numericals / tracing the algorithm."
        else:
            rec = "Focus on clear theory answers (definitions, explanation, diagrams)."
        out[f"{topic} > {sub}"] = dict(topic=topic, subtopic=sub, papers=int(g.paper_id.nunique()),
                                       patterns=pats, recommendation=rec)
    return out
