"""Content-based map from study material (each slide / page) to syllabus subtopics.
Works on what the material says, not on file names. Cached per material fingerprint."""
import hashlib, json, pathlib
import numpy as np, pandas as pd

THRESH = {"sbert": 0.30, "hybrid": 0.20, "tfidf": 0.06}

def _fingerprint(retriever, syl):
    h = hashlib.sha1()
    for fn, loc, t in retriever.pages:
        h.update(f"{fn}|{loc}|{len(t)}".encode())
    h.update(json.dumps([r["subtopic"] for r in syl]).encode())
    return h.hexdigest()[:16]

def build(retriever, syl, out_dir, emb_mode="hybrid"):
    """DataFrame[file, location, source, module, topic, subtopic, score, words]: best syllabus subtopic per unit."""
    from .mapping import Embedder
    cache = pathlib.Path(out_dir) / "topic_map.csv"
    fp = _fingerprint(retriever, syl)
    if cache.exists():
        df = pd.read_csv(cache)
        if len(df) and df.get("fingerprint", pd.Series([""])).iloc[0] == fp:
            return df
    units = _units(retriever.pages)
    if not units or not syl:
        return pd.DataFrame(columns=["file", "location", "source", "module", "topic", "subtopic", "score", "words"])
    labels = [f"{r['topic']}. {r.get('group', '')}. {r['subtopic']}. {r['text']}" for r in syl]
    texts = [" ".join(t.split()[:300]) for _, _, t in units]
    emb = Embedder(mode=emb_mode).fit(labels + texts)
    L, U = emb.encode(labels), emb.encode(texts)
    S = _smooth(U @ L.T, [fn for fn, _, _ in units], [len(t.split()) for _, _, t in units])
    th = THRESH.get(emb.kind, 0.2)
    rows = []
    for (fn, loc, t), s in zip(units, S):
        j = int(np.argmax(s))
        r = syl[j]
        rows.append(dict(file=fn, location=loc, source=f"{fn}, {loc}", module=r["module"], topic=r["topic"],
                         subtopic=r["subtopic"] if s[j] >= th else "(not in syllabus)", score=round(float(s[j]), 3),
                         words=len(t.split())))
    df = pd.DataFrame(rows); df["fingerprint"] = fp
    df.to_csv(cache, index=False)
    return df

def _units(pages, min_words=40):
    """Slides/pages as units (>= 8 words); short paragraphs of DOCX/TXT merged into ~40+ word chunks."""
    out, buf = [], None
    for fn, loc, t in pages:
        if loc.startswith("para"):
            if buf and buf[0] == fn and len(buf[2].split()) < min_words:
                buf = (fn, buf[1].split("-")[0] + "-" + loc.split()[1] if " " in loc else buf[1], buf[2] + " " + t)
                continue
            if buf:
                out.append(buf)
            buf = (fn, loc, t); continue
        if buf:
            out.append(buf); buf = None
        if len(t.split()) >= 8:
            out.append((fn, loc, t))
    if buf:
        out.append(buf)
    return [u for u in out if len(u[2].split()) >= 8]

def _smooth(S, files, words, full_info=60):
    """Blend each unit's similarities with its neighbours (same file) and the whole document.
    Units with little text (e.g. a heading over an equation image) lean more on their context."""
    S2 = S.copy()
    idx_by_file = {}
    for i, f in enumerate(files):
        idx_by_file.setdefault(f, []).append(i)
    for f, idx in idx_by_file.items():
        doc = S[idx].mean(0)
        for k, i in enumerate(idx):
            nb = [idx[j] for j in (k - 2, k - 1, k + 1, k + 2) if 0 <= j < len(idx)]
            nbm = S[nb].mean(0) if nb else S[i]
            w_self = 0.15 + 0.45 * min(1.0, words[i] / full_info)
            rest = 1 - w_self
            S2[i] = w_self * S[i] + rest * 0.55 * nbm + rest * 0.45 * doc
    return S2

def summary_by_subtopic(tm):
    """subtopic -> 'file: slides 3-7, 12' style listing."""
    out = {}
    for sub, g in tm[tm.subtopic != "(not in syllabus)"].groupby("subtopic"):
        parts = []
        for fn, gg in g.groupby("file"):
            locs = gg.location.tolist()
            parts.append(f"{fn}: {_compress(locs)}")
        out[sub] = dict(units=len(g), sources="; ".join(parts))
    return out

def _compress(locs):
    """['slide 3','slide 4','slide 5','slide 9'] -> 'slides 3-5, 9'"""
    import re
    if not locs:
        return ""
    kind = locs[0].split()[0]; nums = set()
    for l in locs:
        n = [int(x) for x in re.findall(r"\d+", l)]
        nums.update(range(n[0], n[1] + 1) if len(n) == 2 and n[1] >= n[0] else n[:1])
    nums = sorted(nums)
    rng, start, prev = [], None, None
    for n in nums:
        if start is None:
            start = prev = n
        elif n == prev + 1:
            prev = n
        else:
            rng.append(f"{start}-{prev}" if prev > start else str(start)); start = prev = n
    if start is not None:
        rng.append(f"{start}-{prev}" if prev > start else str(start))
    plural = {"slide": "slides", "p.": "pp.", "para": "paras"}.get(kind, kind)
    return f"{plural if len(nums) > 1 else kind} {', '.join(rng)}"
