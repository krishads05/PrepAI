"""PrepAI Layer 1 - run the full analysis pipeline.
Usage: python run_layer1.py --syllabus syl.pdf --pyqs pyq_folder/ --material notes/ --out outputs/
PYQ file names must contain the year, e.g. OS_2023.pdf
"""
import argparse, json, pathlib, pandas as pd
from prepai.ingest import read_text, parse_syllabus, parse_pyq, chunk_material, year_from_name, paper_date
from prepai.mapping import Embedder, subtopic_vectors, map_questions, classify_type, llm_refine, acronyms
from prepai.analysis import build_features, priority, recurring_patterns

def files(p):
    p = pathlib.Path(p)
    return sorted(x for x in (p.iterdir() if p.is_dir() else [p]) if x.suffix.lower() in {".pdf", ".txt", ".md", ".pptx", ".ppt", ".docx"})

def main(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from prepai import extract as _ex
    _m, _ex.MODE = _ex.MODE, "run"
    try:
        syl_text = read_text(a.syllabus)
    finally:
        _ex.MODE = _m
    syl = parse_syllabus(syl_text)
    if not syl:
        raise ValueError("NO_TOPICS: no topics could be found in the syllabus file. "
                         "It may be a scanned PDF (no text), or use an unusual layout.\n---EXTRACTED TEXT---\n"
                         + syl_text[:2500])
    questions = []
    from prepai import extract
    mode, extract.MODE = extract.MODE, "run"                  # papers are small: always OCR scanned pages
    try:
        for f in files(a.pyqs):
            t = read_text(f)
            yr, _ = paper_date(t, f.name)
            qs = parse_pyq(t, paper_id=f.stem, year=yr)
            if f.suffix.lower() == ".pdf":
                try:                                          # crop tables/figures as printed
                    from prepai import pyqimg
                    qs = pyqimg.attach_images(f, qs, out / "pyq_images")
                except Exception as e:
                    print(f"[WARN] could not crop question images from {f.name}: {e}")
            questions += qs
    finally:
        extract.MODE = mode
    chunks = []
    for f in (files(a.material) if a.material else []):
        try:
            chunks += chunk_material(read_text(f))
        except Exception as e:                                # one unreadable file must not stop the analysis
            print(f"[WARN] skipped {f.name}: {e}")
    print(f"Syllabus: {len(syl)} subtopics | PYQs: {len(questions)} questions from "
          f"{len({q['paper_id'] for q in questions})} papers | Material chunks: {len(chunks)}")

    emb = Embedder(a.model, a.mode, a.w).fit([r["text"] for r in syl] + [q["question"] for q in questions] + chunks)
    print("Embedder:", emb.kind, "|", emb.model if emb.kind != "tfidf" else "")
    assigned = None
    if a.material:                                            # content-based material -> syllabus map
        from prepai.qgen import Retriever
        from prepai import topicmap
        ret = Retriever(a.material)
        if ret.chunks:
            tm = topicmap.build(ret, syl, out, emb_mode=a.mode)
            pg = {(f, l): t for f, l, t in ret.pages}
            assigned = {}
            for r in tm[tm.subtopic != "(not in syllabus)"].itertuples():
                assigned.setdefault(r.subtopic, []).append(pg.get((r.file, r.location), ""))
            print(f"Topic map: {len(tm)} slides/pages -> {len(assigned)} subtopics")
    T = subtopic_vectors(emb, syl, chunks, alpha=a.alpha, assigned=assigned)
    acr = acronyms(syl, chunks)
    questions = map_questions(emb, questions, syl, T, acr)
    questions = llm_refine(questions, syl)  # no-op without ANTHROPIC_API_KEY
    for q in questions:
        q["types"] = classify_type(q["question"]); q["primary_type"] = q["types"][0]
    qdf = pd.DataFrame(questions)

    pr = priority(build_features(qdf, syl))
    pats = recurring_patterns(emb, qdf)

    qdf.assign(types=qdf.types.str.join(", "), candidates=qdf.candidates.str.join(" | ")).to_csv(out / "questions_mapped.csv", index=False)
    pr.to_csv(out / "topic_priority.csv", index=False)
    json.dump(pats, open(out / "patterns.json", "w", encoding="utf-8"), indent=2)
    json.dump(dict(syllabus=syl, embedder=emb.kind), open(out / "state.json", "w", encoding="utf-8"), indent=2)  # for Layer 2
    write_report(pr, pats, qdf, out / "report.md")
    print(pr[["topic", "subtopic", "papers", "trend", "question_types", "avg_marks", "importance", "priority_label"]]
          .head(12).to_string(index=False))
    print(f"\nLow-confidence mappings to review: {int(qdf.low_confidence.sum())} | Unmapped: {int((qdf.topic=='UNMAPPED').sum())}")

def write_report(pr, pats, qdf, path):
    L = ["# PrepAI - Topic Importance Report", "",
         "_Priority = how strongly to prioritize a topic for preparation, not a prediction that a question will appear._", "",
         "| Topic | Subtopic | PYQ freq | Trend | Question types | Avg marks | Importance | Priority |", "|---|---|---|---|---|---|---|---|"]
    for r in pr.itertuples():
        L.append(f"| {r.topic} | {r.subtopic} | {r.papers}/{r.n_papers} | {r.trend} | {r.question_types or '-'} | {r.avg_marks:.0f} | {r.importance} | {r.priority_label} |")
    L += ["", "## Recurring question patterns", ""]
    order = {f"{r.topic} > {r.subtopic}": r.importance for r in pr.itertuples()}
    for k, v in sorted(pats.items(), key=lambda kv: -order.get(kv[0], 0)):
        L.append(f"### {k} - appeared {v['papers']}/{qdf.paper_id.nunique()} papers")
        for p in v["patterns"]:
            L.append(f"- **{p['pattern']}** ({', '.join(p['types'])}; x{p['count']}; {p['years']})")
        L += [f"\n_Recommendation: {v['recommendation']}_", ""]
    un = qdf[qdf.topic == "UNMAPPED"]
    if len(un):
        L += ["## Unmapped PYQs (review - possibly outside current syllabus wording)", ""] + [f"- {q} ({y})" for q, y in zip(un.question, un.year)]
    pathlib.Path(path).write_text("\n".join(L), encoding="utf-8")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--syllabus", required=True); ap.add_argument("--pyqs", required=True)
    ap.add_argument("--material"); ap.add_argument("--out", default="outputs")
    ap.add_argument("--mode", default="hybrid", choices=["tfidf", "sbert", "hybrid"])
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--w", type=float, default=0.5, help="sbert weight in hybrid mode")
    ap.add_argument("--alpha", type=float, default=0.65, help="1.0 = ignore study material")
    main(ap.parse_args())
