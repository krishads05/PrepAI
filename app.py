"""PrepAI dashboard.  Run:  streamlit run app.py"""
import contextlib, io, json, pathlib, pickle, shutil, time, datetime as dt
from types import SimpleNamespace
import os
import pandas as pd, streamlit as st

import run_layer1, run_layer2
from prepai.planner import plan, score
from prepai.agents import Orchestrator, DEFAULT_CHANNELS
from prepai.scope import match_scope, ScopeStore
from prepai.ingest import read_text
from prepai import extract, vision, topicmap
from prepai.ingest import year_from_name
from prepai import llm

st.set_page_config(page_title="PrepAI", page_icon="📚", layout="wide")
# ---------------------------------------------------------------- subjects (saved per subject, no re-uploading)
import re as _re
SUBJECTS = pathlib.Path("subjects"); SUBJECTS.mkdir(exist_ok=True)
DEMO, NEW = "Sample OS course (demo)", "➕ New subject…"
LAST = SUBJECTS / ".last_subject"

def subject_list():
    return sorted(p.name for p in SUBJECTS.iterdir() if p.is_dir())

def slugify(name):
    return _re.sub(r"\s+", " ", _re.sub(r"[^\w\- ]", "", name)).strip()[:60]

if "_goto" in st.session_state:                       # switch subject on the next run (before the widget exists)
    st.session_state.subject = st.session_state.pop("_goto")
if "subject" not in st.session_state:
    last = LAST.read_text(encoding="utf-8").strip() if LAST.exists() else ""
    st.session_state.subject = last if last in subject_list() else DEMO
if st.session_state.subject not in [DEMO, NEW] + subject_list():   # folder removed outside the app
    st.session_state.subject = DEMO
SUBJ = st.session_state.subject
SDIR = SUBJECTS / SUBJ if SUBJ not in (DEMO, NEW) else None
OUT = SDIR / "outputs" if SDIR else (SUBJECTS / ".new" if SUBJ == NEW else pathlib.Path("outputs_app"))
WS = SDIR if SDIR else pathlib.Path("workspace")

def subject_files(sdir):
    syl_files = sorted((sdir / "syllabus").glob("*")) if (sdir / "syllabus").exists() else []
    pyqs = sorted((sdir / "pyqs").glob("*")) if (sdir / "pyqs").exists() else []
    mats = sorted(p for p in (sdir / "material").glob("*") if p.is_file()) if (sdir / "material").exists() else []
    return (syl_files[0] if syl_files else None), pyqs, mats

if st.session_state.get("_active") != SUBJ:            # subject changed: reset per-subject state
    for k in ("orch", "quiz", "material", "scope_name", "scope_draft", "last_inputs", "engine"):
        st.session_state.pop(k, None)
    st.session_state._active = SUBJ
    if SDIR:
        LAST.write_text(SUBJ, encoding="utf-8")
        _s, _p, _m = subject_files(SDIR)
        st.session_state.material = str(SDIR / "material")
        if _s:
            st.session_state.last_inputs = (str(_s), str(SDIR / "pyqs"))

# ---------------------------------------------------------------- helpers
def load_tables():
    if not (OUT / "topic_table.csv").exists():
        return None, None, None, None
    tbl = pd.read_csv(OUT / "topic_table.csv")
    qdf = pd.read_csv(OUT / "questions_mapped.csv")
    pats = json.load(open(OUT / "patterns.json"))
    syl = json.load(open(OUT / "state.json"))["syllabus"]
    return tbl, qdf, pats, syl

def mastery_now(tbl, student):
    m, att = run_layer2.current_mastery(OUT, tbl, student)
    t = tbl.copy()
    t["mastery"] = t.subtopic.map(m)
    t["attempts"] = t.subtopic.map(att.subtopic.value_counts()).fillna(0).astype(int) if len(att) else 0
    return t, att

def orch():
    if "orch" not in st.session_state:
        st.session_state.orch = Orchestrator(OUT, st.session_state.get("material"))
    st.session_state.orch.bb.state["llm_critic"] = st.session_state.get("llm_critic", False)
    return st.session_state.orch

def run_analysis(syllabus, pyq_dir, material_dir, out=None):
    out = pathlib.Path(out or OUT)
    st.session_state.last_inputs = (str(syllabus), str(pyq_dir))
    st.session_state.material = str(material_dir) if material_dir else None
    st.session_state.orch = Orchestrator(out, st.session_state.material)
    st.session_state.pop("quiz", None)
    return st.session_state.orch.analyze_course(syllabus, pyq_dir, st.session_state.material)

LIB = SDIR / "material" if SDIR else pathlib.Path("library")
MAT_TYPES = ["pdf", "pptx", "ppt", "docx", "txt", "md"]

@st.cache_data(show_spinner=False)
def extracted_words(path, mtime):
    from prepai.ingest import read_text
    try:
        return len(read_text(path).split()), ""
    except Exception as e:
        return 0, f"{type(e).__name__}: {e}"[:120]

@st.cache_data(show_spinner=False)
def pending_cached(folder, files_sig, n_cached):
    return extract.pending_report(folder)

def refresh_agents(note):
    st.session_state.orch = Orchestrator(OUT, st.session_state.material)
    st.session_state.pop("quiz", None)
    st.session_state.orch.bb.post("Orchestrator", "materials updated", note)
    return st.session_state.orch.auditor.audit()

def video_items(sub, k=2):
    row = tbl.set_index("subtopic").loc[sub].to_dict(); row["subtopic"] = sub
    res = orch().resources.recommend(row, st.session_state.get("channels", DEFAULT_CHANNELS), st.session_state.get("course", ""))
    return res["items"][:k], res["mode"]

def save_uploads(files, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for f in files:
        (folder / f.name).write_bytes(f.getbuffer())
    return folder

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📚 PrepAI")
    student = st.text_input("Student name", "me")
    exam = st.date_input("Exam date", dt.date.today() + dt.timedelta(days=14))
    days_left = (exam - dt.date.today()).days
    st.metric("Days remaining", max(days_left, 0))
    with st.container(border=True):
        if llm.backend() == "none":
            st.markdown("🧠 **LLM: not detected** — agents use rules + ML")
            st.caption("Install Ollama and run `ollama pull qwen2.5:3b`, then click Re-check.")
        else:
            st.markdown(f"🧠 **LLM: {llm.backend()}** · `{llm.model_name()}`")
        if st.button("Re-check LLM", use_container_width=True):
            llm.reset(); st.rerun()
        st.session_state.llm_critic = st.toggle("LLM double-checks MCQ answers (slower)", value=False,
                                                disabled=llm.backend() == "none")
    with st.expander("🎥 Video settings"):
        chans = st.text_input("Preferred YouTube channels (comma-separated)", ", ".join(DEFAULT_CHANNELS))
        st.session_state.channels = [c.strip() for c in chans.split(",") if c.strip()]
        st.session_state.course = st.text_input("Course name (improves searches)", "Operating System")
        yk = st.text_input("YouTube API key (optional)", type="password",
                           help="Free from Google Cloud Console -> enable 'YouTube Data API v3' -> Credentials -> API key")
        if yk:
            os.environ["YOUTUBE_API_KEY"] = yk
        st.caption("With a key: ranked real videos. Without: tailored YouTube search links.")
    st.divider()
    st.subheader("📚 Subject")
    opts = [DEMO] + subject_list() + [NEW]
    st.selectbox("Your subjects", opts, key="subject", help="Each subject keeps its files, analysis, quiz history and scopes")

    def analyse(syllabus, pyq_dir, material_dir, out):
        """Run the analysis and report problems in the sidebar."""
        try:
            with st.spinner("Extracting, mapping PYQs, scoring topics, training mastery model..."):
                log = run_analysis(syllabus, pyq_dir, material_dir, out)
            skipped = [l.split("skipped ", 1)[1] for l in log.splitlines() if "[WARN] skipped" in l]
            if skipped:
                st.warning("Some files were skipped:  \n" + "  \n".join(f"• {x}" for x in skipped)
                           + "  \nFor old .ppt files: open in PowerPoint → File → Save As → .pptx, then upload that instead.")
            with st.expander("Pipeline log"):
                st.code(log)
            return True
        except ValueError as e:
            msg = str(e)
            if msg.startswith("NO_TOPICS"):
                st.error("No topics were found in your syllabus file.")
                st.markdown("Likely causes: it's a **scanned PDF** (no selectable text), or an unusual layout. "
                            "Try a text-based PDF, or a .txt file with lines like `Module 1: Topic` "
                            "followed by comma-separated subtopics.")
                with st.expander("What PrepAI extracted from your syllabus"):
                    st.code(msg.split("---EXTRACTED TEXT---", 1)[-1] or "(no text at all - probably a scanned PDF)")
            else:
                st.error(f"Analysis failed: {msg[:300]}")
            return False

    def year_note(files):
        bad = [f.name for f in files or [] if year_from_name(f.name) is None]
        if bad:
            st.caption(f"No year in the name of: {', '.join(bad)}. The year printed on the paper will be used.")

    if SUBJ == NEW:
        name = st.text_input("Subject name", placeholder="e.g. Machine Learning CSE 3138")
        syl_f = st.file_uploader("Syllabus", type=["pdf", "txt", "docx", "md"])
        pyq_f = st.file_uploader("Previous-year papers", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        mat_f = st.file_uploader("Study material (optional)", type=["pdf", "txt", "pptx", "ppt", "docx", "md"],
                                 accept_multiple_files=True)
        year_note(pyq_f)
        sname = slugify(name or "")
        if sname and sname in subject_list():
            st.warning(f"'{sname}' already exists. Pick it from the list, or choose another name.")
        ready = bool(sname) and syl_f and pyq_f and sname not in subject_list()
        if st.button("💾 Save subject & analyse", type="primary", disabled=not ready, use_container_width=True):
            sdir = SUBJECTS / sname
            syl_path = save_uploads([syl_f], sdir / "syllabus") / syl_f.name
            save_uploads(pyq_f, sdir / "pyqs")
            (sdir / "material").mkdir(parents=True, exist_ok=True)
            if mat_f:
                save_uploads(mat_f, sdir / "material")
            if analyse(syl_path, sdir / "pyqs", sdir / "material" if mat_f else None, sdir / "outputs"):
                st.session_state._goto = sname
                st.rerun()
    elif SUBJ == DEMO:
        st.caption("Built-in sample course for trying PrepAI.")
        if st.button("▶ Run analysis", type="primary", use_container_width=True):
            if analyse("sample_data/syllabus.txt", "sample_data/pyqs", "sample_data/material", OUT):
                st.rerun()
    else:
        syl_file, pyqs, mats = subject_files(SDIR)
        analysed = (OUT / "topic_table.csv").exists()
        when = dt.datetime.fromtimestamp((OUT / "topic_table.csv").stat().st_mtime).strftime("%d %b %Y, %H:%M") if analysed else "never"
        st.caption(f"Saved: syllabus `{syl_file.name if syl_file else '—'}` · {len(pyqs)} paper(s) · "
                   f"{len(mats)} material file(s)  \nLast analysed: {when}")
        if not analysed and syl_file and st.button("▶ Run analysis", type="primary", use_container_width=True):
            if analyse(syl_file, SDIR / "pyqs", SDIR / "material" if mats else None, OUT):
                st.rerun()
        with st.expander("➕ Add or replace files"):
            new_syl = st.file_uploader("Replace syllabus", type=["pdf", "txt", "docx", "md"], key=f"rs_{SUBJ}")
            add_pyq = st.file_uploader("Add past papers", type=["pdf", "txt", "docx"], accept_multiple_files=True, key=f"ap_{SUBJ}")
            add_mat = st.file_uploader("Add study material", type=["pdf", "txt", "pptx", "ppt", "docx", "md"],
                                       accept_multiple_files=True, key=f"am_{SUBJ}")
            year_note(add_pyq)
            drop = st.multiselect("Remove papers", [p.name for p in pyqs], key=f"rp_{SUBJ}")
            if st.button("💾 Save changes & re-analyse", disabled=not (new_syl or add_pyq or add_mat or drop)):
                if new_syl:
                    shutil.rmtree(SDIR / "syllabus", ignore_errors=True)
                    save_uploads([new_syl], SDIR / "syllabus")
                if add_pyq:
                    save_uploads(add_pyq, SDIR / "pyqs")
                if add_mat:
                    save_uploads(add_mat, SDIR / "material")
                for d in drop:
                    (SDIR / "pyqs" / d).unlink(missing_ok=True)
                syl_file, pyqs, mats = subject_files(SDIR)
                if syl_file and pyqs and analyse(syl_file, SDIR / "pyqs", SDIR / "material" if mats else None, OUT):
                    st.session_state.pop("orch", None); st.rerun()
        with st.expander("🗑️ Delete this subject"):
            st.caption("Deletes its files, analysis, quiz history and scopes from this computer.")
            sure = st.checkbox(f"Yes, delete '{SUBJ}'", key=f"del_{SUBJ}")
            if st.button("Delete subject", disabled=not sure):
                shutil.rmtree(SDIR, ignore_errors=True)
                st.session_state._goto = DEMO
                st.rerun()

tbl, qdf, pats, syl = load_tables()
if tbl is None:
    st.info("👈 Choose **➕ New subject…** in the sidebar to save your syllabus, papers and notes once, "
            "or try the sample course. Saved subjects open instantly next time.")
    st.stop()
if "material" not in st.session_state:
    if LIB.exists() and any(LIB.iterdir()):
        st.session_state.material = str(LIB)
    else:
        st.session_state.material = "sample_data/material" if pathlib.Path("sample_data/material").exists() else None
T, att = mastery_now(tbl, student)
scopes = ScopeStore(OUT).all()
with st.sidebar:
    st.divider()
    st.subheader("🎯 Exam scope")
    names = ["Full syllabus"] + list(scopes)
    cur_scope = st.session_state.get("scope_name", "Full syllabus")
    pick = st.selectbox("Prepare for", names, index=names.index(cur_scope) if cur_scope in names else 0,
                        help="Create mid-sem / internal scopes in the 🎯 Exam scope tab")
    if pick != st.session_state.get("scope_name"):
        st.session_state.pop("quiz", None)
    st.session_state.scope_name = pick
    desired = set(scopes[pick]) if pick != "Full syllabus" else None
    if orch().bb.state.get("scope") != desired:          # also re-applies after agents are rebuilt
        orch().set_scope(desired)
SCOPE = set(scopes.get(st.session_state.get("scope_name"), [])) if st.session_state.get("scope_name", "Full syllabus") != "Full syllabus" else None
if SCOPE:
    T = T[T.subtopic.isin(SCOPE)]
    seen = T                                              # in-scope topics shown even if never asked in PYQs
    st.info(f"🎯 **{st.session_state.scope_name}** — showing {len(T)} of {len(tbl)} subtopics. "
            "Importance comes from full-syllabus past papers; switch scope in the sidebar.")
else:
    seen = T[T.papers > 0]

# ---------------------------------------------------------------- header metrics
c = st.columns(4)
c[0].metric("Topics analysed", len(T))
c[1].metric("PYQs mapped", f"{(qdf.topic != 'UNMAPPED').sum()}/{len(qdf)}")
c[2].metric("Very-high priority", int((T.priority_label == "Very High").sum()))
c[3].metric("Questions answered", len(att))

tabs = st.tabs(["📊 Important topics", "🔁 Question patterns", "📝 Quiz Me", "📈 Mastery", "🗓️ Study plan", "📂 My materials", "🎯 Exam scope", "🎥 Resources", "🤖 Agents", "🧪 Model evaluation"])

# ---------------------------------------------------------------- 1. importance
with tabs[0]:
    st.caption("Priority = how strongly to prioritise a topic for preparation — **not** a prediction that a question will appear.")
    show = seen.assign(**{"PYQ freq": seen.papers.astype(str) + "/" + seen.n_papers.astype(str),
                          "Mastery": [f"{m:.0%}" if n else "—" for m, n in zip(seen.mastery, seen.attempts)]})
    st.dataframe(show[["topic", "subtopic", "PYQ freq", "trend", "question_types", "avg_marks", "difficulty",
                       "Mastery", "importance", "priority_label"]].rename(columns=str.title),
                 use_container_width=True, hide_index=True,
                 column_config={"Importance": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f"),
                                "Avg_Marks": st.column_config.NumberColumn(format="%.1f")})
    st.subheader("Why is each topic important? (score breakdown)")
    contrib = seen.head(12).set_index("subtopic")[[c for c in seen.columns if c.startswith("contrib_")]]
    st.bar_chart(contrib.rename(columns=lambda c: c.replace("contrib_", "")), horizontal=True)
    never = T[T.papers == 0].subtopic.tolist()
    if never:
        st.caption(f"In syllabus but never asked in the analysed papers: {', '.join(never)}")

# ---------------------------------------------------------------- 2. patterns
with tabs[1]:
    order = seen.set_index("subtopic").importance.to_dict()
    for key, v in sorted(pats.items(), key=lambda kv: -order.get(kv[1]["subtopic"], 0)):
        if SCOPE and v["subtopic"] not in SCOPE:
            continue
        lab = T.set_index("subtopic").priority_label.get(v["subtopic"], "")
        with st.expander(f"**{v['subtopic']}** ({v['topic']}) — {v['papers']}/{qdf.paper_id.nunique()} papers · {lab}"):
            for p in v["patterns"]:
                st.markdown(f"- **{p['pattern']}**  \n  `{', '.join(p['types'])}` · asked {p['count']}× · {p['years']}")
                if p["count"] > 1:
                    st.caption("Also worded as: " + " | ".join(e for e in p["examples"] if e != p["pattern"]))
            st.info(v["recommendation"])
    un = qdf[qdf.topic == "UNMAPPED"]
    if len(un):
        st.warning("Unmapped PYQs (may be outside current syllabus wording):")
        st.dataframe(un[["year", "question"]], hide_index=True)

# ---------------------------------------------------------------- 3. quiz
def show_result(res, q):
    ev = res["evaluation"]
    if q["kind"] == "mcq":
        right = q["options"][q["answer_index"]]
        (st.success if ev["correct"] else st.error)(("Correct! " if ev["correct"] else f"Answer: **{right}**. ") + ev.get("feedback", ""))
        if q.get("source"):
            st.caption(f"📄 Source: {q['source']}")
    elif ev.get("gradable"):
        (st.success if ev["correct"] else st.warning)(f"**Score: {ev['score']}/10** · graded by {ev['method']}")
        st.write(ev.get("feedback", ""))
        if ev.get("breakdown"):
            b = ev["breakdown"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Key points covered", f"{b['key_points']:.0%}"); c2.metric("Key terms used", f"{b['key_terms']:.0%}")
            c3.metric("Semantic match", f"{b['similarity']:.0%}")
        if ev.get("misconceptions"):
            st.error("Misconceptions: " + "; ".join(ev["misconceptions"]))
        src = ev.get("sources", {})
        with st.expander("What your materials expected (with sources)"):
            for x in ev.get("covered", []):
                st.markdown(f"✅ {x}  \n<small>📄 {src.get(x, '')}</small>", unsafe_allow_html=True)
            for x in ev.get("missed", []):
                st.markdown(f"❌ {x}  \n<small>📄 {src.get(x, '')}</small>", unsafe_allow_html=True)
        if src:
            st.caption("Graded only against: " + "; ".join(dict.fromkeys(v for v in src.values() if v)))
    if "mastery" in res:
        b, a_ = res["mastery"]
        st.caption(f"🧠 Learner Model: {q['sb']} mastery {b:.0%} → {a_:.0%}")
    if res.get("tutor"):
        t = res["tutor"]
        with st.container(border=True):
            st.markdown("**🧑‍🏫 Tutor**")
            if t.get("explanation"):
                st.write(t["explanation"])
            for p, sp in zip(t.get("key_points", []), t.get("sources", [""] * 3)):
                st.markdown(f"- {p}  \n  <small>📄 {sp}</small>", unsafe_allow_html=True)
            if t.get("exam_tip"):
                st.info(t["exam_tip"])
            items, _ = video_items(q["sb"])
            st.markdown("🎥 Watch: " + " · ".join(f"[{i['purpose']}]({i['url']})" for i in items))
    if res.get("videos"):
        st.markdown("🎥 **Suggested videos:** " + " · ".join(
            f"[{v.get('title', v.get('purpose'))[:60]}]({v['url']})" for v in res["videos"]))
    if res.get("coordinator"):
        c = res["coordinator"]
        st.caption(f"🧭 Coordinator ({c['mode']}) chose: **{', '.join(c['actions'])}** — {c['reason']}")
    if res.get("replanned"):
        st.caption("🗓️ Study Planner updated your priorities — see the Study plan tab.")

with tabs[2]:
    O = orch()
    qs = st.session_state.get("quiz")
    if not qs or qs["done"]:
        a, b, c3, d = st.columns(4)
        mode = a.selectbox("Questions from", ["mixed", "important", "weak", "all"],
                           format_func={"mixed": "Mixed (importance × weakness)", "important": "Important topics",
                                        "weak": "Weak topics", "all": "Entire syllabus"}.get)
        topic = b.selectbox("Topic filter", ["(any)"] + sorted(seen.topic.unique()) + sorted(seen.subtopic.unique()))
        diff = c3.selectbox("Difficulty", ["Adaptive", "Easy", "Medium", "Hard"])
        n = d.selectbox("Questions", [5, 10, 20])
        O.generator.engine.pyq_share = st.slider("Share of written PYQ-style questions", 0.0, 1.0, 0.3, 0.1)
        if qs and qs["done"]:
            st.success(f"Last quiz: {qs['score']}/{qs['n']} correct. Mastery updated — see the Mastery tab.")
        if st.button("Start quiz", type="primary"):
            st.session_state.quiz = dict(mode=mode, topic=None if topic == "(any)" else topic, diff=diff, n=n,
                                         i=0, score=0, recent=[], q=None, res=None, done=False)
            st.rerun()
    else:
        def next_q():
            q = O.next_question(student, mode=qs["mode"], topic=qs["topic"], difficulty=qs["diff"], recent=qs["recent"])
            if q is None:
                qs["done"] = True
            qs.update(q=q, res=None, t0=time.time())

        def submit(answer=None, self_grade=None):
            res = O.submit(student, qs["q"], answer, time.time() - qs["t0"], self_grade)
            qs["res"] = res
            if "mastery" in res:                             # attempt recorded
                qs["i"] += 1; qs["score"] += int(res["evaluation"]["correct"]); qs["recent"].append(qs["q"]["sb"])

        if qs["q"] is None and not qs["done"]:
            next_q()
        if not qs["done"]:
            q, res = qs["q"], qs["res"]
            recorded = res is not None and "mastery" in res
            st.progress(min(qs["i"] / qs["n"], 1.0), text=f"Question {qs['i'] + (0 if recorded else 1)} of {qs['n']} · correct so far {qs['score']}")
            st.caption(f"{q['sb']} ({q['tp']}) · {q['df']} · current mastery {q['ms']:.0%}")
            if q.get("image"):
                from prepai.pyqimg import clean_question
                st.markdown(f"#### {clean_question(q['question'])}")
                st.image(q["image"], caption="As printed in the original paper", use_container_width=True)
            else:
                st.markdown(f"#### {q['question']}")
            if q.get("source"):
                st.caption(("📜 From " if q["kind"] == "pyq" else "📄 Generated from your materials: ") + q["source"]
                           + (f" · {q['marks']} marks" if q.get("marks") else ""))
            if q["kind"] == "mcq":
                choice = st.radio("Choose one", q["options"], index=None, disabled=recorded, key=f"r_{abs(hash(q['question']))}")
                if not recorded and st.button("Submit", disabled=choice is None):
                    with st.spinner("Agents checking..."):
                        submit(q["options"].index(choice))
                    st.rerun()
            else:
                ans = st.text_area("Write your answer (the Answer Evaluator agent will grade it against your notes)",
                                   height=180, disabled=recorded, key=f"t_{abs(hash(q['question']))}")
                if not recorded:
                    x, y, z, _ = st.columns([1.3, 1, 1, 2])
                    if x.button("🤖 Grade my answer", type="primary", disabled=len(ans.split()) < 3):
                        with st.spinner("Agents grading and deciding next steps" + (" (LLM on CPU can take ~30-60 s)..." if llm.available() else "...")):
                            submit(ans)
                        st.rerun()
                    if res is not None and not res["evaluation"].get("gradable"):
                        st.info(res["evaluation"]["feedback"])
                    if y.button("✅ Self-grade: right"):
                        submit(self_grade=True); st.rerun()
                    if z.button("❌ Self-grade: wrong"):
                        submit(self_grade=False); st.rerun()
            if recorded:
                show_result(res, q)
                if qs["i"] >= qs["n"]:
                    if st.button("Finish", type="primary"):
                        qs["done"] = True; st.rerun()
                elif st.button("Next question ➜", type="primary"):
                    next_q(); st.rerun()
            if st.button("End quiz early"):
                qs["done"] = True; st.rerun()

# ---------------------------------------------------------------- 4. mastery
with tabs[3]:
    tried = T[T.attempts > 0]
    if tried.empty:
        st.info("No quiz attempts yet — take a quiz to start tracking mastery.")
    else:
        st.bar_chart(tried.set_index("subtopic")["mastery"].mul(100).rename("Mastery %"), horizontal=True)
        st.dataframe(tried[["topic", "subtopic", "attempts", "mastery", "importance", "difficulty"]]
                     .assign(mastery=lambda d: d.mastery * 100).sort_values("mastery"), hide_index=True, use_container_width=True,
                     column_config={"mastery": st.column_config.ProgressColumn("Mastery", min_value=0, max_value=100, format="%.0f%%")})
        hist = att.copy(); hist["when"] = pd.to_datetime(hist.ts, unit="s")
        st.subheader("Recent answers")
        st.dataframe(hist.sort_values("ts", ascending=False)[["when", "subtopic", "difficulty", "correct"]].head(20), hide_index=True)
    st.caption("Untested topics use a prior estimate; mastery comes from the monotone PFA model (Model evaluation tab).")

# ---------------------------------------------------------------- 5. plan
with tabs[4]:
    mins = st.slider("How much time do you have today? (minutes)", 30, 300, 120, 15)
    text, scored = orch().daily_plan(student, mins)
    st.subheader(f"Your {mins}-minute plan")
    for line in text.splitlines():
        (st.warning if line.startswith("Deprioritized") else st.markdown)(line if line.startswith("Deprioritized") else f"- {line.strip()}")
    st.caption("Study priority ∝ importance × difficulty × knowledge gap — a planning heuristic, not a law.")
    st.markdown("**🎥 Videos for today's focus topics**")
    for sub in orch().bb.state.get("plan_top", []):
        items, _ = video_items(sub, k=2)
        st.markdown(f"- {sub}: " + " · ".join(f"[{i['purpose']}]({i['url']})" for i in items))
    with st.expander("Full priority ranking"):
        st.dataframe(scored[["subtopic", "importance", "difficulty", "mastery", "study_priority"]], hide_index=True)

# ---------------------------------------------------------------- 6. materials
with tabs[5]:
    st.subheader("📂 Reference materials")
    st.caption("Notes, slides, PDFs or textbook chapters. Quiz questions, answer grading and tutor explanations "
               "use ONLY these files (no outside knowledge), and always cite the file and page/slide they came from.")
    cur = pathlib.Path(st.session_state.material) if st.session_state.get("material") else None
    files = sorted(p for p in cur.iterdir() if p.is_file()) if cur and cur.exists() else []
    st.markdown(f"Active folder: `{cur if cur else 'none'}` · **{len(files)} file(s)**")

    up = st.file_uploader("Add materials", type=MAT_TYPES, accept_multiple_files=True, key="lib_up")
    if st.button("➕ Add to my materials", type="primary", disabled=not up):
        LIB.mkdir(exist_ok=True)
        if cur and cur.exists() and cur.resolve() != LIB.resolve():          # keep what is already loaded
            for p in files:
                if not (LIB / p.name).exists():
                    shutil.copy(p, LIB / p.name)
        for f in up:
            (LIB / f.name).write_bytes(f.getbuffer())
        st.session_state.material = str(LIB)
        with st.spinner("Updating agents with the new materials..."):
            cov = refresh_agents(f"added {len(up)} file(s): {', '.join(f.name for f in up)[:150]}")
        st.success(f"Added {len(up)} file(s). Notes now cover {(cov.coverage == 'Covered').sum()}/{len(cov)} examined topics.")
        st.rerun()

    if files:
        rows = []
        for p in files:
            words, err = extracted_words(str(p), p.stat().st_mtime)
            status = f"⚠️ {err}" if err else "⚠️ no text - scanned PDF?" if words < 50 else "✅ ok"
            rows.append(dict(file=p.name, type=p.suffix.lstrip(".").upper(), size_kb=round(p.stat().st_size / 1024, 1),
                             words_extracted=words, status=status))
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if any(r["words_extracted"] < 50 for r in rows):
            st.warning("Files with no extracted text are probably scanned images. Use a text-based PDF, "
                       "or export the notes from Word/PowerPoint instead.")
        if cur.resolve() == LIB.resolve():
            c1, c2 = st.columns([3, 1])
            victim = c1.selectbox("Remove a file", [p.name for p in files], label_visibility="collapsed")
            if c2.button("🗑️ Remove", use_container_width=True):
                (LIB / victim).unlink()
                refresh_agents(f"removed {victim}")
                st.rerun()
        else:
            st.caption("These are the sample/analysis materials. Adding a file copies them into your own library, "
                       "which you can then edit.")
    st.divider()
    st.markdown("### 🔍 Reading text and equations inside slide images")
    eng = vision.engines()
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"**Text OCR:** {'✅ RapidOCR' if eng['rapidocr'] else '✅ Tesseract' if eng['tesseract'] else '❌ none'}")
    c2.markdown(f"**Equation OCR:** {'✅' if eng['math_ocr'] else '❌'}")
    c3.markdown(f"**Vision model (optional):** {'✅ `' + eng['vision_model'] + '`' if eng['vision_model'] else '—'}")
    missing = []
    if not (eng["rapidocr"] or eng["tesseract"]):
        missing.append("`pip install rapidocr_onnxruntime`  (text in images)")
    if not eng["math_ocr"]:
        missing.append("`pip install rapid_latex_ocr`  (equations rendered as images)")
    if missing:
        st.info("To read images, install (one time) in a terminal, then restart the app:  \n" + "  \n".join(missing))
    if cur and cur.exists():
        todo = pending_cached(str(cur), tuple(sorted((p.name, p.stat().st_mtime) for p in cur.iterdir())),
                              len(list((extract.CACHE / "ocr").glob("*.json"))) if (extract.CACHE / "ocr").exists() else 0)
        n_img = len(todo["images"])

        def run(engine, include_scanned, n_items, sec_each):
            bar = st.progress(0.0, text=f"Reading {n_items} items (~{max(1, round(n_items * sec_each / 60))} min)...")
            t0 = time.time()
            def prog(i, n, label):
                left = (time.time() - t0) / i * (n - i) / 60 if i else 0
                bar.progress(min(i / max(n, 1), 1.0), text=f"{i}/{n} · {label[:40]} · ~{left:.0f} min left")
            r = extract.process_folder(cur, prog, include_scanned=include_scanned, engine=engine)
            refresh_agents(f"read {r['done']} images/pages ({engine})")
            st.cache_data.clear(); st.rerun()

        if n_img:
            st.warning(f"**{n_img} slide images not read yet.**")
            if st.button(f"⚡ Quick read: text & equations in {n_img} slide images (~{max(1, round(n_img * 1.5 / 60))} min)",
                         type="primary", disabled=not (eng["rapidocr"] or eng["tesseract"] or eng["math_ocr"])):
                run("fast", False, n_img, 1.5)
        else:
            st.success("All slide images have been read.")
        with st.expander("Optional, slow: handwritten notes and detailed diagram descriptions"):
            st.caption(f"{len(todo['pages'])} scanned/handwritten pages are skipped by default. Reading handwriting needs a "
                       "local vision model (`ollama pull qwen2.5vl:3b`) and takes roughly 20-60 s per page on a laptop CPU.")
            inc = st.checkbox("Include handwritten / scanned pages", value=False)
            n_slow = len(todo["pages"]) if inc else 0
            if st.button(f"🔍 Detailed read with vision model ({n_slow + n_img} items)",
                         disabled=not eng["vision_model"] or not (n_slow + n_img)):
                run("vision", inc, n_slow + n_img, 40)
            if st.button("Re-check installed engines"):
                vision.reset(); st.cache_data.clear(); st.rerun()

    tm = orch().bb.state.get("topic_map")
    if tm is not None and len(tm):
        st.divider()
        st.markdown("### 🗺️ How your materials map to the syllabus")
        st.caption("Matched by what each slide/page says - file names don't matter. Each syllabus topic shows where it is covered.")
        summ = topicmap.summary_by_subtopic(tm)
        for mod, g in pd.DataFrame(syl).groupby("module", sort=True):
            have = sum(1 for s in g.subtopic if s in summ)
            with st.expander(f"Module {mod}: {g.topic.iloc[0]} — material for {have}/{len(g)} subtopics"):
                st.dataframe(pd.DataFrame([dict(subtopic=s, slides_pages=summ.get(s, {}).get("units", 0),
                                                found_in=summ.get(s, {}).get("sources", "— nothing found"))
                                           for s in g.subtopic]), hide_index=True, use_container_width=True)
        off = tm[tm.subtopic == "(not in syllabus)"]
        if len(off):
            st.caption(f"{len(off)} slides/pages didn't match any syllabus topic (title slides, examples, or off-syllabus content).")
        per_file = (tm[tm.subtopic != "(not in syllabus)"].groupby("file").subtopic
                    .agg(lambda x: ", ".join(x.value_counts().head(3).index)).rename("main topics detected"))
        with st.expander("Main topics detected in each file"):
            st.dataframe(per_file.reset_index(), hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**New materials are already used** by quiz questions, grading, the tutor and the coverage audit.")
    if st.session_state.get("last_inputs"):
        if st.button("🔄 Also re-run full analysis with these materials"):
            with st.spinner("Re-running analysis..."):
                syl_p, pyq_p = st.session_state.last_inputs
                run_analysis(syl_p, pyq_p, st.session_state.material)
            st.success("Analysis updated."); st.rerun()
        st.caption("Optional: refreshes topic difficulty and PYQ-mapping grounding. Topic importance comes from past papers, so it barely changes.")

# ---------------------------------------------------------------- 7. exam scope
with tabs[6]:
    st.subheader("🎯 Exam scope")
    st.caption("Optional. Keep the full syllabus as your master analysis, and create targeted scopes for mid-sems "
               "or internals. The whole app (topics, quiz, plan, videos) then focuses on just those portions.")
    left, right = st.columns(2)
    with left:
        st.markdown("**📘 Full syllabus**")
        for mod, g in pd.DataFrame(syl).groupby("module", sort=True):
            st.markdown(f"- **Module {mod}: {g.topic.iloc[0]}** — {', '.join(g.subtopic)}")
    with right:
        st.markdown("**🎯 Create a targeted scope**")
        sname = st.text_input("Scope name", "Mid-sem")
        pf = st.file_uploader("Upload portions (PDF / DOCX / TXT)", type=["pdf", "docx", "txt", "md"], key="portions")
        ptxt = st.text_area("…or paste / type the portions",
                            placeholder="e.g. Modules 1 and 2, Deadlocks: Banker's algorithm only, page replacement (FIFO, LRU)")
        if st.button("🔍 Detect topics", disabled=not (pf or ptxt.strip())):
            text = ptxt
            if pf:
                tmp = WS / "portions" / pf.name; tmp.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(pf.getbuffer()); text = read_text(tmp) + "\n" + ptxt
            st.session_state.scope_draft = match_scope(text, syl)
    draft = st.session_state.get("scope_draft")
    if draft:
        st.markdown("**How your portions were matched**")
        if draft["matches"]:
            st.dataframe(pd.DataFrame(draft["matches"], columns=["portion text", "matched to", "how"]),
                         hide_index=True, use_container_width=True)
        if draft["unmatched"]:
            st.warning("Not found in your syllabus: " + "; ".join(draft["unmatched"]))
        allsubs = [r["subtopic"] for r in syl]
        chosen = st.multiselect("Review and edit the subtopics in this scope", allsubs,
                                default=[x for x in allsubs if x in draft["subtopics"]])
        if st.button(f"💾 Save scope '{sname}'", type="primary", disabled=not chosen or not sname.strip()):
            ScopeStore(OUT).save(sname.strip(), chosen)
            st.session_state.scope_name = sname.strip(); st.session_state.pop("scope_draft", None)
            st.success(f"Saved '{sname}' with {len(chosen)} subtopics — now active."); st.rerun()
    else:
        with st.expander("Or pick topics manually"):
            allsubs = [r["subtopic"] for r in syl]
            mname = st.text_input("Scope name ", "Internal 1")
            man = st.multiselect("Subtopics", allsubs)
            if st.button("💾 Save manual scope", disabled=not man):
                ScopeStore(OUT).save(mname.strip(), man); st.session_state.scope_name = mname.strip(); st.rerun()
    if scopes:
        st.markdown("**Saved scopes**")
        for n, subs in scopes.items():
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"- **{n}** ({len(subs)} subtopics): {', '.join(subs)}")
            if c2.button("🗑️", key=f"del_{n}"):
                ScopeStore(OUT).delete(n)
                if st.session_state.get("scope_name") == n:
                    st.session_state.scope_name = "Full syllabus"
                st.rerun()

# ---------------------------------------------------------------- 8. resources
with tabs[7]:
    O = orch()
    cov, recs = O.study_resources(student, st.session_state.get("channels", DEFAULT_CHANNELS),
                                  st.session_state.get("course", ""), k=6)
    st.subheader("Recommended videos")
    st.caption("Chosen by the Resource Recommender for topics that are important and either weak for you or missing from your notes.")
    for r, res in recs:
        gap = cov.set_index("subtopic").coverage.get(r["subtopic"], "")
        with st.container(border=True):
            st.markdown(f"**{r['subtopic']}** ({r['topic']}) · importance {r['importance']:.0f} · "
                        f"mastery {1 - r['gap']:.0%} · notes: {gap}")
            if res["mode"] == "api":
                for v in res["items"]:
                    st.markdown(f"- [{v['title']}]({v['url']}) — {v['channel']} · {v['minutes']} min · {v['views']:,} views · *{v['purpose']}*")
                if not res["items"]:
                    st.caption("No relevant videos found.")
            else:
                st.markdown(" · ".join(f"[{i['purpose']}]({i['url']})" for i in res["items"]))
    st.subheader("Do your notes cover what gets examined?")
    st.caption("Coverage Auditor: sentences in your study material that discuss each examined subtopic.")
    st.dataframe(cov.sort_values("importance", ascending=False), hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 9. agents
with tabs[8]:
    st.subheader("Multi-agent architecture")
    st.markdown("""
| Agent | Responsibility |
|---|---|
| **Orchestrator** | Runs workflows and routes work between agents via a shared blackboard |
| **Coordinator (LLM)** | After each answer, decides which agents to involve (tutor, videos, difficulty, topic switch, re-plan) and explains why; rule-based fallback without an LLM |
| **Exam Analyst** | Extracts papers, maps PYQs to topics, ranks importance, finds patterns, estimates difficulty |
| **Question Generator** | Picks the next high-value topic and creates MCQ / cloze / PYQ questions |
| **Critic** | Reviews every generated question (giveaways, duplicates, inconsistent options) and rejects flawed ones |
| **Answer Evaluator** | Checks MCQs; grades written answers against your notes (+LLM if available) |
| **Learner Model** | Records attempts and updates mastery with the ML model |
| **Tutor** | Explains missed concepts from your notes and links them to exam patterns |
| **Study Planner** | Builds the time-limited plan and re-plans when mastery changes |
| **Coverage Auditor** | Checks whether your notes cover each examined topic |
| **Resource Recommender** | Suggests YouTube videos for weak, important, or uncovered topics |
""")
    st.caption("Quiz workflow: Question Generator → Critic → (student) → Answer Evaluator → Learner Model → **Coordinator decides** → Tutor / Resource Recommender / difficulty change / topic switch / Study Planner")
    st.caption(f"LLM-powered agents right now: {'Coordinator, Answer Evaluator, Tutor, Question Generator' + (', Critic' if st.session_state.get('llm_critic') else '') if llm.available() else 'none (no LLM detected) — all agents use rules + ML'}")
    st.caption("Resources workflow: Coverage Auditor → Learner Model → Resource Recommender")
    st.subheader("Live agent activity (this session)")
    tr = pd.DataFrame(orch().bb.trace)
    if tr.empty:
        st.info("No activity yet — run the analysis or take a quiz.")
    else:
        st.dataframe(tr.iloc[::-1], hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- 10. evaluation
with tabs[9]:
    st.subheader("Mastery model (ML Component B)")
    if (OUT / "mastery_eval.csv").exists():
        st.dataframe(pd.read_csv(OUT / "mastery_eval.csv"), hide_index=True)
        st.caption("Held-out simulated learners. Retrain on real data once attempts.csv has a few hundred answers.")
    st.subheader("Topic importance (ML Component A) — feature weights")
    st.dataframe(pd.DataFrame({"feature": ["PYQ frequency", "recency", "marks", "repetition", "question-type diversity"],
                               "weight": [0.35, 0.25, 0.20, 0.10, 0.10]}), hide_index=True)
    st.subheader("PYQ → topic mapping")
    st.write(f"Low-confidence mappings: **{int(qdf.low_confidence.sum())}** of {len(qdf)} (review these)")
    st.dataframe(qdf[qdf.low_confidence][["year", "question", "subtopic", "sim", "margin"]], hide_index=True)
    gold = pathlib.Path("sample_data/gold_mapping.csv")
    if gold.exists() and pathlib.Path(st.session_state.material or "").as_posix().startswith("sample_data"):
        g = pd.read_csv(gold, dtype=str).fillna("")
        mm = qdf.fillna("").astype(str).merge(g, on=["paper_id", "qno", "part"])
        st.metric("Mapping accuracy vs hand labels (sample)", f"{(mm.subtopic == mm.gold_subtopic).mean():.1%}")
