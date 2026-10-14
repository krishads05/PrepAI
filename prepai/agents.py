"""PrepAI multi-agent system (blackboard architecture).
Agents share state on a Blackboard and log every action to a trace, so the
collaboration is visible.  Each agent works offline; the Tutor and Evaluator
use an LLM (Ollama/Anthropic) automatically when one is available."""
import contextlib, io, json, pathlib, pickle, time
from dataclasses import dataclass, field
from types import SimpleNamespace
import pandas as pd
from . import llm
from .quiz import QuizEngine
from .qgen import ClozeGenerator
from .evaluator import AnswerEvaluator
from .mastery import estimate_mastery
from .planner import plan as make_plan

@dataclass
class Blackboard:
    out: pathlib.Path
    state: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    def post(self, agent, action, detail=""):
        self.trace.append(dict(time=time.strftime("%H:%M:%S"), agent=agent, action=action, detail=str(detail)[:300]))

class Agent:
    name = "Agent"
    def __init__(self, bb):
        self.bb = bb
    def log(self, action, detail=""):
        self.bb.post(self.name, action, detail)

class ExamAnalystAgent(Agent):
    """Reads syllabus/PYQs/notes; maps questions, ranks topics, finds patterns, estimates difficulty."""
    name = "Exam Analyst"
    def analyze(self, syllabus, pyqs, material):
        import run_layer1, run_layer2
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run_layer1.main(SimpleNamespace(syllabus=str(syllabus), pyqs=str(pyqs), material=material, out=str(self.bb.out),
                                            mode="hybrid", model="all-MiniLM-L6-v2", w=0.5, alpha=0.65))
            run_layer2.cmd_prepare(SimpleNamespace(out=str(self.bb.out), no_llm=False, material=material))
        self.load()
        t = self.bb.state["table"]
        self.log("analysed course", f"{len(self.bb.state['qdf'])} PYQs -> {len(t)} subtopics; top: {', '.join(t.subtopic.head(3))}")
        return buf.getvalue()
    def load(self):
        o = self.bb.out
        self.bb.state.update(table=pd.read_csv(o / "topic_table.csv"), qdf=pd.read_csv(o / "questions_mapped.csv"),
                             syl=json.load(open(o / "state.json"))["syllabus"],
                             patterns=json.load(open(o / "patterns.json")))

class LearnerModelAgent(Agent):
    """Owns the student's attempt history and the ML mastery model."""
    name = "Learner Model"
    def __init__(self, bb):
        super().__init__(bb)
        self.model = pickle.load(open(bb.out / "mastery_model.pkl", "rb"))
    def attempts(self, student):
        p = self.bb.out / "attempts.csv"
        a = pd.read_csv(p) if p.exists() else pd.DataFrame(columns=["student", "subtopic", "ts", "correct"])
        return a[a.student == student]
    def mastery(self, student):
        return estimate_mastery(self.model, self.attempts(student), self.bb.state["table"].subtopic.tolist())
    def record(self, engine, student, q, correct, secs):
        before = self.mastery(student)[q["sb"]]
        engine.log(student, q["tp"], q["sb"], q["df"], correct, secs)
        after = self.mastery(student)[q["sb"]]
        self.log("updated mastery", f"{q['sb']}: {before:.0%} -> {after:.0%} ({'correct' if correct else 'wrong'})")
        return before, after

class QuestionGeneratorAgent(Agent):
    """Chooses the next high-value topic and produces a question (LLM MCQ / cloze MCQ / PYQ)."""
    name = "Question Generator"
    def __init__(self, bb, material):
        super().__init__(bb)
        s = bb.state
        self.engine = QuizEngine(s["table"], s["qdf"], bb.out, material, s["syl"])
    def next(self, mastery, attempts, mode="mixed", topic=None, difficulty="Adaptive", recent=()):
        avoid = list(recent) + list(self.bb.state.get("avoid", []))
        for _ in range(8):
            tp, sb, ms = self.engine.pick_topic(mode, mastery, topic, avoid)
            df = self.bb.state.get("diff_override", {}).pop(sb, None) or self.engine.pick_difficulty(sb, difficulty, attempts)
            q = self.engine.get_question(tp, sb, df)
            if q:
                q.update(tp=tp, sb=sb, df=df, ms=ms)
                self.log("generated question", f"{q['kind'].upper()} on {sb} ({df}) - chosen by '{mode}' policy, mastery {ms:.0%}")
                return q
        self.log("no question available", topic or mode)
        return None

class AnswerEvaluatorAgent(Agent):
    """Grades MCQs exactly and written answers against the student's notes (+LLM if available)."""
    name = "Answer Evaluator"
    def __init__(self, bb, retriever, terms):
        super().__init__(bb)
        self.ev = AnswerEvaluator(retriever, terms)
    def evaluate(self, q, answer):
        if q["kind"] == "mcq":
            ok = answer == q["answer_index"]
            self.log("checked MCQ", "correct" if ok else f"wrong (answer: {q['options'][q['answer_index']]})")
            return dict(gradable=True, correct=ok, score=10.0 if ok else 0.0, method="exact", feedback=q.get("explanation", ""))
        r = self.ev.grade(q["question"], answer, q["tp"], q["sb"])
        if r["gradable"]:
            self.log("graded written answer", f"{r['score']}/10 via {r['method']}; missing: {', '.join(r.get('missing_terms', [])[:3]) or '-'}")
        else:
            self.log("could not grade", "no matching notes - asked student to self-grade")
        return r

class TutorAgent(Agent):
    """Explains weak concepts from the student's notes and links them to PYQ patterns."""
    name = "Tutor"
    def explain(self, q, evaluation):
        missed = evaluation.get("missed") or self.ret.top_sentences(f"{q['question']} {q['sb']}", k=3, min_score=0.1, sub=q["sb"])
        pats = self.bb.state["patterns"]
        p = next((v for v in pats.values() if v["subtopic"] == q["sb"]), None)
        exp = dict(key_points=missed[:3], sources=[self.ret.src.get(x, "") for x in missed[:3]],
                   exam_tip=(f"Asked in {p['papers']} past papers. {p['recommendation']}" if p else ""))
        if llm.available() and missed:
            try:
                exp["explanation"] = llm.ask(
                    "Explain these points simply to a student who got this exam question wrong, in under 120 words. "
                    "Use ONLY the material below - add no facts that are not in it.\n"
                    f"Question: {q['question']}\nMaterial: {' '.join(missed[:4])}", max_tokens=300)
            except Exception:
                pass
        self.log("explained concept", f"{q['sb']}: {len(exp['key_points'])} key points from notes")
        return exp
    def __init__(self, bb, retriever):
        super().__init__(bb); self.ret = retriever

class StudyPlannerAgent(Agent):
    """Builds the time-constrained plan and re-plans when mastery changes."""
    name = "Study Planner"
    def plan(self, mastery, minutes):
        t = self.bb.state["table"].copy(); t["mastery"] = t.subtopic.map(mastery)
        text, scored = make_plan(t, minutes)
        top = scored[scored.papers > 0].subtopic.head(3).tolist()
        self.bb.state["plan_top"] = top
        self.log("built plan", f"{minutes} min; focus: {', '.join(top)}")
        return text, scored
    def replan_if_needed(self, mastery, minutes=120):
        old = self.bb.state.get("plan_top")
        _, scored = make_plan(self.bb.state["table"].assign(mastery=lambda d: d.subtopic.map(mastery)), minutes)
        new = scored[scored.papers > 0].subtopic.head(3).tolist()
        if old is not None and new != old:
            self.log("re-planned", f"priorities changed: {', '.join(new)}")
        self.bb.state["plan_top"] = new
        return new != old

class Orchestrator:
    """Coordinates the agents through three workflows: analyse, quiz turn, daily plan."""
    def __init__(self, out, material=None):
        self.bb = Blackboard(pathlib.Path(out)); self.material = material
        self.analyst = ExamAnalystAgent(self.bb)
        if (self.bb.out / "topic_table.csv").exists():
            self.analyst.load(); self._spawn()
    def _spawn(self):
        self.learner = LearnerModelAgent(self.bb)
        self.generator = QuestionGeneratorAgent(self.bb, self.material)
        ret = self.generator.engine.ret
        terms = ClozeGenerator(self.bb.state["syl"], ret).terms if ret.chunks else []
        self.evaluator = AnswerEvaluatorAgent(self.bb, ret, terms)
        self.tutor = TutorAgent(self.bb, ret)
        self.planner = StudyPlannerAgent(self.bb)
        self.bb.post("Orchestrator", "agents ready", f"LLM backend: {llm.backend()}")

    # workflow 1
    def analyze_course(self, syllabus, pyqs, material):
        self.material = material
        self.bb.post("Orchestrator", "workflow: analyse course")
        log = self.analyst.analyze(syllabus, pyqs, material)
        self._spawn()
        return log

    # workflow 2a
    def next_question(self, student, **kw):
        return self.generator.next(self.learner.mastery(student), self.learner.attempts(student), **kw)

    # workflow 2b: Evaluator -> Learner Model -> Tutor (if wrong) -> Planner
    def submit(self, student, q, answer, secs=0.0, self_grade=None):
        self.bb.post("Orchestrator", "workflow: answer submitted", q["sb"])
        ev = dict(gradable=False, correct=bool(self_grade), method="self-graded") if self_grade is not None \
            else self.evaluator.evaluate(q, answer)
        result = dict(evaluation=ev)
        if not ev.get("gradable") and self_grade is None:
            return result                                  # UI asks the student to self-grade
        result["mastery"] = self.learner.record(self.generator.engine, student, q, ev["correct"], secs)
        if not ev["correct"]:
            result["tutor"] = self.tutor.explain(q, ev)
        result["replanned"] = self.planner.replan_if_needed(self.learner.mastery(student))
        return result

    # workflow 3
    def daily_plan(self, student, minutes):
        self.bb.post("Orchestrator", "workflow: daily plan")
        return self.planner.plan(self.learner.mastery(student), minutes)


# ---------------------------------------------------------------- additional agents
import re as _re
from .resources import recommend as _recommend, DEFAULT_CHANNELS

class CriticAgent(Agent):
    """Reviews every generated question before the student sees it; rejects flawed ones."""
    name = "Critic"
    def review(self, q):
        if q["kind"] != "mcq":
            return True, "PYQ from past paper - accepted"
        opts, ans = q["options"], q["options"][q["answer_index"]]
        low = [o.lower().strip() for o in opts]
        problems = []
        if len(set(low)) < 4:
            problems.append("duplicate options")
        if any(o != ans.lower() and (o in ans.lower() or ans.lower() in o) for o in low):
            problems.append("an option overlaps the answer")
        stem = q["question"].lower()
        if any(_re.search(rf"(?<![\w-]){_re.escape(o)}(?![\w-])", stem) for o in low):
            problems.append("an option appears in the question (giveaway)")
        shapes = {("A" if o.isupper() else "M" if " " in o.strip() else "S") for o in opts}
        if len(shapes) > 2:
            problems.append("options have inconsistent form (answer may stand out)")
        if len(q["question"].split()) < 6:
            problems.append("question too short to be meaningful")
        if problems:
            self.log("rejected question", f"{q['sb']}: {'; '.join(problems)}")
            return False, problems
        self.log("approved question", f"{q['sb']}: passed 5 quality checks")
        return True, []

class CoverageAuditorAgent(Agent):
    """Checks whether the study material actually covers each examined subtopic."""
    name = "Coverage Auditor"
    def __init__(self, bb, retriever):
        super().__init__(bb); self.ret = retriever
    GENERIC = {"and", "of", "the", "problem", "problems", "methods", "method", "algorithm", "algorithms",
               "structure", "process", "processes", "system", "management", "file", "files"}

    def _hits(self, sub, text):
        """Sentences that name the subtopic, one of its listed terms, or >= 2 of its distinctive words."""
        stem = lambda w: w[:-1] if len(w) > 4 and w.endswith("s") else w
        m = _re.search(r"\((.*?)\)", text)
        phrases = [sub] + ([x.strip() for x in m.group(1).split(",")] if m else [])
        words = {stem(w) for w in _re.findall(r"[a-z0-9']+", sub.lower()) if w not in self.GENERIC and len(w) > 2}
        hits, sents = [], self.ret.sents
        for i, s in enumerate(sents):
            low = s.lower()
            sw = {stem(w) for w in _re.findall(r"[a-z0-9']+", low)}
            if any(_re.search(rf"(?<![\w-]){_re.escape(p.lower())}s?(?![\w-])", low) for p in phrases if len(p) > 2) \
                    or (len(words) >= 2 and len(words & sw) >= 2):
                # sentence window: the next sentences usually continue the same topic
                for x in sents[i:i + 3]:
                    if x not in hits and (x == s or words & {stem(w) for w in _re.findall(r"[a-z0-9']+", x.lower())}):
                        hits.append(x)
        return hits

    def audit(self):
        t = self.bb.state["table"]
        text = {r["subtopic"]: r["text"] for r in self.bb.state["syl"]}
        rows = []
        for r in t.itertuples():                              # every syllabus subtopic, examined or not
            hits = self._hits(r.subtopic, text.get(r.subtopic, r.subtopic)) if self.ret.chunks else []
            n = len(hits)
            status = "Covered" if n >= 2 else "Thin" if n == 1 else "Missing"
            rows.append(dict(topic=r.topic, subtopic=r.subtopic, importance=r.importance, papers=r.papers,
                             matching_sentences=n, coverage=status, example=hits[0][:120] if hits else "",
                             found_in=", ".join(dict.fromkeys(self.ret.src.get(h, "") for h in hits))[:120]))
        df = pd.DataFrame(rows)
        gaps = df[(df.coverage != "Covered") & (df.importance >= 45)]
        self.bb.state["coverage"] = df
        self.log("audited notes", f"{(df.coverage == 'Covered').sum()}/{len(df)} covered; important gaps: "
                 f"{', '.join(gaps.subtopic.head(4)) or 'none'}")
        return df

class ResourceRecommenderAgent(Agent):
    """Recommends YouTube videos for weak, important, or uncovered topics."""
    name = "Resource Recommender"
    def recommend(self, row, channels=DEFAULT_CHANNELS, course=""):
        res = _recommend(row["topic"], row["subtopic"], row.get("question_types", ""), channels, course, self.bb.out)
        self.log("recommended videos", f"{row['subtopic']}: {len(res['items'])} {'videos (API)' if res['mode'] == 'api' else 'search links'}")
        return res
    def targets(self, mastery, coverage, k=6):
        """Priority = importance x (knowledge gap + notes gap)."""
        t = self.bb.state["table"]; t = t[t.papers > 0].copy()
        t["gap"] = 1 - t.subtopic.map(mastery)
        cov = coverage.set_index("subtopic").coverage if coverage is not None and len(coverage) else pd.Series(dtype=str)
        t["notes_gap"] = t.subtopic.map({"Missing": 1.0, "Thin": 0.5, "Covered": 0.0}).fillna(0) if cov.empty else \
            t.subtopic.map(cov).map({"Missing": 1.0, "Thin": 0.5, "Covered": 0.0}).fillna(0)
        t["video_priority"] = t.importance * (t.gap + t.notes_gap)
        return t.sort_values("video_priority", ascending=False).head(k)

# wire new agents into the orchestrator
_orig_spawn = Orchestrator._spawn
def _spawn(self):
    _orig_spawn(self)
    self.critic = CriticAgent(self.bb)
    self.auditor = CoverageAuditorAgent(self.bb, self.generator.engine.ret)
    self.resources = ResourceRecommenderAgent(self.bb)
Orchestrator._spawn = _spawn

def _next_question(self, student, **kw):
    """Generator proposes, Critic reviews; rejected questions are regenerated."""
    for _ in range(6):
        q = self.generator.next(self.learner.mastery(student), self.learner.attempts(student), **kw)
        if q is None:
            return None
        ok, _ = self.critic.review(q)
        if ok:
            return q
    return q
Orchestrator.next_question = _next_question

def _resources(self, student, channels=DEFAULT_CHANNELS, course="", k=6):
    """Workflow: Coverage Auditor -> Learner Model -> Resource Recommender."""
    self.bb.post("Orchestrator", "workflow: study resources")
    cov = self.auditor.audit()
    tg = self.resources.targets(self.learner.mastery(student), cov, k)
    return cov, [(r, self.resources.recommend(r, channels, course)) for r in tg.to_dict("records")]
Orchestrator.study_resources = _resources


# ---------------------------------------------------------------- LLM coordinator
from . import llm as _llm

class CoordinatorAgent(Agent):
    """Decides which agents to involve after each answer. Uses the LLM when available,
    otherwise a transparent rule-based policy. Every decision is logged with its reason."""
    name = "Coordinator"
    ACTIONS = {
        "tutor": "Tutor explains the missed concepts from the notes",
        "videos": "Resource Recommender suggests YouTube videos for this topic",
        "easier": "next question on this topic one difficulty level lower",
        "harder": "next question on this topic one difficulty level higher",
        "switch_topic": "move away from this topic for the next few questions",
        "replan": "Study Planner recomputes today's priorities",
        "continue": "no intervention needed",
    }

    def rules(self, c):
        acts = []
        if not c["correct"]:
            acts.append("tutor")
            if c["wrong_streak"] >= 2:
                acts += ["easier", "videos"]
            if c["notes_coverage"] in ("Missing", "Thin"):
                acts.append("videos")
        elif c["right_streak"] >= 2:
            acts += ["harder", "switch_topic"]
        if abs(c["mastery_after"] - c["mastery_before"]) >= 0.05:
            acts.append("replan")
        return list(dict.fromkeys(acts)) or ["continue"]

    def decide(self, c):
        if _llm.available():
            prompt = ("You coordinate agents in an exam-preparation tutor. Based on the student's latest answer, choose 1-3 "
                      "actions that best help them learn efficiently.\nAvailable actions:\n" +
                      "\n".join(f"- {k}: {v}" for k, v in self.ACTIONS.items()) +
                      f"\nSituation: {json.dumps(c)}\n"
                      'Return {"actions": ["..."], "reason": "one sentence"}')
            try:
                out = _llm.ask_json(prompt, max_tokens=200)
                acts = [a for a in out.get("actions", []) if a in self.ACTIONS][:3]
                if acts:
                    reason = str(out.get("reason", ""))[:200]
                    self.log("decided (LLM)", f"{', '.join(acts)} - {reason}")
                    return acts, reason, f"LLM ({_llm.model_name()})"
                self.log("LLM answer invalid", "falling back to rules")
            except Exception as e:
                self.log("LLM unavailable", f"{type(e).__name__} - falling back to rules")
        acts = self.rules(c)
        self.log("decided (rules)", ", ".join(acts))
        return acts, "rule-based policy", "rules"

def _llm_check(self, q):
    """Critic second opinion: does the LLM agree with the answer key?"""
    if q["kind"] != "mcq" or not (self.bb.state.get("llm_critic") and _llm.available()):
        return True
    try:
        out = _llm.ask_json("Answer this multiple-choice question. Question: " + q["question"] + "\nOptions: " +
                            json.dumps(q["options"]) + '\nReturn {"answer_index": n, "confident": true/false}', max_tokens=60)
        n = int(out.get("answer_index", -1))
        if out.get("confident", True) and n in range(4) and n != q["answer_index"]:
            self.log("rejected question", f"{q['sb']}: LLM disagrees with answer key ({q['options'][n]} vs {q['options'][q['answer_index']]})")
            return False
    except Exception:
        pass
    return True

_rule_review = CriticAgent.review
def _review(self, q):
    ok, why = _rule_review(self, q)
    if ok and not _llm_check(self, q):
        return False, ["LLM disagrees with answer key"]
    return ok, why
CriticAgent.review = _review

_spawn2 = Orchestrator._spawn
def _spawn_c(self):
    _spawn2(self)
    self.coordinator = CoordinatorAgent(self.bb)
    self.bb.state.setdefault("diff_override", {}); self.bb.state.setdefault("avoid", [])
Orchestrator._spawn = _spawn_c

def _context(self, student, q, ev, before, after):
    a = self.learner.attempts(student)
    hist = a[a.subtopic == q["sb"]].sort_values("ts").correct.astype(int).tolist()[-5:]
    streak = lambda v: next((i for i, x in enumerate(reversed(hist)) if x != v), len(hist))
    row = self.bb.state["table"].set_index("subtopic").loc[q["sb"]]
    cov = self.bb.state.get("coverage")
    covs = cov.set_index("subtopic").coverage.get(q["sb"], "unknown") if cov is not None and len(cov) else "unknown"
    return dict(subtopic=q["sb"], topic=q["tp"], question_type=q["kind"], difficulty=q["df"], correct=bool(ev["correct"]),
                score_out_of_10=ev.get("score"), mastery_before=round(before, 2), mastery_after=round(after, 2),
                recent_results_on_topic=hist, wrong_streak=streak(0), right_streak=streak(1),
                topic_importance=float(row.importance), topic_difficulty=row.difficulty, notes_coverage=covs)

def _submit(self, student, q, answer, secs=0.0, self_grade=None):
    """Evaluator -> Learner Model -> Coordinator decides -> chosen agents act."""
    self.bb.post("Orchestrator", "workflow: answer submitted", q["sb"])
    ev = dict(gradable=False, correct=bool(self_grade), method="self-graded") if self_grade is not None \
        else self.evaluator.evaluate(q, answer)
    result = dict(evaluation=ev)
    if not ev.get("gradable") and self_grade is None:
        return result
    before, after = self.learner.record(self.generator.engine, student, q, ev["correct"], secs)
    result["mastery"] = (before, after)
    if "coverage" not in self.bb.state:
        self.auditor.audit()
    acts, reason, mode = self.coordinator.decide(_context(self, student, q, ev, before, after))
    result["coordinator"] = dict(actions=acts, reason=reason, mode=mode)
    levels = ["Easy", "Medium", "Hard"]
    for a in acts:
        if a == "tutor":
            result["tutor"] = self.tutor.explain(q, ev)
        elif a == "videos":
            row = self.bb.state["table"].set_index("subtopic").loc[q["sb"]].to_dict(); row["subtopic"] = q["sb"]
            result["videos"] = self.resources.recommend(row)["items"][:3]
        elif a in ("easier", "harder"):
            i = levels.index(q["df"]) + (-1 if a == "easier" else 1)
            self.bb.state["diff_override"][q["sb"]] = levels[max(0, min(2, i))]
        elif a == "switch_topic":
            self.bb.state["avoid"] = (self.bb.state["avoid"] + [q["sb"]])[-2:]
        elif a == "replan":
            result["replanned"] = self.planner.replan_if_needed(self.learner.mastery(student))
    return result
Orchestrator.submit = _submit


# ---------------------------------------------------------------- exam scope (mid-sem / internals)
def _scoped(bb, floor=20):
    """Full table, or only the active scope's subtopics (importance floored so in-scope topics never vanish)."""
    t = bb.state["table"]
    sc = bb.state.get("scope")
    if not sc:
        return t, True
    t = t[t.subtopic.isin(sc)].copy()
    t["importance"] = t.importance.clip(lower=floor)
    return t, False

def _set_scope(self, subtopics):
    self.bb.state["scope"] = set(subtopics) if subtopics else None
    self.generator.engine.scope = self.bb.state["scope"]
    self.bb.state.pop("plan_top", None)
    self.bb.post("Orchestrator", "exam scope set", f"{len(subtopics)} subtopics" if subtopics else "full syllabus")
Orchestrator.set_scope = _set_scope

def _plan(self, mastery, minutes):
    t, only_seen = _scoped(self.bb); t = t.copy(); t["mastery"] = t.subtopic.map(mastery)
    text, scored = make_plan(t, minutes, only_seen=only_seen)
    top = scored.subtopic.head(3).tolist() if not only_seen else scored[scored.papers > 0].subtopic.head(3).tolist()
    self.bb.state["plan_top"] = top
    self.log("built plan", f"{minutes} min; focus: {', '.join(top)}" + ("" if only_seen else " (scoped)"))
    return text, scored
StudyPlannerAgent.plan = _plan

def _replan(self, mastery, minutes=120):
    old = self.bb.state.get("plan_top")
    t, only_seen = _scoped(self.bb)
    _, scored = make_plan(t.assign(mastery=lambda d: d.subtopic.map(mastery)), minutes, only_seen=only_seen)
    new = scored.subtopic.head(3).tolist()
    if old is not None and new != old:
        self.log("re-planned", f"priorities changed: {', '.join(new)}")
    self.bb.state["plan_top"] = new
    return old is not None and new != old
StudyPlannerAgent.replan_if_needed = _replan

_orig_targets = ResourceRecommenderAgent.targets
def _targets(self, mastery, coverage, k=6):
    sc = self.bb.state.get("scope")
    if not sc:
        return _orig_targets(self, mastery, coverage, k)
    full = self.bb.state["table"]
    self.bb.state["table"], _ = _scoped(self.bb)
    self.bb.state["table"] = self.bb.state["table"].assign(papers=lambda d: d.papers.clip(lower=1))
    try:
        return _orig_targets(self, mastery, coverage, k)
    finally:
        self.bb.state["table"] = full
ResourceRecommenderAgent.targets = _targets


# ---------------------------------------------------------------- content-based topic map (material -> syllabus)
from . import topicmap as _tmap

_spawn3 = Orchestrator._spawn
def _spawn_tm(self):
    _spawn3(self)
    ret = self.generator.engine.ret
    try:
        tm = _tmap.build(ret, self.bb.state["syl"], self.bb.out) if ret.chunks else None
    except Exception as e:
        tm = None
        self.bb.post("Coverage Auditor", "topic map failed", type(e).__name__)
    self.bb.state["topic_map"] = tm
    if tm is not None and len(tm):
        ret.topic_src = {sub: set(g.source) for sub, g in tm[tm.subtopic != "(not in syllabus)"].groupby("subtopic")}
        self.bb.post("Coverage Auditor", "mapped materials to syllabus",
                     f"{(tm.subtopic != '(not in syllabus)').sum()}/{len(tm)} slides/pages matched to {len(ret.topic_src)} subtopics")
Orchestrator._spawn = _spawn_tm

_lex_audit = CoverageAuditorAgent.audit
def _audit(self):
    """Coverage = content-based topic map (works even when names differ) + lexical evidence."""
    df = _lex_audit(self)
    tm = self.bb.state.get("topic_map")
    if tm is None or not len(tm) or df.empty:
        return df
    summ = _tmap.summary_by_subtopic(tm)
    df["mapped_units"] = df.subtopic.map(lambda s: summ.get(s, {}).get("units", 0))
    lex = df.matching_sentences
    df["coverage"] = ["Covered" if (u >= 2 or l >= 2) else "Thin" if (u or l) else "Missing" for u, l in zip(df.mapped_units, lex)]
    df["found_in"] = [summ.get(s, {}).get("sources", "") or f for s, f in zip(df.subtopic, df.found_in)]
    self.bb.state["coverage"] = df
    return df
CoverageAuditorAgent.audit = _audit
