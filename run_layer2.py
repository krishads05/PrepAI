"""PrepAI Layer 2: difficulty, mastery model, adaptive quiz, study planner.
  python run_layer2.py prepare --material sample_data/material   # difficulty + train/evaluate mastery model
  python run_layer2.py quiz --n 10 --mode mixed --material sample_data/material
  python run_layer2.py status             # topics with current mastery
  python run_layer2.py plan --minutes 120
"""
import argparse, pickle, pathlib, time, pandas as pd
from prepai.difficulty import add_difficulty
from prepai.mastery import train_and_save, estimate_mastery
from prepai.planner import plan
from prepai.quiz import QuizEngine

def load(out):
    out = pathlib.Path(out)
    qdf = pd.read_csv(out / "questions_mapped.csv")
    tbl = pd.read_csv(out / "topic_table.csv") if (out / "topic_table.csv").exists() else None
    return out, qdf, tbl

def current_mastery(out, tbl, student):
    model = pickle.load(open(out / "mastery_model.pkl", "rb"))
    att = pd.read_csv(out / "attempts.csv") if (out / "attempts.csv").exists() else pd.DataFrame(columns=["student", "subtopic", "ts", "correct"])
    att = att[att.student == student]
    return estimate_mastery(model, att, tbl.subtopic.tolist()), att

def cmd_prepare(a):
    out, qdf, _ = load(a.out)
    pr = pd.read_csv(out / "topic_priority.csv")
    tbl = add_difficulty(qdf, pr, use_llm=not a.no_llm, material_dir=a.material)
    tbl = tbl[tbl.topic != "UNMAPPED"]
    tbl.to_csv(out / "topic_table.csv", index=False)
    from prepai import llm; print("LLM backend:", llm.backend() if not a.no_llm else "disabled")
    print(tbl[["subtopic", "importance", "diff_pyq", "diff_text", "diff_llm", "difficulty"]].head(12).to_string(index=False))
    report, coef = train_and_save(out / "mastery_model.pkl")
    print("\nMastery model (simulated learners, held-out students):\n", report.to_string(index=False))
    print("\nFeature weights:\n", coef.to_string())
    report.to_csv(out / "mastery_eval.csv", index=False)

def cmd_status(a):
    out, _, tbl = load(a.out)
    m, att = current_mastery(out, tbl, a.student)
    tbl["mastery"] = tbl.subtopic.map(m); tbl["attempts"] = tbl.subtopic.map(att.subtopic.value_counts()).fillna(0).astype(int)
    print(tbl[tbl.papers > 0][["topic", "subtopic", "importance", "difficulty", "attempts", "mastery"]]
          .assign(mastery=lambda d: [f"{m:.0%}" if n else "- (no data)" for m, n in zip(d.mastery, d.attempts)]).to_string(index=False))

def cmd_quiz(a):
    out, qdf, tbl = load(a.out)
    import json
    syl = json.load(open(out / "state.json"))["syllabus"] if (out / "state.json").exists() else []
    eng = QuizEngine(tbl, qdf, out, a.material, syl); recent, score = [], 0
    for i in range(a.n):
        m, att = current_mastery(out, tbl, a.student)            # mastery updates after every answer
        for _ in range(8):                                        # retry if a topic has no question left
            topic, sub, ms = eng.pick_topic(a.mode, m, a.topic, recent)
            diff = eng.pick_difficulty(sub, a.difficulty, att)
            q = eng.get_question(topic, sub, diff)
            if q:
                break
        if not q:
            print("No more questions available for this selection."); break
        print(f"\nQ{i+1} [{sub} | {diff} | mastery {ms:.0%}]\n{q['question']}")
        t0 = time.time()
        if q["kind"] == "mcq":
            for j, o in enumerate(q["options"]):
                print(f"  {'ABCD'[j]}) {o}")
            ans = ""
            while ans not in ("A", "B", "C", "D"):
                ans = input("Your answer (A-D): ").strip().upper()[:1]
            ok = ans == "ABCD"[q["answer_index"]]
            print(("Correct! " if ok else f"Wrong - answer {'ABCD'[q['answer_index']]}. ") + q.get("explanation", ""))
        else:
            input("(Offline PYQ practice) Attempt it, then press Enter...")
            yn = ""
            while yn not in ("y", "n"):
                yn = input("Did you get it fully right? (y/n): ").strip().lower()[:1]
            ok = yn == "y"
        eng.log(a.student, topic, sub, diff, ok, time.time() - t0); recent.append(sub); score += ok
    print(f"\nScore: {score}/{a.n}"); cmd_status(a)

def cmd_plan(a):
    out, _, tbl = load(a.out)
    m, _ = current_mastery(out, tbl, a.student); tbl["mastery"] = tbl.subtopic.map(m)
    text, _ = plan(tbl, a.minutes)
    print(f"Study plan for {a.minutes} min:\n{text}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["prepare", "quiz", "status", "plan"])
    ap.add_argument("--out", default="outputs"); ap.add_argument("--student", default="me")
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--mode", default="mixed", choices=["mixed", "important", "weak", "all"])
    ap.add_argument("--topic"); ap.add_argument("--difficulty", default="Adaptive", choices=["Easy", "Medium", "Hard", "Adaptive"])
    ap.add_argument("--material"); ap.add_argument("--minutes", type=int, default=120); ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args(); {"prepare": cmd_prepare, "quiz": cmd_quiz, "status": cmd_status, "plan": cmd_plan}[a.cmd](a)
