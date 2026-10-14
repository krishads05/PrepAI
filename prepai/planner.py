"""Time-constrained study planner.  Priority ~ importance x difficulty x knowledge gap (heuristic)."""
import numpy as np

BASE_MIN = {"Easy": 20, "Medium": 30, "Hard": 45}   # time to cover a topic at mastery 0

def score(tbl):
    t = tbl.copy()
    t["gap"] = 1 - t["mastery"]
    t["study_priority"] = (t["importance"] / 100) * (0.5 + t["difficulty_score"]) * (0.15 + t["gap"])
    t["study_priority"] = (100 * t.study_priority / t.study_priority.max()).round(1)
    return t.sort_values("study_priority", ascending=False)

def plan(tbl, minutes, min_block=15, quiz_share=0.15, only_seen=True):
    t = score(tbl[tbl.papers > 0]) if ("papers" in tbl and only_seen) else score(tbl)
    quiz = int(round(minutes * quiz_share / 5) * 5) if minutes >= 60 else 0
    budget, blocks, skipped = minutes - quiz, [], []
    for r in t.itertuples():
        need = BASE_MIN[r.difficulty] * (0.3 + 0.7 * r.gap)          # less time if partly mastered
        need = max(min_block, int(round(need / 5) * 5))
        if budget >= min_block and r.study_priority >= 25:
            alloc = min(need, budget)
            if alloc >= min_block:
                blocks.append((r.subtopic, r.topic, alloc, r)); budget -= alloc; continue
        skipped.append(r.subtopic)
    if budget > 0 and blocks:                              # leftover -> top topic practice
        quiz += budget
    lines = []
    for sub, top, m, r in blocks:
        mode = "PYQ/numerical practice" if any(x in str(r.question_types) for x in ("numerical", "tracing")) else "theory revision"
        lines.append(f"{m:>3} min - {sub} ({top}) | {r.difficulty}, mastery {r.mastery:.0%}, importance {r.importance:.0f} -> {mode}")
    if quiz:
        lines.append(f"{quiz:>3} min - Quiz on weak areas (updates your mastery)")
    if skipped:
        lines.append(f"Deprioritized today: {', '.join(skipped[:8])}{' ...' if len(skipped) > 8 else ''}")
    return "\n".join(lines), t
