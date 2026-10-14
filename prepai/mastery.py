"""ML Component B: student mastery.
Model predicts P(correct on next question | history on that topic, question difficulty).
Mastery(topic) = predicted P(correct) on a Medium question.
Until real quiz logs exist, it is trained/evaluated on SIMULATED learners (state this in the report)."""
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

# PFA-style features (Pavlik et al., 2009): successes and failures counted separately
FEATS = ["log_correct", "log_wrong", "ewma", "difficulty"]
MONO = [1, -1, 1, -1]   # more correct -> up, more wrong -> down, recent accuracy -> up, harder -> down

def make_model(kind="monotone_gbm"):
    if kind == "logistic_pfa":
        return LogisticRegression(max_iter=1000)
    return HistGradientBoostingClassifier(monotonic_cst=MONO, max_iter=200, learning_rate=0.05,
                                          max_depth=3, random_state=0)
DIFF = {"Easy": 1, "Medium": 2, "Hard": 3}

def history_features(prev_correct, difficulty, decay=0.6):
    n = len(prev_correct)
    acc = (sum(prev_correct) + 1) / (n + 2)            # Laplace smoothing
    ew, w = 0.5, 0.0
    for c in prev_correct:
        ew = decay * ew + (1 - decay) * c
    k = sum(prev_correct)
    return dict(log_correct=np.log1p(k), log_wrong=np.log1p(n - k), ewma=ew,
                difficulty=difficulty, acc_smooth=acc)

def build_dataset(log):
    """log: DataFrame[student, subtopic, ts, difficulty(1-3), correct] -> one row per attempt."""
    rows = []
    for (_, _), g in log.sort_values("ts").groupby(["student", "subtopic"]):
        hist = []
        for r in g.itertuples():
            f = history_features(hist, r.difficulty); f["correct"] = r.correct
            f["student"] = r.student
            if "true_mastery" in g:
                f["true_mastery"] = r.true_mastery
            rows.append(f); hist.append(r.correct)
    return pd.DataFrame(rows)

def simulate(n_students=300, n_topics=8, seed=0):
    """Learners with latent mastery that improves with practice; 4-option MCQs (guess=0.25)."""
    rng = np.random.default_rng(seed); out = []; ts = 0
    for s in range(n_students):
        ability = rng.normal(0, 1)
        for t in range(n_topics):
            theta = np.clip(rng.beta(2, 2.5) + 0.1 * ability, 0.02, 0.98)
            lr = rng.uniform(0.02, 0.12)
            for _ in range(rng.integers(4, 20)):
                d = int(rng.integers(1, 4))
                logit = 6 * (theta - 0.5) - 1.2 * (d - 2)
                p = 0.25 + 0.75 / (1 + np.exp(-logit))
                c = int(rng.random() < p)
                # true mastery = P(correct on Medium question) - comparable to model output
                tm = 0.25 + 0.75 / (1 + np.exp(-6 * (theta - 0.5)))
                out.append(dict(student=s, subtopic=f"T{t}", ts=ts, difficulty=d, correct=c, true_mastery=tm))
                theta = min(0.99, theta + lr * (1 - theta)); ts += 1   # practice + feedback -> learning
    return pd.DataFrame(out)

def evaluate(log, seed=0):
    ds = build_dataset(log)
    rng = np.random.default_rng(seed)
    test_students = set(rng.choice(ds.student.unique(), size=len(ds.student.unique()) // 4, replace=False))
    te = ds.student.isin(test_students); tr, ts_ = ds[~te], ds[te]
    models = {k: make_model(k).fit(tr[FEATS], tr.correct) for k in ("logistic_pfa", "monotone_gbm")}
    preds = {"global mean": np.full(len(ts_), tr.correct.mean()), "running accuracy": ts_["acc_smooth"].values}
    preds.update({k: m.predict_proba(ts_[FEATS])[:, 1] for k, m in models.items()})
    med = ts_[FEATS].assign(difficulty=2)
    mast = {"running accuracy": ts_.acc_smooth.values}
    mast.update({k: m.predict_proba(med)[:, 1] for k, m in models.items()})
    res = []
    for name, p in preds.items():
        res.append(dict(model=name,
                        AUC=roc_auc_score(ts_.correct, p) if name != "global mean" else 0.5,
                        log_loss=log_loss(ts_.correct, np.clip(p, 1e-4, 1 - 1e-4)),
                        accuracy=accuracy_score(ts_.correct, p > 0.5),
                        mastery_MAE=np.mean(np.abs(mast[name] - ts_.true_mastery)) if name in mast and "true_mastery" in ts_ else np.nan))
    coef = pd.Series(models["logistic_pfa"].coef_[0], index=FEATS).round(3)
    return models, pd.DataFrame(res).round(3), coef

def train_and_save(path, log=None):
    log = simulate() if log is None else log
    _, report, coef = evaluate(log)
    d = build_dataset(log)
    model = make_model("monotone_gbm").fit(d[FEATS], d.correct)   # final model on all data
    pickle.dump(model, open(path, "wb"))
    return report, coef

def estimate_mastery(model, attempts, subtopics):
    """attempts: DataFrame of the current student's quiz log. Returns {subtopic: mastery 0-1}."""
    out = {}
    for s in subtopics:
        g = attempts[attempts.subtopic == s].sort_values("ts") if len(attempts) else attempts
        hist = list(g.correct) if len(g) else []
        f = pd.DataFrame([history_features(hist, 2)])[FEATS]
        out[s] = float(model.predict_proba(f)[0, 1])
    return out
