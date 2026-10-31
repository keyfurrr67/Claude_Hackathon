"""Walks training settings in 0.05 steps, selecting on train out-of-fold results only.

Usage: python tune.py

Round by round, each setting moves one 0.05 step from the current best. A step is accepted
only if it beats the current best by more than 2 standard errors of the paired fold
difference. The walk stops when no step does. Recall targets are then swept without
retraining. The test file is never read. Writes outputs/tuning.csv and
outputs/tuning_thresholds.csv.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

import pipeline
from pipeline import (
    CRITICAL_LOG_PC, CRITICAL_MISS_M, ELEVATED_LOG_PC, FEATURES, N_FOLDS, OUT, SAFE_MISS_M, TARGET,
    calibrate, decide, score, tier_from_pc,
)

STEP = 0.05
MAX_ROUNDS = 6
START = {"learning_rate": 0.06, "rsm": 1.0, "subsample": 1.0}
LIMITS = {"learning_rate": (0.01, 0.31), "rsm": (0.5, 1.0), "subsample": (0.5, 1.0)}
RECALLS = [0.75, 0.80, 0.85, 0.90, 0.95]
HELD_ELEVATED_RECALL = 0.90


def as_params(config):
    params = {"learning_rate": config["learning_rate"], "rsm": config["rsm"]}
    if config["subsample"] < 1:
        params.update(bootstrap_type="Bernoulli", subsample=config["subsample"])
    return params


def evaluate_config(train, config):
    """Returns out-of-fold p4/p6 and the per-fold score used for selection."""
    pipeline.PARAMS.clear()
    pipeline.PARAMS.update(as_params(config))
    scope = train.time_to_tca >= 2
    oof, folds = [], []
    for fit_idx, val_idx in GroupKFold(N_FOLDS).split(train, groups=train.event_id):
        X, risk = train.iloc[fit_idx][FEATURES], train.iloc[fit_idx][TARGET]
        val = train.iloc[val_idx]
        p4 = pipeline.fit_model(X, (risk >= CRITICAL_LOG_PC).astype(int), "Logloss").predict_proba(pipeline.pool(val[FEATURES]))[:, 1]
        p6 = pipeline.fit_model(X, (risk >= ELEVATED_LOG_PC).astype(int), "Logloss").predict_proba(pipeline.pool(val[FEATURES]))[:, 1]
        oof.append(pd.DataFrame({"p4": p4, "p6": p6}, index=val.index))
        hard4 = (scope.iloc[val_idx] & (val.miss_distance >= CRITICAL_MISS_M)).to_numpy()
        hard6 = (scope.iloc[val_idx] & (val.miss_distance > SAFE_MISS_M)).to_numpy()
        folds.append(np.mean([
            average_precision_score(val[TARGET][hard4] >= CRITICAL_LOG_PC, p4[hard4]),
            average_precision_score(val[TARGET][hard6] >= ELEVATED_LOG_PC, p6[hard6]),
        ]))
    return pd.concat(oof).sort_index(), np.array(folds)


def neighbours(config, seen):
    for key, (low, high) in LIMITS.items():
        for delta in (-STEP, STEP):
            value = round(config[key] + delta, 2)
            candidate = {**config, key: value}
            if low <= value <= high and tuple(candidate.values()) not in seen:
                seen.add(tuple(candidate.values()))
                yield candidate


def walk(train):
    best = dict(START)
    best_oof, best_folds = evaluate_config(train, best)
    seen = {tuple(best.values())}
    rows = [{"round": 0, **best, "score": best_folds.mean(), "gain": 0.0, "two_se": 0.0, "accepted": True}]
    print(rows[-1], flush=True)
    for round_no in range(1, MAX_ROUNDS + 1):
        accepted = None
        for candidate in neighbours(best, seen):
            oof, folds = evaluate_config(train, candidate)
            diff = folds - best_folds
            two_se = 2 * diff.std(ddof=1) / np.sqrt(N_FOLDS)
            passed = diff.mean() > two_se
            rows.append({"round": round_no, **candidate, "score": folds.mean(), "gain": diff.mean(), "two_se": two_se, "accepted": False})
            print(rows[-1], "beats noise" if passed else "", flush=True)
            if passed and (accepted is None or diff.mean() > accepted[3]):
                accepted = (candidate, oof, folds, diff.mean(), len(rows) - 1)
        if accepted is None:
            break
        best, best_oof, best_folds = accepted[:3]
        rows[accepted[4]]["accepted"] = True
    pd.DataFrame(rows).to_csv(OUT / "tuning.csv", index=False)
    return best, best_oof


def sweep_recalls(train, oof):
    scope = (train.time_to_tca >= 2).to_numpy()
    frame, raw = train[scope], oof.assign(q10=0.0, q50=0.0, q90=0.0)[scope]
    truth = tier_from_pc(frame[TARGET], frame.miss_distance)
    hard = ((frame[TARGET] >= CRITICAL_LOG_PC) & (frame.miss_distance >= CRITICAL_MISS_M)).to_numpy()
    rows = []
    for critical_recall in RECALLS:
        for elevated_recall in RECALLS:
            iso4, iso6, thresholds = calibrate(oof, train, critical_recall, elevated_recall)
            pred = decide(raw, frame, iso4, iso6, thresholds).predicted_tier.to_numpy()
            s = score(truth, pred)
            rows.append({
                "critical_recall_target": critical_recall, "elevated_recall_target": elevated_recall,
                "macro_f1": s["macro_f1"], "hard_critical_recall": (pred[hard] == "Critical").mean(),
                "missed_critical": s["missed_critical"], "false_critical": s["false_critical"],
                "safe_but_not_safe": s["safe_called_safe_but_not"],
            })
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    table.to_csv(OUT / "tuning_thresholds.csv", index=False)
    return table


def main():
    train = pipeline.add_features(pipeline.load("train_data.csv"))
    best, oof = walk(train)
    table = sweep_recalls(train, oof)
    print("best settings", best)
    print(table.round(4).to_string(index=False))
    # Macro F1 alone favours a lower Elevated target, which more than doubles risky rows called
    # Safe. So hold that target and take the best catch rate, then macro F1.
    choices = table[table.elevated_recall_target == HELD_ELEVATED_RECALL]
    print("selected", choices.sort_values(["hard_critical_recall", "macro_f1"], ascending=False).iloc[0].round(4).to_dict())


if __name__ == "__main__":
    main()
