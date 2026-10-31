"""Collision risk evaluator for Airbus Challenge #1.

Usage: python pipeline.py

Reads train_data.csv and test_data.csv, writes predictions, metrics and failure
cases to outputs/. Miss-distance rules are applied deterministically; CatBoost
only estimates the Pc side of each tier boundary.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
SEED = 42
N_FOLDS = 5
PARAMS = {"learning_rate": 0.06}  # overrides chosen by tune.py

BRIEF_COLUMNS = [
    "event_id", "time_to_tca", "miss_distance", "relative_speed", "relative_position_r",
    "relative_position_t", "relative_position_n", "c_object_type", "c_sigma_r", "t_sigma_r",
    "max_risk_estimate", "F10", "AP",
]
TARGET = "risk"

CRITICAL_LOG_PC = -4.0
ELEVATED_LOG_PC = -6.0
CRITICAL_MISS_M = 1000.0
SAFE_MISS_M = 5000.0
SIGMA_PLACEHOLDER_M = 6.3e7  # Earth radius in metres, used in the data when no covariance exists
CRITICAL_RECALL = 0.95  # chosen by tune.py
ELEVATED_RECALL = 0.90
REVIEW_BAND = 0.5  # flag rows whose probability is within this factor of a decision threshold
TIERS = ["Safe", "Elevated", "Critical"]


def load(name):
    df = pd.read_csv(ROOT / name, usecols=BRIEF_COLUMNS + [TARGET])
    df.insert(0, "row_id", np.arange(len(df)))
    return df


def tier_from_pc(log_pc, miss_m):
    """Brief's tier table. Critical wins over Elevated, Elevated over Safe."""
    log_pc, miss_m = np.asarray(log_pc), np.asarray(miss_m)
    critical = (log_pc >= CRITICAL_LOG_PC) | (miss_m < CRITICAL_MISS_M)
    safe = (log_pc < ELEVATED_LOG_PC) & (miss_m > SAFE_MISS_M)
    return np.where(critical, "Critical", np.where(safe, "Safe", "Elevated"))


def add_features(df):
    """Per-message features. History features only look at earlier messages of the same event."""
    d = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False]).copy()
    g = d.groupby("event_id", sort=False)

    d["c_sigma_placeholder"] = (d.c_sigma_r > SIGMA_PLACEHOLDER_M).astype(int)
    d["t_sigma_placeholder"] = (d.t_sigma_r > SIGMA_PLACEHOLDER_M).astype(int)
    c_sigma = d.c_sigma_r.where(d.c_sigma_r <= SIGMA_PLACEHOLDER_M)
    t_sigma = d.t_sigma_r.where(d.t_sigma_r <= SIGMA_PLACEHOLDER_M)
    radial_sigma = np.sqrt(c_sigma**2 + t_sigma**2)

    d["log_miss"] = np.log10(d.miss_distance.clip(lower=1))
    d["log_speed"] = np.log10(d.relative_speed.clip(lower=1))
    d["log_c_sigma_r"] = np.log10(c_sigma)
    d["log_t_sigma_r"] = np.log10(t_sigma)
    d["log_radial_sigma"] = np.log10(radial_sigma)
    d["abs_r"] = d.relative_position_r.abs()
    d["abs_t"] = d.relative_position_t.abs()
    d["abs_n"] = d.relative_position_n.abs()
    d["radial_sigmas_away"] = d.abs_r / radial_sigma
    d["miss_sigmas_away"] = d.miss_distance / radial_sigma
    # Shape of a Gaussian encounter Pc. Only radial sigmas are in the brief's columns, so this
    # is a model input and not a Pc estimate in its own right.
    d["log_gaussian_decay"] = -0.5 * d.radial_sigmas_away**2 / np.log(10) - 2 * d.log_radial_sigma
    d["speed_x_sigma"] = d.log_speed + d.log_radial_sigma

    d["msg_index"] = g.cumcount()
    d["delta_log_miss"] = g.log_miss.diff()
    d["delta_max_risk"] = g.max_risk_estimate.diff()
    d["delta_log_sigma"] = g.log_c_sigma_r.diff()
    d["delta_time"] = -g.time_to_tca.diff()
    d["min_miss_so_far"] = g.log_miss.cummin()
    d["max_risk_so_far"] = g.max_risk_estimate.cummax()
    d["first_log_miss_gap"] = d.log_miss - g.log_miss.transform("first")

    return d.sort_values("row_id")


FEATURES = [
    "time_to_tca", "relative_position_r", "relative_position_t", "relative_position_n",
    "max_risk_estimate", "F10", "AP", "c_object_type", "c_sigma_placeholder", "t_sigma_placeholder",
    "log_miss", "log_speed", "log_c_sigma_r", "log_t_sigma_r", "log_radial_sigma", "abs_r", "abs_t",
    "abs_n", "radial_sigmas_away", "miss_sigmas_away", "log_gaussian_decay", "speed_x_sigma",
    "msg_index", "delta_log_miss", "delta_max_risk", "delta_log_sigma", "delta_time",
    "min_miss_so_far", "max_risk_so_far", "first_log_miss_gap",
]


def pool(X, y=None):
    return Pool(X, y, cat_features=["c_object_type"])


def fit_model(X, y, loss):
    cls = CatBoostRegressor if loss.startswith("Quantile") else CatBoostClassifier
    return cls(
        loss_function=loss, iterations=350, depth=6,
        random_seed=SEED, verbose=False, allow_writing_files=False, **PARAMS,
    ).fit(pool(X, y))


def fit_all(X, risk):
    return {
        "p4": fit_model(X, (risk >= CRITICAL_LOG_PC).astype(int), "Logloss"),
        "p6": fit_model(X, (risk >= ELEVATED_LOG_PC).astype(int), "Logloss"),
        "q10": fit_model(X, risk, "Quantile:alpha=0.1"),
        "q50": fit_model(X, risk, "Quantile:alpha=0.5"),
        "q90": fit_model(X, risk, "Quantile:alpha=0.9"),
    }


def predict_raw(models, X):
    data = pool(X)
    return pd.DataFrame(
        {
            "p4": models["p4"].predict_proba(data)[:, 1],
            "p6": models["p6"].predict_proba(data)[:, 1],
            "q10": models["q10"].predict(data),
            "q50": models["q50"].predict(data),
            "q90": models["q90"].predict(data),
        },
        index=X.index,
    )


def out_of_fold(train):
    """Every train row is predicted by models that never saw any message of its event."""
    oof = []
    for fold, (fit_idx, val_idx) in enumerate(
        GroupKFold(N_FOLDS).split(train, groups=train.event_id), 1
    ):
        models = fit_all(train.iloc[fit_idx][FEATURES], train.iloc[fit_idx][TARGET])
        oof.append(predict_raw(models, train.iloc[val_idx][FEATURES]))
        print(f"fold {fold}/{N_FOLDS} done", flush=True)
    return pd.concat(oof).sort_index()


def threshold_for_recall(p, y, recall):
    """Highest threshold that still catches `recall` of the positives."""
    return float(np.sort(p[y == 1])[int(np.floor((1 - recall) * (y == 1).sum()))])


def calibrate(oof, train, critical_recall=CRITICAL_RECALL, elevated_recall=ELEVATED_RECALL):
    """Isotonic calibration and decision thresholds, fitted on out-of-fold rows only."""
    in_scope = (train.time_to_tca >= 2).to_numpy()  # test only holds messages 2+ days out
    y4 = (train[TARGET] >= CRITICAL_LOG_PC).to_numpy().astype(int)
    y6 = (train[TARGET] >= ELEVATED_LOG_PC).to_numpy().astype(int)
    iso4 = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.p4, y4)
    iso6 = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof.p6, y6)
    p4, p6 = iso4.predict(oof.p4), iso6.predict(oof.p6)
    miss = train.miss_distance.to_numpy()
    mask4 = in_scope & (miss >= CRITICAL_MISS_M)
    mask6 = in_scope & (miss > SAFE_MISS_M)
    thresholds = {
        "critical": threshold_for_recall(p4[mask4], y4[mask4], critical_recall),
        "elevated": threshold_for_recall(p6[mask6], y6[mask6], elevated_recall),
    }
    return iso4, iso6, thresholds


def decide(raw, frame, iso4, iso6, thresholds):
    """Combine calibrated Pc probabilities with the deterministic miss-distance rules."""
    p4 = iso4.predict(raw.p4)
    p6 = np.maximum(iso6.predict(raw.p6), p4)
    miss = frame.miss_distance.to_numpy()
    close, mid = miss < CRITICAL_MISS_M, (miss >= CRITICAL_MISS_M) & (miss <= SAFE_MISS_M)

    p_critical = np.where(close, 1.0, p4)
    p_safe = np.where(close | mid, 0.0, 1 - p6)
    p_elevated = 1 - p_critical - p_safe

    critical = close | (p4 >= thresholds["critical"])
    elevated = mid | (p6 >= thresholds["elevated"])
    tier = np.where(critical, "Critical", np.where(elevated, "Elevated", "Safe"))
    trigger = np.select(
        [close, critical, mid, elevated],
        ["rule: miss < 1 km", "model: Pc >= 1e-4 likely", "rule: miss 1-5 km", "model: Pc >= 1e-6 likely"],
        default="clear: miss > 5 km and Pc < 1e-6 likely",
    )

    placeholder = (frame.c_sigma_placeholder + frame.t_sigma_placeholder).to_numpy() > 0
    near4 = (~close) & (p4 >= thresholds["critical"] * REVIEW_BAND) & (p4 < thresholds["critical"] / REVIEW_BAND)
    near6 = (miss > SAFE_MISS_M) & (p6 >= thresholds["elevated"] * REVIEW_BAND) & (p6 < thresholds["elevated"] / REVIEW_BAND)
    # max_risk_estimate is an upper bound on Pc, so disagreement with it is not safe to clear silently.
    bound_disagrees = (~critical) & (frame.max_risk_estimate.to_numpy() >= CRITICAL_LOG_PC)
    reason = np.select(
        [placeholder, bound_disagrees, near4, near6],
        ["no usable covariance", "upper-bound Pc above Critical line", "near Critical threshold", "near Elevated threshold"],
        default="",
    )
    # Quantile models are fitted separately, so they can cross.
    q10, q50, q90 = np.sort(raw[["q10", "q50", "q90"]].to_numpy(), axis=1).T
    return pd.DataFrame(
        {
            "log10_pc_pred": q50, "log10_pc_p10": q10,
            "log10_pc_p90": q90, "p_safe": p_safe, "p_elevated": p_elevated,
            "p_critical": p_critical, "predicted_tier": tier, "trigger": trigger,
            "needs_review": reason != "", "review_reason": reason,
        },
        index=frame.index,
    )


def score(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=TIERS)
    actual, flagged = (y_true == "Critical"), (y_pred == "Critical")
    tp = int((actual & flagged).sum())
    return {
        "rows": int(len(y_true)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=TIERS, average="macro")),
        "critical_recall": tp / max(int(actual.sum()), 1),
        "critical_precision": tp / max(int(flagged.sum()), 1),
        "missed_critical": int((actual & ~flagged).sum()),
        "false_critical": int((~actual & flagged).sum()),
        "safe_called_safe_but_not": int(((y_pred == "Safe") & (y_true != "Safe")).sum()),
        "confusion_rows_true_cols_pred": cm.tolist(),
    }


def bootstrap_ci(truth, pred, events, stat, n=500):
    """Resamples whole events, since messages from one event are not independent."""
    rng = np.random.default_rng(SEED)
    index = pd.Series(np.arange(len(events))).groupby(np.asarray(events)).indices
    groups = list(index.values())
    values = []
    for _ in range(n):
        picks = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        values.append(stat(truth[picks], pred[picks]))
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def evaluate(frame, decisions, name):
    truth = tier_from_pc(frame[TARGET], frame.miss_distance)
    pred = decisions.predicted_tier.to_numpy()
    miss, risk = frame.miss_distance.to_numpy(), frame[TARGET].to_numpy()
    y4, y6 = risk >= CRITICAL_LOG_PC, risk >= ELEVATED_LOG_PC
    p4, p6 = decisions.p_critical.to_numpy(), 1 - decisions.p_safe.to_numpy()
    hard4, hard6 = miss >= CRITICAL_MISS_M, miss > SAFE_MISS_M

    result = {
        "model": score(truth, pred),
        "baseline_miss_distance_only": score(truth, tier_from_pc(np.full(len(frame), -30.0), miss)),
        "baseline_max_risk_estimate": score(truth, tier_from_pc(frame.max_risk_estimate, miss)),
        # Rows where the miss-distance rule cannot settle the answer.
        "model_dependent": {
            "pc_critical_rows_miss_ge_1km": int(y4[hard4].sum()),
            "critical_recall": float((pred[hard4 & y4] == "Critical").mean()),
            "critical_precision": float(y4[hard4 & (pred == "Critical")].mean()),
            "critical_pr_auc": float(average_precision_score(y4[hard4], p4[hard4])),
            "critical_pr_auc_max_risk_estimate": float(
                average_precision_score(y4[hard4], frame.max_risk_estimate.to_numpy()[hard4])
            ),
            "critical_base_rate": float(y4[hard4].mean()),
            "critical_brier": float(brier_score_loss(y4[hard4], p4[hard4])),
            "elevated_pr_auc": float(average_precision_score(y6[hard6], p6[hard6])),
            "elevated_pr_auc_max_risk_estimate": float(
                average_precision_score(y6[hard6], frame.max_risk_estimate.to_numpy()[hard6])
            ),
            "elevated_base_rate": float(y6[hard6].mean()),
        },
        "log10_pc": {
            "mae_excluding_floor": float(np.abs(risk - decisions.log10_pc_pred)[risk > -30].mean()),
            "mae_operational_range": float(np.abs(risk - decisions.log10_pc_pred)[risk >= -8].mean()),
            "p10_p90_coverage": float(
                ((risk >= decisions.log10_pc_p10) & (risk <= decisions.log10_pc_p90)).mean()
            ),
        },
        "review": {
            "share_flagged": float(decisions.needs_review.mean()),
            "critical_recall_unflagged_rows": float(
                (pred[(truth == "Critical") & ~decisions.needs_review] == "Critical").mean()
            ),
        },
    }
    print(f"\n== {name} ==\n{json.dumps(result, indent=1)}", flush=True)
    return result


def slices(frame, decisions):
    truth = tier_from_pc(frame[TARGET], frame.miss_distance)
    pred = decisions.predicted_tier.to_numpy()
    groups = {
        "object_type": frame.c_object_type.astype(str).to_numpy(),
        "days_to_tca": np.floor(frame.time_to_tca).astype(int).astype(str).to_numpy(),
        "covariance": np.where(frame.c_sigma_placeholder + frame.t_sigma_placeholder > 0, "placeholder", "usable"),
    }
    rows = []
    for name, values in groups.items():
        for value in sorted(set(values)):
            m = values == value
            rows.append({"slice": name, "value": value, **{
                k: v for k, v in score(truth[m], pred[m]).items() if k != "confusion_rows_true_cols_pred"
            }})
    return rows


def failure_cases(frame, decisions):
    truth = tier_from_pc(frame[TARGET], frame.miss_distance)
    d = pd.concat([frame[BRIEF_COLUMNS[:7] + ["c_object_type", "max_risk_estimate", TARGET]], decisions], axis=1)
    d["true_tier"] = truth
    missed = d[(d.true_tier == "Critical") & (d.predicted_tier != "Critical")].assign(failure="missed Critical")
    false = d[(d.true_tier != "Critical") & (d.predicted_tier == "Critical")].assign(failure="false Critical")
    return pd.concat([missed.sort_values(TARGET, ascending=False), false.sort_values(TARGET).head(50)])


def save_models(models, iso4, iso6, thresholds):
    MODELS.mkdir(exist_ok=True)
    for name, model in models.items():
        model.save_model(str(MODELS / f"{name}.cbm"))
    joblib.dump({"iso4": iso4, "iso6": iso6, "thresholds": thresholds}, MODELS / "calibration.joblib")


def load_models():
    models = {
        name: (CatBoostClassifier() if name in ("p4", "p6") else CatBoostRegressor()).load_model(
            str(MODELS / f"{name}.cbm")
        )
        for name in ("p4", "p6", "q10", "q50", "q90")
    }
    calibration = joblib.load(MODELS / "calibration.joblib")
    return models, calibration["iso4"], calibration["iso6"], calibration["thresholds"]


def main():
    OUT.mkdir(exist_ok=True)
    train = add_features(load("train_data.csv"))
    test = add_features(load("test_data.csv"))
    test_truth = test[TARGET]
    test = test.drop(columns=TARGET)  # unavailable until predictions are on disk

    oof = out_of_fold(train)
    iso4, iso6, thresholds = calibrate(oof, train)
    print("thresholds", thresholds, flush=True)

    models = fit_all(train[FEATURES], train[TARGET])
    save_models(models, iso4, iso6, thresholds)
    decisions = decide(predict_raw(models, test[FEATURES]), test, iso4, iso6, thresholds)
    contrib = models["p4"].get_feature_importance(pool(test[FEATURES]), type="ShapValues")[:, :-1]
    top = np.argsort(-np.abs(contrib), axis=1)[:, :3]
    decisions["top_drivers"] = [
        "; ".join(f"{FEATURES[j]} {'+' if contrib[i, j] > 0 else '-'}" for j in top[i])
        for i in range(len(top))
    ]
    predictions = pd.concat([test[["row_id", "event_id", "time_to_tca", "miss_distance"]], decisions], axis=1)
    predictions.to_csv(OUT / "test_predictions.csv", index=False)

    test[TARGET] = test_truth
    oof_mask = (train.time_to_tca >= 2).to_numpy()
    oof_frame = train[oof_mask]
    oof_decisions = decide(oof[oof_mask], oof_frame, iso4, iso6, thresholds)

    metrics = {
        "thresholds": thresholds,
        "train_cv_events_grouped_time_ge_2d": evaluate(oof_frame, oof_decisions, "train out-of-fold (t >= 2d)"),
        "test": evaluate(test, decisions, "test"),
    }
    truth = tier_from_pc(test[TARGET], test.miss_distance)
    pred = decisions.predicted_tier.to_numpy()
    metrics["test_ci95"] = {
        "macro_f1": bootstrap_ci(truth, pred, test.event_id, lambda t, p: f1_score(t, p, labels=TIERS, average="macro")),
        "critical_recall": bootstrap_ci(truth, pred, test.event_id, lambda t, p: (p[t == "Critical"] == "Critical").mean()),
    }
    metrics["test_slices"] = slices(test, decisions)

    final = train.sort_values("time_to_tca").groupby("event_id").first()
    held = train[train.time_to_tca >= 2].sort_values("time_to_tca").groupby("event_id").first()
    both = held.join(final, rsuffix="_final", how="inner")
    both = both[both.time_to_tca_final < 1]
    at_2d = tier_from_pc(both[TARGET], both.miss_distance)
    at_end = tier_from_pc(both[f"{TARGET}_final"], both.miss_distance_final)
    metrics["tier_drift_2d_to_final_train"] = {
        "events": int(len(both)),
        "ended_critical": int((at_end == "Critical").sum()),
        "ended_critical_not_critical_at_2d": int(((at_end == "Critical") & (at_2d != "Critical")).sum()),
    }

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    failure_cases(test, decisions).to_csv(OUT / "failure_cases.csv", index=False)
    print(json.dumps(metrics["test_ci95"]), json.dumps(metrics["tier_drift_2d_to_final_train"]))


if __name__ == "__main__":
    main()
