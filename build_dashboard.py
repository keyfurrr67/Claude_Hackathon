"""Builds outputs/dashboard.html from outputs/ and test_data.csv.

Usage: python build_dashboard.py (run pipeline.py first)

relative_velocity_* sit outside the brief's column list. They are read here only to draw
the approach line and never reach a model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import CRITICAL_RECALL, OUT, ROOT, TIERS

MODEL_CRITICAL = "model: Pc >= 1e-4 likely"
ACTIONS = {
    "Critical": "Send to maneuver planning. Analyst decides.",
    "Elevated": "Keep monitoring and re-check on the next message.",
    "Safe": "No action proposed.",
}
SAFE_REVIEW_ACTION = "Analyst to check before clearing."
WINDOW_DAYS = range(2, 7)
TRIGGERS = {
    "rule: miss < 1 km": "Predicted to pass within 1 km",
    "rule: miss 1-5 km": "Predicted to pass within 1 to 5 km",
    MODEL_CRITICAL: "Model expects a collision probability of at least 1 in 10,000",
    "model: Pc >= 1e-6 likely": "Model expects a collision probability of at least 1 in 1 million",
    "clear: miss > 5 km and Pc < 1e-6 likely": "Passes beyond 5 km and the model expects a collision probability below 1 in 1 million",
}
REVIEW_REASONS = {
    "": "",
    "no usable covariance": "the tracking uncertainty for this message is missing",
    "upper-bound Pc above Critical line": "the worst-case estimate is above the Critical line but the model disagrees",
    "near Critical threshold": "the model is close to the Critical cut-off",
    "near Elevated threshold": "the model is close to the Elevated cut-off",
}
FEATURE_LABELS = {
    "time_to_tca": "time to closest approach", "relative_position_r": "vertical separation",
    "relative_position_t": "separation along the orbit", "relative_position_n": "separation across the orbit",
    "max_risk_estimate": "worst-case estimate", "F10": "solar activity", "AP": "geomagnetic activity",
    "c_object_type": "object type", "c_sigma_placeholder": "missing object tracking data",
    "t_sigma_placeholder": "missing satellite tracking data", "log_miss": "miss distance",
    "log_speed": "relative speed", "log_c_sigma_r": "object position uncertainty",
    "log_t_sigma_r": "satellite position uncertainty", "log_radial_sigma": "combined position uncertainty",
    "abs_r": "vertical separation", "abs_t": "separation along the orbit",
    "abs_n": "separation across the orbit", "radial_sigmas_away": "vertical gap versus uncertainty",
    "miss_sigmas_away": "miss distance versus uncertainty", "log_gaussian_decay": "gap-based probability signal",
    "speed_x_sigma": "speed and uncertainty together", "msg_index": "number of earlier messages",
    "delta_log_miss": "change in miss distance", "delta_max_risk": "change in worst-case estimate",
    "delta_log_sigma": "change in object uncertainty", "delta_time": "time since last message",
    "min_miss_so_far": "closest prediction so far", "max_risk_so_far": "highest worst-case estimate so far",
    "first_log_miss_gap": "change in miss distance since first message",
}


def plain_drivers(text):
    drivers = []
    for item in text.split("; "):
        feature, sign = item.rsplit(" ", 1)
        drivers.append(f"{FEATURE_LABELS[feature]} ({'raises' if sign == '+' else 'lowers'} risk)")
    return ", ".join(drivers)


def load():
    predictions = pd.read_csv(OUT / "test_predictions.csv", keep_default_na=False)
    raw = pd.read_csv(ROOT / "test_data.csv", usecols=[
        "relative_position_r", "relative_position_t", "relative_position_n", "relative_speed",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n", "c_object_type",
    ])
    return pd.concat([predictions, raw], axis=1), json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))


def queue_order(e):
    """Critical first, then every event needing review, then the rest."""
    m = e["msgs"][-1]
    group = 0 if m[11] == TIERS.index("Critical") else 1 if e["review"] else 2
    return group, not e["review"], -m[11], -m[10], m[1]


def build_events(d):
    d = d.assign(tier_index=d.predicted_tier.map(TIERS.index), source_line=d.row_id + 2)
    events = []
    for event_id, g in d.groupby("event_id"):
        g = g.sort_values("time_to_tca", ascending=False)
        last = g.iloc[-1]
        msgs = g[[
            "time_to_tca", "miss_distance", "relative_position_r", "relative_position_t",
            "relative_position_n", "log10_pc_pred", "log10_pc_p10", "log10_pc_p90", "p_safe",
            "p_elevated", "p_critical", "tier_index",
        ]].round(4).values.tolist()
        critical_now = last.predicted_tier == "Critical"
        events.append({
            "id": int(event_id), "type": last.c_object_type, "msgs": msgs, "review": REVIEW_REASONS[last.review_reason],
            "model": bool(last.trigger == MODEL_CRITICAL),
            "was_critical": bool(not critical_now and (g.predicted_tier == "Critical").any()),
            "trigger": TRIGGERS[last.trigger], "drivers": plain_drivers(last.top_drivers), "row": int(last.source_line),
            "speed": float(last.relative_speed),
            "action": SAFE_REVIEW_ACTION if last.predicted_tier == "Safe" and last.review_reason else ACTIONS[last.predicted_tier],
            "velocity": [float(last.relative_velocity_r), float(last.relative_velocity_t), float(last.relative_velocity_n)],
        })
    events.sort(key=queue_order)
    return events


def build_windows(d):
    windows = []
    for day in WINDOW_DAYS:
        w = d[(d.time_to_tca >= day) & (d.time_to_tca < day + 1)]
        near, wide = w[w.miss_distance < 1000], w[w.miss_distance < 5000]
        windows.append({
            "label": f"{day}-{day + 1} days",
            "within_1km": int(near.event_id.nunique()),
            "within_5km": int(wide.event_id.nunique()),
            "msgs_1km": len(near),
            "msgs_5km": len(wide),
        })
    return windows


def build_insights(d, latest, metrics):
    critical = latest[latest.predicted_tier == "Critical"]
    by_model = int((critical.trigger == MODEL_CRITICAL).sum())
    review = latest[latest.review_reason != ""]
    no_covariance = int((review.review_reason == "no usable covariance").sum())
    ever = d[d.predicted_tier == "Critical"].event_id.unique()
    cleared = int((~latest[latest.event_id.isin(ever)].predicted_tier.eq("Critical")).sum())
    drift = metrics["tier_drift_2d_to_final_train"]
    return [
        f"{len(critical)} of {len(latest):,} events are Critical on their latest message. {by_model} of those are "
        "Critical only because the model expects a collision probability of at least 1 in 10,000 at 1 km or more. "
        "A distance-only screen would pass them, so check these first.",
        f"{len(review)} events need review. {no_covariance} of them have no usable tracking uncertainty, so their "
        "probabilities stay unreliable until a fresh tracking update arrives.",
        f"{cleared} events were Critical on an earlier message but are not on their latest one. "
        "Do not close an event on a single quieter update.",
        f"In the training data, {drift['ended_critical_not_critical_at_2d']} of {drift['ended_critical']} events "
        "that ended Critical were not Critical at the 2-day mark. A Safe or Elevated tier here is not a promise "
        "about the final message, so keep re-checking as new messages arrive.",
    ]


def over_called(m):
    """True Safe messages raised to Elevated or Critical."""
    return sum(m["confusion_rows_true_cols_pred"][0][1:])


def scorecard(test):
    def row(name, m):
        return {"name": name, "f1": f"{m['macro_f1']:.3f}", "missed": m["missed_critical"],
                "false_critical": m["false_critical"], "over": f"{over_called(m):,}"}
    return [
        row("This model", test["model"]),
        row("Worst-case estimate with distance rules", test["baseline_max_risk_estimate"]),
        row("Miss distance only", test["baseline_miss_distance_only"]),
    ]


def model_only(test):
    """The Critical rows the 1 km rule cannot catch, where the model has to decide alone."""
    md = test["model_dependent"]
    critical, rows = sum(test["model"]["confusion_rows_true_cols_pred"][2]), md["pc_critical_rows_miss_ge_1km"]
    return {
        "critical": critical, "by_rule": critical - rows, "rows": rows,
        "caught": round(md["critical_recall"] * rows), "false": f"{1 - md['critical_precision']:.0%}",
        "auc": f"{md['critical_pr_auc']:.2f}", "auc_baseline": f"{md['critical_pr_auc_max_risk_estimate']:.2f}",
    }


def impact(test):
    m, b = test["model"], test["baseline_max_risk_estimate"]
    return (
        f"Against the worst-case estimate with distance rules, the model raises "
        f"{1 - m['false_critical'] / b['false_critical']:.0%} fewer false Critical alarms "
        f"({m['false_critical']} vs {b['false_critical']}) and {1 - over_called(m) / over_called(b):.0%} fewer "
        f"Safe messages as risky ({over_called(m):,} vs {over_called(b):,}). The cost is "
        f"{m['missed_critical']} missed Critical messages, against {b['missed_critical']}."
    )


def limits(metrics, pc_error):
    test = metrics["test"]
    return [
        "The worst-case estimate missed no Critical messages because it nearly always overstates the risk. "
        f"The model accepts {test['model']['missed_critical']} misses in exchange for far fewer false alarms.",
        "Every miss went to review, but only because of a review rule added after seeing these misses. "
        "It changes no tiers, and its catch rate on test is not independent proof.",
        f"The collision probability shown is rough. It is typically off by a factor of about {pc_error} "
        "on the rows that matter, so rely on the tier and not the exact figure.",
        "Some messages have a placeholder instead of real tracking uncertainty. Their probabilities "
        "look unreliable, so they always go to review.",
        "The test data stops 2 days before closest approach, so behaviour in the final 2 days is untested.",
    ]


def failures():
    f = pd.read_csv(OUT / "failure_cases.csv", keep_default_na=False)
    f = f[f.failure == "missed Critical"]
    return [{
        "id": int(r.event_id), "days": f"{r.time_to_tca:.2f}", "miss": f"{r.miss_distance / 1000:.2f}",
        "risk": float(r.risk), "p": f"{r.p_critical:.1%}", "review": REVIEW_REASONS[r.review_reason],
    } for r in f.itertuples()]


def main():
    d, metrics = load()
    latest = d.sort_values("time_to_tca").groupby("event_id", as_index=False).first()
    windows = build_windows(d)
    counts = latest.predicted_tier.value_counts()
    pc_error = int(round(10 ** metrics["test"]["log10_pc"]["mae_operational_range"], -1))
    data = {
        "scope": f"Test data: {len(d):,} warning messages across {len(latest):,} close approaches, 2 to 7 days out. "
                 "It has no messages from the final 2 days.",
        "tiles": [
            {"label": "Events", "value": len(latest)},
            *[{"tier": t, "value": int(counts.get(t, 0))} for t in reversed(TIERS)],
            {"label": "Need analyst review", "value": int((latest.review_reason != "").sum())},
        ],
        "messages": len(d),
        "attention": int(((latest.predicted_tier == "Critical") | (latest.review_reason != "")).sum()),
        "critical_cut": f"{metrics['thresholds']['critical']:.0%}",
        "critical_recall": f"{CRITICAL_RECALL:.0%}",
        "pc_error": pc_error,
        "windows": windows,
        "insights": build_insights(d, latest, metrics),
        "events": build_events(d),
        "model_only": model_only(metrics["test"]),
        "scorecard": scorecard(metrics["test"]),
        "impact": impact(metrics["test"]),
        "limits": limits(metrics, pc_error),
        "failures": failures(),
    }
    payload = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    template = (ROOT / "dashboard_template.html").read_text(encoding="utf-8")
    three = (ROOT / "vendor" / "three.min.js").read_text(encoding="utf-8")
    (OUT / "dashboard.html").write_text(template.replace("/*THREE*/", three).replace("/*DATA*/", payload), encoding="utf-8")
    print(json.dumps({k: data[k] for k in (
        "scope", "tiles", "attention", "windows", "insights", "model_only", "scorecard", "impact", "limits", "failures",
    )}, indent=1))


if __name__ == "__main__":
    main()
