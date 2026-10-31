"""Builds outputs/dashboard.html from outputs/ and test_data.csv.

Usage: python build_dashboard.py (run pipeline.py first)

relative_velocity_* sit outside the brief's column list. They are read here only to draw
the approach line and never reach a model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import OUT, ROOT, TIERS

ACTIONS = {
    "Critical": "Send to maneuver planning. Analyst decides.",
    "Elevated": "Keep monitoring and re-check on the next message.",
    "Safe": "No action proposed.",
}
WINDOW_DAYS = range(2, 7)
TRIGGERS = {
    "rule: miss < 1 km": "Predicted to pass within 1 km",
    "rule: miss 1-5 km": "Predicted to pass within 1 to 5 km",
    "model: Pc >= 1e-4 likely": "Model expects a collision probability of at least 1 in 10,000",
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
    return pd.concat([predictions, raw], axis=1), json.loads((OUT / "metrics.json").read_text())


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
        events.append({
            "id": int(event_id), "type": last.c_object_type, "msgs": msgs, "review": REVIEW_REASONS[last.review_reason],
            "trigger": TRIGGERS[last.trigger], "drivers": plain_drivers(last.top_drivers), "row": int(last.source_line),
            "speed": float(last.relative_speed), "action": ACTIONS[last.predicted_tier],
            "velocity": [float(last.relative_velocity_r), float(last.relative_velocity_t), float(last.relative_velocity_n)],
        })
    events.sort(key=lambda e: (-e["msgs"][-1][11], not e["review"], -e["msgs"][-1][10], e["msgs"][-1][1]))
    return events


def build_windows(d):
    windows = []
    for day in WINDOW_DAYS:
        w = d[(d.time_to_tca >= day) & (d.time_to_tca < day + 1)]
        windows.append({
            "label": f"{day}-{day + 1} days",
            "within_1km": int(w[w.miss_distance < 1000].event_id.nunique()),
            "within_5km": int(w[w.miss_distance < 5000].event_id.nunique()),
        })
    return windows


def build_insights(d, latest, windows, metrics):
    critical = latest[latest.predicted_tier == "Critical"]
    by_model = int((critical.trigger == "model: Pc >= 1e-4 likely").sum())
    ever = d[d.predicted_tier == "Critical"].event_id.unique()
    cleared = int((~latest[latest.event_id.isin(ever)].predicted_tier.eq("Critical")).sum())
    peak = max(windows, key=lambda w: w["within_1km"])
    drift = metrics["tier_drift_2d_to_final_train"]
    types = latest.groupby("c_object_type").predicted_tier.agg(
        share=lambda s: (s == "Critical").mean(), events="size"
    ).query("events >= 30").sort_values("share", ascending=False)
    return [
        f"{len(critical)} of {len(latest):,} events are Critical on their latest message. "
        f"{by_model} of those are flagged by the model alone, at a miss distance of 1 km or more, "
        "which a distance-only screen would pass.",
        f"The busiest window is {peak['label']} before closest approach, with {peak['within_1km']} events inside 1 km.",
        f"{cleared} events were Critical on an earlier message but are not on their latest one, "
        "so tiers move between updates.",
        f"{types.index[0].capitalize()} has the highest Critical share at {types.share.iloc[0]:.1%} "
        f"of {int(types.events.iloc[0]):,} events.",
        f"In the training data, {drift['ended_critical_not_critical_at_2d']} of {drift['ended_critical']} events "
        "that ended Critical were not Critical at the 2-day mark. A Safe or Elevated tier here is "
        "not a promise about the final message.",
    ]


def scorecard(test):
    def row(name, m):
        over = m["confusion_rows_true_cols_pred"][0][1] + m["confusion_rows_true_cols_pred"][0][2]
        return {"name": name, "f1": f"{m['macro_f1']:.3f}", "missed": m["missed_critical"],
                "false_critical": m["false_critical"], "over": f"{over:,}"}
    return [
        row("This model", test["model"]),
        row("Worst-case estimate with distance rules", test["baseline_max_risk_estimate"]),
        row("Miss distance only", test["baseline_miss_distance_only"]),
    ]


def limits(metrics):
    test, md = metrics["test"], metrics["test"]["model_dependent"]
    return [
        f"Most Critical rows are settled by the 1 km rule. Where the model must decide alone "
        f"({md['pc_critical_rows_miss_ge_1km']} rows), it catches {md['critical_recall']:.1%} and "
        f"{1 - md['critical_precision']:.0%} of its model-only Critical flags are false.",
        "The worst-case estimate missed no Critical rows, because it nearly always overstates the risk. "
        "The model accepts a few misses for far fewer false alarms, and every miss on test went to review.",
        f"The collision probability shown is rough. It is typically off by a factor of about "
        f"{round(10 ** test['log10_pc']['mae_operational_range'], -1):.0f} on the rows that matter, so rely on the tier and not the exact figure.",
        "Some messages have a placeholder instead of real tracking uncertainty. Their probabilities "
        "look unreliable, so they always go to review.",
        "The review rule for a high worst-case estimate was added after seeing the test failures. "
        "It changes no tiers, but its perfect catch rate on test is not independent proof.",
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
    data = {
        "scope": f"Test data: {len(d):,} warning messages across {len(latest):,} close approaches, 2 to 7 days out. "
                 "It has no messages from the final 2 days.",
        "tiles": [
            {"label": "Events", "value": len(latest)},
            *[{"tier": t, "value": int(counts.get(t, 0))} for t in reversed(TIERS)],
            {"label": "Need analyst review", "value": int((latest.review_reason != "").sum())},
        ],
        "windows": windows,
        "insights": build_insights(d, latest, windows, metrics),
        "events": build_events(d),
        "scorecard": scorecard(metrics["test"]),
        "limits": limits(metrics),
        "failures": failures(),
    }
    payload = json.dumps(data, separators=(",", ":"), default=float).replace("</", "<\\/")
    template = (ROOT / "dashboard_template.html").read_text()
    three = (ROOT / "vendor" / "three.min.js").read_text()
    (OUT / "dashboard.html").write_text(template.replace("/*THREE*/", three).replace("/*DATA*/", payload))
    print(json.dumps({k: data[k] for k in ("scope", "tiles", "windows", "insights", "scorecard", "limits", "failures")}, indent=1))


if __name__ == "__main__":
    main()
