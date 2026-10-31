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


def load():
    predictions = pd.read_csv(OUT / "test_predictions.csv", keep_default_na=False)
    raw = pd.read_csv(ROOT / "test_data.csv", usecols=[
        "relative_position_r", "relative_position_t", "relative_position_n", "relative_speed",
        "relative_velocity_t", "relative_velocity_n", "c_object_type",
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
            "id": int(event_id), "type": last.c_object_type, "msgs": msgs, "review": last.review_reason,
            "trigger": last.trigger, "drivers": last.top_drivers, "row": int(last.source_line),
            "speed": float(last.relative_speed), "action": ACTIONS[last.predicted_tier],
            "velocity": [float(last.relative_velocity_t), float(last.relative_velocity_n)],
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
        f"{by_model} of those are flagged by the Pc model at a miss distance of 1 km or more, "
        "which a distance-only screen would pass.",
        f"The busiest window is {peak['label']} before TCA, with {peak['within_1km']} events inside 1 km.",
        f"{cleared} events were Critical on an earlier message but are not on their latest one, "
        "so tiers move between updates.",
        f"{types.index[0]} objects have the highest Critical share at {types.share.iloc[0]:.1%} "
        f"of {int(types.events.iloc[0]):,} events.",
        f"On train, {drift['ended_critical_not_critical_at_2d']} of {drift['ended_critical']} events "
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
        row("max_risk_estimate + rules", test["baseline_max_risk_estimate"]),
        row("Miss distance only", test["baseline_miss_distance_only"]),
    ]


def limits(metrics):
    test, md = metrics["test"], metrics["test"]["model_dependent"]
    return [
        f"Most Critical rows are settled by the 1 km rule. Where the model must decide alone "
        f"({md['pc_critical_rows_miss_ge_1km']} rows), it catches {md['critical_recall']:.1%} and "
        f"{1 - md['critical_precision']:.0%} of its model-only Critical flags are false.",
        "max_risk_estimate missed no Critical rows, because it is an upper bound. The model trades "
        "a few misses for far fewer false alarms, which is the decision to challenge.",
        f"Pc values are rough: the error is {test['log10_pc']['mae_operational_range']:.1f} orders of "
        "magnitude on rows with log10 Pc of -8 or above. Rely on tiers and probabilities, not on "
        "the point estimate.",
        "Rows with an Earth-radius placeholder sigma have no usable covariance and their risk "
        "labels look like artefacts, so they always go to review.",
        "The review flag for max_risk_estimate above the Critical line was added after seeing the "
        "test failures. Tier scores are unaffected, but its catch rate on test is not a clean holdout.",
    ]


def failures():
    f = pd.read_csv(OUT / "failure_cases.csv", keep_default_na=False)
    f = f[f.failure == "missed Critical"]
    return [{
        "id": int(r.event_id), "days": f"{r.time_to_tca:.2f}", "miss": f"{r.miss_distance / 1000:.2f}",
        "risk": f"{r.risk:.2f}", "p": f"{r.p_critical:.1%}", "review": r.review_reason,
    } for r in f.itertuples()]


def main():
    d, metrics = load()
    latest = d.sort_values("time_to_tca").groupby("event_id", as_index=False).first()
    windows = build_windows(d)
    counts = latest.predicted_tier.value_counts()
    data = {
        "scope": f"Test set: {len(d):,} messages across {len(latest):,} events, 2 to 7 days before TCA. "
                 "It holds nothing inside 2 days of TCA.",
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
    (OUT / "dashboard.html").write_text(template.replace("/*DATA*/", payload))
    print(json.dumps({k: data[k] for k in ("scope", "tiles", "windows", "insights", "scorecard", "limits", "failures")}, indent=1))


if __name__ == "__main__":
    main()
