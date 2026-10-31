"""Scores a CSV of conjunction messages with the saved models, without retraining.

Usage: python predict.py input.csv output.csv

The input needs the brief's columns. A `risk` column is ignored if present.
"""
import sys

import pandas as pd

from pipeline import BRIEF_COLUMNS, FEATURES, add_features, decide, load_models, predict_raw


def main(source, target):
    frame = pd.read_csv(source, usecols=BRIEF_COLUMNS)
    frame.insert(0, "row_id", range(len(frame)))
    frame = add_features(frame)
    models, iso4, iso6, thresholds = load_models()
    decisions = decide(predict_raw(models, frame[FEATURES]), frame, iso4, iso6, thresholds)
    pd.concat([frame[["row_id", "event_id", "time_to_tca", "miss_distance"]], decisions], axis=1).to_csv(target, index=False)
    print(decisions.predicted_tier.value_counts().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
