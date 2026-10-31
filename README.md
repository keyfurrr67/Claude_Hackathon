# Conjunction Risk Desk (Airbus Challenge #1)

Triage for the on-duty conjunction analyst. It reads conjunction messages, classifies each as
Safe, Elevated or Critical with probabilities, and queues events for review. The maneuver
decision stays with the analyst.

## Run

Put `train_data.csv` and `test_data.csv` in this folder, then:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python pipeline.py          # trains, saves models/, writes outputs/ (about 5 min)
.venv/bin/python build_dashboard.py   # writes outputs/dashboard.html
open outputs/dashboard.html
```

To score a new file with the saved models, without retraining:

```
.venv/bin/python predict.py input.csv output.csv
```

No API keys are needed. Nothing calls an LLM.

## Method

| Step | Done by | Why |
|---|---|---|
| Miss-distance tiers (`< 1 km`, `1-5 km`) | Fixed rules | They are definitions in the brief, so a model must not override them |
| `P(Pc >= 1e-4)`, `P(Pc >= 1e-6)` | CatBoost classifiers, isotonic calibration | The part that needs estimating |
| log10 Pc with 80% interval | CatBoost quantile regression | Shows uncertainty |
| Review flag | Rules | Sends uncertain rows to a person |

Tiers follow the brief, with Critical winning over Elevated and Elevated over Safe. Decision
thresholds are set on out-of-fold train predictions to catch 95% of Critical and 90% of Elevated
positives, so a false alarm is accepted over a miss.

Rows are flagged for review when the covariance is an Earth-radius placeholder, when
`max_risk_estimate` is at or above the Critical line but the model disagrees, or when the
probability is near a threshold.

Only the brief's columns reach a model. `relative_velocity_*` is read by the dashboard to draw
the approach line. `risk` is the target and is never an input.

## Results

Scored on `test_data.csv`: 24,484 messages, 2,167 events, 2 to 7 days before TCA.

| Method | Macro F1 | Missed Critical | False Critical | Safe over-called |
|---|---|---|---|---|
| This model | 0.967 (95% CI 0.962-0.972) | 5 | 58 | 687 |
| `max_risk_estimate` + rules | 0.841 | 0 | 173 | 4,534 |
| Miss distance only | 0.952 | 91 | 0 | 0 |

The rows the rule cannot settle are the real test. With miss distance of 1 km or more, there
are 91 Pc-Critical rows:

| Measure | Result |
|---|---|
| Recall | 94.5% |
| Precision | 59.7% |
| PR-AUC, model | 0.781 |
| PR-AUC, `max_risk_estimate` | 0.591 |

Grouped 5-fold cross-validation on train gave macro F1 0.966 and recall 98.0% on the same kind
of rows. The recall gap to test is 6 of 304 misses against 5 of 91, a small sample on test.

## Tuning

`tune.py` steps training settings by 0.05, selecting on train out-of-fold results only.

- Learning rate, `rsm` and `subsample`: no step beat the starting settings by more than
  2 standard errors, so they are unchanged.
- Recall targets: raising the Critical target from 0.90 to 0.95 cut misses on train from 18 to 6
  for 29 more false Critical rows. The highest macro F1 came from an Elevated target of 0.75,
  which was rejected because it raises risky rows called Safe from 380 to 946.
- On test, the same change left misses at 5 and added 11 false Critical rows. It was kept,
  because reverting it on that result would mean selecting on the test set.

## Known limits

- The model missed 5 Critical rows that `max_risk_estimate` caught. All 5 are within 0.09 of the
  line, and event 1930 got a probability near 0%. The baseline's clean record comes from being
  an upper bound, and it raises 3 times the false Critical flags.
- Precision is 59.7% on model-only Critical flags, so about 2 in 5 are false alarms.
- Point Pc is rough, with an error of 1.6 orders of magnitude on rows with log10 Pc of -8 or
  above. Use tiers and probabilities instead.
- A tier describes the current message and is not a forecast. On train, 99 of 551 events that
  ended Critical were not Critical at the 2-day mark.
- The test file stops 2 days before TCA, so behaviour inside that window is not tested here.
- Test has more high-risk rows than train (9.1% vs 7.2% Critical). Scores held up, but other
  mixes may differ.
- The data has no shared clock, so 24-hour windows are days before each event's own TCA.
- Trajectories are straight-line relative motion, which only holds near TCA.
- The `TBA` object type has 29 rows, too few to judge.

## Evaluation notes

- Cross-validation splits by event, so no event is on both sides.
- Event IDs restart between files and are unrelated. No test row matches a train row.
- Test `risk` is removed before predicting and restored only after predictions are saved.
- Tier thresholds were never tuned on test.
- The `max_risk_estimate` review flag was added after seeing the test failures. It flags all 5
  misses on test and all 6 on out-of-fold train, where it was checked. It changes no tier, but
  its test catch rate is not a clean holdout.

## Files

- `pipeline.py`: features, training, calibration, evaluation
- `predict.py`: scoring with saved models
- `tune.py`: 0.05-step settings search, train only
- `build_dashboard.py`, `dashboard_template.html`: dashboard
- `models/`: trained CatBoost models, calibrators and thresholds
- `outputs/`: predictions, metrics, failure cases, dashboard
