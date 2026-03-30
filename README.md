# Contact Center Forecasting — NeuralNomads (Team 064)

**Top-7 finish** in a multi-team contact center forecasting competition. The task: predict call volume, average handle time (CCT), and abandon rate at **30-minute interval granularity** across four call queues for every day in August 2025.

This repo has the full pipeline, EDA analysis, and the reasoning behind every design decision — not just the code that worked, but why it works.

---

## The Problem in Plain English

Contact centers hire agents on schedules set days or weeks in advance. Get the forecast wrong and you're either burning budget on idle agents or watching customers hang up unanswered. Both hurt — but understaffing costs more in real-world penalties, which shaped how we built the model.

What makes this hard isn't forecasting the daily total. That's manageable. The hard part is accurately distributing those calls across **48 half-hour windows** per day, separately for **4 queues**, for **31 straight days**. That's 5,952 individual predictions, each with three metrics attached.

---

## Results

| Metric | Score |
|---|---|
| Competition Rank | **#7** |
| Composite Score | 15.42 |
| Volume SMAPE (interval-level) | 34.14% |
| CCT Error | 14.03% |
| Abandon Rate Error | 1.31% |
| Workload Penalty | 0.130 |

---

## The Core Idea

Most teams approach this as a time-series forecasting problem — predict each interval autoregressively using lag features. We tried that first. It scored above 140 on the leaderboard because errors compound: each predicted interval feeds into the next, and you're 48 slots deep before the end of the day.

The insight that changed everything: **we already know the daily totals for August.** That information is given. So the real question isn't "how many calls tomorrow?" — it's "what fraction of tomorrow's calls land in each 30-minute window?"

That reframing turned the problem into learning an **intraday shape** from historical data and multiplying:

```
interval_calls = daily_total × shape(queue, day_of_week, slot) × bias
```

No compounding. No autoregressive error propagation. The daily total is ground truth; we're only estimating the distribution.

---

## What Actually Moved the Leaderboard

We went from rank ~25 to rank 7 in a single change: **fixing null values in the training data**.

The interval dataset had 90–282 null `Call_Volume` and `CCT` rows per queue. Dropping them (which is what you'd do by default) silently biases the shape — the missing rows weren't random, they clustered in specific time slots, making those slots look artificially quiet. The shape would then under-allocate calls to exactly the intervals that needed the most.

The fix was a two-tier median imputation: match on `(queue, day-of-week, month, interval slot)` first, fall back to `(queue, day-of-week, interval slot)` across any month if that's empty. After imputation, zero nulls entered the shape calculation.

**Before imputation:** rank 25, Volume SMAPE 35.4%, Composite 16.00  
**After imputation:** rank 7, Volume SMAPE 34.1%, Composite 15.42

No model change. No new features. Just clean data.

---

## Project Structure

```
.
├── main.py                    # entry point
├── requirements.txt
│
├── src/                       # pipeline modules
│   ├── config.py              # all constants (BIAS, CCT params, excluded dates)
│   ├── data_loader.py         # read Excel sheets, normalise interval format
│   ├── utils.py               # trimmed_mean, smape, cyclic encoding, impute_nulls
│   ├── shape.py               # build intraday shape + optional XGBoost refinement
│   ├── forecast.py            # apply shape to August daily totals
│   └── validate.py            # cross-check aggregated predictions vs actuals
│
├── analysis/                  # EDA — every assumption in config.py has a plot here
│   ├── run_eda.py             # run all analyses in one go
│   ├── holidays.py            # why 6 specific dates are excluded
│   ├── intraday_shape.py      # shape stability, DOW differences, smoothing effect
│   ├── cct_patterns.py        # CCT by interval, alpha sweep, threshold justification
│   ├── bias_scoring.py        # asymmetric penalty simulation → justifies BIAS=1.044
│   ├── data_quality.py        # null distribution, imputation quality check
│   └── trends.py              # volume trends, DOW heatmap, metric correlations
│
└── plots/
    └── eda/                   # 17 saved charts from run_eda.py
```

---

## Setup

You'll need Python 3.10+ and the data file (`data.xlsx`) in the project root. The data isn't included here for obvious reasons, but the pipeline structure is fully reproducible if you have contact center interval data in the same format.

```bash
# clone and create environment
git clone https://github.com/mavito/NeuralNomads.git
cd NeuralNomads
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt

# drop your data.xlsx in the project root, then run
python main.py
```

The output is `submission.csv` — 1,488 rows (31 days × 48 slots) in wide format with columns for each queue's calls, abandoned calls, abandon rate, and CCT.

---

## Running the EDA

```bash
python -m analysis.run_eda
```

This runs all seven analysis modules and saves 17 charts to `plots/eda/`. Each chart maps directly to a parameter in `src/config.py` — the idea being that nothing in the pipeline is a magic number; everything has a visualisation that explains it.

| Chart | Justifies |
|---|---|
| `holiday_volume_impact.png` | `EXCLUDE_DATES` in config.py — Easter is −49%, Memorial Day −50% |
| `shape_stability_across_months.png` | Pooling Apr/May/Jun for shape calculation |
| `shape_by_dow.png` + `dow_volume_heatmap.png` | Separate shape per day-of-week |
| `smoothing_effect.png` | `SMOOTH_KERNEL` and `SMOOTH_ALPHA` values |
| `cct_variance_vs_volume.png` | `CCT_THRESHOLD = 15` |
| `cct_stability_months.png` | `CCT_BLEND_ALPHA = 0.9` |
| `asymmetric_penalty_bias.png` | `BIAS = 1.044` |
| `null_distribution.png` + `null_by_slot_heatmap.png` | The imputation approach in `utils.py` |
| `abandon_rate_stability.png` | `ABD_ALPHA = 1.0` (no blending needed) |

---

## Key Design Decisions

### 1. Ratio-of-sums, not ratio-of-means

The shape is computed as:
```
shape[slot] = Σ(calls in slot across all training days) / Σ(total daily calls)
```
rather than the mean of per-day fractions. High-volume days contribute proportionally more — which is correct, because busier days are more representative of the "true" shape than quiet days following a holiday.

### 2. Circular kernel smoothing

Raw ratio-of-sums shapes have slot-to-slot noise from small sample sizes (only ~60 Tuesdays across April-June). We smooth with a 5-element circular kernel `[0.10, 0.20, 0.40, 0.20, 0.10]`, treating the 48 slots as a ring so midnight wraps correctly. The smoothed shape is blended 50/50 with the raw shape (`SMOOTH_ALPHA=0.5`).

### 3. XGBoost as a refinement layer (not a replacement)

`src/shape.py` has a full XGBoost model (`_xgb_refine_shape`) that learns residuals from the statistical shape using cyclic time features and portfolio dummies. It's there, it was evaluated, and it works. But at `blend_alpha=0.0` (the default), it isn't used — because once the training data was clean (post-imputation), the statistical shape was already within the noise floor of what XGBoost could correct.

To experiment with blending:
```python
# in main.py
shape = build_shape(interval, blend_alpha=0.3)  # 30% XGBoost, 70% statistical
```

### 4. CCT blending, not direct prediction

For intervals with ≥15 predicted calls:
```
CCT_pred = 0.9 × historical_interval_CCT + 0.1 × august_daily_CCT
```
For quieter intervals, the historical average is too noisy (CV > 25%), so we fall back to the flat daily average. The `0.9` weight was chosen by sweeping alpha from 0 to 1 on daily-level SMAPE — see `analysis/cct_patterns.py`.

### 5. Deliberate upward bias

The competition penalises understaffing more than overstaffing. We factored this into the forecast directly: `BIAS=1.044` means every volume prediction is ~4.4% above the expected value. The `analysis/bias_scoring.py` module simulates the asymmetric cost function and shows the optimal bias sits at 4–5% above 1.0.

---

## Data Format

The expected Excel file has one sheet per queue per data type:

| Sheet name | Contents |
|---|---|
| `A - Interval`, `B - Interval`, ... | 30-min interval rows for Apr–Jun 2025: `Day`, `Month`, `Interval`, `Call_Volume`, `CCT`, `Abandoned_Rate` |
| `A - Daily`, `B - Daily`, ... | Daily rows Jan 2024–Aug 2025: `Date`, `Call_Volume`, `CCT`, `Abandon_Rate` |
| `Daily Staffing` | Headcount by date (loaded but not used in final model) |

---

## Adapting This to Your Own Data

The model is domain-agnostic — if you have any system where:
- You know the daily total for the forecast period
- You have historical 30-min or hourly data to learn a shape from
- There's a meaningful intraday distribution worth capturing

...then this approach applies. Change the sheet names in `src/data_loader.py`, update `EXCLUDE_DATES` in `src/config.py` to match your relevant holidays, and point the model at your data.

The one assumption that won't generalise without modification is the specific Excel structure. Everything downstream of `load_data()` is format-agnostic.

---

## Dependencies

| Package | Role |
|---|---|
| `pandas` | Data manipulation throughout |
| `numpy` | Shape arithmetic, smoothing |
| `xgboost` | Optional shape refinement model |
| `matplotlib` + `seaborn` | EDA visualisations |
| `openpyxl` | Reading the Excel data file |
| `scikit-learn` | Not in the final model, available for extension |
| `python-pptx` | Presentation generation |

Full pinned versions in `requirements.txt`.

---

## What We Learned

**Data quality is the highest-leverage thing you can do.** Fixing the nulls — 90 to 282 rows per queue — moved us from rank 25 to rank 7. No model change has ever come close to that impact.

**Domain knowledge matters more than model complexity.** The decision to anchor predictions on known daily totals came from understanding the problem structure, not from trying more algorithms. It reduced the search space dramatically.

**Match your loss function to the actual cost.** We were scoring on an asymmetric penalty where understaffing hurt more than overstaffing. Optimising for symmetric SMAPE would have given us a worse real-world result. The 4.4% upward bias was deliberate, not an artifact.

**Use your real evaluation metric as ground truth.** Several internal changes (CCT ratio scaling, recency weighting) looked good on our cross-check SMAPE. They didn't move the leaderboard. We learned to trust the leaderboard as the definitive signal and discard anything that didn't show up there.

---

*Built for the NeuralNomads team as part of a contact center analytics competition — March 2026.*
