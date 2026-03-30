import numpy as np
import pandas as pd
import xgboost as xgb
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import EXCLUDE_DATES, SMOOTH_KERNEL, SMOOTH_ALPHA
from src.utils import trimmed_mean, impute_nulls, encode_cyclic


def build_shape(interval, blend_alpha=0.0):
    """Build an intraday shape table from Apr-Jun interval data.

    The 'shape' tells us what fraction of a day's total call volume
    lands in each 30-minute slot, separately per queue and day-of-week.
    We compute it as a ratio-of-sums (high-volume days dominate naturally)
    then apply a small circular smoothing pass to reduce slot-to-slot noise.

    CCT and abandon rate shapes are trimmed means across all training days
    for that same (queue, DOW, slot) combination.

    blend_alpha controls how much of the XGBoost-refined shape to mix in.
    0.0 = pure statistical (what scored rank 7), 1.0 = pure XGBoost.
    Tuning showed 0.0 was optimal — XGBoost adds no value over the clean
    statistical shape once nulls are properly imputed.
    """
    iv = interval.copy()
    iv['date_str']    = iv['Date'].dt.strftime('%Y-%m-%d')
    iv['day_of_week'] = iv['Date'].dt.day_name()

    # only use Apr-Jun 2025, skip the known holidays
    mask = (
        (iv['Date'].dt.year  == 2025) &
        (iv['Date'].dt.month.isin([4, 5, 6])) &
        (~iv['date_str'].isin(EXCLUDE_DATES))
    )
    iv = iv[mask].copy()

    # fill nulls before computing any aggregates
    # without this, queues with many null rows get a shape based on partial data
    iv = impute_nulls(iv, ['Call_Volume', 'CCT', 'Abandoned_Rate'])
    remaining = iv['Call_Volume'].isna().sum() + iv['CCT'].isna().sum()
    print(f"  Shape training: {len(iv)} rows, {remaining} null values remaining after imputation")

    # --- call volume shape ---
    # ratio-of-sums: busy days get proportionally more weight, which is what we want
    slot_totals = (iv.groupby(['Portfolio', 'day_of_week', 'Interval'])['Call_Volume']
                     .sum().reset_index().rename(columns={'Call_Volume': 'slot_cv'}))
    day_totals  = (iv.groupby(['Portfolio', 'day_of_week'])['Call_Volume']
                     .sum().reset_index().rename(columns={'Call_Volume': 'day_cv'}))

    shape = slot_totals.merge(day_totals, on=['Portfolio', 'day_of_week'])
    shape['shape_cv'] = shape['slot_cv'] / shape['day_cv']

    # sanity check — each (queue, DOW) should sum to exactly 1.0
    sums = shape.groupby(['Portfolio', 'day_of_week'])['shape_cv'].sum().round(4)
    assert (sums == 1.0).all(), f"shape_cv doesn't sum to 1 for: {sums[sums != 1.0].index.tolist()}"

    # circular smoothing: treat intervals as a ring (23:30 wraps to 0:00)
    shape = _smooth_shape(shape)

    # optionally layer in an XGBoost-refined shape on top of the statistical one
    if blend_alpha > 0:
        shape = _xgb_refine_shape(iv, shape, blend_alpha)
    else:
        shape['final_shape_cv'] = shape['shape_cv']
        print("  Shape blend: 0% XGBoost + 100% statistical")

    # --- CCT shape ---
    cct_shape = (iv.groupby(['Portfolio', 'day_of_week', 'Interval'])['CCT']
                   .agg(trimmed_mean).reset_index()
                   .rename(columns={'CCT': 'interval_cct'}))

    # --- abandon rate shape ---
    abd_shape = (iv.groupby(['Portfolio', 'day_of_week', 'Interval'])['Abandoned_Rate']
                   .agg(trimmed_mean).reset_index()
                   .rename(columns={'Abandoned_Rate': 'interval_abd'}))

    shape = shape.merge(cct_shape, on=['Portfolio', 'day_of_week', 'Interval'], how='left')
    shape = shape.merge(abd_shape, on=['Portfolio', 'day_of_week', 'Interval'], how='left')
    shape.drop(columns=['slot_cv', 'day_cv'], inplace=True)

    print(f"  Built shape: {len(shape)} cells across {len(shape) // 48} (queue, DOW) combos")
    return shape


def _smooth_shape(shape):
    """Apply circular kernel smoothing to the call volume shape.

    The kernel is centred (+-2 slots from current) so adjacent half-hours
    contribute but the current slot still contributes the most (weight 0.4).
    Blends 50/50 with the original so we don't lose too much structure.
    """
    half = len(SMOOTH_KERNEL) // 2
    smoothed_parts = []

    for (q, dow), grp in shape.groupby(['Portfolio', 'day_of_week']):
        grp  = grp.sort_values('Interval').copy()
        vals = grp['shape_cv'].values.astype(float)
        n    = len(vals)

        # wrap around at midnight so we don't create an artificial boundary
        smoothed = np.array([
            sum(SMOOTH_KERNEL[k] * vals[(i + k - half) % n]
                for k in range(len(SMOOTH_KERNEL)))
            for i in range(n)
        ])
        grp['shape_cv'] = (1 - SMOOTH_ALPHA) * vals + SMOOTH_ALPHA * smoothed
        smoothed_parts.append(grp)

    return pd.concat(smoothed_parts).reset_index(drop=True)


def _xgb_refine_shape(iv, shape, blend_alpha):
    """Train XGBoost to learn shape residuals and blend with the statistical shape.

    The model takes (interval_num, weekday_num, portfolio) as features and
    predicts the fraction of daily volume per slot. Trained on the same Apr-Jun
    imputed data used for the statistical shape.

    Key finding from tuning: blend_alpha=0.0 (pure statistical) was optimal.
    Once the null imputation was in place, the statistical shape was already
    accurate enough that XGBoost couldn't improve on it — its nonlinear
    corrections introduced as much variance as they removed. This function
    exists so the full methodology is in code, not just in the presentation.
    """
    # compute per-interval fraction of daily volume for each training date
    daily_totals = (iv.groupby(['Portfolio', 'Date'])['Call_Volume']
                      .sum().rename('daily_cv').reset_index())
    df = iv.merge(daily_totals, on=['Portfolio', 'Date'], how='left')
    df['actual_shape'] = np.where(df['daily_cv'] > 0, df['Call_Volume'] / df['daily_cv'], 0)

    # cyclic features: hour-of-day and day-of-week encoded as sin/cos
    df['interval_num'] = (pd.to_datetime(df['Interval'], format='%H:%M').dt.hour * 2 +
                          pd.to_datetime(df['Interval'], format='%H:%M').dt.minute // 30)
    df['weekday_num']  = df['Date'].dt.weekday
    df['is_weekend']   = (df['weekday_num'] >= 5).astype(int)

    df = encode_cyclic(df, 'interval_num', 48)
    df = encode_cyclic(df, 'weekday_num', 7)

    port_dummies = pd.get_dummies(df['Portfolio'], prefix='P').astype(int)
    df = pd.concat([df, port_dummies], axis=1)

    feature_cols = [
        'interval_num_sin', 'interval_num_cos',
        'weekday_num_sin',  'weekday_num_cos',
        'is_weekend',
        'P_A', 'P_B', 'P_C', 'P_D',
    ]

    X = df[feature_cols]
    y = df['actual_shape']

    # regularised to avoid overfitting on just 3 months of interval data
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=10,
        reg_alpha=1.0,
        reg_lambda=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    valid = np.isfinite(y) & np.isfinite(X).all(axis=1)
    model.fit(X[valid], y[valid])

    # save feature importance chart
    os.makedirs('plots', exist_ok=True)
    importances = model.feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(9, 5))
    plt.barh(range(len(order)), importances[order])
    plt.yticks(range(len(order)), [feature_cols[i] for i in order])
    plt.title('XGBoost Shape Model — Feature Importance')
    plt.xlabel('Importance (Gain)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png', dpi=150)
    plt.close()

    # predict for every (queue, DOW, interval) combo in the shape table
    combos = shape[['Portfolio', 'day_of_week', 'Interval']].copy()
    combos['interval_num'] = (pd.to_datetime(combos['Interval'], format='%H:%M').dt.hour * 2 +
                               pd.to_datetime(combos['Interval'], format='%H:%M').dt.minute // 30)
    day_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    combos['weekday_num'] = combos['day_of_week'].map(day_map)
    combos['is_weekend']  = (combos['weekday_num'] >= 5).astype(int)
    combos = encode_cyclic(combos, 'interval_num', 48)
    combos = encode_cyclic(combos, 'weekday_num', 7)
    port_d = pd.get_dummies(combos['Portfolio'], prefix='P').astype(int)
    combos = pd.concat([combos, port_d], axis=1)

    # ensure all four portfolio columns are present in case a queue is missing
    for col in ['P_A', 'P_B', 'P_C', 'P_D']:
        if col not in combos.columns:
            combos[col] = 0

    combos['xgb_shape'] = model.predict(combos[feature_cols])

    # normalise so XGBoost shape also sums to 1.0 per (queue, DOW)
    for (q, dow), idx in combos.groupby(['Portfolio', 'day_of_week']).groups.items():
        total = combos.loc[idx, 'xgb_shape'].sum()
        if total > 0:
            combos.loc[idx, 'xgb_shape'] /= total

    shape = shape.merge(
        combos[['Portfolio', 'day_of_week', 'Interval', 'xgb_shape']],
        on=['Portfolio', 'day_of_week', 'Interval'],
        how='left',
    )
    shape['final_shape_cv'] = (
        (1 - blend_alpha) * shape['shape_cv'] +
        blend_alpha       * shape['xgb_shape']
    )

    # re-normalise after blending
    for (q, dow), idx in shape.groupby(['Portfolio', 'day_of_week']).groups.items():
        total = shape.loc[idx, 'final_shape_cv'].sum()
        if total > 0:
            shape.loc[idx, 'final_shape_cv'] /= total

    print(f"  Shape blend: {blend_alpha:.0%} XGBoost + {1-blend_alpha:.0%} statistical")
    print("  Saved plots/feature_importance.png")
    return shape
