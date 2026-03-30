import numpy as np
import pandas as pd
from src.config import BIAS, CCT_BLEND_ALPHA, CCT_THRESHOLD


def impute_august_daily(daily):
    """Some queues are missing daily rows at the end of August.
    Fill them with the median for the same day-of-week from the rest of 2025.
    """
    aug = daily[(daily['Date'].dt.year == 2025) & (daily['Date'].dt.month == 8)].copy()
    aug['day_of_week'] = aug['Date'].dt.day_name()

    ref_2025 = daily[daily['Date'].dt.year == 2025].copy()
    ref_2025['day_of_week'] = ref_2025['Date'].dt.day_name()

    for metric in ['Call_Volume', 'CCT', 'Abandon_Rate']:
        for idx in aug.index[aug[metric].isna()]:
            q   = aug.at[idx, 'Portfolio']
            dow = aug.at[idx, 'day_of_week']
            peers = ref_2025[
                (ref_2025['Portfolio']   == q) &
                (ref_2025['day_of_week'] == dow) &
                ref_2025[metric].notna()
            ][metric]
            if not peers.empty:
                aug.at[idx, metric] = peers.median()

    remaining = aug[['Call_Volume', 'CCT', 'Abandon_Rate']].isna().sum().sum()
    if remaining > 0:
        print(f"  Warning: {remaining} August daily nulls couldn't be filled")
    return aug


def build_forecast(shape, daily):
    """Core idea: take the known August daily totals, distribute them
    across 48 half-hour slots using the historical Apr-Jun shape, then
    apply a small upward bias to guard against understaffing.

    CCT is blended: 90% from the Apr-Jun interval average, 10% from the
    actual August daily CCT. For very quiet slots (< CCT_THRESHOLD calls)
    we just use the flat daily average because the interval average is too
    noisy to trust.

    Abandon rate comes straight from the Apr-Jun interval average — the
    historical rates are stable enough that no blending is needed.
    """
    aug = impute_august_daily(daily)
    aug['day_of_week'] = aug['Date'].dt.day_name()

    # each daily row fans out to 48 interval rows after the merge
    fc = aug.merge(shape, on=['Portfolio', 'day_of_week'], how='left')

    # fill anything that didn't match (shouldn't happen but just in case)
    fc['final_shape_cv'] = fc['final_shape_cv'].fillna(0)
    fc['interval_cct'] = fc['interval_cct'].fillna(fc['CCT'])
    fc['interval_abd'] = fc['interval_abd'].fillna(fc['Abandon_Rate'])

    # --- calls ---
    fc['calls'] = (
        fc['Call_Volume'] * fc['final_shape_cv'] * fc['Portfolio'].map(BIAS)
    ).clip(lower=0).fillna(0).round().astype(int)

    # --- CCT ---
    # blend the interval shape CCT with the actual daily average
    # high-volume slots get the blend; quiet overnight slots just use the daily average
    busy = fc['calls'] >= CCT_THRESHOLD
    fc['cct'] = fc['CCT'].copy()
    fc.loc[busy, 'cct'] = (
        CCT_BLEND_ALPHA       * fc.loc[busy, 'interval_cct'] +
        (1 - CCT_BLEND_ALPHA) * fc.loc[busy, 'CCT']
    )
    fc['cct'] = fc['cct'].clip(lower=0).round(2)
    fc.loc[fc['calls'] == 0, 'cct'] = 0

    # --- abandon rate ---
    fc['abd_rate'] = fc['interval_abd'].clip(0, 0.95).round(6)

    # --- abandoned calls derived from the above two ---
    fc['abd_calls'] = (
        fc['abd_rate'] * fc['calls']
    ).round().astype(int).clip(lower=0, upper=fc['calls'])

    return fc


def format_submission(fc):
    """wide format submission format"""
    queues = ['A', 'B', 'C', 'D']

    base = (fc[fc['Portfolio'] == queues[0]]
              .sort_values(['Date', 'Interval'])[['Date', 'Interval']]
              .reset_index(drop=True))

    for q in queues:
        grp = fc[fc['Portfolio'] == q].sort_values(['Date', 'Interval']).reset_index(drop=True)
        base[f'Calls_Offered_{q}']   = grp['calls'].values
        base[f'Abandoned_Calls_{q}'] = grp['abd_calls'].values
        base[f'Abandoned_Rate_{q}']  = grp['abd_rate'].round(2).values
        base[f'CCT_{q}']             = grp['cct'].values

    base['Month'] = 'August'
    base['Day']   = base['Date'].dt.day

    #Expected "0:00" not "00:00"
    base['Interval'] = base['Interval'].str.lstrip('0').str.replace(r'^:', '0:', regex=True)

    col_order = ['Month', 'Day', 'Interval']
    for q in queues:
        col_order += [f'Calls_Offered_{q}', f'Abandoned_Calls_{q}',
                      f'Abandoned_Rate_{q}', f'CCT_{q}']

    return base[col_order].sort_values(['Day', 'Interval']).reset_index(drop=True)
