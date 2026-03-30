import pandas as pd
import numpy as np
from src.utils import smape


def cross_check(submission, daily):
    """Compare aggregated interval predictions against the known August daily totals.

    This doesn't directly match what the scorer evaluates (they use interval-level),
    but it's a quick sanity check that our daily totals and CCT averages are reasonable.
    A positive bias on calls is what we want — understaffing is penalised more.
    """
    aug = daily[(daily['Date'].dt.year == 2025) & (daily['Date'].dt.month == 8)].copy()
    aug['day'] = aug['Date'].dt.day

    header = f"  {'Queue':<6} {'Metric':<8} {'SMAPE':>8} {'AvgPred':>10} {'AvgActual':>10} {'Bias':>8}"
    print("\n" + header)
    print("  " + "-" * 55)

    for q in ['A', 'B', 'C', 'D']:
        actual = aug[aug['Portfolio'] == q].set_index('day')

        # calls — sum intervals back to daily and compare
        pred_daily = submission.groupby('Day')[f'Calls_Offered_{q}'].sum()
        common = pred_daily.index.intersection(actual.index)
        if len(common) > 0:
            s    = smape(actual.loc[common, 'Call_Volume'].values, pred_daily.loc[common].values)
            bias = (pred_daily.loc[common].mean() - actual.loc[common, 'Call_Volume'].mean()) \
                   / actual.loc[common, 'Call_Volume'].mean() * 100
            print(f"  {q:<6} {'Calls':<8} {s:>7.2f}% "
                  f"{pred_daily.loc[common].mean():>10.0f} "
                  f"{actual.loc[common, 'Call_Volume'].mean():>10.0f} {bias:>+7.1f}%")

        # CCT — volume-weighted average across intervals
        tmp = submission.copy()
        tmp['w'] = tmp[f'CCT_{q}'] * tmp[f'Calls_Offered_{q}']
        pred_cct = tmp.groupby('Day')['w'].sum() / tmp.groupby('Day')[f'Calls_Offered_{q}'].sum()
        common = pred_cct.index.intersection(actual.index)
        if len(common) > 0:
            s = smape(actual.loc[common, 'CCT'].values, pred_cct.loc[common].values)
            print(f"  {q:<6} {'CCT':<8} {s:>7.2f}% "
                  f"{pred_cct.loc[common].mean():>10.1f} "
                  f"{actual.loc[common, 'CCT'].mean():>10.1f}")
