"""
Holiday impact analysis.

Assumption in config.py: six specific dates in Apr-Jun 2025 are excluded
from the shape calculation. This script shows *why* — those days have
call volumes 7-55% below the normal level for that day-of-week, which
would make the shape look lighter than it really is for normal days.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

HOLIDAYS = {
    '2025-04-18': 'Good Friday',
    '2025-04-20': 'Easter Sunday',
    '2025-05-11': "Mother's Day",
    '2025-05-26': 'Memorial Day',
    '2025-06-15': "Father's Day",
    '2025-06-19': 'Juneteenth',
}

PLOT_DIR = 'plots/eda'


def analyse_holiday_impact(interval):
    """Compare call volume on holiday dates vs the median for the same day-of-week."""
    iv = interval.copy()
    iv['date_str']    = iv['Date'].dt.strftime('%Y-%m-%d')
    iv['day_of_week'] = iv['Date'].dt.day_name()
    iv['month']       = iv['Date'].dt.month

    # only Apr-Jun
    iv = iv[iv['month'].isin([4, 5, 6])].copy()

    # daily total per portfolio
    daily = (iv.groupby(['Portfolio', 'Date', 'date_str', 'day_of_week'])
               ['Call_Volume'].sum().reset_index())

    # DOW baseline — median across *non-holiday* days
    baseline = (daily[~daily['date_str'].isin(HOLIDAYS)]
                .groupby(['Portfolio', 'day_of_week'])['Call_Volume']
                .median().reset_index()
                .rename(columns={'Call_Volume': 'baseline_cv'}))

    holiday_rows = daily[daily['date_str'].isin(HOLIDAYS)].copy()
    holiday_rows = holiday_rows.merge(baseline, on=['Portfolio', 'day_of_week'])
    holiday_rows['pct_diff'] = (holiday_rows['Call_Volume'] - holiday_rows['baseline_cv']) \
                               / holiday_rows['baseline_cv'] * 100
    holiday_rows['label'] = holiday_rows['date_str'].map(HOLIDAYS)

    # --- figure: % deviation from DOW baseline per queue ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.flatten()
    queues = ['A', 'B', 'C', 'D']
    colours = ['#e05252' if v < 0 else '#52a852' for v in
               holiday_rows.groupby('date_str')['pct_diff'].mean().values]

    for i, q in enumerate(queues):
        sub = holiday_rows[holiday_rows['Portfolio'] == q].copy()
        sub = sub.sort_values('date_str')
        bars = axes[i].barh(sub['label'], sub['pct_diff'],
                            color=['#e05252' if v < 0 else '#52a852' for v in sub['pct_diff']])
        axes[i].axvline(0, color='black', linewidth=0.8)
        axes[i].set_title(f'Queue {q}', fontweight='bold')
        axes[i].set_xlabel('% vs normal day-of-week')
        for bar, val in zip(bars, sub['pct_diff']):
            axes[i].text(val - 1 if val < 0 else val + 0.5,
                         bar.get_y() + bar.get_height() / 2,
                         f'{val:.0f}%', va='center', fontsize=8)

    fig.suptitle('Holiday call volume vs normal day-of-week median (Apr–Jun 2025)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/holiday_volume_impact.png', dpi=150)
    plt.close()
    print('Saved holiday_volume_impact.png')

    # --- summary table ---
    summary = (holiday_rows.groupby(['label', 'date_str'])
               ['pct_diff'].mean().reset_index()
               .sort_values('date_str'))
    summary.columns = ['Holiday', 'Date', 'Avg % vs normal DOW']
    summary['Avg % vs normal DOW'] = summary['Avg % vs normal DOW'].round(1)
    print('\nHoliday impact (avg across all queues):')
    print(summary.to_string(index=False))


def plot_volume_timeline(interval):
    """Show volume across the full Apr-Jun period with holiday dates marked.
    Makes it visually obvious why they're outliers worth excluding.
    """
    iv = interval.copy()
    iv['month'] = iv['Date'].dt.month
    iv = iv[iv['month'].isin([4, 5, 6])].copy()

    daily = (iv.groupby(['Portfolio', 'Date'])['Call_Volume']
               .sum().reset_index())

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    queues = ['A', 'B', 'C', 'D']

    for i, q in enumerate(queues):
        sub  = daily[daily['Portfolio'] == q].sort_values('Date')
        axes[i].plot(sub['Date'], sub['Call_Volume'], lw=1.2, color='#4472c4', label=q)

        # mark holiday dates
        for date_str, name in HOLIDAYS.items():
            dt = pd.to_datetime(date_str)
            row = sub[sub['Date'] == dt]
            if not row.empty:
                axes[i].axvline(dt, color='red', lw=1, alpha=0.6, linestyle='--')
                axes[i].annotate(name, xy=(dt, row['Call_Volume'].values[0]),
                                 xytext=(5, 5), textcoords='offset points',
                                 fontsize=7, color='red')

        axes[i].set_ylabel(f'Queue {q}\nDaily CV')
        axes[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    axes[-1].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
    fig.suptitle('Daily call volume Apr–Jun 2025 (red dashed = excluded holidays)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/volume_timeline_with_holidays.png', dpi=150)
    plt.close()
    print('Saved volume_timeline_with_holidays.png')
