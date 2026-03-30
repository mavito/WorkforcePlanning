"""
General volume, CCT and abandon rate trend analysis.

This is the exploratory overview — useful for understanding what
the data looks like before making any modelling decisions.
Covers:
  - Volume trends across queues and months
  - Day-of-week volume patterns
  - Abandon rate stability and seasonality
  - CCT distribution across queues
  - Correlation between daily CV, CCT, and abandon rate
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

PLOT_DIR = 'plots/eda'


def plot_monthly_volume_trends(daily):
    """Overall volume trend from Jan 2024 to Aug 2025 per queue.

    Gives a feel for seasonality, growth, and whether the Aug target
    period is typical or unusual compared to prior months.
    """
    d = daily.copy()
    d['YearMonth'] = d['Date'].dt.to_period('M')
    monthly = d.groupby(['Portfolio', 'YearMonth'])['Call_Volume'].mean().reset_index()
    monthly['Date'] = monthly['YearMonth'].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    colours = {'A': '#4472c4', 'B': '#ed7d31', 'C': '#70ad47', 'D': '#ffc000'}

    for q in ['A', 'B', 'C', 'D']:
        sub = monthly[monthly['Portfolio'] == q]
        ax.plot(sub['Date'], sub['Call_Volume'], label=f'Queue {q}',
                color=colours[q], lw=2, marker='o', markersize=4)

    ax.axvspan(pd.Timestamp('2025-04-01'), pd.Timestamp('2025-06-30'),
               alpha=0.1, color='green', label='Shape training period')
    ax.axvspan(pd.Timestamp('2025-08-01'), pd.Timestamp('2025-08-31'),
               alpha=0.1, color='orange', label='Forecast target')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.set_ylabel('Avg daily call volume')
    ax.set_title('Monthly average daily call volume — Jan 2024 to Aug 2025',
                 fontweight='bold')
    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/monthly_volume_trends.png', dpi=150)
    plt.close()
    print('Saved monthly_volume_trends.png')


def plot_dow_volume_heatmap(daily):
    """Volume by day-of-week and queue — shows the strong DOW effect
    that makes modelling DOW separately a worthwhile design choice.
    """
    d = daily[daily['Date'].dt.month.isin([4, 5, 6]) &
              (daily['Date'].dt.year == 2025)].copy()
    d['DOW'] = d['Date'].dt.day_name()

    dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot = (d.groupby(['Portfolio', 'DOW'])['Call_Volume'].mean()
              .unstack('DOW').reindex(columns=dow_order))

    # normalise each queue to [0,1] so colours are comparable within each row
    pivot_norm = pivot.div(pivot.max(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 3.5))
    sns.heatmap(pivot_norm, annot=pivot.round(0).astype(int), fmt='d',
                cmap='Blues', ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Relative load'})
    ax.set_title('Average daily call volume by queue and day-of-week (Apr–Jun 2025)\n'
                 'Confirms significant DOW variation → model DOW separately',
                 fontweight='bold')
    ax.set_ylabel('Queue')
    ax.set_xlabel('')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/dow_volume_heatmap.png', dpi=150)
    plt.close()
    print('Saved dow_volume_heatmap.png')


def plot_abandon_rate_stability(interval, daily):
    """Is the abandon rate consistent across months?

    ABD_ALPHA = 1.0 in config.py means we trust the Apr-Jun interval
    abandon rates completely for August. That's only valid if the
    abandon rate doesn't drift materially over time.
    """
    iv = interval.copy()
    iv['month']       = iv['Date'].dt.month
    iv['day_of_week'] = iv['Date'].dt.day_name()

    # daily abandoned rate from daily data
    d = daily.copy()
    d['year']  = d['Date'].dt.year
    d['month'] = d['Date'].dt.month
    d_monthly  = d.groupby(['Portfolio', 'year', 'month'])['Abandon_Rate'].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    axes = axes.flatten()
    colours = {'2024': '#4472c4', '2025': '#ed7d31'}

    for i, q in enumerate(['A', 'B', 'C', 'D']):
        for yr in [2024, 2025]:
            sub = d_monthly[(d_monthly['Portfolio'] == q) &
                            (d_monthly['year']      == yr)]
            axes[i].plot(sub['month'], sub['Abandon_Rate'] * 100,
                         label=str(yr), color=colours[str(yr)],
                         marker='o', lw=1.8)

        axes[i].set_title(f'Queue {q} monthly abandon rate')
        axes[i].set_ylabel('Abandon rate (%)')
        axes[i].set_xlabel('Month')
        axes[i].set_xticks(range(1, 13))
        axes[i].set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'], fontsize=8)
        axes[i].legend(fontsize=8)

    fig.suptitle('Monthly abandon rate — 2024 vs 2025\n'
                 'Rate is stable → ABD_ALPHA = 1.0 (full trust in Apr-Jun intervals) is fine',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/abandon_rate_stability.png', dpi=150)
    plt.close()
    print('Saved abandon_rate_stability.png')


def plot_queue_volume_distribution(interval):
    """Distribution of daily call volumes per queue.

    Shows the scale differences between queues and whether the data
    is roughly bell-shaped (median ≈ mean, so trimmed mean is reasonable)
    or heavily skewed (where trimming matters more).
    """
    iv = interval.copy()
    iv['month'] = iv['Date'].dt.month
    iv = iv[iv['month'].isin([4, 5, 6])].copy()

    daily_cv = iv.groupby(['Portfolio', 'Date'])['Call_Volume'].sum().reset_index()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.5), sharey=False)
    colours = ['#4472c4', '#ed7d31', '#70ad47', '#ffc000']

    for i, q in enumerate(['A', 'B', 'C', 'D']):
        sub = daily_cv[daily_cv['Portfolio'] == q]['Call_Volume'].dropna()
        axes[i].hist(sub, bins=25, color=colours[i], alpha=0.8, edgecolor='white')
        axes[i].axvline(sub.mean(),   color='red',    lw=1.5, linestyle='--', label=f'mean={sub.mean():.0f}')
        axes[i].axvline(sub.median(), color='black',  lw=1.2, linestyle=':',  label=f'med={sub.median():.0f}')
        axes[i].set_title(f'Queue {q}')
        axes[i].set_xlabel('Daily CV')
        axes[i].legend(fontsize=7)
        if i == 0:
            axes[i].set_ylabel('# days')

    fig.suptitle('Daily call volume distribution per queue (Apr–Jun 2025)\n'
                 'Mean ≈ median → trimmed_mean is a robust shape estimator',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/daily_cv_distribution.png', dpi=150)
    plt.close()
    print('Saved daily_cv_distribution.png')


def plot_metric_correlations(daily):
    """Are CCT and abandon rate correlated with call volume?

    Understanding these relationships helps decide whether metrics
    should be forecast jointly or independently.
    """
    d = daily[(daily['Date'].dt.year == 2025) &
              (daily['Date'].dt.month.isin([4, 5, 6]))].copy()

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    colours = {'A': '#4472c4', 'B': '#ed7d31', 'C': '#70ad47', 'D': '#ffc000'}

    for i, q in enumerate(['A', 'B', 'C', 'D']):
        sub = d[d['Portfolio'] == q].dropna(subset=['Call_Volume', 'CCT', 'Abandon_Rate'])

        # CV vs CCT
        axes[0][i].scatter(sub['Call_Volume'], sub['CCT'],
                           alpha=0.5, color=colours[q], s=15)
        m, b = np.polyfit(sub['Call_Volume'], sub['CCT'], 1)
        x_line = np.linspace(sub['Call_Volume'].min(), sub['Call_Volume'].max(), 50)
        axes[0][i].plot(x_line, m * x_line + b, color='black', lw=1)
        corr = sub['Call_Volume'].corr(sub['CCT'])
        axes[0][i].set_title(f'Q{q}: CV vs CCT  (r={corr:.2f})')
        axes[0][i].set_xlabel('Daily CV')
        if i == 0:
            axes[0][i].set_ylabel('Daily CCT (s)')

        # CV vs Abandon Rate
        axes[1][i].scatter(sub['Call_Volume'], sub['Abandon_Rate'] * 100,
                           alpha=0.5, color=colours[q], s=15)
        m, b = np.polyfit(sub['Call_Volume'], sub['Abandon_Rate'], 1)
        axes[1][i].plot(x_line, (m * x_line + b) * 100, color='black', lw=1)
        corr2 = sub['Call_Volume'].corr(sub['Abandon_Rate'])
        axes[1][i].set_title(f'Q{q}: CV vs ABD  (r={corr2:.2f})')
        axes[1][i].set_xlabel('Daily CV')
        if i == 0:
            axes[1][i].set_ylabel('Abandon rate (%)')

    fig.suptitle('Correlation between daily CV, CCT, and abandon rate\n'
                 'Low correlation → modelling metrics independently is reasonable',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/metric_correlations.png', dpi=150)
    plt.close()
    print('Saved metric_correlations.png')
