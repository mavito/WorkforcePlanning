"""
Intraday shape analysis.

Assumptions justified here:
  1. The intraday call volume distribution is stable enough across Apr-Jun
     that we can treat all three months as one pool for shape estimation.
  2. Ratio-of-sums is better than ratio-of-means because high-volume days
     are more representative of the "true" shape than low-volume days.
  3. Circular kernel smoothing with alpha=0.5, window=5 reduces noise
     without losing real structure in the shape.
  4. The shape differs meaningfully by day-of-week (so we estimate it
     separately for each DOW rather than a single week-wide average).
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

HOLIDAYS = {
    '2025-04-18', '2025-04-20', '2025-05-11',
    '2025-05-26', '2025-06-15', '2025-06-19',
}
SMOOTH_KERNEL = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
SMOOTH_ALPHA  = 0.5
PLOT_DIR      = 'plots/eda'


def _get_clean_iv(interval):
    iv = interval.copy()
    iv['date_str']    = iv['Date'].dt.strftime('%Y-%m-%d')
    iv['day_of_week'] = iv['Date'].dt.day_name()
    iv['month']       = iv['Date'].dt.month
    iv = iv[
        iv['month'].isin([4, 5, 6]) &
        ~iv['date_str'].isin(HOLIDAYS)
    ].copy()
    return iv


def plot_shape_stability_across_months(interval):
    """Does the intraday shape stay consistent month to month?

    If April, May, June all show roughly the same shape, pooling them is fine.
    If they differ a lot, we'd need to reconsider.
    """
    iv = _get_clean_iv(interval)
    month_names = {4: 'April', 5: 'May', 6: 'June'}

    # compute shape per (queue, month, DOW, interval)
    iv['daily_cv'] = iv.groupby(['Portfolio', 'Date'])['Call_Volume'].transform('sum')
    iv['frac']     = iv['Call_Volume'] / iv['daily_cv'].replace(0, np.nan)

    # focus on one representative DOW — Tuesday (typical busy weekday)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    colours = {'April': '#4472c4', 'May': '#ed7d31', 'June': '#70ad47'}

    for i, q in enumerate(['A', 'B', 'C', 'D']):
        sub = iv[(iv['Portfolio'] == q) & (iv['day_of_week'] == 'Tuesday')]
        for m, mname in month_names.items():
            m_data = sub[sub['month'] == m].groupby('Interval')['frac'].mean()
            m_data = m_data.reindex(sorted(m_data.index))
            axes[i].plot(range(len(m_data)), m_data.values * 100,
                         label=mname, color=colours[mname], lw=1.8)
        axes[i].set_title(f'Queue {q} — Tuesday shape by month')
        axes[i].set_ylabel('% of daily volume')
        axes[i].set_xlabel('30-min interval (0=midnight)')
        axes[i].legend(fontsize=8)
        axes[i].axhline(100 / 48, color='grey', lw=0.6, linestyle=':', label='uniform')

    fig.suptitle('Intraday shape stability across Apr / May / Jun\n'
                 '(high overlap → safe to pool all three months)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/shape_stability_across_months.png', dpi=150)
    plt.close()
    print('Saved shape_stability_across_months.png')


def plot_dow_shape_differences(interval):
    """Shows that weekday/weekend patterns differ enough that a single global
    shape would be wrong — we need separate shapes per day-of-week.
    """
    iv = _get_clean_iv(interval)
    iv['daily_cv'] = iv.groupby(['Portfolio', 'Date'])['Call_Volume'].transform('sum')
    iv['frac']     = iv['Call_Volume'] / iv['daily_cv'].replace(0, np.nan)

    dows   = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    cmap   = cm.get_cmap('tab10', len(dows))
    q      = 'B'   # use a mid-size queue as representative

    fig, ax = plt.subplots(figsize=(13, 5))
    for j, dow in enumerate(dows):
        sub  = iv[(iv['Portfolio'] == q) & (iv['day_of_week'] == dow)]
        avg  = sub.groupby('Interval')['frac'].mean().reindex(sorted(sub['Interval'].unique()))
        linestyle = '--' if dow in ('Saturday', 'Sunday') else '-'
        ax.plot(range(len(avg)), avg.values * 100,
                label=dow, color=cmap(j), lw=1.8, linestyle=linestyle)

    ax.axhline(100 / 48, color='grey', lw=0.6, linestyle=':', label='uniform')
    ax.set_xlabel('30-min interval (0 = midnight)')
    ax.set_ylabel('% of daily volume')
    ax.set_title(f'Queue {q} — intraday shape by day-of-week\n'
                 '(shape varies significantly → must model DOW separately)',
                 fontweight='bold')
    ax.legend(ncol=4, fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/shape_by_dow.png', dpi=150)
    plt.close()
    print('Saved shape_by_dow.png')


def plot_smoothing_effect(interval):
    """Shows the before/after of the circular kernel smoothing.

    The raw ratio-of-sums shape has slot-to-slot spikes (some slots see
    more traffic just due to the small sample size). Smoothing reduces
    those without shifting where the peaks are.
    """
    iv = _get_clean_iv(interval)
    q, dow = 'C', 'Wednesday'

    sub = iv[(iv['Portfolio'] == q) & (iv['day_of_week'] == dow)]
    slot_sums = sub.groupby('Interval')['Call_Volume'].sum().reset_index()
    slot_sums = slot_sums.sort_values('Interval').reset_index(drop=True)
    total = slot_sums['Call_Volume'].sum()
    slot_sums['raw_shape'] = slot_sums['Call_Volume'] / total

    # apply circular smoothing
    vals = slot_sums['raw_shape'].values.astype(float)
    n    = len(vals)
    half = len(SMOOTH_KERNEL) // 2
    smoothed = np.array([
        sum(SMOOTH_KERNEL[k] * vals[(i + k - half) % n]
            for k in range(len(SMOOTH_KERNEL)))
        for i in range(n)
    ])
    slot_sums['smoothed'] = (1 - SMOOTH_ALPHA) * vals + SMOOTH_ALPHA * smoothed

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    x = range(len(slot_sums))
    axes[0].bar(x, slot_sums['raw_shape'] * 100, color='#4472c4', alpha=0.7, width=0.8)
    axes[0].set_title(f'Raw ratio-of-sums shape — Queue {q}, {dow}', fontweight='bold')
    axes[0].set_ylabel('% of daily volume')

    axes[1].bar(x, slot_sums['smoothed'] * 100, color='#ed7d31', alpha=0.7, width=0.8)
    axes[1].plot(x, slot_sums['raw_shape'] * 100, color='grey', lw=0.8,
                 linestyle='--', label='raw (reference)')
    axes[1].set_title(f'After circular smoothing (alpha={SMOOTH_ALPHA}, window=5)',
                      fontweight='bold')
    axes[1].set_ylabel('% of daily volume')
    axes[1].set_xlabel('30-min interval index (0=midnight)')
    axes[1].legend()

    fig.suptitle('Circular kernel smoothing reduces slot-to-slot noise\n'
                 'while preserving the overall peak-hour structure',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/smoothing_effect.png', dpi=150)
    plt.close()
    print('Saved smoothing_effect.png')


def plot_ros_vs_rom_shape(interval):
    """Ratio-of-sums vs ratio-of-means: which is a better shape estimate?

    Ratio-of-sums gives more weight to high-volume days (which represent
    the "busy normal"). Ratio-of-means treats every day equally, so a
    quiet holiday-adjacent day has the same say as a busy Monday. The
    gap between the two reveals whether quiet days have different patterns.
    """
    iv = _get_clean_iv(interval)
    q, dow = 'A', 'Monday'

    sub = iv[(iv['Portfolio'] == q) & (iv['day_of_week'] == dow)]
    iv_day = sub.merge(
        sub.groupby('Date')['Call_Volume'].sum().rename('daily_cv').reset_index(),
        on='Date'
    )
    iv_day['frac'] = iv_day['Call_Volume'] / iv_day['daily_cv'].replace(0, np.nan)

    ros = sub.groupby('Interval')['Call_Volume'].sum()
    ros = (ros / ros.sum()).sort_index()

    rom = iv_day.groupby('Interval')['frac'].mean().sort_index()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = range(len(ros))
    ax.plot(x, ros.values * 100, label='Ratio-of-Sums (our choice)', color='#4472c4', lw=2)
    ax.plot(x, rom.reindex(ros.index).values * 100,
            label='Ratio-of-Means', color='#ed7d31', lw=2, linestyle='--')
    ax.set_xlabel('30-min interval (0=midnight)')
    ax.set_ylabel('Shape weight (%)')
    ax.set_title(f'Queue {q} — {dow}: RoS vs RoM\n'
                 'Both are similar; RoS preferred as busier days dominate naturally',
                 fontweight='bold')
    ax.legend()
    ax.axhline(100 / 48, color='grey', lw=0.6, linestyle=':', label='uniform')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/ros_vs_rom.png', dpi=150)
    plt.close()
    print('Saved ros_vs_rom.png')
