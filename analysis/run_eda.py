"""
EDA runner — calls every analysis module and saves all plots to plots/eda/.

Run this once to generate all the charts that back up the modelling decisions.
Each plot filename maps to a specific assumption in src/config.py or src/shape.py etc.

Usage:
    python -m analysis.run_eda
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import load_data

from analysis.holidays       import analyse_holiday_impact, plot_volume_timeline
from analysis.intraday_shape import (plot_shape_stability_across_months,
                                     plot_dow_shape_differences,
                                     plot_smoothing_effect,
                                     plot_ros_vs_rom_shape)
from analysis.cct_patterns   import (plot_cct_by_interval,
                                     plot_cct_stability_across_months,
                                     plot_cct_variance_vs_volume,
                                     plot_cct_alpha_sweep)
from analysis.bias_scoring   import plot_asymmetric_penalty, plot_per_queue_bias_calibration
from analysis.data_quality   import (plot_null_distribution,
                                     plot_null_by_slot,
                                     plot_august_daily_completeness,
                                     print_imputation_quality)
from analysis.trends         import (plot_monthly_volume_trends,
                                     plot_dow_volume_heatmap,
                                     plot_abandon_rate_stability,
                                     plot_queue_volume_distribution,
                                     plot_metric_correlations)


def main():
    os.makedirs('plots/eda', exist_ok=True)

    print('Loading data...')
    interval, daily = load_data()

    # --- general overview ---
    print('\n[1/6] General volume & metric trends')
    plot_monthly_volume_trends(daily)
    plot_dow_volume_heatmap(daily)
    plot_abandon_rate_stability(interval, daily)
    plot_queue_volume_distribution(interval)
    plot_metric_correlations(daily)

    # --- holiday impact (justifies EXCLUDE_DATES) ---
    print('\n[2/6] Holiday impact analysis')
    analyse_holiday_impact(interval)
    plot_volume_timeline(interval)

    # --- intraday shape (justifies shape design choices) ---
    print('\n[3/6] Intraday shape analysis')
    plot_shape_stability_across_months(interval)
    plot_dow_shape_differences(interval)
    plot_smoothing_effect(interval)
    plot_ros_vs_rom_shape(interval)

    # --- CCT patterns (justifies blending params) ---
    print('\n[4/6] CCT pattern analysis')
    plot_cct_by_interval(interval)
    plot_cct_stability_across_months(interval)
    plot_cct_variance_vs_volume(interval)
    plot_cct_alpha_sweep(interval, daily)

    # --- bias and scoring (justifies BIAS = 1.044) ---
    print('\n[5/6] Bias & scoring analysis')
    plot_asymmetric_penalty(daily)
    plot_per_queue_bias_calibration(daily)

    # --- data quality (justifies null imputation) ---
    print('\n[6/6] Data quality & null imputation')
    plot_null_distribution(interval)
    plot_null_by_slot(interval)
    plot_august_daily_completeness(daily)
    print_imputation_quality(interval)

    print('\nAll plots saved to plots/eda/')
    print('Each filename maps directly to an assumption in the pipeline.')


if __name__ == '__main__':
    main()
