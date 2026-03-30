"""
Approach: use the known August 2025 daily totals as a base, then distribute
each day's volume across 48 half-hour intervals using a historical "shape"
built from April-June data. CCT and abandon rate use the same historical shapes.

This avoids compounding prediction errors that come with autoregressive models —
since we already know how many calls came in each day, we only need to get the
intraday distribution right.

Usage:
    python main.py                  # generate submission.csv
    python main.py --output my.csv  # write to a different file
"""

import argparse
import os
import sys

import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_data
from src.shape      import build_shape
from src.forecast   import build_forecast, format_submission
from src.validate   import cross_check


def parse_args():
    parser = argparse.ArgumentParser(description='Generate August 2025 contact center forecast')
    parser.add_argument('--data',   default='data.xlsx',      help='Input Excel file (default: data.xlsx)')
    parser.add_argument('--output', default='forecasts/submission.csv', help='Output CSV file (default: forecasts/submission.csv)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 55)
    print("  Contact Center Forecasting — Rank 7 Pipeline")
    print("=" * 55)

    # load raw data
    interval, daily = load_data(args.data)

    # build the intraday shape from Apr-Jun historical data
    #this is the key step — it tells us how call volume spreads across the day
    shape = build_shape(interval)

    #generate August predictions (daily totals × shape × bias)
    fc = build_forecast(shape, daily)

    #pivot to the wide submission format
    submission = format_submission(fc)

    #basic validation
    expected = 31 * 48   # 31 days × 48 half-hour slots
    if len(submission) != expected:
        print(f"WARNING :: expected {expected} rows but got {len(submission)}")

    for col in submission.columns[3:]:
        n_neg = (submission[col] < 0).sum()
        if n_neg:
            print(f"WARNING :: {n_neg} negative values in {col}")

    # compare aggregated predictions against the known August daily actuals
    cross_check(submission, daily)

    # write output
    submission.to_csv(args.output, index=False)
    print(f"\nSaved {len(submission)} rows to {args.output}")
    print("=" * 55)


if __name__ == '__main__':
    main()
