import numpy as np
import pandas as pd


def trimmed_mean(x, trim=1):
    # drop the single highest and lowest value before averaging
    # helps with outlier days without needing explicit outlier detection
    vals = sorted(x.dropna())
    if len(vals) <= 2 * trim:
        return np.mean(vals) if vals else np.nan
    return np.mean(vals[trim:-trim])


def smape(y_true, y_pred):
    # symmetric MAPE so scale differences between queues don't dominate
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask  = denom > 0
    err   = np.zeros_like(y_true)
    err[mask] = np.abs(y_true[mask] - y_pred[mask]) / denom[mask]
    return np.mean(err) * 100


def encode_cyclic(df, col, period):
    # time features like hour-of-day and day-of-week are circular,
    # so sin/cos encoding lets the model see that 23:30 is close to 0:00
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / period)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / period)
    return df


def impute_nulls(iv, metrics):
    """Fill missing interval values by matching on (portfolio, DOW, month, interval).
    Falls back to same (portfolio, DOW, interval) across any month if no exact match.
    
    This matters a lot — queues can have 90-280 null rows in Apr-Jun, and leaving
    them in silently biases the shape calculation.
    """
    iv = iv.copy()
    iv['_dow']   = iv['Date'].dt.weekday
    iv['_month'] = iv['Date'].dt.month

    for metric in metrics:
        null_rows = iv.index[iv[metric].isna()]
        for idx in null_rows:
            port  = iv.at[idx, 'Portfolio']
            dow   = iv.at[idx, '_dow']
            month = iv.at[idx, '_month']
            slot  = iv.at[idx, 'Interval']

            # prefer same month so seasonal patterns are preserved
            same_month = iv[
                (iv['Portfolio'] == port) &
                (iv['_dow']      == dow)  &
                (iv['_month']    == month) &
                (iv['Interval']  == slot)  &
                iv[metric].notna()
            ][metric]

            if not same_month.empty:
                iv.at[idx, metric] = same_month.median()
                continue

            # if that month has no data for this slot, widen to all months
            any_month = iv[
                (iv['Portfolio'] == port) &
                (iv['_dow']      == dow)  &
                (iv['Interval']  == slot)  &
                iv[metric].notna()
            ][metric]

            if not any_month.empty:
                iv.at[idx, metric] = any_month.median()

    iv.drop(columns=['_dow', '_month'], inplace=True)
    return iv
