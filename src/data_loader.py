import pandas as pd


def load_data(excel_path='data.xlsx'):
    """Read interval (Apr-Jun 2025) and daily (Jan 2024-Dec 2025) sheets for all queues."""
    queues = ['A', 'B', 'C', 'D']
    interval_frames, daily_frames = [], []

    for q in queues:
        # interval sheet has day and month in separate columns
        iv = pd.read_excel(excel_path, sheet_name=f'{q} - Interval')
        iv['Portfolio'] = q
        iv['Date'] = pd.to_datetime(
            iv['Day'].astype(str) + ' ' + iv['Month'] + ' 2025',
            format='%d %B %Y'
        )
        iv.columns = iv.columns.str.replace(' ', '_')
        interval_frames.append(iv)

        # daily sheet spans two years and has a proper Date column
        dy = pd.read_excel(excel_path, sheet_name=f'{q} - Daily')
        dy['Portfolio'] = q
        # daily dates vary in format across queues, so let pandas infer
        dy['Date'] = pd.to_datetime(dy['Date'], format='mixed')
        dy.columns = dy.columns.str.replace(' ', '_')
        daily_frames.append(dy)

    interval = pd.concat(interval_frames, ignore_index=True)
    daily    = pd.concat(daily_frames,    ignore_index=True)

    # drop rows with no interval label at all (completely empty rows in the sheet)
    interval = interval[interval['Interval'].notna()].copy()

    # some intervals come through as "0:00" instead of "00:00" — normalise them
    interval['Interval'] = (
        interval['Interval']
        .astype(str)
        .str.replace(r'^(\d):(\d{2})', r'0\1:\2', regex=True)
        .str[:5]  # trim any trailing seconds
    )

    print(f"Loaded {len(interval)} interval rows and {len(daily)} daily rows")
    return interval, daily
