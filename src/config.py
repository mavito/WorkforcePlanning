# constants used throughout the pipeline
# holidays are excluded when building the intraday shape so they don't
# distort the "normal day" volume pattern we want to learn from

import numpy as np

# called queues A-D in the data
QUEUES = ['A', 'B', 'C', 'D']

# upward bias applied to call volume predictions
# keeps us on the safe side of overstaffing vs understaffing,
# since understaffing is penalised more heavily in scoring
#BIAS = {'A': 1.044, 'B': 1.044, 'C': 1.044, 'D': 1.044} # best scored version
BIAS = {'A': 1.0, 'B': 1.0, 'C': 1.0, 'D': 1.0}

# expanded holiday list for 2024-2025 to improve model training
# these days typically have significantly lower call volume
EXCLUDE_DATES = {
    # 2024
    '2024-01-01', '2024-01-15', '2024-03-31', '2024-05-27', '2024-06-19',
    '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-20', '2025-04-18', '2025-04-20', '2025-05-11',
    '2025-05-26', '2025-06-15', '2025-06-19', '2025-07-04', '2025-09-01',
    '2025-11-27', '2025-12-25'
}

# circular kernel for smoothing the call volume shape across intervals
# weights sum to 1, centred on the current slot, so nearby slots contribute
SMOOTH_KERNEL = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
SMOOTH_ALPHA  = 0.5   # how much of the smoothed version to blend in

# if an interval has fewer predicted calls than this, CCT is too noisy
# to trust the Apr-Jun shape — fall back to the flat daily average instead
CCT_BLEND_ALPHA = 0.9   # 90% Apr-Jun interval shape, 10% August daily CCT
CCT_THRESHOLD   = 15    # min calls to use the blended CCT
