# constants used throughout the pipeline
# holidays are excluded when building the intraday shape so they don't
# distort the "normal day" volume pattern we want to learn from

import numpy as np

# called queues A-D in the data
QUEUES = ['A', 'B', 'C', 'D']

# upward bias applied to call volume predictions
# keeps us on the safe side of overstaffing vs understaffing,
# since understaffing is penalised more heavily in scoring
BIAS = {'A': 1.044, 'B': 1.044, 'C': 1.044, 'D': 1.044}

# known holidays in Apr-Jun 2025 that pull volume well below typical levels
# leaving these in would make e.g. every "Monday" pattern look lighter than it is
EXCLUDE_DATES = {
    '2025-04-18',  # Good Friday
    '2025-04-20',  # Easter Sunday
    '2025-05-11',  # Mother's Day
    '2025-05-26',  # Memorial Day
    '2025-06-15',  # Father's Day
    '2025-06-19',  # Juneteenth, tried skipping this one and it didn't help
}

# circular kernel for smoothing the call volume shape across intervals
# weights sum to 1, centred on the current slot, so nearby slots contribute
SMOOTH_KERNEL = np.array([0.10, 0.20, 0.40, 0.20, 0.10])
SMOOTH_ALPHA  = 0.5   # how much of the smoothed version to blend in

# if an interval has fewer predicted calls than this, CCT is too noisy
# to trust the Apr-Jun shape — fall back to the flat daily average instead
CCT_BLEND_ALPHA = 0.9   # 90% Apr-Jun interval shape, 10% August daily CCT
CCT_THRESHOLD   = 15    # min calls to use the blended CCT
