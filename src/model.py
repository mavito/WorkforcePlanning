import pandas as pd
import xgboost as xgb
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error
from src.config import EXCLUDE_DATES, QUEUES
from src.utils import encode_cyclic, smape

def create_features(df):
    """Generate features for the XGBoost model."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # calendar features
    df['dow']   = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    df['day']   = df['Date'].dt.day
    df['woy']   = df['Date'].dt.isocalendar().week.astype(int)
    
    # cyclic encoding
    df = encode_cyclic(df, 'dow', 7)
    df = encode_cyclic(df, 'month', 12)

    # holiday indicator
    df['is_holiday'] = df['Date'].dt.strftime('%Y-%m-%d').isin(EXCLUDE_DATES).astype(int)

    # lag features per portfolio
    for q in QUEUES:
        mask = df['Portfolio'] == q
        # short-term and weekly seasonality
        for lag in [1, 2, 7, 14, 21, 28, 364]:
            df.loc[mask, f'lag_{lag}'] = df.loc[mask, 'Call_Volume'].shift(lag)
        
        # rolling features
        df.loc[mask, 'rolling_mean_7'] = df.loc[mask, 'Call_Volume'].shift(1).rolling(7).mean()
        df.loc[mask, 'rolling_std_7']  = df.loc[mask, 'Call_Volume'].shift(1).rolling(7).std()

    return df

def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Simple Grid Search to find best XGBoost params for a given queue."""
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 800],
        'subsample': [0.8, 1.0],
        'min_child_weight': [1, 3]
    }
    
    best_smape = float('inf')
    best_params = None
    
    for params in ParameterGrid(param_grid):
        model = xgb.XGBRegressor(
            **params,
            random_state=42,
            objective='reg:squarederror'
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        err = smape(y_val, preds)
        
        if err < best_smape:
            best_smape = err
            best_params = params
            
    return best_params, best_smape

def train_predict_queue(train_df, predict_df, queue, val_df=None):
    """Train XGBoost for a single queue and predict. Optionally tune first."""
    q_train = train_df[train_df['Portfolio'] == queue].dropna(subset=['Call_Volume', 'lag_364', 'rolling_mean_7'])
    q_test  = predict_df[predict_df['Portfolio'] == queue]

    features = [
        'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day', 'woy', 'is_holiday',
        'lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_21', 'lag_28', 'lag_364',
        'rolling_mean_7', 'rolling_std_7'
    ]

    X_train = q_train[features]
    y_train = q_train['Call_Volume']
    X_test  = q_test[features]

    # If validation data is provided, tune first
    if val_df is not None:
        q_val = val_df[val_df['Portfolio'] == queue].dropna(subset=['Call_Volume'])
        if not q_val.empty:
            X_val = q_val[features]
            y_val = q_val['Call_Volume']
            best_params, _ = tune_hyperparameters(X_train, y_train, X_val, y_val)
            print(f"    Best params for {queue}: {best_params}")
        else:
            best_params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500}
    else:
        # Default fallback
        best_params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500}

    model = xgb.XGBRegressor(
        **best_params,
        random_state=42,
        objective='reg:squarederror'
    )

    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    return preds, model.feature_importances_

def run_xgboost_forecast(daily):
    """Main entry point for XGBoost daily forecasting."""
    print("\n" + "-"*30)
    print("  XGBoost Daily Forecasting")
    print("-"*30)

    # 1. Feature Engineering
    df_feat = create_features(daily)

    # 2. Validation Step + Tuning (Predict July 2025)
    print("Running Grid Search & Validation (July 2025)...")
    val_train = df_feat[df_feat['Date'] < '2025-07-01']
    val_test  = df_feat[(df_feat['Date'] >= '2025-07-01') & (df_feat['Date'] <= '2025-07-31')]
    
    val_results = []
    best_params_per_queue = {}
    for q in QUEUES:
        # separate tuning/eval for July
        q_val_data = val_test[val_test['Portfolio'] == q].dropna(subset=['Call_Volume'])
        if q_val_data.empty: continue
        
        X_tr = val_train[val_train['Portfolio'] == q].dropna(subset=['Call_Volume', 'lag_364', 'rolling_mean_7'])
        features = [
            'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'day', 'woy', 'is_holiday',
            'lag_1', 'lag_2', 'lag_7', 'lag_14', 'lag_21', 'lag_28', 'lag_364',
            'rolling_mean_7', 'rolling_std_7'
        ]
        
        bp, _ = tune_hyperparameters(X_tr[features], X_tr['Call_Volume'], q_val_data[features], q_val_data['Call_Volume'])
        best_params_per_queue[q] = bp
        print(f"    Queue {q} Best: {bp}")
        
        # final val pred for July
        model = xgb.XGBRegressor(**bp, random_state=42, objective='reg:squarederror')
        model.fit(X_tr[features], X_tr['Call_Volume'])
        v_preds = model.predict(q_val_data[features])
        
        q_val = q_val_data.copy()
        q_val['Predicted_CV'] = v_preds
        val_results.append(q_val)
    
    val_df = pd.concat(val_results)
    actuals = val_df['Call_Volume'].dropna()
    preds_val = val_df.loc[actuals.index, 'Predicted_CV']
    
    if not actuals.empty:
        err = smape(actuals, preds_val)
        print(f"  July Validation sMAPE: {err:.2f}%")
    else:
        print("  Warning: No actuals found for July validation.")

    # 3. Final Prediction (August 2025)
    print("Predicting August 2025 with optimized parameters...")
    final_train = df_feat[df_feat['Date'] < '2025-08-01']
    final_test  = df_feat[(df_feat['Date'] >= '2025-08-01') & (df_feat['Date'] <= '2025-08-31')]
    
    final_results = []
    for q in QUEUES:
        # reuse best params found during validation
        bp = best_params_per_queue.get(q, {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 500})
        
        q_tr = final_train[final_train['Portfolio'] == q].dropna(subset=['Call_Volume', 'lag_364', 'rolling_mean_7'])
        q_ts = final_test[final_test['Portfolio'] == q]
        
        model = xgb.XGBRegressor(**bp, random_state=42, objective='reg:squarederror')
        model.fit(q_tr[features], q_tr['Call_Volume'])
        preds = model.predict(q_ts[features])
        
        q_res = q_ts.copy()
        q_res['Call_Volume'] = preds
        final_results.append(q_res)
    
    aug_fc = pd.concat(final_results)
    
    # Merge August predictions back into the original daily dataframe
    # replacing the old median-imputed values
    daily_updated = daily.copy()
    daily_updated['Date'] = pd.to_datetime(daily_updated['Date'])
    
    # identify August 2025 rows in daily_updated
    mask = (daily_updated['Date'] >= '2025-08-01') & (daily_updated['Date'] <= '2025-08-31')
    
    # map predictions by Portfolio and Date
    aug_fc['Date'] = pd.to_datetime(aug_fc['Date'])
    pred_map = aug_fc.set_index(['Portfolio', 'Date'])['Call_Volume'].to_dict()
    
    def get_pred(row):
        if (row['Portfolio'], row['Date']) in pred_map:
            return pred_map[(row['Portfolio'], row['Date'])]
        return row['Call_Volume']

    daily_updated.loc[mask, 'Call_Volume'] = daily_updated.apply(
        lambda r: pred_map.get((r['Portfolio'], r['Date']), r['Call_Volume']) if mask[daily_updated.index[daily_updated['Date'] == r['Date']][0]] else r['Call_Volume'], 
        axis=1
    )
    # The lambda above is a bit complex, let's simplify it.
    for q in QUEUES:
        q_mask = (daily_updated['Portfolio'] == q) & mask
        q_preds = aug_fc[aug_fc['Portfolio'] == q].set_index('Date')['Call_Volume']
        daily_updated.loc[q_mask, 'Call_Volume'] = daily_updated.loc[q_mask, 'Date'].map(q_preds)

    print(f"  August prediction complete. Predicted {len(aug_fc)} daily values.")
    print("-"*30)

    return daily_updated, val_df
