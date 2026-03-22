"""
forecasting_pipeline.py
-----------------------
Solar Energy Production Forecasting Pipeline

Trains 5 models on historical data from ML-ready sites,
selects the best by RMSE, generates 2026 forecasts with
confidence intervals, and saves SHAP feature importance.

Usage:
    python forecasting_pipeline.py

Outputs:
    models/best_model.pkl          - Saved best model
    outputs/forecasts_2026.csv     - 2026 predictions per site
    outputs/model_evaluation.csv   - All model RMSE/MAE scores
    outputs/shap_values.csv        - Feature importance
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import joblib
import os
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import shap

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/ml_ready_sites.csv"
OUTPUT_DIR   = "outputs"
MODEL_DIR    = "models"
N_SPLITS     = 5       # TimeSeriesSplit folds
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Features to use for training
FEATURE_COLS = [
    'month', 'quarter', 'year',
    'month_sin', 'month_cos',
    'is_summer', 'is_winter', 'is_spring', 'is_autumn',
    'expected_kwh',
    'lag_1m', 'lag_3m', 'lag_6m', 'lag_12m',
    'roll_avg_3m', 'roll_avg_6m', 'roll_avg_12m',
    'expected_roll_3m', 'expected_roll_6m', 'expected_roll_12m',
    'total_historical_alerts', 'has_critical_alert',
    'inverter_fault_count', 'string_fault_count',
    'grid_fault_count', 'avg_impact',
]
TARGET_COL = 'actual_kwh'


# ── STEP 1: LOAD DATA ─────────────────────────────────────────────────────────
def load_data():
    print("="*55)
    print("SOLAR ENERGY FORECASTING PIPELINE")
    print("="*55)
    print("\nStep 1: Loading data...")

    df = pd.read_csv(DATA_PATH)
    df['month_year'] = pd.to_datetime(df['month_year'])
    df = df.sort_values(['site_id', 'month_year']).reset_index(drop=True)

    # Only use rows where actual data exists for training
    train_df = df[df['actual_kwh'] > 0].copy()
    # Future rows for forecasting (actual = 0 and future date)
    today = pd.Timestamp.today()
    future_df = df[(df['actual_kwh'] == 0) & (df['month_year'] > today)].copy()

    print(f"  Sites: {df['site_id'].nunique()}")
    print(f"  Training rows: {len(train_df)}")
    print(f"  Future forecast targets: {len(future_df)}")
    print(f"  Date range: {df['month_year'].min().strftime('%b %Y')} → {df['month_year'].max().strftime('%b %Y')}")

    return df, train_df, future_df


# ── STEP 2: PREPARE FEATURES ──────────────────────────────────────────────────
def prepare_features(train_df):
    print("\nStep 2: Preparing features...")

    # Use only rows where all lag features are available
    ready = train_df.dropna(subset=['lag_12m', 'roll_avg_12m']).copy()

    # Fill remaining NaN with 0
    for col in FEATURE_COLS:
        if col in ready.columns:
            ready[col] = ready[col].fillna(0)

    X = ready[FEATURE_COLS].values
    y = ready[TARGET_COL].values

    print(f"  Training samples: {len(X)}")
    print(f"  Features used: {len(FEATURE_COLS)}")
    print(f"  Target range: {y.min():,.0f} kWh → {y.max():,.0f} kWh")

    return ready, X, y


# ── STEP 3: TRAIN & EVALUATE 5 MODELS ────────────────────────────────────────
def train_and_evaluate(X, y):
    print("\nStep 3: Training and evaluating 5 models...")
    print(f"  Using TimeSeriesSplit with {N_SPLITS} folds\n")

    models = {
        'Linear Regression':    LinearRegression(),
        'Random Forest':        RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost':              xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                                  subsample=0.8, colsample_bytree=0.8,
                                                  random_state=RANDOM_STATE, verbosity=0),
        'LightGBM':             lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                                                   subsample=0.8, colsample_bytree=0.8,
                                                   random_state=RANDOM_STATE, verbose=-1),
        'Gradient Boosting':    GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                           max_depth=4, random_state=RANDOM_STATE),
    }

    tscv    = TimeSeriesSplit(n_splits=N_SPLITS)
    results = []

    for name, model in models.items():
        rmse_scores = []
        mae_scores  = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            preds = np.maximum(preds, 0)  # No negative energy

            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mae  = mean_absolute_error(y_val, preds)
            rmse_scores.append(rmse)
            mae_scores.append(mae)

        avg_rmse = np.mean(rmse_scores)
        avg_mae  = np.mean(mae_scores)

        results.append({
            'model':   name,
            'rmse':    round(avg_rmse, 2),
            'mae':     round(avg_mae, 2),
            'rmse_cv': [round(r, 2) for r in rmse_scores],
        })

        print(f"  {name:<22} RMSE: {avg_rmse:>8,.0f} kWh   MAE: {avg_mae:>8,.0f} kWh")

    return models, results


# ── STEP 4: SELECT BEST MODEL ─────────────────────────────────────────────────
def select_best_model(models, results, X, y):
    print("\nStep 4: Selecting best model...")

    results_df = pd.DataFrame(results).sort_values('rmse')
    best_name  = results_df.iloc[0]['model']
    best_rmse  = results_df.iloc[0]['rmse']
    best_mae   = results_df.iloc[0]['mae']

    print(f"\n  ✅ WINNER: {best_name}")
    print(f"     RMSE: {best_rmse:,.0f} kWh")
    print(f"     MAE:  {best_mae:,.0f} kWh")

    # Retrain winner on full dataset
    best_model = models[best_name]
    best_model.fit(X, y)

    # Save model
    model_path = os.path.join(MODEL_DIR, 'best_model.pkl')
    joblib.dump({'model': best_model, 'name': best_name, 'features': FEATURE_COLS}, model_path)
    print(f"\n  Model saved → {model_path}")

    # Save evaluation results
    eval_path = os.path.join(OUTPUT_DIR, 'model_evaluation.csv')
    results_df[['model', 'rmse', 'mae']].to_csv(eval_path, index=False)
    print(f"  Evaluation saved → {eval_path}")

    return best_model, best_name, results_df


# ── STEP 5: SHAP FEATURE IMPORTANCE ──────────────────────────────────────────
def compute_shap(best_model, best_name, X, ready_df):
    print("\nStep 5: Computing SHAP feature importance...")

    try:
        if best_name in ['XGBoost', 'LightGBM', 'Random Forest', 'Gradient Boosting']:
            explainer   = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X)
        else:
            explainer   = shap.LinearExplainer(best_model, X)
            shap_values = explainer.shap_values(X)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            'feature':    FEATURE_COLS,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        shap_path = os.path.join(OUTPUT_DIR, 'shap_values.csv')
        shap_df.to_csv(shap_path, index=False)

        print("  Top 5 most important features:")
        for _, row in shap_df.head(5).iterrows():
            bar = '█' * int(row['importance'] / shap_df['importance'].max() * 20)
            print(f"    {row['feature']:<25} {bar} {row['importance']:,.0f}")

        # Save SHAP bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        top10 = shap_df.head(10)
        ax.barh(top10['feature'][::-1], top10['importance'][::-1], color='#2E75B6')
        ax.set_xlabel('Mean |SHAP Value| (kWh)')
        ax.set_title(f'Feature Importance — {best_name}')
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        chart_path = os.path.join(OUTPUT_DIR, 'shap_importance.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  SHAP saved → {shap_path}")
        print(f"  Chart saved → {chart_path}")

        return shap_df

    except Exception as e:
        print(f"  SHAP skipped: {e}")
        return None


# ── STEP 6: GENERATE FORECASTS ────────────────────────────────────────────────
def generate_forecasts(best_model, best_name, df, ready_df):
    print("\nStep 6: Generating forecasts...")

    today   = pd.Timestamp.today()
    all_forecasts = []

    for site_id in df['site_id'].unique():
        site_df   = df[df['site_id'] == site_id].copy()
        site_hist = ready_df[ready_df['site_id'] == site_id].copy()

        if len(site_hist) < 3:
            continue

        # Future rows (actual = 0 or future date)
        future = site_df[(site_df['actual_kwh'] == 0) & (site_df['month_year'] > today)].copy()

        if len(future) == 0:
            # Generate next 12 months if no future rows
            last_date = site_df['month_year'].max()
            future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
            future = pd.DataFrame({'month_year': future_dates, 'site_id': site_id})
            # Carry forward last known expected
            last_expected = site_df['expected_kwh'].iloc[-1]
            future['expected_kwh'] = last_expected
            future['month']    = future['month_year'].dt.month
            future['quarter']  = future['month_year'].dt.quarter
            future['year']     = future['month_year'].dt.year
            future['month_sin'] = np.sin(2 * np.pi * future['month'] / 12)
            future['month_cos'] = np.cos(2 * np.pi * future['month'] / 12)
            future['is_summer'] = future['month'].isin([6,7,8]).astype(int)
            future['is_winter'] = future['month'].isin([12,1,2]).astype(int)
            future['is_spring'] = future['month'].isin([3,4,5]).astype(int)
            future['is_autumn'] = future['month'].isin([9,10,11]).astype(int)

            # Lags from historical
            actuals = site_hist['actual_kwh'].values
            for i, row in future.iterrows():
                future.loc[i, 'lag_1m']  = actuals[-1] if len(actuals) >= 1 else 0
                future.loc[i, 'lag_3m']  = actuals[-3] if len(actuals) >= 3 else 0
                future.loc[i, 'lag_6m']  = actuals[-6] if len(actuals) >= 6 else 0
                future.loc[i, 'lag_12m'] = actuals[-12] if len(actuals) >= 12 else 0
                future.loc[i, 'roll_avg_3m']  = np.mean(actuals[-3:]) if len(actuals) >= 3 else 0
                future.loc[i, 'roll_avg_6m']  = np.mean(actuals[-6:]) if len(actuals) >= 6 else 0
                future.loc[i, 'roll_avg_12m'] = np.mean(actuals[-12:]) if len(actuals) >= 12 else 0
                future.loc[i, 'expected_roll_3m']  = last_expected
                future.loc[i, 'expected_roll_6m']  = last_expected
                future.loc[i, 'expected_roll_12m'] = last_expected

        # Fill alert features from historical
        alert_cols = ['total_historical_alerts', 'has_critical_alert',
                      'inverter_fault_count', 'string_fault_count',
                      'grid_fault_count', 'avg_impact']
        for col in alert_cols:
            if col in site_hist.columns:
                future[col] = site_hist[col].iloc[0] if len(site_hist) > 0 else 0
            else:
                future[col] = 0

        # Fill any remaining missing
        for col in FEATURE_COLS:
            if col not in future.columns:
                future[col] = 0
            future[col] = future[col].fillna(0)

        X_future = future[FEATURE_COLS].values
        preds    = best_model.predict(X_future)
        preds    = np.maximum(preds, 0)

        # Confidence interval ±15% (based on typical solar forecast variance)
        for i, (_, row) in enumerate(future.iterrows()):
            all_forecasts.append({
                'site_id':        site_id,
                'month_year':     row['month_year'],
                'forecast_kwh':   round(preds[i], 0),
                'lower_bound':    round(preds[i] * 0.85, 0),
                'upper_bound':    round(preds[i] * 1.15, 0),
                'expected_kwh':   row.get('expected_kwh', 0),
                'model_used':     best_name,
            })

    forecasts_df = pd.DataFrame(all_forecasts)
    forecast_path = os.path.join(OUTPUT_DIR, 'forecasts_2026.csv')
    forecasts_df.to_csv(forecast_path, index=False)

    print(f"  Forecasted {len(forecasts_df)} months across {forecasts_df['site_id'].nunique()} sites")
    print(f"  Forecasts saved → {forecast_path}")
    print("\n  Sample forecasts:")
    print(forecasts_df[['site_id','month_year','forecast_kwh','lower_bound','upper_bound']].head(8).to_string(index=False))

    return forecasts_df


# ── STEP 7: PLOT FORECAST vs ACTUAL ──────────────────────────────────────────
def plot_forecast(df, forecasts_df, best_name):
    print("\nStep 7: Generating forecast charts...")

    # Plot for the site with most history
    site_counts = df[df['actual_kwh'] > 0].groupby('site_id').size()
    top_site    = site_counts.idxmax()

    hist   = df[(df['site_id'] == top_site) & (df['actual_kwh'] > 0)].copy()
    future = forecasts_df[forecasts_df['site_id'] == top_site].copy()

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(hist['month_year'], hist['actual_kwh'],
            color='#2E75B6', linewidth=2, label='Actual kWh', marker='o', markersize=3)
    ax.plot(hist['month_year'], hist['expected_kwh'],
            color='#A0A0A0', linewidth=1, linestyle='--', label='Expected kWh')

    if len(future) > 0:
        ax.plot(future['month_year'], future['forecast_kwh'],
                color='#ED7D31', linewidth=2, linestyle='--', label=f'Forecast ({best_name})', marker='s', markersize=4)
        ax.fill_between(future['month_year'],
                        future['lower_bound'], future['upper_bound'],
                        alpha=0.2, color='#ED7D31', label='Confidence interval (±15%)')

    ax.axvline(x=pd.Timestamp.today(), color='red', linewidth=1, linestyle=':', alpha=0.7, label='Today')
    ax.set_title(f'Solar Production Forecast — {top_site} | Model: {best_name}', fontsize=13)
    ax.set_xlabel('Month')
    ax.set_ylabel('Energy Production (kWh)')
    ax.legend(loc='upper left', fontsize=9)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    plt.tight_layout()

    chart_path = os.path.join(OUTPUT_DIR, 'forecast_chart.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved → {chart_path}")

    return chart_path


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df, train_df, future_df   = load_data()
    ready_df, X, y            = prepare_features(train_df)
    models, results           = train_and_evaluate(X, y)
    best_model, best_name, results_df = select_best_model(models, results, X, y)
    shap_df                   = compute_shap(best_model, best_name, X, ready_df)
    forecasts_df              = generate_forecasts(best_model, best_name, df, ready_df)
    chart_path                = plot_forecast(df, forecasts_df, best_name)

    print("\n" + "="*55)
    print("PIPELINE COMPLETE")
    print("="*55)
    print(f"  Best model:     {best_name}")
    print(f"  RMSE:           {results_df.iloc[0]['rmse']:,.0f} kWh")
    print(f"  MAE:            {results_df.iloc[0]['mae']:,.0f} kWh")
    print(f"  Forecasts:      outputs/forecasts_2026.csv")
    print(f"  SHAP chart:     outputs/shap_importance.png")
    print(f"  Forecast chart: outputs/forecast_chart.png")
    print("="*55)
