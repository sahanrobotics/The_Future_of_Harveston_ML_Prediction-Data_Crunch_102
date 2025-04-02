import pandas as pd
from prophet import Prophet
import numpy as np
import logging
import warnings
import time
from sklearn.metrics import mean_absolute_error
import optuna

# --- Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define file paths
TRAIN_FILE = 'train.csv'
TEST_FILE = 'test.csv'
SUBMISSION_FILE = 'submission_prophet_optuna_feats_v2.csv'

# --- Target Variable Handling ---
BASE_TARGET_VARIABLES = ['Avg_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed']
WIND_DIR_TARGETS = ['Wind_Direction_sin', 'Wind_Direction_cos']
TARGET_VARIABLES_TO_PROCESS = BASE_TARGET_VARIABLES + ['Wind_Direction']
TARGETS_FOR_MODELING = BASE_TARGET_VARIABLES + WIND_DIR_TARGETS

# --- Feature Engineering Config ---
LAGS_TO_CREATE = [1, 2, 3, 7]
ROLLING_WINDOWS = [3, 7, 14]
LAG_PREFIX = 'lag_'
ROLL_PREFIX = 'roll_mean_'

# Define potential base regressors
NON_GEO_REGRESSORS = [
    'Avg_Feels_Like_Temperature', 'Temperature_Range', 'Feels_Like_Temperature_Range',
    'Rain_Duration'
]
GEO_REGRESSORS = ['latitude', 'longitude']

# Define a base year
BASE_YEAR = 2000

# Validation configuration
VALIDATION_SPLIT_PERCENTAGE = 0.8
MIN_VALIDATION_POINTS = 30
OUTLIER_CLIP_PERCENTILE = 0.01

# Optuna Configuration
N_OPTUNA_TRIALS = 30
OPTUNA_TIMEOUT = None

# Variables likely benefiting from multiplicative seasonality
MULTIPLICATIVE_VARS = ['Radiation', 'Rain_Amount', 'Wind_Speed']

# --- Helper Functions ---
def kelvin_to_celsius(temp_k):
    """Converts Kelvin to Celsius."""
    return temp_k - 273.15

def clip_outliers(series, lower_quantile, upper_quantile):
    """Clips outliers in a Series based on quantiles."""
    series_numeric = pd.to_numeric(series, errors='coerce')
    if series_numeric.isnull().all(): return series
    lower_bound = series_numeric.quantile(lower_quantile)
    upper_bound = series_numeric.quantile(upper_quantile)
    if pd.isna(lower_bound) or pd.isna(upper_bound): return series_numeric
    return series_numeric.clip(lower=lower_bound, upper=upper_bound)

def preprocess_data(df, is_train=True):
    """Cleans, preprocesses, adds sin/cos wind, lags, and rolling features."""
    df_processed = df.copy()
    print(f"Preprocessing {'train' if is_train else 'test'} data (Adv Feats)...")

    df_processed['Year'] = pd.to_numeric(df_processed['Year'], errors='coerce')
    df_processed.dropna(subset=['Year'], inplace=True)
    df_processed['Year'] = df_processed['Year'].astype(int) + BASE_YEAR - 1
    df_processed['ds'] = pd.to_datetime( df_processed[['Year', 'Month', 'Day']], errors='coerce')
    initial_rows = len(df_processed)
    df_processed.dropna(subset=['ds'], inplace=True)
    dropped_rows = initial_rows - len(df_processed)
    if dropped_rows > 0: print(f"Dropped {dropped_rows} rows due to invalid dates.")

    if is_train:
        all_current_cols = list(set(TARGET_VARIABLES_TO_PROCESS + NON_GEO_REGRESSORS + GEO_REGRESSORS))

        if 'Avg_Temperature' in df_processed.columns:
            k_mask = df_processed['Avg_Temperature'] > 150
            df_processed.loc[k_mask, 'Avg_Temperature'] = df_processed.loc[k_mask, 'Avg_Temperature'].apply(kelvin_to_celsius)
        if 'Avg_Feels_Like_Temperature' in df_processed.columns:
            k_mask = df_processed['Avg_Feels_Like_Temperature'] > 150
            df_processed.loc[k_mask, 'Avg_Feels_Like_Temperature'] = df_processed.loc[k_mask, 'Avg_Feels_Like_Temperature'].apply(kelvin_to_celsius)

        print("Converting relevant columns to numeric...")
        for col in all_current_cols:
            if col in df_processed.columns: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        if 'Wind_Direction' in df_processed.columns:
            print("Creating Wind_Direction sin/cos components...")
            wd_rad = np.radians(df_processed['Wind_Direction'])
            df_processed['Wind_Direction_sin'] = np.sin(wd_rad)
            df_processed['Wind_Direction_cos'] = np.cos(wd_rad)
            all_current_cols.extend(WIND_DIR_TARGETS)
            all_current_cols = list(set(all_current_cols))

        print("Cleaning NaN/inf...")
        initial_rows_clean = len(df_processed)
        df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
        necessary_cols_pre_feat = [col for col in all_current_cols if col in df_processed.columns]
        df_processed.dropna(subset=necessary_cols_pre_feat, how='any', inplace=True)
        rows_dropped_clean = initial_rows_clean - len(df_processed)
        if rows_dropped_clean > 0: print(f"Dropped {rows_dropped_clean} rows due to NaN/inf.")

        df_processed.sort_values(['kingdom', 'ds'], inplace=True)
        print("Creating lag/rolling features (grouped by kingdom)...")
        lag_feat_names = []
        roll_feat_names = []
        grouped = df_processed.groupby('kingdom', group_keys=False)

        for target_col in TARGETS_FOR_MODELING:
            if target_col in df_processed.columns:
                for lag in LAGS_TO_CREATE:
                    col_name = f"{LAG_PREFIX}{target_col}_lag{lag}"
                    df_processed[col_name] = grouped[target_col].shift(lag)
                    lag_feat_names.append(col_name)
                shifted_target = grouped[target_col].shift(1)
                for window in ROLLING_WINDOWS:
                    col_name = f"{ROLL_PREFIX}{target_col}_w{window}"
                    df_processed[col_name] = shifted_target.rolling(window=window, min_periods=1).mean()
                    roll_feat_names.append(col_name)

        all_feature_cols = lag_feat_names + roll_feat_names

        initial_rows_feat_clean = len(df_processed)
        df_processed.dropna(subset=all_feature_cols, how='any', inplace=True)
        rows_dropped_feat = initial_rows_feat_clean - len(df_processed)
        if rows_dropped_feat > 0: print(f"Dropped {rows_dropped_feat} rows due to lag/rolling feature NaNs.")

        cols_to_keep = (['ds', 'kingdom']
                       + TARGETS_FOR_MODELING
                       + NON_GEO_REGRESSORS
                       + GEO_REGRESSORS
                       + all_feature_cols)
        final_cols_to_keep = [col for col in cols_to_keep if col in df_processed.columns]
        if 'Wind_Direction_sin' in final_cols_to_keep and 'Wind_Direction' in final_cols_to_keep:
             final_cols_to_keep.remove('Wind_Direction')

        cols_to_drop_final = [col for col in df_processed.columns if col not in final_cols_to_keep]
        if cols_to_drop_final: df_processed.drop(columns=cols_to_drop_final, inplace=True)

    else: # Test data
        cols_to_drop_test = ['Year', 'Month', 'Day', 'latitude', 'longitude', 'Wind_Direction']
        df_processed.drop(columns=cols_to_drop_test, inplace=True, errors='ignore')

    df_processed.sort_values('ds', inplace=True)
    print(f"Preprocessing complete. Final shape: {df_processed.shape}")
    print("-" * 30)
    return df_processed

# --- Optuna Objective Function (More Features) ---
def objective(trial, train_data, val_data, target_col,
              regressors_non_geo_geo, regressors_lag, regressors_roll,
              is_multiplicative):
    """Optuna objective - includes lag/roll features & separate prior scales."""

    changepoint_prior_scale = trial.suggest_float('changepoint_prior_scale', 1e-3, 5e-1, log=True)
    seasonality_prior_scale = trial.suggest_float('seasonality_prior_scale', 1e-2, 2e1, log=True)
    n_changepoints = trial.suggest_int('n_changepoints', 15, 40)
    yearly_fourier = trial.suggest_int('yearly_fourier_order', 5, 25)
    weekly_fourier = trial.suggest_int('weekly_fourier_order', 3, 15)
    monthly_fourier = trial.suggest_int('monthly_fourier_order', 3, 12)
    quarterly_fourier = trial.suggest_int('quarterly_fourier_order', 3, 10)

    prior_scale_non_geo_geo = 10.0
    if regressors_non_geo_geo:
        prior_scale_non_geo_geo = trial.suggest_float('prior_scale_non_geo_geo', 1e-2, 2e1, log=True)

    prior_scale_lag = 10.0
    if regressors_lag:
        prior_scale_lag = trial.suggest_float('prior_scale_lag', 1e-2, 2e1, log=True)

    prior_scale_roll = 10.0
    if regressors_roll:
        prior_scale_roll = trial.suggest_float('prior_scale_roll', 1e-2, 2e1, log=True)

    all_trial_regressors = regressors_non_geo_geo + regressors_lag + regressors_roll
    cols_to_select = ['ds', target_col] + all_trial_regressors
    prophet_train_trial = train_data[cols_to_select].rename(columns={target_col: 'y'})
    prophet_train_trial['y'] = clip_outliers(prophet_train_trial['y'], OUTLIER_CLIP_PERCENTILE, 1 - OUTLIER_CLIP_PERCENTILE)
    future_val_trial = val_data[['ds'] + all_trial_regressors].copy()

    if len(prophet_train_trial) < 2: return float('inf')

    seasonality_mode = 'multiplicative' if is_multiplicative else 'additive'
    model_trial = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        n_changepoints=n_changepoints,
        yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
        seasonality_mode=seasonality_mode,
    )
    model_trial.add_seasonality(name='yearly', period=365.25, fourier_order=yearly_fourier)
    model_trial.add_seasonality(name='weekly', period=7, fourier_order=weekly_fourier)
    model_trial.add_seasonality(name='monthly', period=30.5, fourier_order=monthly_fourier)
    model_trial.add_seasonality(name='quarterly', period=365.25 / 4, fourier_order=quarterly_fourier)

    for reg in regressors_non_geo_geo: model_trial.add_regressor(reg, prior_scale=prior_scale_non_geo_geo)
    for reg in regressors_lag: model_trial.add_regressor(reg, prior_scale=prior_scale_lag)
    for reg in regressors_roll: model_trial.add_regressor(reg, prior_scale=prior_scale_roll)

    try:
        model_trial.fit(prophet_train_trial)
        forecast_val = model_trial.predict(future_val_trial)

        validation_results = val_data[['ds', target_col]].merge(forecast_val[['ds', 'yhat']], on='ds', how='inner')
        validation_results.rename(columns={target_col: 'y'}, inplace=True)

        if target_col in ['Radiation', 'Rain_Amount', 'Wind_Speed']: validation_results['yhat'] = validation_results['yhat'].clip(lower=0)
        if target_col in WIND_DIR_TARGETS: validation_results['yhat'] = validation_results['yhat'].clip(lower=-1, upper=1)

        validation_results.dropna(subset=['y', 'yhat'], inplace=True)
        if not validation_results.empty: return mean_absolute_error(validation_results['y'], validation_results['yhat'])
        else: return float('inf')
    except Exception as e: return float('inf')

# --- Main Forecasting Logic ---
start_time = time.time()
validation_maes = {}
best_hyperparams = {}

print("Loading data...")
try:
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
except FileNotFoundError as e: print(f"Error: {e}."); exit()
test_ids = test_df['ID'].copy()

train_processed = preprocess_data(train_df, is_train=True)
test_processed = preprocess_data(test_df, is_train=False)

submission_df = test_processed[['ID', 'kingdom']].copy()
for target in TARGETS_FOR_MODELING + ['Wind_Direction']: submission_df[target] = np.nan

unique_kingdoms = train_processed['kingdom'].unique()
print(f"\nFound {len(unique_kingdoms)} unique kingdoms.")

total_models = len(unique_kingdoms) * len(TARGETS_FOR_MODELING)
model_count = 0

for kingdom in unique_kingdoms:
    print(f"\n--- Processing Kingdom: {kingdom} ---")
    kingdom_train_data_full = train_processed[train_processed['kingdom'] == kingdom].sort_values('ds').copy()
    kingdom_test_data = test_processed[test_processed['kingdom'] == kingdom].copy()

    if kingdom_train_data_full.empty: continue
    n_train_full = len(kingdom_train_data_full)
    perform_validation = n_train_full >= (MIN_VALIDATION_POINTS / (1 - VALIDATION_SPLIT_PERCENTAGE))
    if not perform_validation: print(f"Warning: Insufficient data ({n_train_full}) for validation split after feature eng.")
    if kingdom_test_data.empty: print(f"Warning: No test data found.")
    future_df_final = kingdom_test_data[['ds']].drop_duplicates().sort_values('ds')
    if future_df_final.empty and not kingdom_test_data.empty: print(f"Warning: No valid future dates found.")

    print(f"Full training data shape: {kingdom_train_data_full.shape}")
    if not future_df_final.empty: print(f"Future dates to predict: {len(future_df_final)}")

    all_available_columns = kingdom_train_data_full.columns.tolist()
    all_available_non_geo = [r for r in NON_GEO_REGRESSORS if r in all_available_columns]
    all_available_geo = [r for r in GEO_REGRESSORS if r in all_available_columns]
    all_available_lags = [r for r in all_available_columns if r.startswith(LAG_PREFIX)]
    all_available_rolls = [r for r in all_available_columns if r.startswith(ROLL_PREFIX)]

    for target in TARGETS_FOR_MODELING:
        model_count += 1
        print(f"\n({model_count}/{total_models}) Processing & Optimizing: {kingdom} - {target}")

        if target not in kingdom_train_data_full.columns:
             print(f"  Warning: Target column '{target}' not found. Skipping.")
             validation_maes[(kingdom, target)] = np.nan; best_hyperparams[(kingdom, target)] = None; continue

        exclude_prefix_lag = f"{LAG_PREFIX}{target}_lag"
        exclude_prefix_roll = f"{ROLL_PREFIX}{target}_w"

        valid_regressors_non_geo_geo = [r for r in all_available_non_geo + all_available_geo if r != target]
        valid_regressors_lag = [r for r in all_available_lags if not r.startswith(exclude_prefix_lag)]
        valid_regressors_roll = [r for r in all_available_rolls if not r.startswith(exclude_prefix_roll)]

        current_best_params = None; validation_mae = np.nan
        if perform_validation:
            print(f"  Running Optuna ({N_OPTUNA_TRIALS} trials)...")
            split_index = int(n_train_full * VALIDATION_SPLIT_PERCENTAGE)
            train_split_df = kingdom_train_data_full.iloc[:split_index].copy()
            validation_df = kingdom_train_data_full.iloc[split_index:].copy()

            if len(validation_df) < MIN_VALIDATION_POINTS:
                 print(f"  Warning: Validation set too small ({len(validation_df)}). Skipping optimization.")
            else:
                try:
                    study = optuna.create_study(direction='minimize')
                    is_multiplicative = target in MULTIPLICATIVE_VARS
                    study.optimize(
                        lambda trial: objective(trial, train_split_df, validation_df, target,
                                                valid_regressors_non_geo_geo, valid_regressors_lag, valid_regressors_roll,
                                                is_multiplicative),
                        n_trials=N_OPTUNA_TRIALS, timeout=OPTUNA_TIMEOUT, show_progress_bar=True
                    )
                    if study.best_trial:
                         current_best_params = study.best_trial.params
                         validation_mae = study.best_trial.value
                         print(f"  Optuna Best Validation MAE: {validation_mae:.5f}")
                    else: print(" Optuna study finished without a successful best trial."); current_best_params = None
                except Exception as e: print(f"  Optuna study failed: {e}"); current_best_params = None

            validation_maes[(kingdom, target)] = validation_mae
            best_hyperparams[(kingdom, target)] = current_best_params
        else:
             validation_maes[(kingdom, target)] = np.nan; best_hyperparams[(kingdom, target)] = None

        print(f"  Training final model...")
        prophet_train_full = kingdom_train_data_full[['ds', target]].rename(columns={target: 'y'})
        prophet_train_full['y'] = clip_outliers(prophet_train_full['y'], OUTLIER_CLIP_PERCENTILE, 1 - OUTLIER_CLIP_PERCENTILE)
        if len(prophet_train_full) < 2: continue

        final_params = current_best_params if current_best_params else {}
        seasonality_mode = 'multiplicative' if target in MULTIPLICATIVE_VARS else 'additive'

        model_final = Prophet(
            changepoint_prior_scale=final_params.get('changepoint_prior_scale', 0.1),
            seasonality_prior_scale=final_params.get('seasonality_prior_scale', 10.0),
            n_changepoints=final_params.get('n_changepoints', 25),
            yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False,
            seasonality_mode=seasonality_mode,
        )
        model_final.add_seasonality(name='yearly', period=365.25, fourier_order=final_params.get('yearly_fourier_order', 10))
        model_final.add_seasonality(name='weekly', period=7, fourier_order=final_params.get('weekly_fourier_order', 3))
        model_final.add_seasonality(name='monthly', period=30.5, fourier_order=final_params.get('monthly_fourier_order', 5))
        model_final.add_seasonality(name='quarterly', period=365.25/4, fourier_order=final_params.get('quarterly_fourier_order', 5))

        try: model_final.fit(prophet_train_full)
        except Exception as e: print(f"  Error fitting FINAL model: {e}"); continue
        if future_df_final.empty: continue

        print(f"  Predicting on {len(future_df_final)} test dates...")
        try:
            future_predict_df = future_df_final[['ds']].copy()
            forecast_final = model_final.predict(future_predict_df)
            if target in ['Radiation', 'Rain_Amount', 'Wind_Speed']: forecast_final['yhat'] = forecast_final['yhat'].clip(lower=0)
            if target in WIND_DIR_TARGETS: forecast_final['yhat'] = forecast_final['yhat'].clip(lower=-1, upper=1)
        except Exception as e: print(f"  Error predicting FINAL: {e}"); continue

        predictions_for_target = forecast_final[['ds', 'yhat']].copy()
        kingdom_submission_slice = kingdom_test_data[['ID', 'ds']].merge(predictions_for_target, on='ds', how='left')
        submission_df[target] = submission_df['ID'].map(kingdom_submission_slice.set_index('ID')['yhat']).fillna(submission_df[target])

        print(f"  Final prediction complete for: {kingdom} - {target}")

print("\nConverting Wind Direction sin/cos predictions back to degrees...")
if 'Wind_Direction_sin' in submission_df.columns and 'Wind_Direction_cos' in submission_df.columns:
    sin_comp = pd.to_numeric(submission_df['Wind_Direction_sin'], errors='coerce')
    cos_comp = pd.to_numeric(submission_df['Wind_Direction_cos'], errors='coerce')
    valid_mask = sin_comp.notna() & cos_comp.notna()
    radians = np.full(len(submission_df), np.nan)
    radians[valid_mask] = np.arctan2(sin_comp[valid_mask], cos_comp[valid_mask])
    degrees = np.degrees(radians)
    submission_df['Wind_Direction'] = (degrees + 360) % 360
    submission_df.drop(columns=['Wind_Direction_sin', 'Wind_Direction_cos'], inplace=True, errors='ignore')
else:
    print("Warning: Wind_Direction_sin/cos columns not found for conversion.")
    if 'Wind_Direction' not in submission_df.columns: submission_df['Wind_Direction'] = np.nan


print("\n--- Optuna Best Validation MAE Summary (Adv Feats) ---")
valid_maes = {k: v for k, v in validation_maes.items() if pd.notna(v)}
if valid_maes:
    mae_summary = pd.Series(valid_maes).unstack()
    print(mae_summary.to_string(float_format="%.5f", na_rep='NaN'))
    print("\nAverage Best MAE per Target Variable:")
    print(mae_summary.mean(axis=0).round(5))
    print("\nAverage Best MAE per Kingdom:")
    print(mae_summary.mean(axis=1).round(5))
    overall_mae = pd.Series(valid_maes).mean()
    if pd.notna(overall_mae): print(f"\nOverall Average Best MAE: {overall_mae:.5f}")
    else: print("\nOverall Average Best MAE: N/A")
else: print("No valid validation results to report.")
print("-" * 30)


print("\nFinalizing submission file...")
FINAL_SUBMISSION_COLUMNS = ['ID'] + BASE_TARGET_VARIABLES + ['Wind_Direction']
submission_final = submission_df[FINAL_SUBMISSION_COLUMNS].copy()

nan_counts_before_fill = submission_final[FINAL_SUBMISSION_COLUMNS[1:]].isnull().sum()
for target in FINAL_SUBMISSION_COLUMNS[1:]:
    submission_final[target] = pd.to_numeric(submission_final[target], errors='coerce').round(4)
    if target in ['Radiation', 'Rain_Amount', 'Wind_Speed']:
        submission_final[target] = submission_final[target].clip(lower=0)

submission_final.fillna(0, inplace=True)
if nan_counts_before_fill.sum() > 0: print(f"Warning: Filled {nan_counts_before_fill.sum()} NaN values with 0.")

submission_final['ID'] = submission_final['ID'].astype(int)
submission_final.sort_values('ID', inplace=True)

try:
    submission_final.to_csv(SUBMISSION_FILE, index=False)
    print(f"\nSubmission file saved successfully to '{SUBMISSION_FILE}'")
    print(submission_final.head())
except Exception as e: print(f"Error saving file: {e}")

end_time = time.time()
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")