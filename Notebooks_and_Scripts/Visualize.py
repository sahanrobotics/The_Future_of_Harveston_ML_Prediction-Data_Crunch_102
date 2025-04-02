import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging

# --- Configuration & Setup ---
warnings.filterwarnings('ignore', category=FutureWarning)
# logging.getLogger('prophet').setLevel(logging.ERROR)
# logging.getLogger('cmdstanpy').setLevel(logging.ERROR)

# Define file paths
TRAIN_FILE = 'train.csv'

# Define a base year (consistent with previous preprocessing)
BASE_YEAR = 2000

# Variables to focus on for visualization
VIZ_TARGETS_RAW = ['Avg_Temperature', 'Avg_Feels_Like_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction']
VIZ_TARGETS_PROC = ['Avg_Temperature', 'Avg_Feels_Like_Temperature', 'Radiation', 'Rain_Amount', 'Wind_Speed', 'Wind_Direction_sin', 'Wind_Direction_cos']
VIZ_REGRESSORS = ['Temperature_Range', 'Feels_Like_Temperature_Range', 'Rain_Duration']

# Number of rows to sample for plotting (optional, use None for all)
N_SAMPLE_POINTS = 50000 # Adjust as needed, or set to None

# --- Helper Functions ---
def kelvin_to_celsius(temp_k):
    """Converts Kelvin to Celsius."""
    return temp_k - 273.15

def preprocess_data_for_viz(df):
    """
    Cleans and preprocesses the dataframe for visualization purposes.
    Keeps original columns needed for comparison alongside processed ones.
    Focuses on conversions and essential cleaning, not dropping regressors/features yet.
    """
    df_processed = df.copy()
    print("Preprocessing data for visualization...")

    df_processed['Year'] = pd.to_numeric(df_processed['Year'], errors='coerce')
    df_processed.dropna(subset=['Year'], inplace=True)
    df_processed['Year'] = df_processed['Year'].astype(int) + BASE_YEAR - 1

    print("Creating datetime column ('ds')...")
    df_processed['ds'] = pd.to_datetime(
        df_processed[['Year', 'Month', 'Day']], errors='coerce'
    )
    initial_rows = len(df_processed)
    df_processed.dropna(subset=['ds'], inplace=True)
    dropped_rows = initial_rows - len(df_processed)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to invalid dates during ds creation.")

    print("Converting temperatures to Celsius...")
    if 'Avg_Temperature' in df_processed.columns:
        df_processed['Avg_Temperature_orig'] = df_processed['Avg_Temperature']
        temp_numeric = pd.to_numeric(df_processed['Avg_Temperature'], errors='coerce')
        kelvin_mask = temp_numeric > 150
        df_processed['Avg_Temperature'] = np.where(
            kelvin_mask & temp_numeric.notna(),
            temp_numeric - 273.15,
            temp_numeric
        )
    if 'Avg_Feels_Like_Temperature' in df_processed.columns:
        df_processed['Avg_Feels_Like_Temperature_orig'] = df_processed['Avg_Feels_Like_Temperature']
        temp_numeric = pd.to_numeric(df_processed['Avg_Feels_Like_Temperature'], errors='coerce')
        kelvin_mask = temp_numeric > 150
        df_processed['Avg_Feels_Like_Temperature'] = np.where(
            kelvin_mask & temp_numeric.notna(),
            temp_numeric - 273.15,
            temp_numeric
        )

    if 'Wind_Direction' in df_processed.columns:
        print("Creating Wind_Direction sin/cos components...")
        wd_numeric = pd.to_numeric(df_processed['Wind_Direction'], errors='coerce')
        wd_rad = np.radians(wd_numeric)
        df_processed['Wind_Direction_sin'] = np.sin(wd_rad)
        df_processed['Wind_Direction_cos'] = np.cos(wd_rad)

    print("Ensuring key columns are numeric...")
    cols_to_numeric = (VIZ_TARGETS_RAW + VIZ_REGRESSORS +
                       ['latitude', 'longitude', 'Evapotranspiration'])
    for col in cols_to_numeric:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("\nNaN counts per column after preprocessing steps:")
    print(df_processed.isnull().sum())
    print("-" * 30)

    df_processed.sort_values('ds', inplace=True)

    print("Preprocessing for visualization complete.")
    print("-" * 30)
    return df_processed

# --- Main Script ---
print("Loading raw training data...")
try:
    raw_train_df = pd.read_csv(TRAIN_FILE)
    print(f"Raw data loaded: {raw_train_df.shape}")
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure '{TRAIN_FILE}' is in the script directory.")
    exit()

df_to_process = raw_train_df.copy()

processed_df = preprocess_data_for_viz(df_to_process)

print("Adding 'ds' column to original raw data for plotting...")
raw_train_df['Year_temp'] = pd.to_numeric(raw_train_df['Year'], errors='coerce')
raw_train_df['Year_temp'] = raw_train_df['Year_temp'].fillna(0).astype(int) + BASE_YEAR - 1

raw_train_df.rename(columns={'Year_temp': 'year'}, inplace=True)

raw_train_df['ds'] = pd.to_datetime(
    raw_train_df[['year', 'Month', 'Day']], errors='coerce'
)

raw_train_df.drop(columns=['year'], inplace=True)

raw_train_df_plot = raw_train_df.dropna(subset=['ds']).sort_values('ds')

if N_SAMPLE_POINTS is not None and N_SAMPLE_POINTS < len(raw_train_df_plot):
    print(f"\nSampling {N_SAMPLE_POINTS} points for faster plotting...")
    plot_raw_df = raw_train_df_plot.sample(n=N_SAMPLE_POINTS, random_state=42).sort_values('ds')
    plot_processed_df = processed_df.sample(n=N_SAMPLE_POINTS, random_state=42).sort_values('ds')
else:
    plot_raw_df = raw_train_df_plot
    plot_processed_df = processed_df

print(f"Using {len(plot_raw_df)} points for plots.")
print("-" * 30)

# --- Visualization ---

print("Generating histograms for raw data...")
cols_for_raw_hist = [col for col in VIZ_TARGETS_RAW + VIZ_REGRESSORS if col in plot_raw_df.columns]
plot_raw_df[cols_for_raw_hist] = plot_raw_df[cols_for_raw_hist].apply(pd.to_numeric, errors='coerce')

n_cols_raw = len(cols_for_raw_hist)
n_rows_raw = (n_cols_raw + 2) // 3
plt.figure(figsize=(15, n_rows_raw * 4))
for i, col in enumerate(cols_for_raw_hist):
    plt.subplot(n_rows_raw, 3, i + 1)
    data_to_plot = plot_raw_df[col].dropna()
    if not data_to_plot.empty:
         sns.histplot(data_to_plot, kde=False)
         plt.title(f'Raw {col}')
    else:
         plt.title(f'Raw {col} (No Data)')
         plt.text(0.5, 0.5, 'No Valid Data', ha='center', va='center')
         plt.xticks([])
         plt.yticks([])

plt.suptitle('Histograms of Raw Training Data Features', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("Generating time series plots for raw data...")
n_rows_ts_raw = (n_cols_raw + 1) // 2
plt.figure(figsize=(18, n_rows_ts_raw * 5))
for i, col in enumerate(cols_for_raw_hist):
    plt.subplot(n_rows_ts_raw, 2, i + 1)
    data_to_plot = plot_raw_df[[ 'ds', col]].dropna()
    if not data_to_plot.empty:
        plt.plot(data_to_plot['ds'], data_to_plot[col], marker='.', linestyle='none', markersize=1, alpha=0.6)
        plt.title(f'Raw {col} over Time')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.xticks(rotation=45)
    else:
         plt.title(f'Raw {col} over Time (No Data)')
         plt.text(0.5, 0.5, 'No Valid Data', ha='center', va='center')
         plt.xticks([])
         plt.yticks([])

plt.suptitle('Time Series of Raw Training Data Features', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


print("Generating histograms for processed data...")
cols_for_proc_hist = [col for col in VIZ_TARGETS_PROC + VIZ_REGRESSORS if col in plot_processed_df.columns]
plot_processed_df[cols_for_proc_hist] = plot_processed_df[cols_for_proc_hist].apply(pd.to_numeric, errors='coerce')

n_cols_proc = len(cols_for_proc_hist)
n_rows_proc = (n_cols_proc + 2) // 3
plt.figure(figsize=(15, n_rows_proc * 4))
for i, col in enumerate(cols_for_proc_hist):
    plt.subplot(n_rows_proc, 3, i + 1)
    data_to_plot = plot_processed_df[col].dropna()
    if not data_to_plot.empty:
        sns.histplot(data_to_plot, kde=False)
        plt.title(f'Processed {col}')
    else:
        plt.title(f'Processed {col} (No Data)')
        plt.text(0.5, 0.5, 'No Valid Data', ha='center', va='center')
        plt.xticks([])
        plt.yticks([])

plt.suptitle('Histograms of Processed Training Data Features', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("Generating time series plots for processed data...")
n_rows_ts_proc = (n_cols_proc + 1) // 2
plt.figure(figsize=(18, n_rows_ts_proc * 5))
for i, col in enumerate(cols_for_proc_hist):
    plt.subplot(n_rows_ts_proc, 2, i + 1)
    data_to_plot = plot_processed_df[['ds', col]].dropna()
    if not data_to_plot.empty:
        plt.plot(data_to_plot['ds'], data_to_plot[col], marker='.', linestyle='none', markersize=1, alpha=0.6)
        plt.title(f'Processed {col} over Time')
        plt.xlabel('Date')
        plt.ylabel(col)
        plt.xticks(rotation=45)
    else:
         plt.title(f'Processed {col} over Time (No Data)')
         plt.text(0.5, 0.5, 'No Valid Data', ha='center', va='center')
         plt.xticks([])
         plt.yticks([])

plt.suptitle('Time Series of Processed Training Data Features', fontsize=16, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

print("\nVisualization script finished.")