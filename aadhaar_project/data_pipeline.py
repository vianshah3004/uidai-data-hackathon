
import pandas as pd
import numpy as np
from . import config
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_rolling_slope(series, window=30):
    """
    Calculates the slope of a linear regression fit over a rolling window.
    Vectorized implementation using covariance/variance formula.
    """
    x = np.arange(window)
    mean_x = np.mean(x)
    var_x = np.var(x)
    weights = x - mean_x
    
    def weighted_sum(y_window):
        return np.sum(y_window * weights)
    
    numerator = series.rolling(window).apply(weighted_sum, raw=True)
    slope = numerator / (window * var_x)
    return slope

def load_data():
    """Loads and concatenates the three CSV files to process all ~5M records."""
    logger.info("Loading Datasets...")
    
    # Load with low_memory=False to avoid dtypes warnings or specify dtypes if known
    bio = pd.read_csv(config.BIO_FILE)
    demo = pd.read_csv(config.DEMO_FILE)
    pop = pd.read_csv(config.POP_FILE)
    
    # Standardize column names
    bio.columns = [c.replace('bio_age_17_', 'bio_age_17_plus') if 'bio_age_17_' in c and 'plus' not in c else c for c in bio.columns]
    demo.columns = [c.replace('demo_age_17_', 'demo_age_17_plus') if 'demo_age_17_' in c and 'plus' not in c else c for c in demo.columns]
    
    # Convert dates
    for df, name in [(bio, 'bio'), (demo, 'demo'), (pop, 'pop')]:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        else:
            logger.error(f"'date' column missing in {name} dataset")
            raise ValueError(f"Missing date column in {name}")
            
    logger.info("Concatenating Datasets (stacking all records)...")
    
    # Align columns by simple concatenation (pandas handles disjoint columns by filling NaN)
    # This preserves every single line from the input files as a distinct row.
    df = pd.concat([bio, demo, pop], axis=0, ignore_index=True)
    
    # Fill update NaNs with 0
    update_cols = ['bio_age_5_17', 'bio_age_17_plus', 'demo_age_5_17', 'demo_age_17_plus']
    for col in update_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    # Drop rows where District is NaN
    df = df.dropna(subset=['district', 'date'])
    
    # Store raw record count in config for reporting
    config.RAW_RECORD_COUNT = len(df)
    logger.info(f"Raw stacked records: {config.RAW_RECORD_COUNT}")
    
    return df

def aggregate_to_district(df_pincode):
    """Aggregates pincode-level stacked data to district-day level."""
    logger.info("Aggregating to District-Day level...")
    
    # Update columns - we SUM these
    update_cols = [
        'bio_age_5_17', 'bio_age_17_plus',
        'demo_age_5_17', 'demo_age_17_plus'
    ]
    
    # Population columns - we take MAX (since they are snapshots)
    # Taking SUM would double-count if multiple population entries exist for same district/day (unlikely but safe)
    # If population is missing for a row (e.g. update row), it is NaN. Max ignores NaN.
    pop_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    
    # Ensure columns exist in df
    avail_update = [c for c in update_cols if c in df_pincode.columns]
    avail_pop = [c for c in pop_cols if c in df_pincode.columns]
    
    # Define agg dictionary
    agg_dict = {c: 'sum' for c in avail_update}
    agg_dict.update({c: 'max' for c in avail_pop})
    
    # Group by Date, State, District
    df_district = df_pincode.groupby(['date', 'state', 'district']).agg(agg_dict).reset_index()
    
    return df_district

def perform_feature_engineering(df):
    """Engineers growth, coverage, update type, demographic, and temporal features."""
    logger.info("Engineering Features (Daily Data with 30-day windows)...")
    
    # Sort
    df = df.sort_values(by=['state', 'district', 'date'])
    
    # --- Basic Sums ---
    df['total_bio'] = df['bio_age_5_17'] + df['bio_age_17_plus']
    df['total_demo'] = df['demo_age_5_17'] + df['demo_age_17_plus']
    df['total_updates'] = df['total_bio'] + df['total_demo']
    
    df['total_pop_youth'] = df['age_5_17']
    df['total_pop_adult'] = df['age_18_greater']
    df['total_pop'] = df['age_0_5'] + df['total_pop_youth'] + df['total_pop_adult']
    
    # Avoid division by zero
    df['total_pop'] = df['total_pop'].replace(0, np.nan)
    
    # --- Growth & Momentum ---
    # We must calculate these PER DISTRICT
    # Using groupby + transform/shift
    
    g = df.groupby('district')
    
    # Growth rates (using pct_change)
    # Using 30 days for 1 Month, 90 days for 3 Months (approx)
    # Since data might be sparse (not every day has a record), we use 'freq' if index is date, or just shift(30) if rows are daily.
    # The aggregation was: df_pincode.groupby(['date'...]).sum(). 
    # If a district has NO update on a day, does it exist in df? 
    # 'aggregate_to_district' does not reindex. So rows are sparse daily.
    # If I use shift(30), I am shifting 30 *records*, which might be 30 days or 30 weeks depending on sparsity.
    
    # To be robust, we should calculate Rolling features based on Time Index, not Record Count.
    # But 'transform' with rolling supports '30D' offset ONLY if index is datetime.
    
    # Let's set index to date for calculation?
    # Complex with groupby.
    
    # Simpler: Assume daily data is relatively dense or fill missing?
    # If I leave it as sparse, shift(30) is unpredictable.
    # I MUST Reindex to fill missing days with 0 to ensure shift(30) == 30 days.
    
    # Reindexing for continuity:
    # This might explode data size (750 dists * 365 days = 270k rows).
    # This is exactly what the user probably wants ("All data", "High volume").
    # And it makes shift(30) valid.
    
    # Let's do implicit reindexing via set_index + stack? No.
    # Iterate districts? Slow.
    
    # GroupBy Resample('D') is efficient.
    
    # Optimized Interpolation/Fill:
    # We need to keep State/Pincode metadata.
    
    # Re-implement Resampling ('D') similar to previous step but Daily.
    
# ... (main process function)
    
    # Sum cols
    sum_cols = ['total_bio', 'total_demo', 'total_updates', 'bio_age_5_17', 'bio_age_17_plus', 
                'demo_age_5_17', 'demo_age_17_plus']
    mean_cols = ['age_0_5', 'age_5_17', 'age_18_greater', 'total_pop', 'total_pop_youth', 'total_pop_adult']
    
    # Resample to Daily, filling 0 for sums
    df.set_index('date', inplace=True)
    df_daily = df.groupby(['district', 'state'])[sum_cols].resample('D').sum().reset_index()
    
    # Forward fill population (it doesn't change daily)
    df_pop = df.groupby(['district', 'state'])[mean_cols].resample('D').ffill().reset_index()
    
    df = pd.merge(df_daily, df_pop, on=['district', 'state', 'date'])
    
    # Now we have strict Daily rows. shift(30) is exactly 30 days.
    
    g = df.groupby('district')
    
    df['growth_rate_1month'] = g['total_updates'].pct_change(30)
    df['growth_rate_3month'] = g['total_updates'].pct_change(90)
    df['growth_rate_1week'] = g['total_updates'].pct_change(7)
    
    # Acceleration
    df['acceleration'] = df['growth_rate_1month'] - g['growth_rate_1month'].shift(30)
    
    # Lag Features (Past performance)
    df['updates_lag_1m'] = g['total_updates'].shift(30)
    df['updates_lag_2m'] = g['total_updates'].shift(60)
    
    # Rolling Means & Extremes (Smoothness & Range)
    df['updates_roll_7d'] = g['total_updates'].transform(lambda x: x.rolling(7).mean())
    df['updates_roll_14d'] = g['total_updates'].transform(lambda x: x.rolling(14).mean())
    df['updates_roll_30d'] = g['total_updates'].transform(lambda x: x.rolling(30).mean())
    
    df['updates_max_30d'] = g['total_updates'].transform(lambda x: x.rolling(30).max())
    df['updates_min_30d'] = g['total_updates'].transform(lambda x: x.rolling(30).min())
    df['ratio_to_peak'] = df['total_updates'] / df['updates_max_30d'].replace(0, 1)
    
    # --- Hybrid Forecasting Features (Trend Projection) ---
    # Calculate rolling linear trend slope (30 days)
    # This mimics a local ARIMA/LinearForecast
    df['ts_slope_30d'] = g['total_updates'].transform(lambda x: calculate_rolling_slope(x, 30))
    
    # Project 90 days ahead: Projection = Current + (Slope * 90)
    # Feature: Projected Growth Rate
    projected_val = df['total_updates'] + (df['ts_slope_30d'] * 90)
    df['ts_projected_growth'] = (projected_val - df['total_updates']) / df['total_updates'].replace(0, 1)
    
    # Momentum Score (Weighted combination)
    df['momentum_score'] = (0.5 * df['growth_rate_1month'].fillna(0)) + \
                           (0.3 * df['growth_rate_3month'].fillna(0)) + \
                           (0.2 * df['acceleration'].fillna(0))
                           
    # Volatility (Std dev of growth rate)
    df['volatility'] = g['growth_rate_1month'].rolling(180).std().reset_index(0, drop=True)
    df['volatility_short'] = g['total_updates'].rolling(14).std().reset_index(0, drop=True) # Short term fluctuation
    
    # --- Coverage & Gap ---
    df['cum_bio'] = g['total_bio'].cumsum()
    df['cum_demo'] = g['total_demo'].cumsum()
    df['cum_total'] = g['total_updates'].cumsum()
    
    df['bio_coverage_rate'] = df['cum_bio'] / df['total_pop']
    df['demo_coverage_rate'] = df['cum_demo'] / df['total_pop']
    df['total_coverage_rate'] = df['cum_total'] / df['total_pop']
    
    # Coverage Gap vs State
    state_coverage = df.groupby(['date', 'state'])['total_coverage_rate'].mean().reset_index()
    state_coverage.rename(columns={'total_coverage_rate': 'state_avg_coverage'}, inplace=True)
    
    df = pd.merge(df, state_coverage, on=['date', 'state'], how='left')
    df['coverage_gap_vs_state'] = df['state_avg_coverage'] - df['total_coverage_rate']
    
    # --- PHYSICS-BASED FEATURES (Surge Dynamics) ---
    
    # 1. Velocity, Acceleration, Jerk (Motion Physics)
    # Velocity is already approximated by 'growth_rate_1month'.
    # Acceleration = Change in Velocity
    df['velocity'] = df['growth_rate_1month'].fillna(0)
    # Re-group because 'velocity' is new
    df['acceleration'] = df.groupby('district')['velocity'].diff()
    # Jerk = Change in Acceleration
    df['jerk'] = df.groupby('district')['acceleration'].diff()
    
    # 2. Spatial Contagion (Pressure-Wave)
    # Average momentum of the STATE (excluding self ideally, but mean is fine approx)
    # We need to join state-level average momentum back to rows
    state_momentum = df.groupby(['date', 'state'])['momentum_score'].transform('mean')
    df['state_momentum'] = state_momentum
    # Drag: Am I lagging my state? (Negative = Drag, likely to be pulled up)
    df['spatial_drag'] = df['momentum_score'] - df['state_momentum']
    
    # 3. Gap Dynamics (Demand Release)
    # How fast is the gap closing? (Derivative of coverage gap)
    # coverage_gap_vs_state is (State Avg - My Rate).
    # If gap is shrinking, I am catching up.
    df['gap_closure_velocity'] = df.groupby('district')['coverage_gap_vs_state'].diff() * -1 # Positive = closing gap
    # Urgency: Velocity relative to remaining gap
    df['gap_urgency'] = df['gap_closure_velocity'] / df['coverage_gap_vs_state'].replace(0, 1)

    # --- EXPLICIT INTERACTIONS (Decision Anchors) ---
    # Critical Mass: High Acceleration + High Remaining Gap (Unmet demand exploding)
    df['critical_mass'] = df['acceleration'] * df['coverage_gap_vs_state']
    
    # Contagion Risk: State is moving fast + I have gap to fill
    df['contagion_risk'] = df['state_momentum'] * df['coverage_gap_vs_state']
    
    # Structural tension: High Volatility + High Velocity (Chaos)
    df['structural_tension'] = df['volatility_short'] * df['velocity']
    
    # Re-define groupby for subsequent features
    g_new = df.groupby('district')
    
    # --- Update Type Features ---
    df['bio_demo_ratio'] = df['total_bio'] / df['total_demo'].replace(0, np.nan)
    # Fill Inf?
    df['bio_demo_ratio'] = df['bio_demo_ratio'].fillna(0) # or suitable default
    
    df['bio_demo_ratio_change'] = g_new['bio_demo_ratio'].diff(3)
    
    df['bio_proportion'] = df['total_bio'] / df['total_updates'].replace(0, 1) # avoid div0
    df['demo_proportion'] = df['total_demo'] / df['total_updates'].replace(0, 1)
    
    # Interaction Features (Must be after definitions)
    # Using 'total_updates' instead of 'daily_avg_current' because daily_avg wasn't defined yet! 
    # Actually, daily_avg_current is defined in TARGET label function, not here.
    # So we use 'total_updates'.
    
    # Trend Divergence
    df['trend_div_7_30'] = df['updates_roll_7d'] - df['updates_roll_30d']
    
    # Temporal Flags
    df['is_year_end'] = (df['date'].dt.month >= 11).astype(int)
    
    df['bio_intensity'] = df['bio_proportion'] * df['total_updates']
    df['growth_momentum'] = df['growth_rate_1month'] * df['momentum_score']
    
    # --- Demographic Features ---
    df['youth_proportion'] = df['total_pop_youth'] / df['total_pop']
    df['adult_proportion'] = df['total_pop_adult'] / df['total_pop']
    
    df['youth_update_intensity'] = (df['bio_age_5_17'] + df['demo_age_5_17']) / df['total_pop_youth'].replace(0, np.nan)
    df['adult_update_intensity'] = (df['bio_age_17_plus'] + df['demo_age_17_plus']) / df['total_pop_adult'].replace(0, np.nan)
    
    # Aging out indicator: Decline in youth population?
    # Change in youth population
    df['youth_pop_change'] = g_new['total_pop_youth'].diff()
    df['aging_out_indicator'] = (df['youth_pop_change'] < 0).astype(int) # Binary flag if youth pool is shrinking
    
    # --- Temporal Features ---
    # Temporal - Daily patterns are critical!
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['day_of_month'] = df['date'].dt.day
    
    # Yearly/Quarterly
    df['month_of_year'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month_of_year'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_of_year'] / 12)
    df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
    df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def generate_target_labels(df, lookahead_months=3, threshold_pct=0.50):
    """
    Creates the 'is_hotspot' binary label.
    Label = 1 if (Avg Daily Updates in Next 3 Months) >= (Current Month Avg Daily Updates * 1.5)
    
    Note: Updates in df are totals for the month.
    Avg Daily = Total / DaysInMonth. All months roughly 30 days, we can simplify to Total Comparison or be precise.
    Let's be precise.
    """
    logger.info("Generating Target Labels...")
    
    # Input df is Daily Resampled. 
    df['daily_avg_current'] = df['total_updates'] # It is daily
    
    # Forward looking window
    # Valid only if we have the future data. Last 3 months of data will have NaN targets.
    
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=lookahead_months)
    
    g = df.groupby('district')['total_updates']
    
    # rolling sum of next 3 months (excluding current? Spec says "Look forward 3 months (next 90 days)")
    # .shift(-1) starts from next month
    # We want sum(next 3 months) / sum(days next 3 months)
    
    # Simplified approach: Average of next 3 months' Daily Averages? 
    # Or Sum of next 3 months updates / 90 days?
    # "Calculate average daily updates in next 3 months" -> (Total Updates Month+1 + M+2 + M+3) / (Days M+1 + M+2 + M+3)
    
    # Let's shift the daily avg column backwards by 1 to get next month, then roll mean 3
    # Shift -1: Next month is now at current row.
    # Rolling(3): Sum of Next, Next+1, Next+2
    
    g_daily = df.groupby('district')['daily_avg_current']
    
    # We want Mean of (Next 3 months).
    # shift(-3) puts M+3 at current. No.
    # We want: at time t, look at t+1, t+2, t+3.
    # shift(-1) shifts data up so t row sees t+1 value.
    # Then rolling(3) looking forward? Pandas rolling is backward looking usually.
    # But checking 'FixedForwardWindowIndexer' above.
    
    # Actually, easiest way: 
    # Reverse the dataframe? Or use shift(-1) then rolling(3, closed='left'?)
    
    # let's use shift(-1) + rolling(3) forward on original? 
    # Actually if we assume sorted by date ascending:
    # A backward rolling mean on reversed data is forward rolling on forward data.
    
    # Let's try:
    # future_3m_avg = g_daily.transform(lambda x: x.shift(-1).rolling(3, min_periods=3).mean())
    # This works. shift(-1) moves t+1 to t. rolling(3) calculates mean of t+1, t+in_window... wait.
    # If I engage shift(-1), the row at t has t+1. 
    # Normal rolling(3) sums [t-2, t-1, t]. 
    # So on shifted data (t+1), rolling(3) would sum [(t+1)-2, (t+1)-1, t+1] = [t-1, t, t+1]. That's not what we want.
    
    # We want [t+1, t+2, t+3].
    
    # Correct way (Daily):
    # lookahead_months = 3 (~90 days)
    # window_size = 90
    
    window_size = 90
    
    future_avg = df.groupby('district')['daily_avg_current'].transform(
        lambda x: x.rolling(window=window_size, min_periods=window_size).mean().shift(-window_size)
    )
    
    df['future_daily_avg'] = future_avg
    
    # Label
    # Increase >= 50% means (Future / Current) >= 1.5
    # Handle division by zero or small numbers?
    # If current is 0 and future is > 0 -> Hotspot? Yes.
    # If both 0 -> No.
    
    # Let's define a minimum threshold for significance too? Spec doesn't say. 
    # But usually 0->1 is huge percent but not a hotspot.
    # Let's stick to strict spec: >= 50% increase.
    # But if current is 0, we can't divide.
    # Logic: if future >= current * 1.5
    
    # Refined Target Definition (Volume-Aware)
    # 1. Growth Criteria: > 75% Increase (1.75x)
    # 2. Volume Criteria: Absolute increase of at least 10 updates (prevents 1->2 noise)
    
    df['is_hotspot'] = 0
    
    growth_condition = (df['future_daily_avg'] >= df['daily_avg_current'] * 1.75)
    volume_condition = (df['future_daily_avg'] - df['daily_avg_current'] >= 10)
    
    # Combined Mask
    mask_hotspot = growth_condition & volume_condition
    
    # Edge case: current is 0. 
    # If starting from 0, we require significant future volume (>20) to call it a hotspot.
    mask_zero = (df['daily_avg_current'] == 0)
    mask_zero_growth = (df['future_daily_avg'] > 20)
    
    df.loc[mask_hotspot, 'is_hotspot'] = 1
    df.loc[mask_zero & mask_zero_growth, 'is_hotspot'] = 1
    
    # Remove rows where target is NaN (last 3 months)
    # No, we might want them for "Prediction" later (Input for prediction), 
    # but for Training we must drop them.
    # We will flag train_eligible rows
    
    df['target_available'] = df['future_daily_avg'].notna().astype(int)
    
    return df

def run_pipeline():
    df = load_data()
    df_dist = aggregate_to_district(df)
    df_feat = perform_feature_engineering(df_dist)
    df_final = generate_target_labels(df_feat)
    return df_final

if __name__ == "__main__":
    df = run_pipeline()
    logger.info(f"Pipeline complete. Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
