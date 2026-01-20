
import pandas as pd
import numpy as np
from aadhaar_project import data_pipeline, config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_noise():
    # Load processed data (before feature engineering ideally, but we can reuse pipeline func partially)
    # We'll use run_pipeline but inspecting intermediate steps is harder.
    # Let's just load and aggregate using the pipeline functions.
    
    logger.info("Loading Data...")
    df_raw = data_pipeline.load_data()
    df_dist = data_pipeline.aggregate_to_district(df_raw)
    
    # Calculate total_updates manually as feature engineering does it later
    # Sum relevant columns
    cols = ['bio_age_5_17', 'bio_age_17_plus', 'demo_age_5_17', 'demo_age_17_plus']
    # Ensure they exist and fillna
    for c in cols:
        if c not in df_dist.columns:
            df_dist[c] = 0
            
    df_dist['total_updates'] = df_dist[cols].sum(axis=1)
    
    # Noise Check 1: Low Volume
    total_rows = len(df_dist)
    zeros = (df_dist['total_updates'] == 0).sum()
    low_vol_10 = (df_dist['total_updates'] < 10).sum()
    low_vol_50 = (df_dist['total_updates'] < 50).sum()
    
    print(f"\nNOISE ANALYSIS REPORT")
    print(f"=====================")
    print(f"Total Rows: {total_rows}")
    print(f"Zero Updates: {zeros} ({zeros/total_rows:.1%})")
    print(f"< 10 Updates: {low_vol_10} ({low_vol_10/total_rows:.1%}) --> Extreme Poisson Noise")
    print(f"< 50 Updates: {low_vol_50} ({low_vol_50/total_rows:.1%}) --> High Variance")
    
    # Noise Check 2: Volatility (Coefficient of Variation)
    # Group by district and calc CV
    g = df_dist.groupby('district')['total_updates']
    cv = g.std() / g.mean()
    
    high_cv = (cv > 1.0).sum()
    print(f"\nDistricts with CV > 1.0 (Extremely Volatile): {high_cv}/{len(cv)}")
    
    # Noise Check 3: Spikes (Are there 1-day anomalies?)
    # We define spike as > 5 * 30-day mean
    df_dist['roll_mean'] = g.transform(lambda x: x.rolling(30).mean())
    df_dist['spike_ratio'] = df_dist['total_updates'] / df_dist['roll_mean'].replace(0, 1)
    
    spikes = (df_dist['spike_ratio'] > 5).sum()
    print(f"Single Day Spikes (>5x Monthly Avg): {spikes} ({spikes/total_rows:.1%})")
    
    # Distribution of Labels
    # We need to see if our Label Logic handles this.
    # We can't easily reproduce the label logic here without running full pipeline,
    # but the Low Volume stat is the most critical.

if __name__ == "__main__":
    analyze_noise()
