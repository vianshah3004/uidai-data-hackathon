
import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib
from . import config, data_pipeline, model_engine

class PredictionSystem:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        self.stacker = None
        self.features = []
        self.threshold = 0.5
        self.risk_map = {}
        self.global_risk = 0.5
        
    def load_model(self):
        # Path to ensemble pickle
        ensemble_path = os.path.join(config.ARTIFACTS_DIR, "ensemble_model.pkl")
        
        if not os.path.exists(ensemble_path):
             # Fallback to old path/method or error
            if os.path.exists(config.MODEL_PATH):
                 # This path is mainly for backwards compatibility or single model runs
                 # But we strongly prefer ensemble now.
                 pass
            raise FileNotFoundError(f"Ensemble Model not found at {ensemble_path}. Train first.")
            
        payload = joblib.load(ensemble_path)
        
        self.xgb_model = payload.get('xgb')
        self.lgb_model = payload.get('lgb')
        self.cat_model = payload.get('cat')
        self.features = payload['features']
        self.threshold = payload.get('threshold', 0.5)
        self.risk_map = payload.get('risk_map', {})
        self.global_risk = payload.get('global_risk', 0.5)
        
    def generate_predictions(self):
        # Load fresh data
        df = data_pipeline.run_pipeline()
        
        # We want to predict for the latest month available for each district
        # Sort by date desc and take head(1) per district
        latest_df = df.sort_values('date', ascending=False).groupby('district').head(1).copy()
        
        # Apply Risk Map
        if self.risk_map:
            latest_df['district_risk'] = latest_df['district'].map(self.risk_map).fillna(self.global_risk)
        else:
            latest_df['district_risk'] = self.global_risk
        
        X = latest_df[self.features]
        
        # Predict Ensemble
        p1 = self.xgb_model.predict_proba(X)[:, 1]
        p2 = self.lgb_model.predict_proba(X)[:, 1]
        p3 = self.cat_model.predict_proba(X)[:, 1]
        
        probs = (p1 + p2 + p3) / 3.0
        
        # Categorize
        risks = []
        for p in probs:
            if p > 0.75: risks.append('CRITICAL')
            elif p > 0.50: risks.append('HIGH')
            elif p > 0.25: risks.append('MEDIUM')
            else: risks.append('LOW')
            
        latest_df['hotspot_probability'] = probs
        latest_df['risk_level'] = risks
        
        # Outputs
        self.save_ranked_list(latest_df)
        self.save_heatmap_data(latest_df)
        self.save_detailed_report(latest_df)
        
        return latest_df
        
    def save_ranked_list(self, df):
        # Top 20 districts by probability
        ranked = df.sort_values('hotspot_probability', ascending=False).head(20)
        cols = ['state', 'district', 'hotspot_probability', 'risk_level', 'momentum_score', 'growth_rate_1month', 'coverage_gap_vs_state']
        ranked[cols].to_csv(config.PRIORITY_LIST_PATH, index=False)
        print(f"Ranked list saved to {config.PRIORITY_LIST_PATH}")

    def save_heatmap_data(self, df):
        # Needs coordinates? Pincode centroids? 
        # Dataset doesn't have lat/long. We export State/District/Pincode/Risk.
        # User asked for "Export coordinates". If data doesn't have it, we can't invent it.
        # But we can output the district list for mapping.
        cols = ['state', 'district', 'hotspot_probability', 'risk_level']
        df[cols].to_csv(os.path.join(config.ARTIFACTS_DIR, "heatmap_data.csv"), index=False)

    def save_detailed_report(self, df):
        # Full dump
        df.to_csv(config.PREDICTIONS_PATH, index=False)
        
        # Alert System
        alerts = df[df['risk_level'].isin(['CRITICAL', 'HIGH'])]
        alerts_path = os.path.join(config.ARTIFACTS_DIR, "alerts.csv")
        alerts[['state', 'district', 'risk_level', 'hotspot_probability']].to_csv(alerts_path, index=False)
        print(f"Alerts generated: {len(alerts)} districts flagged.")

if __name__ == "__main__":
    sys = PredictionSystem()
    sys.load_model()
    sys.generate_predictions()
