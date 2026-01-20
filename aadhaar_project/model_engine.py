
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterSampler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
try:
    import shap
except ImportError:
    shap = None
import logging
import os
from . import config

logger = logging.getLogger(__name__)

class HotspotModel:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.cat_model = None
        self.stacker = None # Meta-learner
        self.threshold = 0.5 
        
    def split_data(self, df):
        # ... (Keep existing logic)
        df = df.sort_values(by='date')
        df = df[df['target_available'] == 1].copy()
        
        dates = np.sort(df['date'].unique())
        n_dates = len(dates)
        
        train_end_idx = int(n_dates * 0.80)
        val_end_idx = int(n_dates * 0.90)
        
        train_dates = dates[:train_end_idx]
        val_dates = dates[train_end_idx:val_end_idx]
        test_dates = dates[val_end_idx:]
        
        train_set = df[df['date'].isin(train_dates)].copy()
        val_set = df[df['date'].isin(val_dates)].copy()
        test_set = df[df['date'].isin(test_dates)].copy()
        
        logger.info(f"Split sizes: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
        return train_set, val_set, test_set

    def get_feature_cols(self, df):
        exclude = ['date', 'state', 'district', 'pincode', 'is_hotspot', 'target_available', 'future_daily_avg']
        features = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'int64', 'int32']]
        return features

    # ... (train method exists) ...

    def save_model(self, features, risk_map, global_risk):
        # Save all models and config in one pickle
        payload = {
            'xgb': self.xgb_model,
            'lgb': self.lgb_model,
            'cat': self.cat_model,
            'stacker': self.stacker,
            'features': features,
            'threshold': self.threshold,
            'risk_map': risk_map,
            'global_risk': global_risk
        }
        # Save to ARTIFACTS_DIR/ensemble_model.pkl
        path = os.path.join(config.ARTIFACTS_DIR, "ensemble_model.pkl")
        joblib.dump(payload, path)
        logger.info(f"Ensemble model saved to {path}")
    
    def train(self, df):
        target = 'is_hotspot'
        train_df, val_df, test_df = self.split_data(df)
        
        # Risk Encoding
        logger.info("Applying Target Encoding for District Risk...")
        risk_map = train_df.groupby('district')['is_hotspot'].mean().to_dict()
        global_risk = train_df['is_hotspot'].mean()
        
        for d in [train_df, val_df, test_df]:
            d['district_risk'] = d['district'].map(risk_map).fillna(global_risk)
        
        # Initial Features
        features = self.get_feature_cols(train_df)
        X_train, y_train = train_df[features], train_df[target]
        X_val, y_val = val_df[features], val_df[target]
        X_test, y_test = test_df[features], test_df[target]

        # --- TRAIN BASE LEARNERS ---
        # 1. XGBoost
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.01, max_depth=8,
            subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42, early_stopping_rounds=30
        )
        self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # 2. LightGBM
        logger.info("Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=10,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, random_state=42
        )
        self.lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])

        # 3. CatBoost
        logger.info("Training CatBoost...")
        self.cat_model = CatBoostClassifier(
            iterations=1000, learning_rate=0.02, depth=8,
            loss_function='Logloss', eval_metric='AUC',
            verbose=False, random_seed=42
        )
        self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        
        # Ensemble Threshold Logic
        logger.info("Optimizing Ensemble Threshold...")
        val_probs = self.predict_proba_ensemble(X_val)
        
        thresholds = np.arange(0.2, 0.8, 0.01)
        best_acc = 0
        best_t = 0.5
        for t in thresholds:
            p = (val_probs >= t).astype(int)
            acc = (p == y_val).mean()
            if acc > best_acc:
                best_acc = acc
                best_t = t
                
        self.threshold = best_t
        logger.info(f"Optimal Ensemble Threshold: {self.threshold:.2f} (Val Acc: {best_acc:.2%})")
        
        # Evaluation
        self.evaluate(X_test, y_test)
        self.save_data_stats(df, train_df, val_df, test_df, config.RAW_RECORD_COUNT)
        
        # Save Ensemble
        self.save_model(features, risk_map, global_risk)

    def predict_proba_ensemble(self, X):
        p1 = self.xgb_model.predict_proba(X)[:, 1]
        p2 = self.lgb_model.predict_proba(X)[:, 1]
        p3 = self.cat_model.predict_proba(X)[:, 1]
        return (p1 + p2 + p3) / 3.0

    def save_data_stats(self, total_df, train, val, test, raw_count=0):
        """Generates a markdown file with data statistics."""
        # Stats path: .../aadhaar_project/DATA_STATS.md
        stats_path = os.path.join(config.BASE_DIR, "DATA_STATS.md")
        
        total_rows = len(total_df)
        hotspots = total_df['is_hotspot'].sum()
        
        content = f"""# Aadhaar Hotspot Model Data Statistics

## Dataset Transformation
- **Raw Input Records (Pincode Level)**: {raw_count:,}
- **Aggregated Records (Daily District Level)**: {total_rows:,}
  *Rationale: Raw data was at Pincode level. Aggregated to Daily District level. Features computed using 30-day (1M) and 90-day (3M) sliding windows to capture monthly trends while preserving data volume.*

## Dataset Overview
- **Total Processed Records**: {total_rows:,}
- **Total Hotspots Identified**: {hotspots:,} ({hotspots/total_rows:.1%})
- **Non-Hotspots**: {total_rows - hotspots:,}

## Training Split Information
The data was split temporally to prevent data leakage (using past data to predict future).

| Split | Date Range | Row Count | Hotspots | % of Data |
|-------|------------|-----------|----------|-----------|
| **Training** | {train['date'].min().date()} to {train['date'].max().date()} | {len(train):,} | {train['is_hotspot'].sum():,} | 70% |
| **Validation** | {val['date'].min().date()} to {val['date'].max().date()} | {len(val):,} | {val['is_hotspot'].sum():,} | 15% |
| **Testing** | {test['date'].min().date()} to {test['date'].max().date()} | {len(test):,} | {test['is_hotspot'].sum():,} | 15% |

## Feature Space
- **Total Features Used**: {len(self.get_feature_cols(total_df))}
- **Target Variable**: `is_hotspot` (Binary)

Generated on: {pd.Timestamp.now()}
"""
        with open(stats_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Data statistics saved to {stats_path}")

    def evaluate(self, X_test, y_test):
        probs = self.predict_proba_ensemble(X_test)
        preds = (probs >= self.threshold).astype(int)
        
        acc = (preds == y_test).mean()
        print("\n" + "#"*40)
        print(f"  ENSEMBLE ACCURACY: {acc:.4%}")
        print("#"*40 + "\n")
        print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")
        print(classification_report(y_test, preds))
