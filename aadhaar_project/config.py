
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)  # Parent directory where CSVs are located

BIO_FILE = os.path.join(DATA_DIR, "final_UIDAI_cleaned_biometrics_data.csv")
DEMO_FILE = os.path.join(DATA_DIR, "final_UIDAI_cleaned_demographic_data.csv")
POP_FILE = os.path.join(DATA_DIR, "final_UIDAI_cleaned_enrol_data.csv")

# Output Paths
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "xgboost_hotspot_model.json")
ENCODER_PATH = os.path.join(ARTIFACTS_DIR, "encoders.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.json")
PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "predictions.csv")
PRIORITY_LIST_PATH = os.path.join(ARTIFACTS_DIR, "priority_list.csv")

# Create artifacts dir if not exists
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Hyperparameters
XGB_PARAMS = {
    'n_estimators': 200,          # Tunable: 100-300
    'learning_rate': 0.05,        # Tunable: 0.01-0.1
    'max_depth': 6,               # Tunable: 4-8
    'min_child_weight': 3,        # Tunable: 1-5
    'subsample': 0.8,             # Tunable: 0.7-0.9
    'colsample_bytree': 0.8,      # Tunable: 0.7-0.9
    'objective': 'binary:logistic',
    'eval_metric': ['logloss', 'auc', 'error'],
    'scale_pos_weight': 1         # Will be adjusted dynamically based on class imbalance
}

# Feature groups
FEATURE_COLS = [] # Will be populated
TARGET_COL = 'is_hotspot'

# Experiment Settings
LOOKAHEAD_MONTHS = 3
HOTSPOT_THRESHOLD = 0.50  # 50% increase
RAW_RECORD_COUNT = 0
