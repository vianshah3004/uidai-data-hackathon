# Strategic Roadmap for >85% Accuracy & Generalization

## Current State Analysis
- **Test Accuracy**: ~83%
- **Validation Accuracy**: ~90%
- **Gap**: 7% (Evidence of **Overfitting** or **Domain Shift** in the test month).
- **Diagnosis**: The model has learned the training period (March-Oct) very well but struggles slightly with the specific dynamics of December (Test). This is a classical generalization problem.

## Proposed Advanced Techniques

### 1. Feature Reduction for Generalization (Signal-to-Noise Ratio)
**Problem**: We have thrown ~70+ features at the model. Many (like weak lags or raw counts) might contain noise that the model memorizes.
**Solution**: Implement **Recursive Feature Elimination (RFE)** or Importance-Based Selection.
- *Action*: Train a scout model, identify the bottom 25% of features, and **drop them**.
- *Benefit*: Forces the model to rely on strong causal drivers (Physics, Gap) rather than coincidental correlations.

### 2. Stacking Ensemble (Meta-Learning)
**Problem**: Simple averaging `(A+B+C)/3` assumes all models are equally good.
**Solution**: Train a **Logistic Regression Meta-Learner** on the predictions of XGB/LGBM/CatBoost.
- *Action*: The Meta-Learner learns *when* to trust XGB vs CatBoost.
- *Benefit*: sophisticated combination usually yields +1-2% accuracy.

### 3. Robust Scaling & Transformation
**Problem**: Features like `total_updates` or `acceleration` likely have extreme outliers (power law). Even tree models can destabilize with extreme values in interaction terms.
**Solution**: Apply `QuantileTransformer` (Gaussian Output).
- *Benefit*: Compresses outliers and makes distributions normal, stabilizing the decision boundaries.

### 4. Time-Aware Cross-Validation
**Problem**: Standard splits might be lucky/unlucky.
**Solution**: While expensive, we will ensure our validation set is strictly the *latest* possible data before Test to minimize concept drift. (Already done, but we will reinforce parameters).

## Implementation Plan

1.  **Data Pipeline**: Add `QuantileTransformer` to handle skewness.
2.  **Model Engine**:
    -   Step A: Train "Scout" XGBoost.
    -   Step B: Select Top features (Importance > Threshold).
    -   Step C: Train Base Learners (XGB, LGB, Cat) on *Selected* features.
    -   Step D: Train `StackingClassifier` (LogisticRegression) on validation probabilities.
3.  **Execution**: Run and Evaluate.
