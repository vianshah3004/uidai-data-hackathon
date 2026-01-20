# implementation Plan: Physics-Informed XGBoost for Aadhaar Hotspot Detection

## 1. Core Diagnosis
The current model (81% Accuracy, 0.90 AUC) has reached a plateau because it relies on **observed historical growth** (reactionary) rather than **structural tension** (anticipatory). To bridge the gap to >95%, we must model the **physics of the surge**: potential energy (unmet demand), kinetic energy (current velocity), and contagion (pressure waves).

## 2. Feature Engineering Strategy

### A. Multi-Timeframe Acceleration Physics (The "Engine")
*Objective: Detect pre-surge tension before the surge becomes visible.*
- **Velocity ($v$)**: Improved `growth_rate_1month` (normalized).
- **Acceleration ($a$)**: Rate of change of velocity ($v_t - v_{t-1}$). *High positive $a$ indicates a text-book exponential takeoff.*
- **Jerk ($j$)**: Rate of change of acceleration ($a_t - a_{t-1}$). *Positive jerk signalizes "instability" — the acceleration itself is accelerating.*
- **Resistance**: `volatility_roll_14d`. *High volatility with low velocity = friction/confusion. High volatility + high velocity = chaotic surge.*

### B. Pseudo-Spatial Contagion (The "Wave")
*Objective: Model the "pressure wave" of updates moving across a state without lat/long data.*
- **State-Level Pulse**: Average momentum of *other* districts in the same state.
- **Relativity**: `district_momentum` vs `state_momentum`.
  - *If $D > S$: Leading the wave.*
  - *If $D < S$ but S is rising: Drag effect (likely to be pulled up).*
- **Cluster Count**: % of other districts in state currently flagged as High Risk.

### C. Coverage Gap Dynamics (The "Fuel")
*Objective: Convert static "coverage gap" into a dynamic "release valve" signal.*
- **Gap Closure Velocity**: How fast is `coverage_gap` shrinking? *Rapid shrinking = Release event.*
- **Urgency Index**: `gap_closure_velocity` / `remaining_gap`. *Are we closing the last mile? (Often fastest).*

### D. Explicit Interactions (The "Decision Anchors")
*Objective: Force tree splits on critical compound logic.*
1. **Critical Mass**: `Acceleration` $\times$ `Coverage Gap` (*Fast moving + High demand*).
2. **Contagion Risk**: `State Pulse` $\times$ `Gap` (*State is moving + I have inventory*).
3. **Sustained Burn**: `Momentum` $\times$ `Jerk` (*Already moving + speeding up*).

## 3. Label & Evaluation Refinement
- **Stricter Target**: Raise definition of "Hotspot" to require **>75% growth** AND **>State Average Growth**. This removes noise and focuses on *exceptional* events.
- **Metric Priority**: Optimize for **Recall @ Precision 0.85**. maximizing AUC is good, but we need to catch the surges (Recall) without flooding operations with false alarms.

## 4. Implementation Steps
1. **Refactor Data Pipeline**: Implement Physics, Contagion, and Gap Dynamic features.
2. **Refine Target**: Tighten `is_hotspot` logic.
3. **Retrain Ensemble**: Train XGB+LGBM+CatBoost on new feature space.
4. **Validation**: Measure Recall improvement on "Jerk" and "Contagion" features specifically.
