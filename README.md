# DID-based Pricing Decision System

This project builds an explainable pricing decision pipeline based on
Difference-in-Differences (DID).

We first use Event Study and 2×2 DID to verify whether historical price
changes have a **causal impact** on sales.  
Only when the effect is statistically reliable do we translate the result
into weekly pricing recommendations.

Why DID?
In real retail data, sales fluctuate due to seasonality, trends and noise.
DID helps isolate the causal impact of price changes from these confounders,
making pricing decisions safer and more interpretable.

## Pipeline Overview

1. Prepare event-level price change data
2. Aggregate to event-time × treat/control level
3. Event Study DID (dynamic effects & pre-trend check)
4. 2×2 DID (average treatment effect)
5. Build store-level pricing policy table
6. Generate weekly item-level pricing recommendations

### Event Study DID

We estimate the following model on event-level differences:

diff_{e,t} = α + Σ τ_t · 1{event_time = t} + ε_{e,t}

where:
- diff_{e,t} = mean(sales_treated) − mean(sales_control)
- baseline_time = −1 (last pre-treatment period)

This allows us to:
- Check pre-trends (parallel trends assumption)
- Observe dynamic treatment effects over time

### 2×2 DID (Average Treatment Effect)

For each event:
ATE_e = (post mean diff) − (pre mean diff)

We then aggregate across events using weighted least squares
with cluster-robust standard errors at the event level.

### Pre-trend Check

If significant effects appear before price changes,
the DID assumption may be violated.

We therefore flag stores with strong pre-trend signals as
"audit_only" to avoid unsafe automated pricing decisions.

## Pricing Recommendation Logic

Pricing actions are determined by combining:
- DID confidence (significance + pre-trend)
- Directional effects (price up vs price down)
- Recent sales level and short-term trend

Possible actions:
- hold
- cut (small / medium)
- raise (small)

## Outputs

- Event Study plots (per store / direction)
- Store-level pricing policy table
- Weekly item-level pricing recommendations

## How to Run

```bash
pip install -r requirements.txt

# Step 4: DID
python src/step4_run_did.py

# Step 5: Build policy table
python src/step5_build_policy.py

# Step 6: Weekly pricing recommendation
python src/step6_run_pricing.py
