# DriftDetect

A modular data drift detection, simulation, and coverage reporting toolkit for tabular data.

Built for Systems & Software Engineering I — Goethe University Frankfurt  
Authors: Melisa Lugji & Dilekcan Pamir  
Supervisor: Prof. Dr. Visvanathan Ramesh

---

## What it does

DriftDetect answers two questions:

1. **Did my data drift?** — Compare a new data window against a baseline profile using four statistical tests (KS, Wasserstein, Chi-square, Jensen-Shannon).
2. **Would my monitoring have caught it?** — Simulate controlled drift with known ground truth, run detection, and measure FPR, Detection Power, and Time-to-Detect.

---

## Installation

```bash
git clone <your-repo-url>
cd driftdetect_final

python3 -m venv .venv
source .venv/bin/activate       # Mac/Linux
# .venv\Scripts\activate        # Windows

python3 -m pip install -e ".[dev,dashboard]"
```

---

## Quick Start

```bash
# Verify everything works end-to-end
python demo.py
```

This generates `demo_output/` with baseline, drift results, and an HTML report.

---

## Two Operating Modes

### Production Mode — detect drift on real data

```bash
# Step 1: Build baseline from your training data
python -m driftdetect.cli baseline-build --data your_data.csv --out baseline.json

# Step 2: Detect drift on a new window
python -m driftdetect.cli detect \
  --baseline baseline.json \
  --window new_data.csv \
  --out drift.json \
  --append-history history.json   # optional: builds time-series for dashboard

# Step 3: Generate HTML report
python -m driftdetect.cli report --drift drift.json --out report.html

# Step 4: CI gate — exits with code 1 if drift detected
python -m driftdetect.cli detect --baseline baseline.json --window new_data.csv --ci
```

### Engineering Mode — simulate drift and measure detector quality

```bash
# Step 1: Simulate controlled drift from a YAML scenario
python -m driftdetect.cli simulate \
  --scenario scenarios/income_drift.yaml \
  --outdir sim_out \
  --overwrite

# Step 2: Build baseline from clean reference data
python -m driftdetect.cli baseline-build --data your_data.csv --out baseline.json

# Step 3: Run detection on all simulated windows
python -m driftdetect.cli detect-suite \
  --baseline baseline.json \
  --manifest sim_out/manifest.json \
  --outdir sim_out

# Step 4: Compute coverage metrics (FPR, Power, TTD)
python -m driftdetect.cli coverage \
  --manifest sim_out/manifest.json \
  --drifts-dir sim_out/drifts \
  --out sim_out/coverage.json
```

---

## Income Simulation Experiment

Demonstrates gradual drift, new category detection, and variance drift on synthetic demographic data.

```bash
# Build income baseline
python -c "
import json, numpy as np
from pathlib import Path
from driftdetect.simulate import _generate_income_window
from driftdetect.baseline import build_baseline_profile

rng = np.random.default_rng(999)
df = _generate_income_window(rng, 2000)
baseline = build_baseline_profile(df)
Path('income_baseline.json').write_text(
    json.dumps(baseline.model_dump(by_alias=True), indent=2))
print('Done')
"

# Run simulation pipeline
python -m driftdetect.cli simulate \
  --scenario scenarios/income_drift.yaml \
  --outdir income_sim --overwrite

python -m driftdetect.cli detect-suite \
  --baseline income_baseline.json \
  --manifest income_sim/manifest.json \
  --outdir income_sim

python -m driftdetect.cli coverage \
  --manifest income_sim/manifest.json \
  --drifts-dir income_sim/drifts \
  --out income_sim/coverage.json
```

**Expected results:**

| Event | Type | Power | TTD |
|-------|------|-------|-----|
| income_gradual_shift | numeric_shift | 1.0 | 2 windows |
| new_occupation_category | new_category | 1.0 | 0 windows |
| hours_variance_drift | variance_shift | 1.0 | 0 windows |

---

## Logistics Experiment

Validates DriftDetect on realistic logistics data using the course order simulator.

**Requires:** Place `order_simulator.py` (prof's file) in the project root, then:

```bash
pip install scikit-learn
python logistics_drift_test.py
```

**Three drift scenarios:**
1. Weight mean shift — CUST_A switches to heavier packages (supplier change)
2. New unit type — CUST_B starts using vehicle type 80 (new fleet)
3. Day pattern shift — CUST_A moves preferred delivery day Tue→Fri

**Key finding:** The day pattern shift was never detected (Power=0.0). CUST_B orders on all 7 days, diluting the aggregate Day distribution signal. Coverage analysis revealed this blind spot and points to the fix: monitor per-customer sub-windows instead of aggregate data.

---

## Dashboard

```bash
streamlit run dashboard/app.py
```

Use the **Experiment** dropdown in the sidebar to switch between:
- **Income Drift** — income simulation results
- **Logistics Experiment** — logistics experiment results  
- **Demo (Synthetic)** — quick demo data
- **Custom** — point to any artifact files manually

---

## Scenario YAML Format

```yaml
scenario_id: my_scenario
dataset: income        # "income" or "logistics"
seed: 42
window_count: 12
window_size: 500
output_format: csv

events:
  - event_id: wage_shift
    drift_type: numeric_shift    # numeric_shift | variance_shift | missingness | new_category | correlation_drift
    start_window: 4
    features: [income]
    intensity:
      mean_shift: 15000
      mode: gradual              # abrupt (default) or gradual
      ramp_windows: 4
```

---

## Project Structure

```
driftdetect/
  models.py           — Pydantic data models (baseline, features, histograms)
  baseline.py         — Build baseline profile from DataFrame
  detector_core.py    — Statistical tests (KS, Wasserstein, Chi-square, JSD)
  detector_facade.py  — Orchestrates detection workflow, writes drift.json
  policy.py           — Global drift decision rules (schema / severity / count)
  simulate.py         — Scenario-driven data generator with drift injection
  scenario.py         — YAML scenario validation
  coverage.py         — FPR, Power, TTD computation from ground truth
  reporter.py         — HTML report generation (Jinja2)
  cli.py              — Command-line interface (Typer)

dashboard/
  app.py              — Streamlit dashboard (artifact-only, no recomputation)

scenarios/
  income_drift.yaml   — Income simulation scenario (gradual + abrupt drift)
  demo.yaml           — Simple demo scenario

tests/                — 25 unit tests covering all core components
demo.py               — End-to-end sanity check
logistics_drift_test.py — Logistics experiment (requires order_simulator.py)
```

---

## Statistical Tests

| Feature Type | Test | Measures |
|-------------|------|---------|
| Numeric | Kolmogorov-Smirnov | Maximum CDF gap |
| Numeric | Wasserstein Distance | Total distribution displacement |
| Categorical | Chi-Square | Count deviation from expected |
| Categorical | Jensen-Shannon Divergence | Information-theoretic distance, handles new categories |

Each test produces a severity score in [0, 1]. Per-feature severity is the maximum across all applicable tests. Global drift triggers if: schema changed, OR max severity ≥ S_global, OR ≥ K features drifted.

---

## Running Tests

```bash
pytest tests/ -v
```

25 tests, all passing.