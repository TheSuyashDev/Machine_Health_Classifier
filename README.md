# CNC Care — Machine Health & Predictive Maintenance

A lightweight, zero-leakage predictive maintenance system that calculates mixed-attribute similarity between CNC machines using a custom **Gower metric**, classifies machine health into **Normal**, **Warning**, or **Critical** with a **$k$-Nearest Neighbors ($k$-NN)** model, and generates actionable maintenance work-orders.

---

## Key Highlights

- **Custom Gower Similarity Engine**: Natively handles 4 data types without artificial encoding distortion:
  - *Nominal* (`machine_type`, `controller_type`, `cooling_system`) $\to$ exact match.
  - *Binary* (`lubrication_alert`, `abnormal_noise`, `maintenance_overdue` [symmetric]; `coolant_failure` [asymmetric]).
  - *Ordinal* (`wear_level`, `maintenance_condition`, `degradation_level`) $\to$ normalized ranks ($0$ to $1$).
  - *Numeric Interval* (`temp`, `vibration`, `RPM`, `hours`, `kW`, `load`) $\to$ $1 - \frac{|x_i - x_j|}{R_k}$.
- **Zero Data Leakage**: Fixed engineering ranges ($R_k$), static ordinal mappings, and chronological train/val/test splits ($70\% / 15\% / 15\%$). Target is never in the feature space.
- **Explainable Predictions**: Explains predictions using the $k$ nearest reference machines, vote distributions, and sensor evidence.
- **Prescriptive Guidance**: Direct mapping to technician SOP checklists and urgency levels (`ROUTINE`, `SCHEDULED`, `IMMEDIATE`).

---

## Quickstart & How to Run

### 1. Setup Environment
```bash
# Clone and enter directory
cd Machine_Health_Classifier

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser.
- **Dashboard**: View fleet condition counts (Normal, Warning, Critical) and the prioritized "Machines Requiring Attention" table.
- **Analyze Machine**: Select an existing machine or enter custom telemetry to receive instantaneous health prediction, nearest neighbors, and a prescriptive maintenance checklist.

### 3. Run Command-Line Interface (CLI Pipeline)
```bash
# Emergency Critical scenario
python src/main.py --scenario critical

# Developing Warning scenario
python src/main.py --scenario warning

# Query specific machine ID
python src/main.py --machine_id CNC-042 -k 3
```

### 4. Run Automated Tests
```bash
pytest -v
```
*All 218 unit tests cover preprocessing, Gower math, similarity search, k-NN voting, evaluation, and recommendations.*

### 5. Run Model Evaluation & Benchmarks
```bash
# Run model evaluation across Train/Val/Test
python evaluation/evaluate_model.py

# Generate 5 publication figures in evaluation/figures/
python evaluation/generate_visualizations.py
```

---

## Project Structure

```
Machine_Health_Classifier/
├── app.py                          # Streamlit UI dashboard
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── data/
│   ├── cnc_machines.csv            # 300-machine fleet dataset
│   ├── generate_dataset.py         # Dataset generator
│   └── data_dictionary.md          # 22-column schema details
│
├── src/
│   ├── data_preprocessing.py       # Validation, encoding, chronological split
│   ├── gower_similarity.py         # Custom weighted Gower metric
│   ├── similarity_search.py        # Top-k ranking and tie-breaking
│   ├── knn_classifier.py           # Custom k-NN classifier
│   ├── maintenance_recommendations.py # Work-order & SOP recommendation engine
│   └── main.py                     # CLI pipeline entry point
│
├── evaluation/
│   ├── evaluate_model.py           # Hyperparameter tuning (k=3) & baseline tests
│   ├── generate_visualizations.py  # Confusion matrix & tuning curves generator
│   └── metrics_summary.json        # Precomputed evaluation metrics
│
└── tests/                          # 218 pytest unit tests
```

---

## Model Benchmark Summary (Unseen Test Set)

| Model | Accuracy | Macro F1 | Critical Recall | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **Majority Baseline** | $64.44\%$ | $26.13\%$ | $0.00\%$ | Blindly predicts Warning; misses all Critical machines. |
| **Numeric-Only Euclidean** | $53.33\%$ | $30.41\%$ | $20.00\%$ | Ignores tool wear, coolant failure, and kinematics. |
| **Champion Gower $k$-NN ($k=3$)** | **$60.00\%$** | **$55.30\%$** | **$40.00\%$** | Best balance; $+24.89\%$ Macro F1 gain over Euclidean. |

---

## License
MIT License.
