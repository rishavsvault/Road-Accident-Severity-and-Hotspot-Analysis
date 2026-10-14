# 🚦 Explainable Road Accident Severity & Hotspot Analysis for Kolkata (KMDA)

An end-to-end explainable machine learning and spatial intelligence platform developed for the **Kolkata Metropolitan Development Authority (KMDA)**. The system links crash severity to road network topology, weather dynamics, and temporal signals, explaining fatal risks via **TreeSHAP** and prescribing targeted civil engineering interventions.

---

## 🏛️ Project Architecture & Pipeline

```text
Road Accident Severity and Hotspot Analysis/
├── configs/                            # Centralized configurations (hydra/yaml)
│   ├── config.yaml                     # Master paths, CRS (EPSG:32645 UTM 45N), seeds
│   ├── data_config.yaml                # Kolkata corridors, OSM tags, weather API params
│   ├── model_config.yaml               # XGBoost params, Spatial CV, Optuna search space
│   └── intervention_rules.yaml         # Rules mapping SHAP drivers -> KMDA engineering fixes
│
├── data/                               # Data directory (managed by .gitignore)
│   ├── raw/                            # Immutable raw inputs (crashes, weather, boundaries)
│   ├── interim/                        # Cleaned parquet crashes & road network segments
│   └── processed/                      # Feature matrices, risk tiers, and blackspot lists
│
├── src/                                # Modular core Python package
│   ├── data/                           # Ingestion, cleaning, OSM segments, spatial snapper
│   ├── features/                       # Road, temporal, weather, participant feature builders
│   ├── hotspots/                       # 1D Network KDE, Getis-Ord Gi*, segment risk tiering
│   ├── models/                         # Spatial Grouped CV, XGBoost trainer, evaluators, Optuna
│   ├── explainability/                 # TreeSHAP explainer & KMDA intervention engine
│   └── utils/                          # Metric geo utilities (UTM 45N), logging
│
├── dashboard/                          # Interactive Streamlit Multi-Page Web Application
│   ├── app.py                          # Executive overview & KPIs
│   └── pages/
│       ├── 1_🗺️_Hotspot_Explorer.py     # 3D interactive Pydeck map of Kolkata risk tiers
│       ├── 2_🔍_Segment_Deep_Dive.py    # Local SHAP waterfall & segment countermeasure cards
│       ├── 3_🛠️_What_If_Simulator.py   # Counterfactual simulator for lighting/speed/medians
│       └── 4_📄_KMDA_Policy_Brief.py   # Tender-ready Top-20 blackspot priority action plan
│
├── tests/                              # Pytest test suite
├── docs/                               # Methodology whitepapers & briefs
├── run_pipeline.py                     # Master one-click end-to-end execution runner
├── requirements.txt                    # Pinned Python package dependencies
├── environment.yml                     # Conda geospatial environment specification
└── setup.py                            # Package installer (`pip install -e .`)
```

---

## ⚡ Quick Start & Reproduction

### 1. Installation
Using Conda (recommended for Geospatial C-libraries):
```bash
conda env create -f environment.yml
conda activate kolkata_accident_risk
```

Or using standard Python virtual environment:
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. Execute the End-to-End Pipeline
Run the master script to generate mock/clean data, snap to Kolkata road corridors, build features, calculate 1D Network-KDE, train XGBoost with Spatial Grouped Cross-Validation, and export KMDA interventions:
```bash
python run_pipeline.py
```

### 3. Launch the Interactive Dashboard
Launch the multi-page Streamlit portal:
```bash
streamlit run dashboard/app.py
```

### 4. Run Unit Tests
```bash
pytest tests/
```

---

## 🔬 Core Methodology & Differentiators

1. **Spatial Grouped Cross-Validation:** Partitioned strictly by Kolkata planning zone/ward (`SpatialGroupKFold`) to prevent spatial autocorrelation leakage.
2. **1D Network-Constrained KDE (NetKDE):** Evaluates risk strictly along road corridors rather than flat 2D maps that erroneously bleed risk into the Hooghly river or parks.
3. **Severe Error Penalization:** Optimizes **Fatal-Class Recall** and **Quadratic Weighted Kappa (QWK)** with sample reweighting.
4. **SHAP-to-KMDA Intervention Engine:** Directly translates positive SHAP risk drivers into actionable civil engineering fixes:
   - High speed + high risk $\rightarrow$ Speed calming & automated cameras.
   - Night + unlit $\rightarrow$ High-mast LED illumination upgrade.
   - High pedestrian conflict $\rightarrow$ Tabletop raised crosswalks / Foot Overbridges (FOB).
   - Wet weather / monsoon $\rightarrow$ High-friction micro-surfacing (HFST) and curb drainage.
