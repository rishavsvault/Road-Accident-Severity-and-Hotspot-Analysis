# Explainable Road Accident Severity & Hotspot Analysis for Kolkata (KMDA)

> **[CRITICAL NOTICE: SYNTHETIC DATA DISCLOSURE]**  
> All crash records, severity classifications, and road segment rankings in this repository are currently derived from **statistically calibrated synthetic data** modeled after national Ministry of Road Transport and Highways (MoRTH) accident profiles for the Kolkata Metropolitan Area.  
> Official Kolkata Police First Information Report (FIR) logs and KMDA field databases have not yet been ingested. All findings, SHAP attributions, and blackspot rankings reflect simulated distributions for methodology demonstration and must not be used for physical civil deployment without operational ground-truth validation.

---

An end-to-end explainable machine learning and spatial intelligence platform developed for the **Kolkata Metropolitan Development Authority (KMDA)**. The system links crash severity to road network topology, multi-station weather dynamics, and temporal signals, explaining fatal risks via **TreeSHAP** and prescribing targeted civil engineering interventions.

---

## Project Architecture & Pipeline

```text
Road Accident Severity and Hotspot Analysis/
├── configs/                            # Centralized configurations (hydra/yaml)
│   ├── config.yaml                     # Master paths, CRS (EPSG:32645 UTM 45N), seeds
│   ├── data_config.yaml                # Kolkata corridors, OSM tags, weather API params
│   ├── model_config.yaml               # XGBoost params, Spatial CV, Optuna search space
│   ├── best_tuned_params.yaml          # Best hyperparameters discovered via Optuna
│   └── intervention_rules.yaml         # Rules mapping SHAP drivers -> KMDA engineering fixes
│
├── data/                               # Data directory (managed by .gitignore)
│   ├── raw/                            # Immutable raw inputs (crashes, weather, OSM graphs)
│   ├── interim/                        # Cleaned parquet crashes & road network segments
│   └── processed/                      # Feature matrices, risk tiers, blackspots, calibration curves
│
├── src/                                # Modular core Python package
│   ├── data/                           # Ingestion, multi-vertex polygon PiP, OSMnx loader, snapper
│   ├── features/                       # Road, temporal, multi-station IDW weather builders
│   ├── hotspots/                       # 1D NetKDE, Getis-Ord Gi*, exposure-normalized ranker
│   ├── models/                         # Spatial Grouped CV, XGBoost trainer, Ordinal model, Optuna
│   ├── explainability/                 # TreeSHAP explainer & KMDA intervention engine
│   └── utils/                          # Metric geo utilities (UTM 45N), logging, feature names
│
├── dashboard/                          # Interactive Streamlit Multi-Page Web Application
│   ├── app.py                          # Executive overview, KPIs & reliability diagrams
│   ├── components/                     # Reusable UI disclaimer banners & cards
│   └── pages/
│       ├── 1_Hotspot_Explorer.py       # 3D interactive Pydeck map with unmonitored segment styling
│       ├── 2_Segment_Deep_Dive.py      # Local SHAP waterfall & segment countermeasure cards
│       ├── 3_What_If_Simulator.py      # Counterfactual simulator for lighting/speed/medians
│       └── 4_KMDA_Policy_Brief.py      # Tender-ready Top-20 blackspot priority action plan
│
├── tests/                              # Pytest test suite
├── docs/                               # Methodology whitepapers & briefs
├── run_pipeline.py                     # Master one-click end-to-end execution runner
├── requirements.txt                    # Pinned Python package dependencies
├── environment.yml                     # Conda geospatial environment specification
└── setup.py                            # Package installer (`pip install -e .`)
```

---

## Quick Start & Reproduction

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
Run the master script to clean crash records with polygon point-in-polygon filtering, interpolate multi-station weather, snap to Kolkata road corridors, compute 1D NetKDE, evaluate exposure-adjusted risk scores, train calibrated XGBoost with Optuna best params, benchmark against Ordinal models, and export calibration curves and KMDA interventions:
```bash
python run_pipeline.py
```
To run live Optuna hyperparameter re-tuning (adds ~2 minutes):
```bash
python run_pipeline.py --tune
```

### 3. Launch the Interactive Dashboard
Launch the multi-page Streamlit portal:
```bash
streamlit run dashboard/app.py
```

### 4. Run Unit Tests & Explainer Diagnostics
Run test suite across spatial stats, feature encoding, models, and SHAP:
```bash
python -m pytest tests/
```
Run standalone CLI SHAP diagnostic on trained model artifacts:
```bash
python -m src.explainability.shap_explainer
```

---

## Core Methodology & Technical Highlights

1. **Polygon Point-in-Polygon Filtering:** Replaces rectangular bounding-box cropping with a multi-vertex administrative boundary polygon tracing the Kolkata Municipal Corporation (KMC) perimeter.
2. **Multi-Station Weather Interpolation:** Interpolates precipitation, visibility, and temperature at each crash's exact GPS coordinate via 2D Inverse Distance Weighting (IDW) across four metropolitan meteorological stations (Alipore, Dum Dum, Salt Lake, Behala).
3. **Traffic & Population Exposure Normalization:** Normalizes segment risk by Annual Average Daily Traffic (AADT), 2-year Million Vehicle Kilometers Traveled (MVKT), and ward population density rather than raw crash counts.
4. **Spatial Grouped Cross-Validation:** Partitioned strictly by geographic zone (`SpatialGroupKFold`) to prevent spatial autocorrelation leakage.
5. **Probability Calibration (Isotonic / Platt):** Calibrates predicted probabilities via `CalibratedClassifierCV` and exports reliability curves across all three severity tiers.
6. **Ordinal XGBoost Model Comparison:** Benchmarks standard multiclass XGBoost against an ordinal cumulative binary architecture enforcing severity monotonicity ($P(\text{Fatal}) \le P(\ge \text{Grievous})$).
7. **Severe Error Penalization & PR-AUC:** Optimizes Fatal-Class Recall and Precision-Recall AUC (PR-AUC) to penalize false optimism on rare fatal accidents.
8. **SHAP-to-KMDA Intervention Engine:** Directly translates positive SHAP risk drivers into actionable civil engineering fixes:
   - High speed + high risk: Speed calming & automated enforcement cameras.
   - Night + unlit: High-mast LED illumination upgrade.
   - High pedestrian conflict: Tabletop raised crosswalks / Foot Overbridges (FOB).
   - Wet weather / monsoon: High-friction micro-surfacing (HFST) and curb drainage.
