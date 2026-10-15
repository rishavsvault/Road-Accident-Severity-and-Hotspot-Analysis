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

---

## Methodology Whitepaper & Mathematical Framework

### 1. Problem Formulation
India accounts for over 11% of global road traffic fatalities. Conventional municipal intervention relies heavily on aggregate crash tally frequencies, often leading to retroactive, suboptimal countermeasure allocation. This platform develops an end-to-end explainable machine learning and spatial intelligence architecture tailored for the **Kolkata Metropolitan Development Authority (KMDA)**.

### 2. Spatial Discretization & Network Segmentation
Rather than treating roads as continuous infinite geometries or treating urban space as an unconstrained 2D plane, roads are segmented into discrete analytical units $S_i$ of 200–500 meters, split at junctions and geometry bends.

### 3. 1D Network-Constrained Kernel Density Estimation (NetKDE)
Standard planar 2D KDE produces spatial bleeding into unnavigable spaces (e.g., the Hooghly River, Maidan parklands, East Kolkata Wetlands). We implement 1D Network-constrained KDE along graph edges:

$$KDE(S_i) = \frac{1}{|S_i|} \sum_{c \in C, \, d(c, S_i) \le h} w(c) \cdot \frac{3}{4}\left(1 - \left(\frac{d(c, S_i)}{h}\right)^2\right)$$

where:
- $h = 400$ meters (search bandwidth along network distance)
- $w(c)$ is the severity weighting: $\text{Fatal} = 10$, $\text{Grievous} = 3$, $\text{Minor} = 1$
- $|S_i|$ is the segment length in meters.

### 4. Length-Normalized & Exposure-Adjusted Segment Risk Score (0–100)
To establish an intuitive, comparable metric across road corridors of variable geometries, the raw severity-weighted crash density per 100 meters is standardized onto a continuous 0–100 scale:

$$\text{Raw Score}(S_i) = \frac{10 \cdot N_{\text{fatal}} + 3 \cdot N_{\text{grievous}} + 1 \cdot N_{\text{minor}}}{\max(50, |S_i|) / 100}$$

$$\text{Standardized Risk Score}(S_i) = \left( \frac{\text{Raw Score}(S_i)}{\max_{j} \text{Raw Score}(S_j)} \right) \times 100 \in [0, 100]$$

Where:
- **0.0**: Baseline corridor with zero crash occurrences.
- **100.0**: Peak hazardous blackspot corridor in the KMDA network.
- **Priority Tiers**: High ($\ge 85^{\text{th}}$ percentile), Medium ($50^{\text{th}}–85^{\text{th}}$ percentile), Low ($< 50^{\text{th}}$ percentile).

### 5. Statistical Hotspot Significance: Getis-Ord $G_i^*$ & Moran's $I$
Local spatial clustering is measured using the Getis-Ord $G_i^*$ statistic:

$$G_i^* = \frac{\sum_{j=1}^n w_{ij} x_j - \bar{X}\sum_{j=1}^n w_{ij}}{S \sqrt{\frac{n\sum_{j=1}^n w_{ij}^2 - (\sum_{j=1}^n w_{ij})^2}{n - 1}}}$$

Segments with $Z(G_i^*) \ge +1.96$ ($p < 0.05$) are classified as statistically significant blackspots. Global spatial autocorrelation is validated via Global Moran's $I$.

### 6. Severity Modeling: Calibrated XGBoost with Spatial Grouped Cross-Validation
To prevent spatial data leakage where proximate crashes on the same corridor artificially inflate model performance, validation is partitioned via **Spatial Grouped K-Fold** based on planning zones / police jurisdictions:

$$\text{Train}_k \cap \text{Val}_k = \emptyset \quad \forall \text{ Geographic Zones}$$

Evaluation strictly monitors:
- **Macro-F1 & PR-AUC:** Balanced capability across all three severity classes, penalizing false optimism on rare fatal accidents.
- **Fatal-Class Recall:** Minimizes false negatives on life-threatening incidents.
- **Quadratic Weighted Kappa (QWK):** Quadradically penalizes extreme ordinal errors (predicting minor when actual was fatal).
- **Probability Calibration (Isotonic Regression):** Calibrates predicted probabilities via `CalibratedClassifierCV` to ensure empirical reliability.

### 7. Explainability & Civil Countermeasures Mapping
Using TreeSHAP, Shapley values quantify the exact marginal contribution of road and environmental features ($x_j$):

$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left( f_x(S \cup \{j\}) - f_x(S) \right)$$

Positive SHAP attributions for fatal outcomes are algorithmically translated by the `KMDAInterventionEngine` into municipal civil engineering fixes:
- **High speed + high risk**: Speed calming & automated enforcement cameras.
- **Night + unlit**: High-mast smart LED illumination upgrade.
- **High pedestrian conflict**: Tabletop raised crosswalks / Foot Overbridges (FOB).
- **Wet weather / monsoon**: High-friction micro-surfacing (HFST) and curb drainage.
