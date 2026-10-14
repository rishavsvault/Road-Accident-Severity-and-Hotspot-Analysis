# Analytical Notebooks Guide

This directory contains the sequential analytical workflow for researchers, competition judges, and KMDA technical officers:

1. `01_data_audit_and_eda.ipynb`
   - Exploratory analysis of Kolkata crash records.
   - Missing data audit, spatial distribution across wards, time-of-day and monsoon seasonal trends.

2. `02_osm_road_segmentation.ipynb`
   - Road network graph extraction and 200–500m segment partitioning.
   - Junction tagging and projection to UTM Zone 45N (`EPSG:32645`).

3. `03_hotspot_spatial_stats.ipynb`
   - 1D Network-Constrained Kernel Density Estimation (NetKDE).
   - Getis-Ord Gi* hotspot detection and DBSCAN crash clustering.

4. `04_model_training_and_cv.ipynb`
   - Spatial Grouped Cross-Validation (`SpatialGroupKFold`).
   - Baseline benchmarks (Logistic Regression, Random Forest) vs. XGBoost.
   - Hyperparameter exploration and evaluation on Macro-F1, Fatal Recall, and Quadratic Weighted Kappa.

5. `05_shap_and_policy_brief.ipynb`
   - Global TreeSHAP feature importance ranking.
   - Local waterfall SHAP force plots on priority blackspot stretches.
   - Translating SHAP values into KMDA infrastructure tenders and engineering fixes.
