# Methodology Whitepaper: Explainable Road Accident Severity & Hotspot Analysis for Kolkata (KMDA)

## 1. Problem Formulation
India accounts for over 11% of global road traffic fatalities. Conventional municipal intervention relies heavily on aggregate crash tally frequencies, often leading to retroactive, suboptimal countermeasure allocation. This project develops an end-to-end explainable machine learning and spatial intelligence architecture tailored for the **Kolkata Metropolitan Development Authority (KMDA)**.

## 2. Mathematical Framework & Pipeline Design

### 2.1 Spatial Discretization & Road Segmentation
Rather than treating roads as continuous infinite geometries or treating urban space as a flat 2D plane, roads are segmented into discrete analytical units $S_i$ of 200–500 meters, split at junctions and geometry bends.

### 2.2 1D Network-Constrained Kernel Density Estimation (NetKDE)
Standard planar 2D KDE produces spatial bleeding into unnavigable spaces (e.g., the Hooghly River, Maidan parklands, salt lakes). We implement 1D Network-constrained KDE along graph edges:

$$KDE(S_i) = \frac{1}{|S_i|} \sum_{c \in C, d(c, S_i) \le h} w(c) \cdot \frac{3}{4}\left(1 - \left(\frac{d(c, S_i)}{h}\right)^2\right)$$

where:
- $h = 400$ meters (bandwidth)
- $w(c)$ is the severity weighting: $\text{Fatal} = 10$, $\text{Grievous} = 3$, $\text{Minor} = 1$
- $|S_i|$ is the segment length in meters.

### 2.3 Statistical Hotspot Significance: Getis-Ord Gi*
Local spatial autocorrelation is measured using the Getis-Ord $G_i^*$ statistic:

$$G_i^* = \frac{\sum_{j=1}^n w_{ij} x_j - \bar{X}\sum_{j=1}^n w_{ij}}{S \sqrt{\frac{n\sum_{j=1}^n w_{ij}^2 - (\sum_{j=1}^n w_{ij})^2}{n - 1}}}$$

Segments with $Z(G_i^*) \ge +1.96$ ($p < 0.05$) are classified as statistically significant blackspots.

### 2.4 Severity Prediction: XGBoost with Spatial Grouped Cross-Validation
To prevent spatial data leakage where proximate crashes on the same corridor artificially inflate model performance, validation is partitioned via **Spatial Grouped K-Fold** based on planning zones / police jurisdictions:
$$\text{Train}_k \cap \text{Val}_k = \emptyset \quad \forall \text{ Geographic Zones}$$

Evaluation strictly monitors:
- **Macro-F1:** Balanced capability across all three severity classes.
- **Fatal-Class Recall:** Minimizes false negatives on life-threatening incidents.
- **Quadratic Weighted Kappa (QWK):** Quadradically penalizes extreme ordinal errors (predicting minor when actual was fatal).

### 2.5 Explainability & Civil Countermeasures Mapping
Using TreeSHAP, Shapley values quantify the exact marginal contribution of road features ($x_j$):
$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left( f_x(S \cup \{j\}) - f_x(S) \right)$$

Positive SHAP attributions for fatal outcomes are algorithmically translated by the `KMDAInterventionEngine` into municipal civil fixes (smart LED illumination, tabletop crosswalks, New Jersey median barriers, high-friction micro-surfacing).
