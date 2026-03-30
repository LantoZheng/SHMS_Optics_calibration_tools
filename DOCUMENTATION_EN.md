# shms_optics_calibration – Detailed Technical Documentation

## Table of Contents

1. [Physics Background](#1-physics-background)
2. [Package Architecture Overview](#2-package-architecture-overview)
3. [Data I/O Module (data_io)](#3-data-io-module-data_io)
4. [Data Preprocessing Module (preprocessing)](#4-data-preprocessing-module-preprocessing)
5. [Clustering Module (clustering)](#5-clustering-module-clustering)
6. [Calibration Module (calibration)](#6-calibration-module-calibration)
7. [Evaluation Module (evaluation)](#7-evaluation-module-evaluation)
8. [Visualization Module (visualization)](#8-visualization-module-visualization)
9. [Configuration System (config)](#9-configuration-system-config)
10. [Impact of Parameter Choices on Results](#10-impact-of-parameter-choices-on-results)
11. [Complete Workflow Examples](#11-complete-workflow-examples)
12. [Parameter Quick-Reference Tables](#12-parameter-quick-reference-tables)

---

## 1. Physics Background

### 1.1 Why SHMS Optics Calibration Is Necessary

The SHMS (Super High Momentum Spectrometer) is one of the primary detector systems in Hall C at Jefferson Lab. It is designed for high-precision electron-scattering experiments in the momentum range of 2.0–11.0 GeV/c. Its magnet configuration is **dQQQD** (one horizontal deflecting magnet + three quadrupole magnets + one dipole magnet).

As particles travel from the target to the focal plane, their trajectories are shaped by these magnets. The goal of **optics calibration** is to precisely determine the mathematical transformation—the **reconstruction matrix**—that maps focal-plane measured quantities (directly recorded by the detectors) to target-plane physical quantities (kinematic variables at the particle-scattering vertex).

If the calibration is inaccurate, the reconstructed sieve-hole positions will be blurry or distorted, introducing systematic errors into all downstream physics results.

### 1.2 Coordinate System Definitions

| Coordinate system | Variable | Direction | ROOT branch |
|---|---|---|---|
| Focal Plane | $x_{fp}$ | Vertical (dispersive direction) | `P.dc.x_fp` |
| Focal Plane | $y_{fp}$ | Horizontal (non-dispersive direction) | `P.dc.y_fp` |
| Focal Plane | $x'_{fp} = dx/dz$ | Vertical angle | `P.dc.xp_fp` |
| Focal Plane | $y'_{fp} = dy/dz$ | Horizontal angle | `P.dc.yp_fp` |
| Target | $\delta = dp/p$ | Momentum deviation | `P.gtr.dp` |
| Target | $x'_{tar}$ (θ) | Vertical angle | `P.gtr.th` |
| Target | $y'_{tar}$ (φ) | Horizontal angle | `P.gtr.ph` |
| Target | $y_{tar}$ | Horizontal position | `P.gtr.y` |
| Target | $z_{react}$ | Reaction vertex Z position | `P.react.z` |

**Important convention**: For SHMS, the $+y_{tar}$ direction points upstream (toward the beam dump).

### 1.3 The Sieve Slit Method

The standard experimental procedure for optics calibration:

1. **Data taking**: Insert a **sieve slit** in front of the spectrometer entrance—a heavy-metal plate with a precision-drilled array of holes in a known geometric pattern. Data are taken using a carbon-foil target.
2. **Initial reconstruction**: Replay the data with existing matrix elements (e.g., from a COSY simulation). If the optics are off, the reconstructed sieve-hole image will appear blurry or distorted.
3. **Event selection**: Identify events belonging to each sieve hole and determine the "true" position of each hole (given by the known geometry).
4. **Minimization (SVD)**: Define a $\chi^2$ function and use Singular Value Decomposition (SVD) to minimize the difference between reconstructed and true hole positions, yielding the optimized matrix elements.

The core task of this software package corresponds to **step 3**: automatically identifying, from experimental data, the groups of events (clusters) that correspond to each sieve hole, and building a grid index for them in the target-plane coordinate system.

---

## 2. Package Architecture Overview

```
shms_optics_calibration/
├── __init__.py        # Public API exports
├── config.py          # All configuration parameters and defaults
├── data_io.py         # ROOT file loading and sieve-plane projection
├── preprocessing.py   # Foil-position classification and data preparation
├── clustering.py      # Clustering algorithms (DBSCAN / HDBSCAN / Two-Entry)
├── calibration.py     # Grid-index construction and hole-position estimation
├── evaluation.py      # Benchmark evaluation (efficiency / purity / separability)
├── visualization.py   # Visualization functions
├── README.md          # Quick-start guide (English)
├── DOCUMENTATION.md   # Detailed technical documentation (Chinese)
└── DOCUMENTATION_EN.md  # This document (English)
```

### Typical Workflow

```
ROOT file
    │
    ▼ data_io.load_and_prepare_data()
    │  (load data → sieve-plane projection → range filter)
    │
    ▼ preprocessing.classify_foils_with_range()
    │  (identify multiple foil positions from P_gtr_y distribution)
    │
    ▼ (for each foil position)
    │
    ▼ clustering.cluster_by_foil_position()
    │  Available algorithms:
    │  ├─ auto_dbscan_clustering()       # DBSCAN with automatic parameter search
    │  ├─ peel_and_cluster_edges()       # Second-stage edge clustering
    │  ├─ two_entry_dbscan()             # Two-Entry DBSCAN (recommended)
    │  └─ auto_hdbscan_clustering()      # Hierarchical HDBSCAN clustering
    │
    ▼ calibration.build_full_grid_index()
    │  (construct row/column grid index from cluster centers)
    │
    ▼ evaluation.calculate_cluster_metrics()  (optional – requires simulation data)
       (efficiency / purity / separability evaluation)
```

---

## 3. Data I/O Module (`data_io`)

### 3.1 `load_root_file()`

Loads data from a ROOT file into a pandas DataFrame.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file_path` | str | — | Path to the ROOT file |
| `tree_name` | str | `"T"` | TTree name. Trees produced by the standard Hall C replay software (hcana) are typically named `"T"` |
| `branches` | list\[str\] \| None | None | List of branch names to read. `None` reads all branches (high memory usage; specifying branches is strongly recommended) |
| `verbose` | bool | True | Whether to print file path, event count, etc. |

**Dependency**: requires `uproot` (`pip install uproot`).

**Usage tips**:
- For large ROOT files (>10 GB), specifying only the required columns via `branches` can reduce memory usage by more than 10×.
- If the file contains vector-type branches (e.g., multi-track data), `uproot` will convert them to `awkward` arrays; using the `library="pd"` option automatically expands them into multiple rows.

---

### 3.2 `project_to_sieve()` and `add_sieve_projection()`

Projects reconstructed variables onto sieve-plane coordinates.

**Projection formulas (from the SHMS optics model):**

```
sieve_x = P_gtr_x + P_gtr_th × z_coefficient

sieve_y = (-0.019 × dp + 0.00019 × dp² + 213.0 × ph + y)
         + 40.0 × (-0.00052 × dp + 0.0000052 × dp² + ph)
```

where `z_coefficient = 253.0` is the sieve-to-target distance (in cm).

**`TargetProjectionConfig` parameters:**

| Parameter | Default | Physical meaning |
|---|---|---|
| `x_z_coefficient` | 253.0 | Sieve-to-target Z distance (cm). Changing this value is equivalent to shifting the assumed Z position of the sieve, directly affecting the accuracy of `sieve_x` |
| `y_dp_linear` | −0.019 | First-order coefficient of $\delta$ (momentum deviation) contribution to `sieve_y` (dispersion correction) |
| `y_dp_quadratic` | 0.00019 | Second-order coefficient of $\delta$ (non-linear dispersion correction) |
| `y_ph_coefficient` | 213.0 | Coefficient of $\phi$ (horizontal angle) contribution (= 138.0 + 75.0, from SHMS optical geometry) |
| `y_offset_dp_linear` | −0.00052 | First-order coefficient of $\delta$ in the offset term |
| `y_offset_dp_quadratic` | 0.0000052 | Second-order coefficient of $\delta$ in the offset term |
| `y_offset_multiplier` | 40.0 | Overall amplification factor of the offset term (from the sieve-offset geometry) |

> **Parameter impact analysis**:
> - A 1 cm error in `x_z_coefficient` will shift `sieve_x` by ~2.5 mm for angles of order $x'_{tar} \sim 10^{-2}$ rad—a significant effect on sieve-hole identification.
> - The value 213.0 = 138.0 + 75.0 for `y_ph_coefficient` comes from the path length after the main bending magnet plus a correction term. An error in this coefficient shifts the entire `sieve_y` pattern horizontally.
> - All coefficients were carefully fitted to real SHMS data. They should not be modified unless there is a clear physical justification (e.g., a different magnetic-field configuration).

**`DataLoadingConfig` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `feature_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | Focal-plane variable column names (used for 4D separability analysis) |
| `target_cols` | `['P_gtr_dp', 'P_gtr_th', 'P_gtr_ph', 'P_react_z']` | Target-plane variable column names |
| `sieve_x_range` | (−20.0, 20.0) | Valid range for `sieve_x` (cm); events outside this range are filtered out |
| `sieve_y_range` | (−20.0, 20.0) | Valid range for `sieve_y` (cm) |

---

### 3.3 `load_simulation_data()`

Reads COSY/SIMC simulation ROOT files, simultaneously computing the reconstructed sieve pattern and truth information.

**Simulation branches read:**

| Branch | Physical quantity |
|---|---|
| `psdelta` | Momentum deviation $\delta$ |
| `psyptar` | Horizontal angle $\phi$ |
| `psxptar` | Vertical angle $\theta$ |
| `psytar` | Target horizontal position $y_{tar}$ |
| `fry` | Beam raster Y position |
| `xsieve`, `ysieve` | True physical sieve-hole coordinates (cm) |
| `xsnum`, `ysnum` | True sieve-hole indices (row/column) |

The output **`truth_hole_id`** is computed as `xsnum * 100 + (ysnum % 100)`, forming a unique hole identifier used for comparison against clustering results.

---

## 4. Data Preprocessing Module (`preprocessing`)

### 4.1 `classify_foils_with_range()`

Automatically identifies multiple foil positions from peaks in the `P_gtr_y` distribution and classifies events accordingly.

**Background**: Experiments typically use multiple carbon foils (e.g., 3 foils at different Z positions) to extend target coverage. Events from each foil produce a distinct peak in the `P_gtr_y` distribution.

**Algorithm steps:**

1. Build a histogram of `col_name` (default `P_gtr_y`) within `y_range`, using `bins` bins.
2. Find peaks with `scipy.signal.find_peaks`: peak height ≥ maximum peak height × `peak_height_fraction`, and peaks are separated by at least `peak_distance` bins.
3. Compute the Full Width at Half Maximum (FWHM) of each peak with `peak_widths`, then convert to standard deviation: $\sigma = \text{FWHM} / 2.355$.
4. Classify events whose value falls in `[peak_center − sigma\_factor × σ,\ peak\_center + sigma\_factor × σ]` as belonging to that foil.

**`FoilClassificationConfig` parameters:**

| Parameter | Default | Description and impact |
|---|---|---|
| `col_name` | `'P_gtr_y'` | Column name used for classification. In standard Hall C analysis, `P_gtr_y` corresponds to the target horizontal position; foils at different Z positions are clearly separated in this variable. |
| `bins` | 50 | Number of histogram bins. **Too few** (e.g., 20) blurs the peak shape, making nearby foils hard to distinguish; **too many** (e.g., 200) produce spurious peaks when statistics are low. |
| `sigma_factor` | 2.5 | Classification half-width factor: events within ±N × σ of the peak center are assigned to that foil. **Increasing** it (e.g., 3.0) captures more tail events but may introduce contamination from neighboring foils; **decreasing** it (e.g., 1.5) gives purer classification but lower efficiency. |
| `y_range` | (−5.0, 5.0) | Valid range for `P_gtr_y` (cm). Should be set to cover the foil positions in the experiment. Events outside this range are labeled `foil_position = -1`. |
| `peak_height_fraction` | 0.05 | Minimum peak height relative to the tallest peak. Used to suppress spurious small peaks from statistical noise. Reduce this if a particular foil has very low statistics (e.g., an end foil). |
| `peak_distance` | 10 | Minimum separation between neighboring peaks (in histogram bins). Prevents a single broad peak from being identified as multiple peaks. |

> **Parameter impact analysis (`sigma_factor`)**:
>
> | sigma_factor | Fraction of Gaussian captured | Recommended use case |
> |---|---|---|
> | 1.0 | ~68% | Very strict; suitable when foils are very close together |
> | 2.0 | ~95% | Moderate |
> | 2.5 (default) | ~99% | Recommended; balances efficiency and purity |
> | 3.0 | ~99.7% | Relaxed; useful with low statistics |
> | 3.5+ | >99.9% | May introduce cross-foil contamination |

---

## 5. Clustering Module (`clustering`)

This module is the core of the package. It provides three families of clustering algorithms, all targeting the same goal: identifying the dense groups of events (clusters) that correspond to individual sieve holes.

### 5.1 Automatic DBSCAN Parameter Search (`auto_dbscan_clustering`)

**DBSCAN algorithm principle:**

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) defines clusters via density reachability: a point is a *core point* if at least `min_samples` points lie within its $\varepsilon$-neighborhood; all points density-reachable from a core point belong to the same cluster; points not belonging to any cluster are labeled as noise (`is_noise = True`).

**Key advantages:**
- No need to specify the number of clusters in advance (the number of visible sieve holes may be unknown due to occlusion)
- Can detect clusters of arbitrary shape
- Automatically classifies sparse-region points as noise

**Automatic optimization strategy (grid search):**

1. Generate eps candidates: 10 equally spaced values in `eps_range`.
2. Generate `min_samples` candidates: if not specified, use `len(data)/1000` as a baseline and take 50%, 75%, 100%, 125%, 150% of that value (5 candidates).
3. Run DBSCAN for every `(eps, min_samples)` combination.
4. Check physical constraints:
   - The x-extent and y-extent of each cluster must not exceed `max_cluster_size`.
   - Neighboring cluster centers must be separated by at least `distance_threshold`.
5. Among all parameter combinations that satisfy the constraints and produce a cluster count within `target_clusters`, select the one whose cluster count is closest to the midpoint of the target range.

**`DBSCANConfig` parameters:**

| Parameter | Default | Description and impact |
|---|---|---|
| `x_col` | `'sieve_x'` | Column name for the X coordinate |
| `y_col` | `'sieve_y'` | Column name for the Y coordinate |
| `eps_range` | (0.01, 0.2) | Search range for eps (cm). eps defines the neighborhood radius: **too small** → each point forms its own cluster (over-segmentation); **too large** → neighboring holes merge (under-segmentation). Sieve-hole diameters are ~0.4–0.8 cm; typical optimal eps is ~0.05–0.15 cm. |
| `target_clusters` | (50, 70) | Target cluster count range. The SHMS sieve plate hole count is determined by its geometry (typically ~63 holes in a 7×9 pattern), but edge holes may not be visible. Set this based on the actual sieve plate specifications. |
| `min_samples` | None | Minimum number of neighbors required for a core point. When `None`, automatically computed as `data_size / 1000`. **Increasing** this makes the algorithm more robust to noise, but may classify low-statistics holes as noise; **decreasing** it finds more holes but is more noise-sensitive. |
| `max_iterations` | 10 | Maximum search iterations (the current implementation uses a fixed grid search, so this parameter does not control loop iterations directly). |
| `distance_threshold` | 1.0 | Minimum distance between cluster centers (cm). Sieve-hole pitch is ~1.27–2.54 cm; setting this to 50%–80% of the hole pitch is appropriate. **Too large**: forces nearby clusters to merge; **too small**: allows multiple clusters to correspond to a single hole. |
| `max_cluster_size` | 2.2 | Maximum extent of a single cluster (cm). Corresponds to the physical size upper bound of a sieve hole. Setting this to 2–3 times the hole diameter prevents events from two adjacent holes from merging into an oversized cluster. |

---

### 5.2 Edge Region Clustering (`peel_and_cluster_edges`)

**Problem background**: Due to the non-uniform acceptance of SHMS (far higher statistics in the center than at the edges), a standard DBSCAN tuned for the overall dataset tends to classify sparse edge-region points as noise, causing edge sieve holes to be missed.

**Algorithm (convex-hull peeling):**

1. Compute the convex hull of all points belonging to the already-identified core clusters.
2. Expand the convex-hull boundary by a buffer `radius` (a positive `radius` shrinks the boundary inward so that fewer points are considered "inside").
3. Extract "edge points" that lie outside the expanded boundary.
4. Run DBSCAN independently on the edge points (with an independently searched parameter set).
5. Merge newly discovered edge clusters with the existing core clusters, renumbering them sequentially.

**`EdgeClusteringConfig` parameters:**

| Parameter | Default | Description and impact |
|---|---|---|
| `radius_candidates` | [0.3, 0.5, 0.8, 1.0, 1.5] | Candidate values for the convex-hull buffer radius (cm). **Smaller values** (0.3): peel only a thin outer shell, leaving more points available for edge clustering; **larger values** (1.5): peel a thicker edge band, appropriate when edge holes are well separated from the core. |
| `eps_candidates` | [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20] | Candidate eps values for the edge DBSCAN. Edge-region density is lower, so the optimal eps is typically slightly larger than for the core region. |
| `target_new_clusters` | (5, 15) | Target number of newly discovered edge clusters. Set according to the number of edge holes on the sieve plate (the SHMS sieve typically has 10–16 edge holes not found by the core DBSCAN pass). |
| `distance_threshold` | 1.0 | Minimum center distance between new edge clusters and existing core clusters (cm). Prevents the edge DBSCAN from re-discovering already-identified core holes. |

---

### 5.3 Two-Entry DBSCAN (`two_entry_dbscan`)

This is the recommended standard workflow, chaining core and edge clustering in sequence:

```python
# Stage 1: Core-region DBSCAN (high-density area)
df, eps_core, n_core = auto_dbscan_clustering(df, config=core_config)

# Stage 2: Edge-region peeling and clustering (low-density area)
df, eps_edge, n_total = peel_and_cluster_edges(df, config=edge_config)
```

**Return value structure:**

```python
params = {
    'core_eps':      float,   # Optimal eps for the core region
    'edge_eps':      float,   # Optimal eps for the edge region
    'core_clusters': int,     # Number of core clusters
    'edge_clusters': int      # Number of newly added edge clusters
}
```

---

### 5.4 Hierarchical HDBSCAN (`auto_hdbscan_clustering`)

**HDBSCAN algorithm principle:**

HDBSCAN (Hierarchical DBSCAN) identifies clusters by constructing a density hierarchy tree (minimum spanning tree → condensed tree). Compared with DBSCAN, it offers:
- Insensitivity to the eps parameter (automatically learns local density)
- Handles **non-uniform density** data (e.g., where the central sieve-hole region is much denser than the edges)
- Cluster-selection strategy controllable via `cluster_selection_method`

**`HDBSCANConfig` parameters:**

| Parameter | Default | Description and impact |
|---|---|---|
| `min_cluster_size_range` | (30, 100) | Search range for `min_cluster_size`. HDBSCAN's `min_cluster_size` is the minimum number of events required for a meaningful cluster. **Increasing**: ignores small clusters (holes with few events), resulting in less noise; **decreasing**: finds more small clusters, but may produce spurious ones. |
| `min_samples_range` | (30, 80) | Search range for `min_samples`. Controls how conservatively core points are defined (analogous to DBSCAN's `min_samples`); larger values are more conservative. |
| `target_clusters` | (50, 70) | Target cluster count range; same meaning as in DBSCAN. |
| `max_iterations` | 10 | Maximum search iterations (current implementation uses a fixed grid search). |
| `distance_threshold` | 1.0 | Minimum distance between cluster centers (cm). |
| `max_cluster_size` | 2.2 | Maximum cluster extent (cm). |
| `cluster_selection_method` | `'leaf'` | Cluster-selection strategy. `'leaf'`: select the finest-grained leaf nodes from the hierarchy tree (produces more, smaller, more compact clusters—ideal for sieve-hole identification); `'eom'` (Excess of Mass): selects stable large-scale structures (fewer, larger clusters—may merge adjacent holes). |
| `metric` | `'euclidean'` | Distance metric. Euclidean distance is appropriate for physical coordinates measured in cm. |
| `alpha` | 1.0 | Controls how single-linkage splits are converted into the condensed tree. Standard value is 1.0. |

> **DBSCAN vs. HDBSCAN selection guide:**
>
> | Situation | Recommendation |
> |---|---|
> | Standard data quality, roughly uniform density | Two-Entry DBSCAN (faster, more stable) |
> | Extreme center-to-edge density ratio (>10×) | HDBSCAN (adapts to local density) |
> | Need precise control over cluster count | DBSCAN (more intuitive parameters) |
> | Very large dataset (>1 million events per foil) | HDBSCAN (better memory efficiency) |

---

### 5.5 Per-Foil Batch Clustering (`cluster_by_foil_position`)

Applies the chosen clustering algorithm to each foil position separately, returning a dictionary:  
`{foil_position: {'df': DataFrame, 'params': dict, 'n_clusters': int}}`.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `method` | `'dbscan'` | Clustering method: `'dbscan'`, `'hdbscan'`, or `'two_entry'`. When `method='dbscan'` and `use_two_entry=True` (default), the Two-Entry approach is used automatically. |
| `use_two_entry` | True | Whether to enable the second-stage edge clustering when `method='dbscan'`. |
| `foil_col` | `'foil_position'` | Column name for the foil-position label. |

---

## 6. Calibration Module (`calibration`)

### 6.1 `build_grid_index_from_centers()`

Infers row/column grid indices from cluster-center coordinates. This is the key step that converts an unordered set of clustering results into a structured sieve-hole array.

**Algorithm steps:**

1. **Extract cluster centers**: Compute the mean coordinate of each cluster; filter out noise points.
2. **Merge close centers** (`merge_threshold`): Use hierarchical clustering (average linkage) to merge centers closer than the threshold.
3. **PCA alignment** (optional, `use_pca_alignment`): Rotate the point cloud onto its principal axes to remove small tilts of the sieve plate.
4. **Estimate grid spacing**: Use the median of k-nearest-neighbor distances (k=5) as the grid spacing.
5. **Determine origin**: The cluster center closest to the centroid of the point cloud is assigned `(row=0, col=0)`.
6. **Assign row/column indices**:
   ```python
   row = round((aligned_y - origin_y) / grid_spacing)
   col = round((aligned_x - origin_x) / grid_spacing)
   ```
7. **Find missing positions**: Enumerate all expected positions within the row/column range; missing ones are the set difference from the detected positions.

**`GridIndexConfig` parameters:**

| Parameter | Default | Description and impact |
|---|---|---|
| `cluster_col` | `'cluster'` | Column name for cluster labels |
| `x_col` | `'cluster_center_x'` | Column name for cluster center X coordinate |
| `y_col` | `'cluster_center_y'` | Column name for cluster center Y coordinate |
| `use_pca_alignment` | False | Whether to apply PCA axis alignment. **When enabled**: rotates the point cloud to its principal axes before assigning grid indices—appropriate when the sieve plate has a noticeable tilt (>2°); **disabled** (default): assumes sieve rows and columns are approximately parallel to the coordinate axes. |
| `merge_threshold` | 0.5 | Center-merging threshold (cm). Two centers closer than 0.5 cm are merged into one. **Too large**: may merge adjacent real holes (for a plate with ~1 cm pitch, set to 30%–40% of the pitch); **too small**: a single hole may be double-indexed. |

> **Impact of PCA alignment**:
>
> If the sieve plate is rotated ~5° relative to the coordinate axes, not applying PCA introduces a grid-index error of approximately `sin(5°) × spacing ≈ 0.09 × spacing`. For a plate with 1.27 cm pitch, this error is ~1.1 mm—still within an acceptable range. However, for tilt angles larger than ~10°, enabling PCA alignment is strongly recommended.

---

### 6.2 `get_missing_holes()`

Identifies missing positions in the sieve-hole grid.

**Parameters:**

| Parameter | Default | Description |
|---|---|---|
| `only_internal` | True | Whether to return only missing holes that lie inside the convex hull of the detected holes (i.e., truly blocked holes, not edge effects). `True`: report only holes with (row, col) positions inside the convex hull of detected holes; `False`: also report all positions in the bounding rectangle of the grid. |

---

### 6.3 `estimate_hole_positions()`

Estimates the physical coordinates of missing holes based on the fitted grid parameters (spacing, origin, rotation angle):

```python
aligned_x = origin_x + col * grid_spacing
aligned_y = origin_y + row * grid_spacing

# If PCA rotation was applied, transform back to the original coordinate system
if rotation_angle != 0:
    est_x = aligned_x * cos(-θ) - aligned_y * sin(-θ)
    est_y = aligned_x * sin(-θ) + aligned_y * cos(-θ)
```

The estimated positions can be overlaid on the original scatter plot to guide manual inspection of the raw data in the predicted location, helping to determine whether the hole is genuinely absent or was erroneously classified as noise.

---

## 7. Evaluation Module (`evaluation`)

### 7.1 `calculate_cluster_metrics()`

Computes clustering quality metrics using simulation truth labels (`truth_hole_id`).

**Metric definitions:**

- **Purity**: For cluster $C_k$, let $h^*$ be the dominant truth hole (the one with the most events in $C_k$):
  $$\text{Purity}(C_k) = \frac{|C_k \cap h^*|}{|C_k|}$$
  High purity (≈1.0) indicates that the cluster genuinely corresponds to a single sieve hole.

- **Efficiency**: For truth hole $H_j$:
  $$\text{Efficiency}(H_j) = \frac{\text{events correctly assigned to the cluster dominated by }H_j}{|H_j|}$$
  High efficiency (≈1.0) means that most events from the hole were correctly recovered.

**`BenchmarkConfig` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `truth_col` | `'truth_hole_id'` | Truth-label column name (in simulation data) |
| `cluster_col` | `'cluster'` | Clustering-result column name |

---

### 7.2 `calculate_separability_metrics()`

Evaluates cluster separability in the four-dimensional focal-plane phase space ($x_{fp}, y_{fp}, x'_{fp}, y'_{fp}$).

**Metrics computed:**

| Metric | Range | Interpretation |
|---|---|---|
| Silhouette Score | [−1, +1] | Higher is better: +1 = tight clusters well separated; 0 = clusters on the boundary; negative = points likely misassigned |
| Davies-Bouldin Index | [0, ∞) | Lower is better: small values indicate compact, well-separated clusters |
| Calinski-Harabasz Score | [0, ∞) | Higher is better: ratio of between-cluster to within-cluster variance |
| Separability Ratio | any | min_inter_cluster_dist / intra_cluster_dist; >1 indicates inter-cluster distance exceeds intra-cluster radius (good separation) |

**`SeparabilityConfig` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `fp_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | Focal-plane variable column names |
| `normalize` | True | Whether to normalize features before computing distances (using `StandardScaler`). **Strongly recommended to keep `True`**: the four focal-plane variables have different units (positions in cm, angles in rad); without normalization, position variables dominate the distance calculation. |

---

## 8. Visualization Module (`visualization`)

### 8.1 `visualize_dbscan_results()`

Plots a clustering scatter plot with signal points colored by cluster ID, noise points in black, and cluster centers marked with white crosses.

**`VisualizationConfig` parameters:**

| Parameter | Default | Description |
|---|---|---|
| `figsize` | (10, 10) | Figure size (inches) |
| `xlim` | (−20.0, 20.0) | X-axis range (cm); should match `sieve_x_range` |
| `ylim` | (−20.0, 20.0) | Y-axis range (cm) |
| `point_size` | 0.2 | Signal point size (use a small value at high event density to avoid overplotting) |
| `noise_point_size` | 0.1 | Noise point size |
| `noise_alpha` | 0.3 | Noise point transparency |
| `signal_alpha` | 0.6 | Signal point transparency |
| `cmap` | `'nipy_spectral'` | Color map. `nipy_spectral` provides high-contrast multi-color mapping suitable for distinguishing many clusters (60+ works well). |
| `grid_alpha` | 0.3 | Grid-line transparency |

### 8.2 Other Visualization Functions

| Function | Purpose |
|---|---|
| `visualize_clustering_summary()` | 2×2 panel: clustering results + cluster-center distribution + cluster-size histogram + statistics text |
| `visualize_clusters_in_focal_plane()` | Show clustering results projected onto focal-plane variables (4 subplots: $x_{fp}$-$y_{fp}$, $x_{fp}$-$x'_{fp}$, etc.) |
| `visualize_foil_classification()` | `P_gtr_y` distribution histogram with foil classification ranges annotated |
| `visualize_sieve_plane()` | 2D heat map of sieve-plane coordinates |
| `visualize_benchmark_comparison()` | Bar charts comparing metrics across algorithms |
| `plot_efficiency_map()` | Efficiency heat map at sieve-hole grid positions (requires grid index) |

---

## 9. Configuration System (`config`)

### 9.1 Configuration Class Structure

All configuration classes are Python `dataclass` instances that can be instantiated and customized directly:

```python
from shms_optics_calibration import DBSCANConfig

# Default configuration
config = DBSCANConfig()

# Custom configuration
config = DBSCANConfig(
    eps_range=(0.05, 0.25),
    target_clusters=(55, 75),
    distance_threshold=1.2,
    max_cluster_size=2.5
)
```

### 9.2 Configuration Priority

When a function accepts both individual keyword arguments and a `config` object, **the `config` object takes priority** (overrides the individual arguments):

```python
# The config object overrides eps_range and target_clusters
df, eps, n = auto_dbscan_clustering(
    df,
    eps_range=(0.01, 0.1),    # Ignored when config is provided
    target_clusters=(40, 50), # Ignored when config is provided
    config=DBSCANConfig(eps_range=(0.05, 0.2), target_clusters=(55, 65))
)
```

### 9.3 Global Constants

| Constant | Value | Description |
|---|---|---|
| `RANDOM_SEED` | 42 | Random seed for reproducibility |
| `FEATURE_COLS` | `['P_dc_x_fp', ...]` | List of focal-plane feature variables |
| `TARGET_COLS` | `['P_gtr_dp', ...]` | List of target-plane variables |

---

## 10. Impact of Parameter Choices on Results

### 10.1 Effect of the DBSCAN `eps` Parameter

`eps` is the most critical DBSCAN parameter and directly determines the clustering granularity:

```
eps too small (< 0.03 cm):
  ↓ Neighborhood radius too small
  ↓ Many normal events classified as noise
  ↓ Each sieve hole may split into multiple sub-clusters
  Result: cluster count >> hole count; high purity but very low efficiency

eps in the optimal range (0.05–0.15 cm, recommended):
  ↓ Dense sieve-hole regions correctly identified
  ↓ Noise events (secondary scattering, mis-reconstruction) correctly excluded
  Result: cluster count ≈ hole count; both purity and efficiency are high

eps too large (> 0.2 cm):
  ↓ Adjacent sieve holes are merged
  ↓ Noise events are absorbed into clusters
  Result: cluster count << hole count; inflated efficiency but very low purity
```

**Role of automatic search**: `auto_dbscan_clustering` uses grid search to find the optimal `eps` that satisfies the physical constraints (`max_cluster_size`, `distance_threshold`) and keeps the cluster count within `target_clusters`, eliminating the need for manual trial-and-error tuning.

### 10.2 Effect of `target_clusters`

```
target_clusters set too low (e.g., 40–50 when the actual hole count is 63):
  → Algorithm prefers larger eps
  → Adjacent holes merge; real holes are missed

target_clusters set too high (e.g., 80–100):
  → Algorithm uses smaller eps
  → Holes are over-segmented into sub-regions
  → Post-processing merging is required

Recommended strategy:
  1. Consult the sieve-plate geometry documentation to determine the hole count (e.g., 63)
  2. Estimate how many holes lie outside the acceptance (typically 5–10 edge holes)
  3. Set target_clusters = (lower bound of visible holes, total hole count)
```

### 10.3 Advantages of Two-Entry DBSCAN

Single-pass DBSCAN fails in the following situations:
- **Center-to-edge density ratio > 5:1**: the central holes have far more statistics than the edge holes.
- Enlarging `eps` to detect edge holes causes central holes to merge.

The two-entry strategy:
1. Use a strict `eps` (suited for the high-density center region) to identify core holes.
2. Apply a looser `eps` to only the remaining edge points to identify residual edge holes.
3. Result: both central and edge holes are accurately identified.

### 10.4 Effect of HDBSCAN `cluster_selection_method`

```
'leaf' (leaf-node selection, recommended default):
  → Selects clusters at the finest granularity of the hierarchy tree
  → Produces more, smaller, more compact clusters
  → Ideal for sieve-hole detection (holes are physically distinct, small clusters)

'eom' (Excess of Mass):
  → Selects "stable" large-scale structures
  → Produces fewer, larger clusters
  → May merge adjacent holes
  → Better suited for detecting continuous structures (not appropriate for discrete holes)
```

### 10.5 Effect of `sigma_factor` on Foil Classification

Example with three foils at positions ≈ −2.5, 0, +2.5 cm and peak width $\sigma \approx 0.5$ cm:

```
sigma_factor = 1.5:
  Foil 0 classification range: [−3.25, −1.75]
  Unclassified gap: [−1.75, −0.75] (events labeled −1)
  → Strict, but ~13% of events may be lost

sigma_factor = 2.5 (default):
  Foil 0 classification range: [−3.75, −1.25]
  Foil 1 range [−1.25, +1.25] just touches foil 0
  → Recommended

sigma_factor = 3.0:
  Foil 0 classification range: [−4.0, −1.0]
  Foil 1 range [−1.5, +1.5] overlaps foil 0 by 0.5 cm
  → Cross-foil contamination; holes from two foils become mixed in clustering
```

### 10.6 Effect of `merge_threshold` on Grid Indexing

```
merge_threshold = 0 (no merging):
  → If one hole splits into two sub-clusters (when eps is too small),
    two centers < 0.5 cm apart may be assigned the same grid position,
    causing an index conflict

merge_threshold = 0.5 cm (default, ~40% of hole pitch):
  → Merges truly duplicate nearby centers
  → Adjacent real holes (~1.27 cm apart) are preserved as separate entries

merge_threshold = 1.5 cm (too large):
  → Merges adjacent holes with pitch ~1.27 cm
  → Severe loss of grid-index entries
```

---

## 11. Complete Workflow Examples

### 11.1 Standard Experimental Data Analysis

```python
from shms_optics_calibration import (
    load_and_prepare_data,
    classify_foils_with_range,
    cluster_by_foil_position,
    build_full_grid_index,
    get_missing_holes,
    estimate_hole_positions,
    visualize_dbscan_results,
    DBSCANConfig,
    EdgeClusteringConfig,
    FoilClassificationConfig,
    TargetProjectionConfig,
)

# ---- Step 1: Load data ----
df = load_and_prepare_data(
    "run12345.root",
    tree_name="T",
    add_projection=True,
    filter_range=True
)
print(f"Events loaded: {len(df):,}")

# ---- Step 2: Foil classification ----
foil_config = FoilClassificationConfig(
    bins=60,              # More bins for better peak resolution
    sigma_factor=2.5,     # Keep default
    y_range=(-6.0, 6.0),  # Wider range to cover three foil positions
    peak_distance=8       # Reduce if foil spacing is small
)
df = classify_foils_with_range(df, config=foil_config, plot=True)
print(f"Foils identified: {df['foil_position'].nunique() - 1}")  # Exclude -1

# ---- Step 3: Configure clustering parameters ----
# Core-region DBSCAN
core_config = DBSCANConfig(
    eps_range=(0.03, 0.18),
    target_clusters=(45, 65),    # Expect ~55 visible core holes
    distance_threshold=0.9,      # ~70% of the hole pitch (~1.27 cm)
    max_cluster_size=2.0         # ~3× hole diameter (~0.4 cm)
)

# Edge-region supplemental clustering
edge_config = EdgeClusteringConfig(
    radius_candidates=[0.3, 0.6, 1.0],
    eps_candidates=[0.06, 0.10, 0.15, 0.20],
    target_new_clusters=(5, 12)
)

# ---- Step 4: Cluster by foil position ----
results = cluster_by_foil_position(
    df,
    method='two_entry',       # Recommended two-entry approach
    dbscan_config=core_config,
    edge_config=edge_config,
    verbose=True
)

# ---- Step 5: Build grid index ----
from shms_optics_calibration import GridIndexConfig

grid_config = GridIndexConfig(
    use_pca_alignment=False,  # False when sieve plate has no significant tilt
    merge_threshold=0.4       # ~30% of hole pitch (~1.27 cm)
)
full_index, grid_params = build_full_grid_index(
    results,
    config=grid_config
)

# ---- Step 6: Inspect missing holes ----
for foil_pos, result in results.items():
    centers = full_index[full_index['foil_position'] == foil_pos]
    params = grid_params[foil_pos]

    missing = get_missing_holes(centers, params, only_internal=True)
    if missing:
        estimated = estimate_hole_positions(centers, params, missing)
        print(f"Foil {foil_pos}: {len(missing)} internally missing hole(s)")
        print(estimated[['row', 'col', 'estimated_x', 'estimated_y']])

# ---- Step 7: Visualize ----
for foil_pos, result in results.items():
    fig = visualize_dbscan_results(
        result['df'],
        n_clusters=result['n_clusters'],
        title_prefix=f"Foil {foil_pos}",
        show=True
    )
```

### 11.2 Simulation Data Benchmark

```python
from shms_optics_calibration import (
    load_simulation_data,
    auto_dbscan_clustering,
    auto_hdbscan_clustering,
    two_entry_dbscan,
    calculate_cluster_metrics,
    calculate_separability_metrics,
    compare_algorithms,
)

# Load simulation data with truth labels
df_sim = load_simulation_data("simc_output.root")
print(f"Simulation events: {len(df_sim):,}")
print(f"Truth holes: {df_sim['truth_hole_id'].nunique()}")

# Run three algorithms for comparison
results_comparison = {}

# Method 1: DBSCAN
df_db = df_sim.copy()
df_db, eps, n = auto_dbscan_clustering(df_db, target_clusters=(55, 70))
cm_db, tm_db, overall_db = calculate_cluster_metrics(df_db)
results_comparison['DBSCAN'] = {
    'cluster_metrics': cm_db,
    'truth_metrics': tm_db,
    'overall': overall_db
}

# Method 2: Two-Entry DBSCAN
df_2e = df_sim.copy()
df_2e, params_2e, n_2e = two_entry_dbscan(df_2e)
cm_2e, tm_2e, overall_2e = calculate_cluster_metrics(df_2e)
results_comparison['Two-Entry DBSCAN'] = {
    'cluster_metrics': cm_2e,
    'truth_metrics': tm_2e,
    'overall': overall_2e
}

# Method 3: HDBSCAN
df_hd = df_sim.copy()
df_hd, params_hd, n_hd = auto_hdbscan_clustering(
    df_hd,
    min_cluster_size_range=(30, 80),
    target_clusters=(55, 70)
)
cm_hd, tm_hd, overall_hd = calculate_cluster_metrics(df_hd)
results_comparison['HDBSCAN'] = {
    'cluster_metrics': cm_hd,
    'truth_metrics': tm_hd,
    'overall': overall_hd
}

# Generate comparison table
comparison_table = compare_algorithms(results_comparison)
print(comparison_table.to_string(index=False))

# 4D focal-plane separability analysis
global_metrics, per_cluster = calculate_separability_metrics(
    df_2e,
    normalize=True  # Must normalize (different units)
)
print(f"Silhouette Score: {global_metrics['silhouette_score']:.4f}")
print(f"Mean Separability Ratio: {global_metrics['mean_separability_ratio']:.3f}")
```

---

## 12. Parameter Quick-Reference Tables

### 12.1 All Configuration Class Parameters

#### `DataLoadingConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `feature_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | list | Focal-plane feature variables |
| `target_cols` | `['P_gtr_dp', 'P_gtr_th', 'P_gtr_ph', 'P_react_z']` | list | Target-plane variables |
| `sieve_x_range` | (−20.0, 20.0) | tuple | `sieve_x` filter range (cm) |
| `sieve_y_range` | (−20.0, 20.0) | tuple | `sieve_y` filter range (cm) |

#### `TargetProjectionConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `x_z_coefficient` | 253.0 | float | Sieve-to-target Z distance (cm) |
| `y_dp_linear` | −0.019 | float | First-order dispersion correction coefficient for δ |
| `y_dp_quadratic` | 0.00019 | float | Second-order dispersion correction coefficient for δ |
| `y_ph_coefficient` | 213.0 | float | φ contribution coefficient |
| `y_offset_dp_linear` | −0.00052 | float | First-order δ coefficient in offset term |
| `y_offset_dp_quadratic` | 0.0000052 | float | Second-order δ coefficient in offset term |
| `y_offset_multiplier` | 40.0 | float | Amplification factor of offset term |

#### `FoilClassificationConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `col_name` | `'P_gtr_y'` | str | Column name used for classification |
| `bins` | 50 | int | Number of histogram bins |
| `sigma_factor` | 2.5 | float | Classification half-width factor (× σ) |
| `y_range` | (−5.0, 5.0) | tuple | Valid value range (cm) |
| `peak_height_fraction` | 0.05 | float | Minimum peak height relative to maximum |
| `peak_distance` | 10 | int | Minimum distance between adjacent peaks (bins) |

#### `DBSCANConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `x_col` | `'sieve_x'` | str | X coordinate column name |
| `y_col` | `'sieve_y'` | str | Y coordinate column name |
| `eps_range` | (0.01, 0.2) | tuple | eps search range (cm) |
| `target_clusters` | (50, 70) | tuple | Target cluster count range |
| `min_samples` | None | int\|None | Minimum neighbor count (None = auto) |
| `max_iterations` | 10 | int | Maximum search iterations |
| `distance_threshold` | 1.0 | float | Minimum cluster-center separation (cm) |
| `max_cluster_size` | 2.2 | float | Maximum single-cluster extent (cm) |

#### `EdgeClusteringConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `radius_candidates` | [0.3, 0.5, 0.8, 1.0, 1.5] | list | Convex-hull buffer radius candidates (cm) |
| `eps_candidates` | [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20] | list | Edge eps candidates (cm) |
| `target_new_clusters` | (5, 15) | tuple | Target count of newly found edge clusters |
| `distance_threshold` | 1.0 | float | Minimum separation between new and existing clusters (cm) |

#### `HDBSCANConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `x_col` | `'sieve_x'` | str | X coordinate column name |
| `y_col` | `'sieve_y'` | str | Y coordinate column name |
| `min_cluster_size_range` | (30, 100) | tuple | Search range for `min_cluster_size` |
| `min_samples_range` | (30, 80) | tuple\|None | Search range for `min_samples` (None = HDBSCAN default) |
| `target_clusters` | (50, 70) | tuple | Target cluster count range |
| `max_iterations` | 10 | int | Maximum search iterations |
| `distance_threshold` | 1.0 | float | Minimum cluster-center separation (cm) |
| `max_cluster_size` | 2.2 | float | Maximum single-cluster extent (cm) |
| `cluster_selection_method` | `'leaf'` | str | Cluster-selection method (`'leaf'` or `'eom'`) |
| `metric` | `'euclidean'` | str | Distance metric |
| `alpha` | 1.0 | float | HDBSCAN alpha parameter |

#### `SoftWeightedDBSCANConfig` (experimental)

> This configuration class is used for a soft-weighted partitioned DBSCAN that applies different weights to points in different `P_gtr_y` regions, processing all foils in a single global clustering pass. Currently an experimental feature.

| Parameter | Default | Description |
|---|---|---|
| `gtry_col` | `'P_gtr_y'` | Column name used for region partitioning |
| `gtry_ranges` | `{'-2.5±1': (-3.5,-1.5), '0±1': (-1,1), '+2.5±1': (1.5,3.5)}` | `P_gtr_y` regions corresponding to three foils |
| `suppression_strength_grid` | (0.7, 0.8, 0.9) | Grid of region suppression strengths to search |
| `sigma_grid` | (0.8, 1.0, 1.2) | Grid of Gaussian weight σ values to search (cm) |
| `active_threshold_grid` | (0.3, 0.4, 0.5) | Grid of activation weight thresholds to search |
| `eps_grid` | (0.05, 0.08, …, 0.20) | Grid of eps values to search (cm) |
| `min_samples_grid` | (30, 50, 70, 90) | Grid of `min_samples` values to search |
| `assign_radius` | 0.8 | Search radius for assigning points to new clusters (cm) |

#### `GridIndexConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `cluster_col` | `'cluster'` | str | Cluster label column name |
| `x_col` | `'cluster_center_x'` | str | Cluster center X column name |
| `y_col` | `'cluster_center_y'` | str | Cluster center Y column name |
| `use_pca_alignment` | False | bool | Whether to enable PCA axis alignment |
| `merge_threshold` | 0.5 | float | Center-merging threshold (cm) |

#### `VisualizationConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `figsize` | (10, 10) | tuple | Figure size (inches) |
| `xlim` | (−20.0, 20.0) | tuple | X-axis range (cm) |
| `ylim` | (−20.0, 20.0) | tuple | Y-axis range (cm) |
| `point_size` | 0.2 | float | Signal point size |
| `noise_point_size` | 0.1 | float | Noise point size |
| `noise_alpha` | 0.3 | float | Noise point transparency |
| `signal_alpha` | 0.6 | float | Signal point transparency |
| `cmap` | `'nipy_spectral'` | str | Color map name |
| `grid_alpha` | 0.3 | float | Grid-line transparency |

#### `BenchmarkConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `truth_col` | `'truth_hole_id'` | str | Truth-label column name |
| `cluster_col` | `'cluster'` | str | Clustering-result column name |

#### `SeparabilityConfig`

| Parameter | Default | Type | Description |
|---|---|---|---|
| `fp_cols` | `['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']` | list | Focal-plane feature variables |
| `normalize` | True | bool | Whether to normalize features before computing distances |

---

### 12.2 Recommended Parameter Configurations by Use Case

#### Scenario A: Standard SHMS sieve plate (~63 holes, pitch ~1.27 cm)

```python
DBSCANConfig(
    eps_range=(0.03, 0.18),
    target_clusters=(50, 68),
    distance_threshold=0.9,
    max_cluster_size=2.0
)
EdgeClusteringConfig(
    radius_candidates=[0.3, 0.5, 0.8],
    target_new_clusters=(5, 15)
)
FoilClassificationConfig(
    sigma_factor=2.5,
    y_range=(-5.0, 5.0)
)
```

#### Scenario B: Low statistics (< 100 events per hole)

```python
DBSCANConfig(
    min_samples=5,           # Relax minimum-neighbor requirement
    eps_range=(0.05, 0.25),  # Allow a larger neighborhood
    target_clusters=(40, 65)
)
HDBSCANConfig(              # Or switch to HDBSCAN
    min_cluster_size_range=(5, 30),
    cluster_selection_method='leaf'
)
```

#### Scenario C: Very high statistics (> 10,000 events per hole)

```python
DBSCANConfig(
    min_samples=100,         # Stricter core-point requirement; suppress edge fluctuations
    eps_range=(0.03, 0.12),  # Narrower search range for faster computation
    max_cluster_size=1.8     # Tighter size constraint
)
```

#### Scenario D: Significantly tilted sieve plate (> 5°)

```python
GridIndexConfig(
    use_pca_alignment=True,
    merge_threshold=0.4
)
```

---

## References

1. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). *A density-based algorithm for discovering clusters in large spatial databases with noise*. KDD, 226–231.
2. Campello, R. J., Moulavi, D., & Sander, J. (2013). *Density-based clustering based on hierarchical density estimates*. PAKDD, 160–172.
3. Jefferson Lab Hall C: https://www.jlab.org/physics/hall-c
4. SHMS optics references: https://github.com/hszumila/SHMS_optics
5. JLab Hall C hcana analysis software: https://github.com/JeffersonLab/hcana
