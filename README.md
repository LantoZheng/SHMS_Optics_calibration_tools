# SHMS Optics Calibration Package

A Python package for SHMS (Super High Momentum Spectrometer) optics calibration using machine learning clustering algorithms.

## Features

- **Data Loading**: Load and preprocess ROOT file data from SHMS experiments
- **Foil Classification**: Automatically classify foil positions based on P_gtr_y distribution
- **Clustering Algorithms**:
  - DBSCAN with automatic parameter optimization
  - HDBSCAN hierarchical clustering
  - Two-Entry DBSCAN (core + edge clustering)
- **Visualization**: Comprehensive plotting functions for clustering results
- **Calibration**: Grid indexing and alignment for sieve hole patterns
- **Evaluation**: Benchmark metrics (efficiency, purity, separability)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/LantoZheng/AI-ML-R-SIDIS.git
cd AI-ML-R-SIDIS

# Install the package
pip install .

# Or install in development mode
pip install -e .
```

### With Optional Dependencies

```bash
# Install with visualization support
pip install .[viz]

# Install with ROOT file reading support
pip install .[root]

# Install with HDBSCAN support
pip install .[hdbscan]

# Install all optional dependencies
pip install .[all]
```

## Dependencies

### Required
- numpy >= 1.20.0
- pandas >= 1.3.0
- scipy >= 1.7.0
- scikit-learn >= 1.0.0

### Optional
- matplotlib >= 3.4.0 (visualization)
- seaborn >= 0.11.0 (visualization)
- uproot >= 4.0.0 (ROOT file reading)
- hdbscan >= 0.8.0 (HDBSCAN clustering)

## Quick Start

### Basic Usage

```python
from shms_optics_calibration import (
    load_and_prepare_data,
    classify_foils_with_range,
    auto_dbscan_clustering,
    visualize_dbscan_results
)

# Load data from ROOT file
df = load_and_prepare_data("path/to/data.root")

# Classify foil positions
df = classify_foils_with_range(df, y_range=(-5, 5))

# Get data for a specific foil
df_foil0 = df[df['foil_position'] == 0].copy()

# Run clustering
df_clustered, eps, n_clusters = auto_dbscan_clustering(
    df_foil0,
    target_clusters=(50, 60)
)

# Visualize results
visualize_dbscan_results(
    df_clustered,
    best_eps=eps,
    n_clusters=n_clusters
)
```

### Clustering by All Foil Positions

```python
from shms_optics_calibration import (
    load_and_prepare_data,
    classify_foils_with_range,
    cluster_by_foil_position,
    build_full_grid_index
)

# Load and prepare data
df = load_and_prepare_data("data.root")
df = classify_foils_with_range(df)

# Cluster all foil positions
results = cluster_by_foil_position(df, method='two_entry')

# Build grid indices
grid_index, params = build_full_grid_index(results)

# View results
print(grid_index[['foil_position', 'cluster', 'row', 'col']].head(20))
```

### Using Configuration Objects

```python
from shms_optics_calibration import (
    DBSCANConfig,
    auto_dbscan_clustering
)

# Create custom configuration
config = DBSCANConfig(
    eps_range=(0.05, 0.25),
    target_clusters=(60, 80),
    distance_threshold=1.5,
    max_cluster_size=2.5
)

# Use configuration
df_clustered, eps, n = auto_dbscan_clustering(df, config=config)
```

### Benchmark Evaluation (with Simulation Data)

```python
from shms_optics_calibration import (
    load_simulation_data,
    auto_dbscan_clustering,
    calculate_cluster_metrics
)

# Load simulation with truth labels
df_sim = load_simulation_data("simulation.root")

# Run clustering
df_clustered, _, _ = auto_dbscan_clustering(df_sim)

# Evaluate against truth
cluster_metrics, truth_metrics, overall = calculate_cluster_metrics(
    df_clustered,
    truth_col='truth_hole_id'
)

print(f"Mean Purity: {overall['mean_purity']:.4f}")
print(f"Mean Efficiency: {overall['mean_efficiency']:.4f}")
```

## Module Documentation

### Data I/O (`data_io`)

| Function | Description |
|----------|-------------|
| `load_root_file` | Load ROOT file into pandas DataFrame |
| `load_and_prepare_data` | Load, project, and filter data |
| `load_simulation_data` | Load simulation data with truth labels |
| `project_to_sieve` | Calculate sieve-plane projection |
| `add_sieve_projection` | Add sieve_x, sieve_y columns |
| `filter_sieve_range` | Filter data to sieve-plane range |

### Preprocessing (`preprocessing`)

| Function | Description |
|----------|-------------|
| `classify_foils_with_range` | Classify foil positions from P_gtr_y |
| `get_foil_positions` | Get list of valid foil positions |
| `get_foil_subset` | Extract data for specific foil |

### Clustering (`clustering`)

| Function | Description |
|----------|-------------|
| `auto_dbscan_clustering` | DBSCAN with automatic parameter search |
| `peel_and_cluster_edges` | Second-stage edge clustering |
| `two_entry_dbscan` | Combined core + edge clustering |
| `auto_hdbscan_clustering` | HDBSCAN with parameter search |
| `cluster_by_foil_position` | Apply clustering to each foil |

### Visualization (`visualization`)

| Function | Description |
|----------|-------------|
| `visualize_dbscan_results` | Plot clustering scatter plot |
| `visualize_clustering_summary` | Multi-panel clustering summary |
| `visualize_clusters_in_focal_plane` | Focal plane projections |
| `visualize_foil_classification` | Foil classification histogram |
| `visualize_sieve_plane` | Sieve-plane 2D histogram |
| `visualize_benchmark_comparison` | Algorithm comparison plots |
| `plot_efficiency_map` | Efficiency map visualization |

### Calibration (`calibration`)

| Function | Description |
|----------|-------------|
| `build_grid_index_from_centers` | Build row/col grid from cluster centers |
| `get_grid_occupancy_table` | Create occupancy pivot table |
| `get_missing_holes` | Identify missing sieve holes |
| `estimate_hole_positions` | Estimate positions of missing holes |
| `build_full_grid_index` | Build grid for all foil positions |

### Evaluation (`evaluation`)

| Function | Description |
|----------|-------------|
| `calculate_cluster_metrics` | Compute efficiency and purity |
| `calculate_separability_metrics` | Compute 4D separability |
| `compare_algorithms` | Create algorithm comparison table |
| `get_low_performance_holes` | Find low-efficiency holes |
| `get_low_purity_clusters` | Find low-purity clusters |

## Configuration Classes

All configuration classes support dataclass features and can be customized:

| Config Class | Description |
|--------------|-------------|
| `DataLoadingConfig` | Data loading parameters |
| `TargetProjectionConfig` | Projection formula coefficients |
| `FoilClassificationConfig` | Foil classification parameters |
| `DBSCANConfig` | DBSCAN clustering parameters |
| `EdgeClusteringConfig` | Edge clustering parameters |
| `HDBSCANConfig` | HDBSCAN clustering parameters |
| `VisualizationConfig` | Plotting parameters |
| `GridIndexConfig` | Grid indexing parameters |
| `BenchmarkConfig` | Benchmark evaluation parameters |
| `SeparabilityConfig` | Separability analysis parameters |

Default configuration instances are available as `DEFAULT_*_CONFIG`.

## Default Parameters

Key default parameters based on the original notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eps_range` | (0.01, 0.2) | DBSCAN eps search range |
| `target_clusters` | (50, 70) | Target cluster count range |
| `distance_threshold` | 1.0 cm | Min cluster center distance |
| `max_cluster_size` | 2.2 cm | Max cluster extent |
| `y_range` (foil) | (-5, 5) | P_gtr_y range for foil classification |
| `sigma_factor` | 2.5 | Foil classification width factor |

## Physics Background

The SHMS (Super High Momentum Spectrometer) is part of the experimental equipment at Jefferson Lab's Hall C. This package is designed for optics calibration using data from a sieve slit collimator.

### Key Variables

- **Focal Plane**: P_dc_x_fp, P_dc_y_fp, P_dc_xp_fp, P_dc_yp_fp
- **Target Variables**: P_gtr_x, P_gtr_y, P_gtr_th, P_gtr_ph, P_gtr_dp
- **Sieve Coordinates**: sieve_x, sieve_y (reconstructed sieve-plane pattern)

### Sieve-Plane Projection Formula

```
sieve_x = x + th * 253.0

sieve_y = (-0.019 * dp + 0.00019 * dp² + 213 * ph + y)
         + 40.0 * (-0.00052 * dp + 0.0000052 * dp² + ph)
```

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

## References

- Jefferson Lab Hall C: https://www.jlab.org/physics/hall-c
- DBSCAN: Ester et al. (1996)
- HDBSCAN: Campello et al. (2013)
- Slides for this :https://docs.google.com/presentation/d/1iP7oNQDBm0SIf55r50uZIg5BEsk-dZVPX9HUFhJIWe8/edit?usp=sharing
