"""
SHMS Optics Calibration Package
================================

A Python package for SHMS (Super High Momentum Spectrometer) optics
calibration using machine learning clustering algorithms.

This package provides tools for:
- Loading and preprocessing ROOT file data
- Foil position classification
- Clustering analysis (DBSCAN, HDBSCAN, Two-Entry DBSCAN)
- Grid indexing and calibration alignment
- Benchmark evaluation and visualization

Quick Start
-----------
>>> from shms_optics_calibration import (
...     load_and_prepare_data,
...     classify_foils_with_range,
...     auto_dbscan_clustering,
...     visualize_dbscan_results
... )
>>> 
>>> # Load data
>>> df = load_and_prepare_data("data.root")
>>> 
>>> # Classify foil positions
>>> df = classify_foils_with_range(df)
>>> 
>>> # Cluster for each foil position
>>> df_foil0 = df[df['foil_position'] == 0].copy()
>>> df_clustered, eps, n_clusters = auto_dbscan_clustering(df_foil0)
>>> 
>>> # Visualize results
>>> visualize_dbscan_results(df_clustered, best_eps=eps, n_clusters=n_clusters)

Module Overview
---------------
data_io : Data loading and file I/O
    Functions for loading ROOT files and preparing data.

preprocessing : Data preprocessing
    Foil classification and data preparation for clustering.

clustering : Clustering algorithms
    DBSCAN, HDBSCAN, and Two-Entry clustering implementations.

visualization : Plotting and visualization
    Functions for visualizing clustering results and distributions.

calibration : Grid indexing and alignment
    Functions for building grid indices from cluster centers.

evaluation : Benchmark evaluation
    Functions for computing efficiency, purity, and separability metrics.

config : Configuration and parameters
    Configuration classes with default parameters.

Version History
---------------
0.1.0 (2024)
    Initial release with core functionality extracted from
    SHMS_Optics_Calibration.ipynb notebook.

References
----------
- SHMS (Super High Momentum Spectrometer) at Jefferson Lab
- DBSCAN: Ester et al. (1996) "A Density-Based Algorithm for
  Discovering Clusters"
- HDBSCAN: Campello et al. (2013) "Hierarchical Density Estimates
  for Data Clustering"
"""

__version__ = '0.1.0'
__author__ = 'SHMS Optics Calibration Team'
__license__ = 'MIT'

# ============================================================
# Data I/O Exports
# ============================================================
from .data_io import (
    load_root_file,
    get_root_file_info,
    project_to_sieve,
    add_sieve_projection,
    filter_sieve_range,
    filter_branch_ranges,
    load_and_prepare_data,
    load_simulation_data,
)

# ============================================================
# Preprocessing Exports
# ============================================================
from .preprocessing import (
    classify_foils_with_range,
    get_foil_positions,
    get_foil_subset,
    prepare_clustering_data,
    initialize_clustering_columns,
)

# ============================================================
# Clustering Exports
# ============================================================
from .clustering import (
    auto_dbscan_clustering,
    peel_and_cluster_edges,
    two_entry_dbscan,
    auto_hdbscan_clustering,
    cluster_by_foil_position,
    suggest_adaptive_clustering_configs,
)

# ============================================================
# Visualization Exports
# ============================================================
from .visualization import (
    visualize_dbscan_results,
    visualize_clustering_summary,
    visualize_clusters_in_focal_plane,
    visualize_foil_classification,
    visualize_sieve_plane,
    visualize_benchmark_comparison,
    plot_efficiency_map,
)

# ============================================================
# Calibration Exports
# ============================================================
from .calibration import (
    build_grid_index_from_centers,
    get_grid_occupancy_table,
    get_missing_holes,
    estimate_hole_positions,
    build_full_grid_index,
    get_row_statistics,
)

# ============================================================
# Evaluation Exports
# ============================================================
from .evaluation import (
    calculate_cluster_metrics,
    calculate_separability_metrics,
    compare_algorithms,
    get_low_performance_holes,
    get_low_purity_clusters,
)

# ============================================================
# Configuration Exports
# ============================================================
from .config import (
    # Configuration classes
    DataLoadingConfig,
    TargetProjectionConfig,
    FoilClassificationConfig,
    DBSCANConfig,
    EdgeClusteringConfig,
    HDBSCANConfig,
    SoftWeightedDBSCANConfig,
    VisualizationConfig,
    GridIndexConfig,
    BenchmarkConfig,
    SeparabilityConfig,
    # Default instances
    DEFAULT_DATA_LOADING_CONFIG,
    DEFAULT_TARGET_PROJECTION_CONFIG,
    DEFAULT_FOIL_CLASSIFICATION_CONFIG,
    DEFAULT_DBSCAN_CONFIG,
    DEFAULT_EDGE_CLUSTERING_CONFIG,
    DEFAULT_HDBSCAN_CONFIG,
    DEFAULT_SOFT_WEIGHTED_DBSCAN_CONFIG,
    DEFAULT_VISUALIZATION_CONFIG,
    DEFAULT_GRID_INDEX_CONFIG,
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_SEPARABILITY_CONFIG,
    # Constants
    RANDOM_SEED,
    FEATURE_COLS,
    TARGET_COLS,
    DEFAULT_BRANCHES,
    DEFAULT_PRESERVED_COLS,
)

# ============================================================
# All Public Exports
# ============================================================
__all__ = [
    # Version info
    '__version__',
    '__author__',
    '__license__',
    
    # Data I/O
    'load_root_file',
    'get_root_file_info',
    'project_to_sieve',
    'add_sieve_projection',
    'filter_sieve_range',
    'filter_branch_ranges',
    'load_and_prepare_data',
    'load_simulation_data',
    
    # Preprocessing
    'classify_foils_with_range',
    'get_foil_positions',
    'get_foil_subset',
    'prepare_clustering_data',
    'initialize_clustering_columns',
    
    # Clustering
    'auto_dbscan_clustering',
    'peel_and_cluster_edges',
    'two_entry_dbscan',
    'auto_hdbscan_clustering',
    'cluster_by_foil_position',
    'suggest_adaptive_clustering_configs',
    
    # Visualization
    'visualize_dbscan_results',
    'visualize_clustering_summary',
    'visualize_clusters_in_focal_plane',
    'visualize_foil_classification',
    'visualize_sieve_plane',
    'visualize_benchmark_comparison',
    'plot_efficiency_map',
    
    # Calibration
    'build_grid_index_from_centers',
    'get_grid_occupancy_table',
    'get_missing_holes',
    'estimate_hole_positions',
    'build_full_grid_index',
    'get_row_statistics',
    
    # Evaluation
    'calculate_cluster_metrics',
    'calculate_separability_metrics',
    'compare_algorithms',
    'get_low_performance_holes',
    'get_low_purity_clusters',
    
    # Configuration classes
    'DataLoadingConfig',
    'TargetProjectionConfig',
    'FoilClassificationConfig',
    'DBSCANConfig',
    'EdgeClusteringConfig',
    'HDBSCANConfig',
    'SoftWeightedDBSCANConfig',
    'VisualizationConfig',
    'GridIndexConfig',
    'BenchmarkConfig',
    'SeparabilityConfig',
    
    # Default configurations
    'DEFAULT_DATA_LOADING_CONFIG',
    'DEFAULT_TARGET_PROJECTION_CONFIG',
    'DEFAULT_FOIL_CLASSIFICATION_CONFIG',
    'DEFAULT_DBSCAN_CONFIG',
    'DEFAULT_EDGE_CLUSTERING_CONFIG',
    'DEFAULT_HDBSCAN_CONFIG',
    'DEFAULT_SOFT_WEIGHTED_DBSCAN_CONFIG',
    'DEFAULT_VISUALIZATION_CONFIG',
    'DEFAULT_GRID_INDEX_CONFIG',
    'DEFAULT_BENCHMARK_CONFIG',
    'DEFAULT_SEPARABILITY_CONFIG',
    
    # Constants
    'RANDOM_SEED',
    'FEATURE_COLS',
    'TARGET_COLS',
    'DEFAULT_BRANCHES',
    'DEFAULT_PRESERVED_COLS',
]
