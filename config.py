"""
Configuration and default parameters for SHMS Optics Calibration.

This module contains all default parameters used throughout the package,
extracted from the SHMS_Optics_Calibration.ipynb notebook.

Default parameters can be overridden by passing custom values to functions.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional

# Random seed for reproducibility
RANDOM_SEED = 42

# ============================================================
# Feature and Target Column Names
# ============================================================

#: Focal plane variables (measured by drift chambers)
FEATURE_COLS = ['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']

#: Target variables (physics quantities at reaction vertex)
TARGET_COLS = ['P_gtr_dp', 'P_gtr_th', 'P_gtr_ph', 'P_react_z']

#: Default branches to read from ROOT files.
#: Includes focal plane variables, target reconstruction variables,
#: and reaction vertex variables.
DEFAULT_BRANCHES: List[str] = [
    # Focal plane (fit inputs — MUST have all 4)
    'P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp',
    # Target reconstruction (to compare against truth)
    'P_gtr_dp', 'P_gtr_ph', 'P_gtr_th', 'P_gtr_y', 'P_gtr_x',
    # Reaction vertex (extended-target / beam-offset correction)
    'P_react_x', 'P_react_y', 'P_react_z',
]


@dataclass
class DataLoadingConfig:
    """Configuration for data loading and file reading.
    
    Attributes:
        feature_cols: List of focal plane variable column names.
        target_cols: List of target variable column names.
        target_x_range: Tuple (min, max) for filtering target_x values.
        target_y_range: Tuple (min, max) for filtering target_y values.
        branches: List of branch names to read from ROOT files.
            Defaults to DEFAULT_BRANCHES (focal plane, target reconstruction,
            and reaction vertex columns). Set to None to read all branches.
    """
    feature_cols: List[str] = field(default_factory=lambda: FEATURE_COLS.copy())
    target_cols: List[str] = field(default_factory=lambda: TARGET_COLS.copy())
    target_x_range: Tuple[float, float] = (-20.0, 20.0)
    target_y_range: Tuple[float, float] = (-20.0, 20.0)
    branches: Optional[List[str]] = field(default_factory=lambda: DEFAULT_BRANCHES.copy())


@dataclass
class TargetProjectionConfig:
    """Configuration for target plane projection calculation.
    
    The projection formula is based on the SHMS optics:
    - Target_x = x + th * z_coefficient
    - Target_y = (-0.019 * dp + 0.00019 * dp² + (138.0 + 75.0) * ph + y) 
                 + 40.0 * (-0.00052 * dp + 0.0000052 * dp² + ph)
    
    Attributes:
        x_z_coefficient: Z coefficient for target_x projection.
        y_dp_linear: Linear dp coefficient for target_y.
        y_dp_quadratic: Quadratic dp coefficient for target_y.
        y_ph_coefficient: ph coefficient for target_y (138.0 + 75.0).
        y_offset_dp_linear: Offset dp linear coefficient.
        y_offset_dp_quadratic: Offset dp quadratic coefficient.
        y_offset_multiplier: Multiplier for offset term.
    """
    x_z_coefficient: float = 253.0
    y_dp_linear: float = -0.019
    y_dp_quadratic: float = 0.00019
    y_ph_coefficient: float = 213.0  # 138.0 + 75.0
    y_offset_dp_linear: float = -0.00052
    y_offset_dp_quadratic: float = 0.0000052
    y_offset_multiplier: float = 40.0


@dataclass
class FoilClassificationConfig:
    """Configuration for foil position classification.
    
    Foil positions are determined by finding peaks in the P_gtr_y distribution
    and classifying events based on their proximity to peak centers.
    
    Attributes:
        col_name: Column name to analyze for foil classification.
        bins: Number of bins for histogram generation.
        sigma_factor: Factor for classification range (peak ± sigma_factor * sigma).
        y_range: Tuple (min, max) for valid P_gtr_y range.
        peak_height_fraction: Minimum peak height as fraction of max peak.
        peak_distance: Minimum distance between peaks in bins.
    """
    col_name: str = 'P_gtr_y'
    bins: int = 50
    sigma_factor: float = 2.5
    y_range: Tuple[float, float] = (-5.0, 5.0)
    peak_height_fraction: float = 0.05
    peak_distance: int = 10


@dataclass
class DBSCANConfig:
    """Configuration for DBSCAN clustering algorithm.
    
    Attributes:
        x_col: Column name for x-coordinate.
        y_col: Column name for y-coordinate.
        eps_range: Tuple (min, max) for eps parameter search.
        target_clusters: Tuple (min, max) for target cluster count.
        min_samples: Minimum samples per cluster. If None, auto-calculated.
        max_iterations: Maximum search iterations.
        distance_threshold: Minimum distance between cluster centers.
        max_cluster_size: Maximum allowed cluster size in cm.
    """
    x_col: str = 'target_x'
    y_col: str = 'target_y'
    eps_range: Tuple[float, float] = (0.01, 0.2)
    target_clusters: Tuple[int, int] = (50, 70)
    min_samples: Optional[int] = None
    max_iterations: int = 10
    distance_threshold: float = 1.0
    max_cluster_size: float = 2.2


@dataclass
class EdgeClusteringConfig:
    """Configuration for edge region clustering (peel and cluster).
    
    This is the second stage of two-entry DBSCAN that processes edge regions
    after core clustering is complete.
    
    Attributes:
        radius_candidates: List of radius values to search.
        eps_candidates: List of eps values to search.
        target_new_clusters: Tuple (min, max) for target new edge clusters.
        distance_threshold: Minimum distance between cluster centers.
    """
    radius_candidates: List[float] = field(
        default_factory=lambda: [0.3, 0.5, 0.8, 1.0, 1.5]
    )
    eps_candidates: List[float] = field(
        default_factory=lambda: [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    )
    target_new_clusters: Tuple[int, int] = (5, 15)
    distance_threshold: float = 1.0


@dataclass
class HDBSCANConfig:
    """Configuration for HDBSCAN clustering algorithm.
    
    Attributes:
        x_col: Column name for x-coordinate.
        y_col: Column name for y-coordinate.
        min_cluster_size_range: Tuple (min, max) for min_cluster_size search.
        min_samples_range: Tuple (min, max) for min_samples search. None uses default.
        target_clusters: Tuple (min, max) for target cluster count.
        max_iterations: Maximum search iterations.
        distance_threshold: Minimum distance between cluster centers.
        max_cluster_size: Maximum allowed cluster size in cm.
        cluster_selection_method: HDBSCAN cluster selection method.
        metric: Distance metric.
        alpha: Alpha parameter for HDBSCAN.
    """
    x_col: str = 'target_x'
    y_col: str = 'target_y'
    min_cluster_size_range: Tuple[int, int] = (30, 100)
    min_samples_range: Optional[Tuple[int, int]] = (30, 80)
    target_clusters: Tuple[int, int] = (50, 70)
    max_iterations: int = 10
    distance_threshold: float = 1.0
    max_cluster_size: float = 2.2
    cluster_selection_method: str = 'leaf'
    metric: str = 'euclidean'
    alpha: float = 1.0


@dataclass
class SoftWeightedDBSCANConfig:
    """Configuration for soft-weighted peel DBSCAN with gtr_y regions.
    
    Attributes:
        gtry_col: Column name for gtr_y values.
        gtry_ranges: Dictionary mapping region labels to (min, max) ranges.
        max_iterations: Maximum iterations for soft-weighted clustering.
        max_cluster_size: Maximum allowed cluster size in cm.
        distance_threshold: Minimum distance between cluster centers.
        suppression_strength_grid: Grid of suppression strengths to search.
        sigma_grid: Grid of sigma values for Gaussian weighting.
        active_threshold_grid: Grid of active thresholds to search.
        eps_grid: Grid of eps values for DBSCAN.
        min_samples_grid: Grid of min_samples values for DBSCAN.
        assign_radius: Radius for assigning points to new clusters.
    """
    gtry_col: str = 'P_gtr_y'
    gtry_ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            '-2.5±1': (-3.5, -1.5),
            '0±1': (-1.0, 1.0),
            '+2.5±1': (1.5, 3.5)
        }
    )
    max_iterations: int = 5
    max_cluster_size: float = 2.2
    distance_threshold: float = 1.0
    suppression_strength_grid: Tuple[float, ...] = (0.7, 0.8, 0.9)
    sigma_grid: Tuple[float, ...] = (0.8, 1.0, 1.2)
    active_threshold_grid: Tuple[float, ...] = (0.3, 0.4, 0.5)
    eps_grid: Tuple[float, ...] = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20)
    min_samples_grid: Tuple[int, ...] = (30, 50, 70, 90)
    assign_radius: float = 0.8


@dataclass
class VisualizationConfig:
    """Configuration for visualization functions.
    
    Attributes:
        figsize: Default figure size (width, height).
        xlim: Default x-axis limits.
        ylim: Default y-axis limits.
        point_size: Default scatter point size.
        noise_point_size: Point size for noise points.
        noise_alpha: Alpha (transparency) for noise points.
        signal_alpha: Alpha (transparency) for signal points.
        cmap: Default colormap for clustering visualization.
        grid_alpha: Grid line transparency.
    """
    figsize: Tuple[int, int] = (10, 10)
    xlim: Tuple[float, float] = (-20.0, 20.0)
    ylim: Tuple[float, float] = (-20.0, 20.0)
    point_size: float = 0.2
    noise_point_size: float = 0.1
    noise_alpha: float = 0.3
    signal_alpha: float = 0.6
    cmap: str = 'nipy_spectral'
    grid_alpha: float = 0.3


@dataclass
class GridIndexConfig:
    """Configuration for grid indexing from cluster centers.
    
    Attributes:
        cluster_col: Column name for cluster labels.
        x_col: Column name for cluster center x-coordinate.
        y_col: Column name for cluster center y-coordinate.
        use_pca_alignment: Whether to use PCA for axis alignment.
        merge_threshold: Distance threshold for merging close centers.
    """
    cluster_col: str = 'cluster'
    x_col: str = 'cluster_center_x'
    y_col: str = 'cluster_center_y'
    use_pca_alignment: bool = False
    merge_threshold: float = 0.5


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation.
    
    Attributes:
        truth_col: Column name for ground truth hole IDs.
        cluster_col: Column name for predicted cluster IDs.
    """
    truth_col: str = 'truth_hole_id'
    cluster_col: str = 'cluster'


@dataclass
class SeparabilityConfig:
    """Configuration for cluster separability analysis.
    
    Attributes:
        fp_cols: List of focal plane column names for 4D analysis.
        normalize: Whether to normalize features before analysis.
    """
    fp_cols: List[str] = field(
        default_factory=lambda: ['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']
    )
    normalize: bool = True


# ============================================================
# Default Configuration Instances
# ============================================================

#: Default data loading configuration
DEFAULT_DATA_LOADING_CONFIG = DataLoadingConfig()

#: Default target projection configuration
DEFAULT_TARGET_PROJECTION_CONFIG = TargetProjectionConfig()

#: Default foil classification configuration
DEFAULT_FOIL_CLASSIFICATION_CONFIG = FoilClassificationConfig()

#: Default DBSCAN configuration
DEFAULT_DBSCAN_CONFIG = DBSCANConfig()

#: Default edge clustering configuration
DEFAULT_EDGE_CLUSTERING_CONFIG = EdgeClusteringConfig()

#: Default HDBSCAN configuration
DEFAULT_HDBSCAN_CONFIG = HDBSCANConfig()

#: Default soft-weighted DBSCAN configuration
DEFAULT_SOFT_WEIGHTED_DBSCAN_CONFIG = SoftWeightedDBSCANConfig()

#: Default visualization configuration
DEFAULT_VISUALIZATION_CONFIG = VisualizationConfig()

#: Default grid index configuration
DEFAULT_GRID_INDEX_CONFIG = GridIndexConfig()

#: Default benchmark configuration
DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()

#: Default separability configuration
DEFAULT_SEPARABILITY_CONFIG = SeparabilityConfig()
