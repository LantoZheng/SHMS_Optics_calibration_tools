"""
Clustering module for SHMS Optics Calibration.

This module provides clustering algorithms for identifying sieve hole
patterns in SHMS optics data, including DBSCAN, HDBSCAN, and two-entry
clustering approaches.
"""

from typing import Optional, Tuple, Dict, List, Any
import warnings
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from scipy.spatial.distance import pdist, cdist
    from scipy.spatial import ConvexHull
except ImportError:
    DBSCAN = None
    NearestNeighbors = None
    pdist = None
    cdist = None
    ConvexHull = None

try:
    import hdbscan as hdbscan_lib
except ImportError:
    hdbscan_lib = None

try:
    from matplotlib.path import Path
except ImportError:
    Path = None

from .config import (
    DBSCANConfig,
    HDBSCANConfig,
    EdgeClusteringConfig,
    SoftWeightedDBSCANConfig,
    DEFAULT_DBSCAN_CONFIG,
    DEFAULT_HDBSCAN_CONFIG,
    DEFAULT_EDGE_CLUSTERING_CONFIG,
    DEFAULT_SOFT_WEIGHTED_DBSCAN_CONFIG,
)
from .preprocessing import initialize_clustering_columns

# Warning thresholds for unusual clustering results
_NOISE_RATIO_THRESHOLD = 0.3
_MIN_CLUSTER_EVENTS = 10
_MIN_CLUSTER_SIZE_RATIO = 0.1
_MAX_CLUSTER_SIZE_RATIO = 5.0
_MAX_ESTIMATION_POINTS = 20000


def _sample_points_for_estimation(
    data: np.ndarray,
    max_points: int = _MAX_ESTIMATION_POINTS,
    random_seed: int = 42
) -> np.ndarray:
    """Subsample points for fast parameter estimation."""
    if len(data) <= max_points:
        return data
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(len(data), size=max_points, replace=False)
    return data[idx]


def _estimate_k_distance(
    data: np.ndarray,
    k: int
) -> Optional[np.ndarray]:
    """Estimate k-NN distance distribution for adaptive eps tuning."""
    if NearestNeighbors is None or len(data) < 3:
        return None

    sampled = _sample_points_for_estimation(data)
    if len(sampled) < 3:
        return None

    k_eff = int(np.clip(k, 2, min(25, len(sampled) - 1)))
    if k_eff < 2:
        return None

    nn = NearestNeighbors(n_neighbors=k_eff + 1)
    nn.fit(sampled)
    distances, _ = nn.kneighbors(sampled)
    kth = distances[:, -1]
    kth = kth[np.isfinite(kth)]
    if len(kth) == 0:
        return None
    return kth


def _estimate_nearest_neighbor_spacing(data: np.ndarray) -> Optional[float]:
    """Estimate characteristic spacing using nearest-neighbor median."""
    if NearestNeighbors is None or len(data) < 2:
        return None

    sampled = _sample_points_for_estimation(data)
    if len(sampled) < 2:
        return None

    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(sampled)
    distances, _ = nn.kneighbors(sampled)
    nearest = distances[:, 1]
    nearest = nearest[(nearest > 0) & np.isfinite(nearest)]
    if len(nearest) == 0:
        return None
    return float(np.median(nearest))


def _build_adaptive_eps_candidates(
    data: np.ndarray,
    min_samples_for_knn: int,
    eps_range: Tuple[float, float],
    n_linear: int = 10
) -> np.ndarray:
    """Build eps candidates using both bounds and k-distance statistics."""
    eps_min, eps_max = eps_range
    if eps_min >= eps_max:
        return np.array([eps_min])

    linear = np.linspace(eps_min, eps_max, n_linear)
    k_dist = _estimate_k_distance(data, k=max(4, int(min_samples_for_knn * 0.5)))
    if k_dist is None:
        return linear

    q30 = float(np.quantile(k_dist, 0.30))
    q97 = float(np.quantile(k_dist, 0.97))
    data_low = max(eps_min, q30 * 0.90)
    data_high = min(eps_max, q97 * 1.15)
    if data_low >= data_high:
        data_low, data_high = eps_min, eps_max

    adaptive_linear = np.linspace(data_low, data_high, n_linear)
    quantile_pts = np.quantile(k_dist, [0.40, 0.55, 0.70, 0.82, 0.90, 0.96])
    merged = np.concatenate([linear, adaptive_linear, quantile_pts])
    merged = np.unique(np.clip(merged, eps_min, eps_max))
    merged = np.sort(merged)

    if len(merged) < 5:
        return np.linspace(eps_min, eps_max, n_linear)
    return merged


def suggest_adaptive_clustering_configs(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    expected_clusters: Optional[int] = None,
    target_margin: float = 0.2,
    verbose: bool = True
) -> Tuple[DBSCANConfig, EdgeClusteringConfig, Dict[str, Any]]:
    """
    Suggest data-adaptive DBSCAN and edge-clustering configurations.

    This helper estimates characteristic density and spacing from the input
    points, then constructs conservative-yet-adaptive search ranges for:
    - core DBSCAN eps / min_samples / distance_threshold / max_cluster_size
    - edge clustering radius / eps / target_new_clusters / distance_threshold

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with x_col and y_col.
    x_col, y_col : str
        Coordinate columns for clustering.
    expected_clusters : int, optional
        If provided, use this as target center; otherwise infer from a quick
        provisional DBSCAN pass.
    target_margin : float
        Relative margin around inferred/expected cluster count.
    verbose : bool
        If True, print estimated statistics and suggested ranges.

    Returns
    -------
    tuple
        (core_config, edge_config, metadata)
    """
    if DBSCAN is None:
        raise ImportError(
            "sklearn is required for adaptive clustering config suggestion. "
            "Install it with: pip install scikit-learn"
        )

    data = df[[x_col, y_col]].dropna().values
    if len(data) < 50:
        raise ValueError("At least 50 valid points are required for adaptive suggestion.")

    n_points = len(data)
    base_min_samples = int(np.clip(np.sqrt(n_points) * 0.25, 8, 80))
    eps_candidates = _build_adaptive_eps_candidates(
        data=data,
        min_samples_for_knn=base_min_samples,
        eps_range=(0.03, 0.50),
        n_linear=12
    )

    eps_mid = float(np.median(eps_candidates))
    db_quick = DBSCAN(eps=eps_mid, min_samples=base_min_samples, n_jobs=-1)
    labels_quick = db_quick.fit_predict(data)
    n_quick_clusters = len(set(labels_quick)) - (1 if -1 in labels_quick else 0)

    if expected_clusters is None:
        if n_quick_clusters > 0:
            target_center = n_quick_clusters
        else:
            target_center = 60
    else:
        target_center = int(expected_clusters)

    low = max(10, int(np.floor(target_center * (1.0 - target_margin))))
    high = max(low + 5, int(np.ceil(target_center * (1.0 + target_margin))))

    centers_quick = []
    for k in set(labels_quick):
        if k == -1:
            continue
        centers_quick.append(data[labels_quick == k].mean(axis=0))
    centers_quick = np.array(centers_quick) if len(centers_quick) > 0 else np.empty((0, 2))

    spacing = _estimate_nearest_neighbor_spacing(centers_quick)
    if spacing is None:
        spacing = _estimate_nearest_neighbor_spacing(data)
    if spacing is None:
        spacing = 1.5

    distance_threshold = float(np.clip(0.60 * spacing, 0.7, 1.8))
    max_cluster_size = float(np.clip(1.35 * spacing, 1.6, 2.8))

    core_cfg = DBSCANConfig(
        x_col=x_col,
        y_col=y_col,
        eps_range=(float(eps_candidates.min()), float(eps_candidates.max())),
        target_clusters=(low, high),
        min_samples=base_min_samples,
        distance_threshold=distance_threshold,
        max_cluster_size=max_cluster_size,
        drop_noise=True
    )

    edge_eps_min = float(np.clip(core_cfg.eps_range[0] * 0.85, 0.03, 0.60))
    edge_eps_max = float(np.clip(core_cfg.eps_range[1] * 1.25, edge_eps_min + 1e-3, 0.60))
    edge_eps_candidates = np.linspace(edge_eps_min, edge_eps_max, 8)

    # Include negative/zero/positive radii to avoid over-expanding hull only.
    radius_base = float(np.clip(0.25 * spacing, 0.15, 0.9))
    radius_candidates = sorted(set([
        -1.4 * radius_base,
        -0.8 * radius_base,
        0.0,
        0.8 * radius_base,
        1.4 * radius_base,
    ]))

    edge_target_high = max(5, int(np.ceil(0.25 * high)))
    edge_cfg = EdgeClusteringConfig(
        radius_candidates=[float(r) for r in radius_candidates],
        eps_candidates=[float(e) for e in edge_eps_candidates],
        target_new_clusters=(0, edge_target_high),
        distance_threshold=distance_threshold,
    )

    metadata = {
        'n_points': n_points,
        'quick_clusters': n_quick_clusters,
        'estimated_spacing': spacing,
        'base_min_samples': base_min_samples,
        'eps_candidates_preview': [float(v) for v in eps_candidates[:6]],
    }

    if verbose:
        print("[Adaptive Suggestion]")
        print(f"  points={n_points:,}, quick_clusters={n_quick_clusters}")
        print(f"  estimated spacing={spacing:.3f} cm")
        print(f"  core eps_range={core_cfg.eps_range}, min_samples={core_cfg.min_samples}")
        print(f"  core target_clusters={core_cfg.target_clusters}")
        print(f"  distance_threshold={distance_threshold:.3f}, max_cluster_size={max_cluster_size:.3f}")
        print(f"  edge target_new_clusters={edge_cfg.target_new_clusters}")

    return core_cfg, edge_cfg, metadata


def auto_dbscan_clustering(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    eps_range: Tuple[float, float] = (0.01, 0.2),
    target_clusters: Tuple[int, int] = (50, 70),
    min_samples: Optional[int] = None,
    max_iterations: int = 10,
    distance_threshold: float = 1.0,
    max_cluster_size: float = 2.2,
    drop_noise: bool = True,
    config: Optional[DBSCANConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, int]:
    """
    Perform DBSCAN clustering with automatic parameter optimization.
    
    This function uses grid search to find optimal DBSCAN parameters
    (eps and min_samples) that produce a target number of clusters
    while satisfying physical constraints on cluster size and separation.
    
    Physical constraints:
    - Single cluster size must not exceed max_cluster_size
    - Cluster centers must be separated by at least distance_threshold
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with coordinate columns.
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    eps_range : tuple of float, optional
        (min, max) range for eps parameter search. Default is (0.01, 0.2).
    target_clusters : tuple of int, optional
        (min, max) target number of clusters. Default is (50, 70).
    min_samples : int, optional
        Minimum samples per cluster. If None, auto-calculated as len(data)/1000.
    max_iterations : int, optional
        Maximum search iterations. Default is 10.
    distance_threshold : float, optional
        Minimum distance between cluster centers. Default is 1.0.
    max_cluster_size : float, optional
        Maximum allowed cluster size in cm. Default is 2.2.
    drop_noise : bool, optional
        If True (default), noise points (cluster == -1) are removed from
        the returned DataFrame. Set to False to retain them with
        ``is_noise=True``.
    config : DBSCANConfig, optional
        Configuration object. If provided, overrides individual parameters.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    tuple
        (df, best_eps, n_clusters) where:
        - df: DataFrame with added clustering columns ('cluster', 'is_noise',
          'cluster_center_x', 'cluster_center_y')
        - best_eps: Optimal eps parameter found
        - n_clusters: Number of clusters detected
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> df = classify_foils_with_range(df)
    >>> df_clustered, eps, n_clusters = auto_dbscan_clustering(
    ...     df, target_clusters=(50, 60)
    ... )
    >>> print(f"Found {n_clusters} clusters with eps={eps:.4f}")
    
    See Also
    --------
    peel_and_cluster_edges : Second-stage edge clustering
    auto_hdbscan_clustering : HDBSCAN alternative
    """
    if DBSCAN is None:
        raise ImportError(
            "sklearn is required for DBSCAN clustering. "
            "Install it with: pip install scikit-learn"
        )
    
    # Use config if provided
    if config is not None:
        x_col = config.x_col
        y_col = config.y_col
        eps_range = config.eps_range
        target_clusters = config.target_clusters
        min_samples = config.min_samples
        max_iterations = config.max_iterations
        distance_threshold = config.distance_threshold
        max_cluster_size = config.max_cluster_size
        drop_noise = config.drop_noise
    
    # Extract data
    df = df.copy()
    data = df[[x_col, y_col]].values
    
    # Set min_samples search range (adaptive by data scale)
    if min_samples is None:
        base_min_samples = int(np.clip(np.sqrt(len(data)) * 0.25, 8, 80))
        min_samples_candidates = [
            max(4, int(base_min_samples * 0.6)),
            max(4, int(base_min_samples * 0.8)),
            base_min_samples,
            max(4, int(base_min_samples * 1.2)),
            max(4, int(base_min_samples * 1.5))
        ]
        min_samples_candidates = sorted(list(set(min_samples_candidates)))
    else:
        min_samples_candidates = [min_samples]
    
    # eps candidates (adaptive + bounded by requested range)
    eps_candidates = _build_adaptive_eps_candidates(
        data=data,
        min_samples_for_knn=min_samples_candidates[len(min_samples_candidates) // 2],
        eps_range=eps_range,
        n_linear=10
    )
    
    target_low, target_high = target_clusters
    best_eps = None
    best_min_samples = None
    best_n = 0
    best_min_dist = 0
    
    if verbose:
        print(f"Starting grid search eps×min_samples "
              f"(range: eps={eps_range}, min_samples={min_samples_candidates})")
        print(f"Target clusters: {target_low}-{target_high}, "
              f"center distance threshold: {distance_threshold:.2f}, "
              f"max cluster size: {max_cluster_size:.2f}")
    
    # Grid search
    total_attempts = 0
    for min_s in min_samples_candidates:
        for eps_val in eps_candidates:
            total_attempts += 1
            db = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
            labels = db.fit_predict(data)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Check physical constraints
            min_center_dist = float('inf')
            valid_size = True
            
            if n_clusters > 0:
                cluster_centers = []
                for k in set(labels):
                    if k == -1:
                        continue
                    cluster_points = data[labels == k]
                    
                    # Check cluster size
                    x_size = cluster_points[:, 0].max() - cluster_points[:, 0].min()
                    y_size = cluster_points[:, 1].max() - cluster_points[:, 1].min()
                    if x_size > max_cluster_size or y_size > max_cluster_size:
                        valid_size = False
                        break
                    
                    cluster_centers.append(cluster_points.mean(axis=0))
                
                cluster_centers = np.array(cluster_centers)
                
                if n_clusters > 1 and len(cluster_centers) > 1:
                    min_center_dist = pdist(cluster_centers).min()
            
            # Check if all constraints are satisfied
            if (valid_size and 
                target_low <= n_clusters <= target_high and 
                (n_clusters == 1 or min_center_dist > distance_threshold)):
                
                target_center = (target_low + target_high) / 2
                current_score = abs(n_clusters - target_center)
                best_score = abs(best_n - target_center) if best_n > 0 else float('inf')
                
                if current_score < best_score or (
                    current_score == best_score and min_center_dist > best_min_dist
                ):
                    best_eps = eps_val
                    best_min_samples = min_s
                    best_n = n_clusters
                    best_min_dist = min_center_dist
                    if verbose:
                        print(f"✓ Found candidate: eps={eps_val:.4f}, "
                              f"min_samples={min_s}, clusters={n_clusters}, "
                              f"min center distance={min_center_dist:.4f}")
    
    # Fallback if no satisfying parameters found
    if best_eps is None:
        if verbose:
            print("No fully satisfying parameters found, searching for closest...")
        best_score = float('inf')
        target_center = (target_low + target_high) / 2
        
        for min_s in min_samples_candidates:
            for eps_val in eps_candidates:
                db = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
                labels = db.fit_predict(data)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                min_center_dist = float('inf')
                valid_size = True
                
                if n_clusters > 0:
                    cluster_centers = []
                    for k in set(labels):
                        if k == -1:
                            continue
                        cluster_points = data[labels == k]
                        
                        x_size = cluster_points[:, 0].max() - cluster_points[:, 0].min()
                        y_size = cluster_points[:, 1].max() - cluster_points[:, 1].min()
                        if x_size > max_cluster_size or y_size > max_cluster_size:
                            valid_size = False
                            break
                        
                        cluster_centers.append(cluster_points.mean(axis=0))
                    
                    cluster_centers = np.array(cluster_centers)
                    
                    if n_clusters > 1 and len(cluster_centers) > 1:
                        min_center_dist = pdist(cluster_centers).min()
                
                if (valid_size and n_clusters > 0 and 
                    (n_clusters == 1 or min_center_dist > distance_threshold)):
                    score = abs(n_clusters - target_center)
                    if score < best_score:
                        best_score = score
                        best_eps = eps_val
                        best_min_samples = min_s
                        best_n = n_clusters
        
        if best_eps is not None and verbose:
            print(f"Using parameters: eps={best_eps:.4f}, "
                  f"min_samples={best_min_samples}, clusters={best_n}")
    
    if verbose:
        print(f"\nSearch complete ({total_attempts} parameter combinations tried)")
    
    # Safety check
    if best_eps is None:
        if verbose:
            print("⚠️ Warning: No valid parameters found, using defaults")
        best_eps = eps_range[0]
        best_min_samples = min_samples_candidates[0] if min_samples_candidates else 50
        best_n = 0
    
    if verbose:
        print(f"Optimal parameters: eps={best_eps:.4f}, "
              f"min_samples={best_min_samples}, expected clusters={best_n}")
    
    # Final clustering with optimal parameters
    db_final = DBSCAN(eps=best_eps, min_samples=best_min_samples, n_jobs=-1)
    final_labels = db_final.fit_predict(data)
    
    # Write results to DataFrame
    df['cluster'] = final_labels
    df['is_noise'] = df['cluster'] == -1
    
    # Calculate cluster centers
    centroids = df[~df['is_noise']].groupby('cluster')[[x_col, y_col]].mean()
    centroids.columns = ['cluster_center_x', 'cluster_center_y']
    
    df['cluster_center_x'] = df['cluster'].map(centroids['cluster_center_x'])
    df['cluster_center_y'] = df['cluster'].map(centroids['cluster_center_y'])
    
    if verbose:
        print(f"\nClustering complete! Found {best_n} clusters")
        print(f"Noise points: {df['is_noise'].sum()}")
    
    # Warn about unusual cluster count
    if best_n < target_low or best_n > target_high:
        warnings.warn(
            f"Unusual cluster count: found {best_n} clusters, "
            f"outside target range [{target_low}, {target_high}]. "
            "Clustering results may be unreliable.",
            UserWarning, stacklevel=2
        )
    
    # Warn about high noise ratio
    n_total = len(final_labels)
    n_noise = (final_labels == -1).sum()
    if n_total > 0:
        noise_ratio = n_noise / n_total
        if noise_ratio > _NOISE_RATIO_THRESHOLD:
            warnings.warn(
                f"Unusual dropout data: {noise_ratio:.1%} of events "
                f"({n_noise:,}) are classified as noise. "
                "Consider adjusting DBSCAN parameters (eps, min_samples).",
                UserWarning, stacklevel=2
            )
    
    # Warn about unusual cluster sizes (event counts)
    if best_n > 0:
        cluster_sizes = [
            (final_labels == k).sum()
            for k in sorted(set(final_labels)) if k != -1
        ]
        mean_size = np.mean(cluster_sizes)
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        if min_size < max(_MIN_CLUSTER_EVENTS, mean_size * _MIN_CLUSTER_SIZE_RATIO):
            warnings.warn(
                f"Unusual cluster size: smallest cluster has {min_size} events "
                f"(mean: {mean_size:.0f}). Some clusters may be spurious.",
                UserWarning, stacklevel=2
            )
        if max_size > mean_size * _MAX_CLUSTER_SIZE_RATIO:
            warnings.warn(
                f"Unusual cluster size: largest cluster has {max_size} events "
                f"(mean: {mean_size:.0f}). Some clusters may be over-merged.",
                UserWarning, stacklevel=2
            )
    
    if drop_noise:
        n_before = len(df)
        df = df[~df['is_noise']].copy()
        if verbose:
            print(f"Dropped {n_before - len(df):,} noise points "
                  f"({len(df):,} remaining)")
    
    return df, best_eps, best_n


def peel_and_cluster_edges(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    radius: Optional[float] = None,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    target_new_clusters: Tuple[int, int] = (5, 15),
    distance_threshold: float = 1.0,
    config: Optional[EdgeClusteringConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, float, int]:
    """
    Second-stage clustering for edge regions using convex hull peeling.
    
    This function performs edge clustering after initial DBSCAN clustering.
    It identifies the convex hull of core clusters and clusters the
    remaining edge points using a second round of DBSCAN.
    
    The algorithm:
    1. Build convex hull from core cluster points
    2. Expand hull boundary by radius (buffer zone)
    3. Identify edge points outside the buffer
    4. Apply DBSCAN to edge points
    5. Merge new clusters with existing ones
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with initial clustering results (must have 'cluster' and
        'is_noise' columns from first DBSCAN pass).
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    radius : float, optional
        Convex hull expansion radius. If None, searches optimal value.
    eps : float, optional
        DBSCAN eps for edge clustering. If None, searches optimal value.
    min_samples : int, optional
        DBSCAN min_samples for edge clustering. If None, auto-calculated.
    target_new_clusters : tuple of int, optional
        (min, max) target number of new edge clusters. Default is (5, 15).
    distance_threshold : float, optional
        Minimum distance between cluster centers. Default is 1.0.
    config : EdgeClusteringConfig, optional
        Configuration object. If provided, overrides individual parameters.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    tuple
        (df, best_eps, n_clusters) where:
        - df: Updated DataFrame with new edge clusters added
        - best_eps: Optimal eps parameter found
        - n_clusters: Total number of clusters (core + edge)
    
    Examples
    --------
    >>> # First, run core clustering
    >>> df, eps_core, n_core = auto_dbscan_clustering(df)
    >>> # Then, cluster edges
    >>> df, eps_edge, n_total = peel_and_cluster_edges(df)
    >>> print(f"Edge clusters added: {n_total - n_core}")
    
    See Also
    --------
    auto_dbscan_clustering : First-stage core clustering
    two_entry_dbscan : Combined core + edge clustering
    """
    if DBSCAN is None or ConvexHull is None or Path is None:
        raise ImportError(
            "sklearn, scipy, and matplotlib are required for edge clustering. "
            "Install with: pip install scikit-learn scipy matplotlib"
        )
    
    # Use config if provided
    if config is not None:
        radius_candidates = config.radius_candidates
        eps_candidates_base = config.eps_candidates
        target_new_clusters = config.target_new_clusters
        distance_threshold = config.distance_threshold
    else:
        if radius is None:
            # Include negative/zero/positive radii for robust edge peeling.
            radius_candidates = [-0.6, -0.3, 0.0, 0.3, 0.6, 1.0]
        else:
            radius_candidates = [radius]
        
        if eps is None:
            eps_candidates_base = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        else:
            eps_candidates_base = [eps]

    # Ensure radius search includes boundary-friendly options.
    if all(r > 0 for r in radius_candidates):
        radius_candidates = sorted(set(radius_candidates + [0.0, -0.5 * min(radius_candidates)]))
    else:
        radius_candidates = sorted(set(radius_candidates))
    
    df = df.copy()
    
    if verbose:
        print("-" * 60)
        print("Stage 2: Peeling center and searching edge clusters...")
    
    # Extract data and core points
    data = df[[x_col, y_col]].values
    core_mask = ~df['is_noise']
    points_core = data[core_mask]
    
    if len(points_core) < 3:
        if verbose:
            print("Core region has too few points, cannot build convex hull.")
        return df, eps if eps else 0.1, 0
    
    # Estimate spacing from core centers to adapt distance threshold.
    core_centers = df[core_mask].groupby('cluster')[[x_col, y_col]].mean().values
    core_spacing = _estimate_nearest_neighbor_spacing(core_centers) if len(core_centers) > 1 else None
    adaptive_distance_threshold = distance_threshold
    if core_spacing is not None:
        adaptive_distance_threshold = min(
            distance_threshold,
            float(np.clip(0.65 * core_spacing, 0.7, 1.8))
        )
    
    if verbose:
        print(f"Parameter search space:")
        print(f"  radius: {radius_candidates}")
        print(f"  eps(base): {eps_candidates_base}")
        print(f"  min_samples: adaptive per edge region")
        print(f"  target new edge clusters: {target_new_clusters[0]}-{target_new_clusters[1]}")
        if core_spacing is not None:
            print(f"  adaptive distance threshold: {adaptive_distance_threshold:.3f} "
                  f"(base={distance_threshold:.3f}, spacing={core_spacing:.3f})")
    
    # Grid search for optimal parameters
    best_params = None
    best_score = float('inf')
    best_new_clusters = 0
    best_edge_labels = None
    best_edge_mask = None
    
    target_low, target_high = target_new_clusters
    target_center = (target_low + target_high) / 2
    
    total_attempts = 0
    
    for r in radius_candidates:
        # Build convex hull mask
        hull = ConvexHull(points_core)
        hull_vertices = points_core[hull.vertices]
        poly_path = Path(hull_vertices)
        
        # Identify edge region
        is_inside = poly_path.contains_points(data, radius=r)
        edge_mask = ~is_inside
        data_edge = data[edge_mask]
        
        if len(data_edge) < 10:
            continue
        
        if min_samples is None:
            local_base_min = int(np.clip(np.sqrt(len(data_edge)) * 0.35, 4, 50))
            min_samples_candidates = sorted(list(set([
                max(3, int(local_base_min * 0.6)),
                max(3, int(local_base_min * 0.8)),
                local_base_min,
                max(3, int(local_base_min * 1.25)),
                max(3, int(local_base_min * 1.5))
            ])))
        else:
            min_samples_candidates = [min_samples]

        if eps is None:
            eps_from_edge = _build_adaptive_eps_candidates(
                data=data_edge,
                min_samples_for_knn=min_samples_candidates[0],
                eps_range=(min(eps_candidates_base), max(eps_candidates_base)),
                n_linear=8
            )
            eps_candidates = sorted(set(
                [float(v) for v in eps_candidates_base] +
                [float(v) for v in eps_from_edge]
            ))
        else:
            eps_candidates = eps_candidates_base

        for eps_val in eps_candidates:
            for min_s in min_samples_candidates:
                total_attempts += 1
                
                db_edge = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
                labels_edge = db_edge.fit_predict(data_edge)
                
                n_new = len(set(labels_edge)) - (1 if -1 in labels_edge else 0)
                
                # Check physical constraints
                if n_new > 0:
                    new_centers = []
                    for k in set(labels_edge):
                        if k == -1:
                            continue
                        new_centers.append(data_edge[labels_edge == k].mean(axis=0))
                    new_centers = np.array(new_centers)
                    
                    # Check distances between new clusters
                    valid = True
                    if len(new_centers) > 1:
                        if pdist(new_centers).min() < adaptive_distance_threshold:
                            valid = False
                    
                    # Check distances to existing core clusters
                    if valid and len(points_core) > 0:
                        core_centers = df[core_mask].groupby('cluster')[
                            [x_col, y_col]
                        ].mean().values
                        if cdist(new_centers, core_centers).min() < adaptive_distance_threshold:
                            valid = False
                    
                    if valid and target_low <= n_new <= target_high:
                        score = abs(n_new - target_center)
                        if score < best_score:
                            best_score = score
                            best_params = (r, eps_val, min_s)
                            best_new_clusters = n_new
                            best_edge_labels = labels_edge.copy()
                            best_edge_mask = edge_mask.copy()
                            if verbose:
                                print(f"✓ Found candidate: radius={r:.2f}, "
                                      f"eps={eps_val:.3f}, min_samples={min_s}, "
                                      f"new clusters={n_new}")
    
    # Fallback search if no satisfying parameters found
    if best_params is None:
        if verbose:
            print(f"\nNo fully satisfying parameters found, searching closest...")
        best_score = float('inf')
        
        for r in radius_candidates:
            hull = ConvexHull(points_core)
            poly_path = Path(points_core[hull.vertices])
            is_inside = poly_path.contains_points(data, radius=r)
            edge_mask = ~is_inside
            data_edge = data[edge_mask]
            
            if len(data_edge) < 10:
                continue
            
            if min_samples is None:
                local_base_min = int(np.clip(np.sqrt(len(data_edge)) * 0.35, 4, 50))
                min_samples_candidates = sorted(list(set([
                    max(3, int(local_base_min * 0.6)),
                    max(3, int(local_base_min * 0.8)),
                    local_base_min,
                    max(3, int(local_base_min * 1.25)),
                    max(3, int(local_base_min * 1.5))
                ])))
            else:
                min_samples_candidates = [min_samples]

            if eps is None:
                eps_from_edge = _build_adaptive_eps_candidates(
                    data=data_edge,
                    min_samples_for_knn=min_samples_candidates[0],
                    eps_range=(min(eps_candidates_base), max(eps_candidates_base)),
                    n_linear=8
                )
                eps_candidates = sorted(set(
                    [float(v) for v in eps_candidates_base] +
                    [float(v) for v in eps_from_edge]
                ))
            else:
                eps_candidates = eps_candidates_base

            for eps_val in eps_candidates:
                for min_s in min_samples_candidates:
                    db_edge = DBSCAN(eps=eps_val, min_samples=min_s, n_jobs=-1)
                    labels_edge = db_edge.fit_predict(data_edge)
                    n_new = len(set(labels_edge)) - (1 if -1 in labels_edge else 0)
                    
                    score = abs(n_new - target_center)
                    if score < best_score and n_new > 0:
                        best_score = score
                        best_params = (r, eps_val, min_s)
                        best_new_clusters = n_new
                        best_edge_labels = labels_edge.copy()
                        best_edge_mask = edge_mask.copy()
        
        if best_params and verbose:
            print(f"Using parameters: radius={best_params[0]:.2f}, "
                  f"eps={best_params[1]:.3f}, min_samples={best_params[2]}")
    
    if verbose:
        print(f"\nSearch complete ({total_attempts} parameter combinations tried)")
    
    # Apply best parameters
    if best_params is None or best_edge_labels is None:
        if verbose:
            print("Could not find valid edge clustering parameters, skipping stage 2.")
        return df, eps if eps else 0.1, len(df[~df['is_noise']].groupby('cluster'))
    
    best_radius, best_eps, best_min_samples = best_params
    if verbose:
        print(f"Optimal parameters: radius={best_radius:.2f}, "
              f"eps={best_eps:.3f}, min_samples={best_min_samples}")
        print(f"New edge clusters: {best_new_clusters}")
    
    # Merge with existing clusters
    max_core_id = df['cluster'].max()
    
    new_labels = best_edge_labels.copy()
    valid_edge_mask = best_edge_labels != -1
    new_labels[valid_edge_mask] += (max_core_id + 1)
    
    df.loc[best_edge_mask, 'cluster'] = new_labels
    df['is_noise'] = df['cluster'] == -1
    
    # Recalculate cluster centers
    centroids = df[~df['is_noise']].groupby('cluster')[[x_col, y_col]].mean()
    centroids.columns = ['cluster_center_x', 'cluster_center_y']
    
    df['cluster_center_x'] = df['cluster'].map(centroids['cluster_center_x'])
    df['cluster_center_y'] = df['cluster'].map(centroids['cluster_center_y'])
    
    if verbose:
        print(f"Total clusters located: {len(centroids)}")
    
    return df, best_eps, len(centroids)


def two_entry_dbscan(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    core_config: Optional[DBSCANConfig] = None,
    edge_config: Optional[EdgeClusteringConfig] = None,
    drop_noise: bool = True,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], int]:
    """
    Two-entry DBSCAN clustering combining core and edge clustering.
    
    This is a convenience function that combines auto_dbscan_clustering
    (for core regions) and peel_and_cluster_edges (for edge regions)
    into a single call.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with coordinate columns.
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    core_config : DBSCANConfig, optional
        Configuration for core clustering.
    edge_config : EdgeClusteringConfig, optional
        Configuration for edge clustering.
    drop_noise : bool, optional
        If True (default), noise points remaining after both clustering
        stages are removed from the returned DataFrame. Set to False to
        retain them with ``is_noise=True``. Note: noise dropping from
        ``core_config`` is ignored here; noise is only dropped once after
        both stages complete.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    tuple
        (df, params, n_clusters) where:
        - df: DataFrame with clustering results
        - params: Dictionary with 'core_eps', 'edge_eps', 'core_clusters',
          'edge_clusters' keys
        - n_clusters: Total number of clusters
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> df = classify_foils_with_range(df)
    >>> df_foil0 = get_foil_subset(df, 0)
    >>> df_clustered, params, n_total = two_entry_dbscan(df_foil0)
    >>> print(f"Total clusters: {n_total}")
    >>> print(f"Core: {params['core_clusters']}, Edge: {params['edge_clusters']}")
    
    See Also
    --------
    auto_dbscan_clustering : Core clustering only
    peel_and_cluster_edges : Edge clustering only
    """
    if verbose:
        print("=" * 60)
        print("Two-Entry DBSCAN Clustering")
        print("=" * 60)
    
    # Stage 1: Core clustering (never drop noise here — peel_and_cluster_edges needs them)
    if verbose:
        print("\n[Stage 1] Core region clustering...")
    
    df, eps_core, n_core = auto_dbscan_clustering(
        df, x_col=x_col, y_col=y_col,
        config=core_config, drop_noise=False, verbose=verbose
    )
    
    # Stage 2: Edge clustering
    if verbose:
        print(f"\n[Stage 2] Edge region clustering...")
    
    df, eps_edge, n_total = peel_and_cluster_edges(
        df, x_col=x_col, y_col=y_col,
        config=edge_config, verbose=verbose
    )
    
    params = {
        'core_eps': eps_core,
        'edge_eps': eps_edge,
        'core_clusters': n_core,
        'edge_clusters': n_total - n_core
    }
    
    if verbose:
        print(f"\nTwo-Entry DBSCAN complete!")
        print(f"  Core clusters: {n_core}")
        print(f"  Edge clusters: {n_total - n_core}")
        print(f"  Total clusters: {n_total}")
    
    if drop_noise:
        n_before = len(df)
        df = df[~df['is_noise']].copy()
        if verbose:
            print(f"Dropped {n_before - len(df):,} noise points "
                  f"({len(df):,} remaining)")
    
    return df, params, n_total


def auto_hdbscan_clustering(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    min_cluster_size_range: Tuple[int, int] = (30, 100),
    min_samples_range: Optional[Tuple[int, int]] = (30, 80),
    target_clusters: Tuple[int, int] = (50, 70),
    max_iterations: int = 10,
    distance_threshold: float = 1.0,
    max_cluster_size: float = 2.2,
    drop_noise: bool = True,
    config: Optional[HDBSCANConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], int]:
    """
    Perform HDBSCAN clustering with automatic parameter optimization.
    
    HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications
    with Noise) is a hierarchical extension of DBSCAN that automatically
    determines cluster structure without requiring eps parameter.
    
    This function uses grid search to find optimal HDBSCAN parameters
    that produce a target number of clusters while satisfying physical
    constraints.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with coordinate columns.
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    min_cluster_size_range : tuple of int, optional
        (min, max) range for min_cluster_size search. Default is (30, 100).
    min_samples_range : tuple of int, optional
        (min, max) range for min_samples search. If None, uses HDBSCAN defaults.
    target_clusters : tuple of int, optional
        (min, max) target number of clusters. Default is (50, 70).
    max_iterations : int, optional
        Maximum search iterations. Default is 10.
    distance_threshold : float, optional
        Minimum distance between cluster centers. Default is 1.0.
    max_cluster_size : float, optional
        Maximum allowed cluster size in cm. Default is 2.2.
    drop_noise : bool, optional
        If True (default), noise points (cluster == -1) are removed from
        the returned DataFrame. Set to False to retain them with
        ``is_noise=True``.
    config : HDBSCANConfig, optional
        Configuration object. If provided, overrides individual parameters.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    tuple
        (df, best_params, n_clusters) where:
        - df: DataFrame with clustering columns
        - best_params: Dictionary with optimal parameters
        - n_clusters: Number of clusters detected
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> df = classify_foils_with_range(df)
    >>> df_clustered, params, n_clusters = auto_hdbscan_clustering(
    ...     df, target_clusters=(70, 80)
    ... )
    >>> print(f"Found {n_clusters} clusters")
    >>> print(f"Parameters: {params}")
    
    See Also
    --------
    auto_dbscan_clustering : DBSCAN alternative
    two_entry_dbscan : Two-stage DBSCAN approach
    """
    if hdbscan_lib is None:
        raise ImportError(
            "hdbscan is required for HDBSCAN clustering. "
            "Install it with: pip install hdbscan"
        )
    
    # Use config if provided
    if config is not None:
        x_col = config.x_col
        y_col = config.y_col
        min_cluster_size_range = config.min_cluster_size_range
        min_samples_range = config.min_samples_range
        target_clusters = config.target_clusters
        max_iterations = config.max_iterations
        distance_threshold = config.distance_threshold
        max_cluster_size = config.max_cluster_size
        cluster_selection_method = config.cluster_selection_method
        metric = config.metric
        alpha = config.alpha
        drop_noise = config.drop_noise
    else:
        cluster_selection_method = 'leaf'
        metric = 'euclidean'
        alpha = 1.0
    
    df = df.copy()
    data = df[[x_col, y_col]].values
    
    # Set parameter search ranges
    min_size_low, min_size_high = min_cluster_size_range
    min_cluster_sizes = np.linspace(min_size_low, min_size_high, 5, dtype=int)
    min_cluster_sizes = sorted(list(set(min_cluster_sizes)))
    
    if min_samples_range is None:
        min_samples_candidates = [None]
    else:
        min_s_low, min_s_high = min_samples_range
        min_samples_candidates = np.linspace(min_s_low, min_s_high, 3, dtype=int)
        min_samples_candidates = sorted(list(set(min_samples_candidates)))
    
    if verbose:
        print("-" * 60)
        print("HDBSCAN parameter grid search starting...")
        print(f"Data size: {len(data)} points")
        print(f"Target clusters: {target_clusters[0]}-{target_clusters[1]}")
        print(f"Search space:")
        print(f"  min_cluster_size: {min_cluster_sizes}")
        print(f"  min_samples: {min_samples_candidates}")
        print("-" * 60)
    
    # Grid search
    best_params = None
    best_score = float('inf')
    best_labels = None
    best_n_clusters = 0
    
    target_low, target_high = target_clusters
    target_center = (target_low + target_high) / 2
    
    total_attempts = 0
    
    for min_size in min_cluster_sizes:
        for min_samp in min_samples_candidates:
            total_attempts += 1
            
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=min_size,
                min_samples=min_samp,
                cluster_selection_method=cluster_selection_method,
                metric=metric,
                alpha=alpha
            )
            
            labels = clusterer.fit_predict(data)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Check physical constraints
            min_center_dist = float('inf')
            valid_size = True
            
            if n_clusters > 0:
                cluster_centers = []
                for k in set(labels):
                    if k == -1:
                        continue
                    cluster_points = data[labels == k]
                    
                    x_size = cluster_points[:, 0].max() - cluster_points[:, 0].min()
                    y_size = cluster_points[:, 1].max() - cluster_points[:, 1].min()
                    if x_size > max_cluster_size or y_size > max_cluster_size:
                        valid_size = False
                        break
                    
                    cluster_centers.append(cluster_points.mean(axis=0))
                
                cluster_centers = np.array(cluster_centers)
                
                if n_clusters > 1 and len(cluster_centers) > 1:
                    min_center_dist = pdist(cluster_centers).min()
            
            if (valid_size and
                target_low <= n_clusters <= target_high and
                (n_clusters == 1 or min_center_dist > distance_threshold)):
                
                score = abs(n_clusters - target_center)
                if score < best_score:
                    best_score = score
                    best_params = {
                        'min_cluster_size': min_size,
                        'min_samples': min_samp
                    }
                    best_labels = labels.copy()
                    best_n_clusters = n_clusters
                    if verbose:
                        print(f"✓ Found candidate: min_cluster_size={min_size}, "
                              f"min_samples={min_samp}, clusters={n_clusters}")
    
    if verbose:
        print(f"\nSearch complete ({total_attempts} parameter combinations tried)")
    
    # Fallback search: if no strict candidate in target range, find the
    # closest physically-valid candidate instead of blindly using smallest
    # min_cluster_size/min_samples (which often over-splits holes).
    if best_params is None:
        if verbose:
            print("⚠️ No strict in-range candidate found, searching closest valid candidate...")

        fallback_best_score = float('inf')
        fallback_best_params = None
        fallback_best_labels = None
        fallback_best_n_clusters = 0

        for min_size in min_cluster_sizes:
            for min_samp in min_samples_candidates:
                clusterer = hdbscan_lib.HDBSCAN(
                    min_cluster_size=min_size,
                    min_samples=min_samp,
                    cluster_selection_method=cluster_selection_method,
                    metric=metric,
                    alpha=alpha
                )

                labels = clusterer.fit_predict(data)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters <= 0:
                    continue

                # Re-check physical constraints
                min_center_dist = float('inf')
                valid_size = True
                cluster_centers = []

                for k in set(labels):
                    if k == -1:
                        continue
                    cluster_points = data[labels == k]

                    x_size = cluster_points[:, 0].max() - cluster_points[:, 0].min()
                    y_size = cluster_points[:, 1].max() - cluster_points[:, 1].min()
                    if x_size > max_cluster_size or y_size > max_cluster_size:
                        valid_size = False
                        break

                    cluster_centers.append(cluster_points.mean(axis=0))

                if not valid_size:
                    continue

                cluster_centers = np.array(cluster_centers)
                if n_clusters > 1 and len(cluster_centers) > 1:
                    min_center_dist = pdist(cluster_centers).min()

                if n_clusters > 1 and min_center_dist <= distance_threshold:
                    continue

                # Score by closeness to target center; tie-break toward less
                # fragmentation (larger min_cluster_size) and fewer clusters.
                score = abs(n_clusters - target_center)
                tie_break = (
                    -min_size,          # prefer larger clusters
                    - (min_samp if min_samp is not None else 0),
                    -min_center_dist if np.isfinite(min_center_dist) else float('-inf'),
                    -n_clusters
                )

                if (score < fallback_best_score or
                    (score == fallback_best_score and fallback_best_params is not None and tie_break > (
                        -fallback_best_params['min_cluster_size'],
                        -(fallback_best_params['min_samples'] if fallback_best_params['min_samples'] is not None else 0),
                        float('-inf'),
                        -fallback_best_n_clusters
                    ))):
                    fallback_best_score = score
                    fallback_best_params = {
                        'min_cluster_size': min_size,
                        'min_samples': min_samp
                    }
                    fallback_best_labels = labels.copy()
                    fallback_best_n_clusters = n_clusters

        if fallback_best_params is not None:
            best_params = fallback_best_params
            best_labels = fallback_best_labels
            best_n_clusters = fallback_best_n_clusters
            if verbose:
                print(f"Using closest valid parameters: {best_params}, clusters={best_n_clusters}")
        else:
            if verbose:
                print("⚠️ Warning: No physically valid candidate found, using conservative defaults")
            conservative_min_size = int(np.median(min_cluster_sizes))
            conservative_min_samples = (
                int(np.median([v for v in min_samples_candidates if v is not None]))
                if min_samples_candidates and min_samples_candidates[0] is not None
                else None
            )
            best_params = {
                'min_cluster_size': conservative_min_size,
                'min_samples': conservative_min_samples
            }

            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                min_samples=best_params['min_samples'],
                cluster_selection_method=cluster_selection_method,
                metric=metric,
                alpha=alpha
            )
            best_labels = clusterer.fit_predict(data)
            best_n_clusters = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    
    if verbose:
        print(f"Optimal parameters: {best_params}, clusters={best_n_clusters}")
    
    # Write results to DataFrame
    df['cluster'] = best_labels
    df['is_noise'] = df['cluster'] == -1
    
    centroids = df[~df['is_noise']].groupby('cluster')[[x_col, y_col]].mean()
    centroids.columns = ['cluster_center_x', 'cluster_center_y']
    
    df['cluster_center_x'] = df['cluster'].map(centroids['cluster_center_x'])
    df['cluster_center_y'] = df['cluster'].map(centroids['cluster_center_y'])
    
    if verbose:
        print(f"\nClustering complete! Found {best_n_clusters} clusters")
        print(f"Noise points: {df['is_noise'].sum()}")
    
    # Warn about unusual cluster count
    if best_n_clusters < target_low or best_n_clusters > target_high:
        warnings.warn(
            f"Unusual cluster count: found {best_n_clusters} clusters, "
            f"outside target range [{target_low}, {target_high}]. "
            "Clustering results may be unreliable.",
            UserWarning, stacklevel=2
        )
    
    # Warn about high noise ratio
    n_total = len(best_labels)
    n_noise = (best_labels == -1).sum()
    if n_total > 0:
        noise_ratio = n_noise / n_total
        if noise_ratio > _NOISE_RATIO_THRESHOLD:
            warnings.warn(
                f"Unusual dropout data: {noise_ratio:.1%} of events "
                f"({n_noise:,}) are classified as noise. "
                "Consider adjusting HDBSCAN parameters (min_cluster_size, min_samples).",
                UserWarning, stacklevel=2
            )
    
    # Warn about unusual cluster sizes (event counts)
    if best_n_clusters > 0:
        cluster_sizes = [
            (best_labels == k).sum()
            for k in sorted(set(best_labels)) if k != -1
        ]
        mean_size = np.mean(cluster_sizes)
        min_size = min(cluster_sizes)
        max_size = max(cluster_sizes)
        if min_size < max(_MIN_CLUSTER_EVENTS, mean_size * _MIN_CLUSTER_SIZE_RATIO):
            warnings.warn(
                f"Unusual cluster size: smallest cluster has {min_size} events "
                f"(mean: {mean_size:.0f}). Some clusters may be spurious.",
                UserWarning, stacklevel=2
            )
        if max_size > mean_size * _MAX_CLUSTER_SIZE_RATIO:
            warnings.warn(
                f"Unusual cluster size: largest cluster has {max_size} events "
                f"(mean: {mean_size:.0f}). Some clusters may be over-merged.",
                UserWarning, stacklevel=2
            )
    
    if drop_noise:
        n_before = len(df)
        df = df[~df['is_noise']].copy()
        if verbose:
            print(f"Dropped {n_before - len(df):,} noise points "
                  f"({len(df):,} remaining)")
    
    return df, best_params, best_n_clusters


def cluster_by_foil_position(
    df: pd.DataFrame,
    method: str = 'dbscan',
    foil_col: str = 'foil_position',
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    dbscan_config: Optional[DBSCANConfig] = None,
    edge_config: Optional[EdgeClusteringConfig] = None,
    hdbscan_config: Optional[HDBSCANConfig] = None,
    use_two_entry: bool = True,
    verbose: bool = True
) -> Dict[int, Dict[str, Any]]:
    """
    Apply clustering to each foil position separately.
    
    This function iterates over all valid foil positions and applies
    the specified clustering method to each subset of data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'foil_position' column.
    method : str, optional
        Clustering method: 'dbscan', 'hdbscan', or 'two_entry'.
        Default is 'dbscan'.
    foil_col : str, optional
        Column name for foil position. Default is 'foil_position'.
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    dbscan_config : DBSCANConfig, optional
        Configuration for DBSCAN clustering.
    edge_config : EdgeClusteringConfig, optional
        Configuration for edge clustering (used with two_entry method).
    hdbscan_config : HDBSCANConfig, optional
        Configuration for HDBSCAN clustering.
    use_two_entry : bool, optional
        If True and method='dbscan', uses two-entry approach.
        Default is True.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    dict
        Dictionary mapping foil_position to clustering results:
        {foil_position: {'df': DataFrame, 'params': dict, 'n_clusters': int}}
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> df = classify_foils_with_range(df)
    >>> results = cluster_by_foil_position(df, method='dbscan')
    >>> for foil, result in results.items():
    ...     print(f"Foil {foil}: {result['n_clusters']} clusters")
    
    See Also
    --------
    auto_dbscan_clustering : Single DBSCAN clustering
    auto_hdbscan_clustering : Single HDBSCAN clustering
    two_entry_dbscan : Two-entry DBSCAN clustering
    """
    if foil_col not in df.columns:
        raise ValueError(f"Column '{foil_col}' not found in DataFrame. "
                        "Run classify_foils_with_range first.")
    
    # Get valid foil positions
    foil_positions = sorted([
        int(v) for v in df[foil_col].dropna().unique() if v != -1
    ])
    
    if verbose:
        print("=" * 60)
        print(f"Clustering by foil position (method: {method})")
        print(f"Foil positions: {foil_positions}")
        print("=" * 60)
    
    results = {}
    
    for foil_pos in foil_positions:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"foil_position = {foil_pos}")
            print("=" * 60)
        
        # Get subset for this foil
        subset_mask = df[foil_col] == foil_pos
        df_subset = df.loc[subset_mask].copy()
        
        if df_subset.empty:
            if verbose:
                print("No data for this foil position, skipping.")
            continue
        
        if verbose:
            print(f"Data points: {len(df_subset):,}")
        
        # Apply clustering method
        if method == 'hdbscan':
            df_clustered, params, n_clusters = auto_hdbscan_clustering(
                df_subset, x_col=x_col, y_col=y_col,
                config=hdbscan_config, verbose=verbose
            )
        elif method == 'two_entry' or (method == 'dbscan' and use_two_entry):
            df_clustered, params, n_clusters = two_entry_dbscan(
                df_subset, x_col=x_col, y_col=y_col,
                core_config=dbscan_config, edge_config=edge_config,
                verbose=verbose
            )
        else:  # dbscan
            df_clustered, eps, n_clusters = auto_dbscan_clustering(
                df_subset, x_col=x_col, y_col=y_col,
                config=dbscan_config, verbose=verbose
            )
            params = {'eps': eps}
        
        results[foil_pos] = {
            'df': df_clustered,
            'params': params,
            'n_clusters': n_clusters
        }
    
    if verbose:
        print(f"\n{'=' * 60}")
        print("Clustering by foil position complete!")
        print("=" * 60)
        print("\nSummary:")
        for foil_pos, result in results.items():
            print(f"  Foil {foil_pos}: {result['n_clusters']} clusters")
    
    return results
