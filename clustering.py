"""
Clustering module for SHMS Optics Calibration.

This module provides clustering algorithms for identifying sieve hole
patterns in SHMS optics data, including DBSCAN, HDBSCAN, and two-entry
clustering approaches.
"""

from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd

try:
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import pdist, cdist
    from scipy.spatial import ConvexHull
except ImportError:
    DBSCAN = None
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


def auto_dbscan_clustering(
    df: pd.DataFrame,
    x_col: str = 'target_x',
    y_col: str = 'target_y',
    eps_range: Tuple[float, float] = (0.01, 0.2),
    target_clusters: Tuple[int, int] = (50, 70),
    min_samples: Optional[int] = None,
    max_iterations: int = 10,
    distance_threshold: float = 1.0,
    max_cluster_size: float = 2.2,
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
        Column name for x-coordinate. Default is 'target_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'target_y'.
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
    
    # Extract data
    df = df.copy()
    data = df[[x_col, y_col]].values
    
    # Set min_samples search range
    if min_samples is None:
        base_min_samples = max(1, int(len(data) / 1000))
        min_samples_candidates = [
            max(1, int(base_min_samples * 0.5)),
            max(1, int(base_min_samples * 0.75)),
            base_min_samples,
            max(1, int(base_min_samples * 1.25)),
            max(1, int(base_min_samples * 1.5))
        ]
        min_samples_candidates = sorted(list(set(min_samples_candidates)))
    else:
        min_samples_candidates = [min_samples]
    
    # eps candidates
    eps_candidates = np.linspace(eps_range[0], eps_range[1], 10)
    
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
    
    return df, best_eps, best_n


def peel_and_cluster_edges(
    df: pd.DataFrame,
    x_col: str = 'target_x',
    y_col: str = 'target_y',
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
        Column name for x-coordinate. Default is 'target_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'target_y'.
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
        eps_candidates = config.eps_candidates
        target_new_clusters = config.target_new_clusters
        distance_threshold = config.distance_threshold
    else:
        if radius is None:
            radius_candidates = [0.3, 0.5, 0.8, 1.0, 1.5]
        else:
            radius_candidates = [radius]
        
        if eps is None:
            eps_candidates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
        else:
            eps_candidates = [eps]
    
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
    
    # Set min_samples candidates
    if min_samples is None:
        base_min_samples = max(1, int(len(data) / 1000))
        min_samples_candidates = [
            max(1, int(base_min_samples * 0.5)),
            max(1, int(base_min_samples * 0.75)),
            base_min_samples,
            max(1, int(base_min_samples * 1.25)),
            max(1, int(base_min_samples * 1.5))
        ]
        min_samples_candidates = sorted(list(set(min_samples_candidates)))
    else:
        min_samples_candidates = [min_samples]
    
    if verbose:
        print(f"Parameter search space:")
        print(f"  radius: {radius_candidates}")
        print(f"  eps: {eps_candidates}")
        print(f"  min_samples: {min_samples_candidates}")
        print(f"  target new edge clusters: {target_new_clusters[0]}-{target_new_clusters[1]}")
    
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
                        if pdist(new_centers).min() < distance_threshold:
                            valid = False
                    
                    # Check distances to existing core clusters
                    if valid and len(points_core) > 0:
                        core_centers = df[core_mask].groupby('cluster')[
                            [x_col, y_col]
                        ].mean().values
                        if cdist(new_centers, core_centers).min() < distance_threshold:
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
    x_col: str = 'target_x',
    y_col: str = 'target_y',
    core_config: Optional[DBSCANConfig] = None,
    edge_config: Optional[EdgeClusteringConfig] = None,
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
        Column name for x-coordinate. Default is 'target_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'target_y'.
    core_config : DBSCANConfig, optional
        Configuration for core clustering.
    edge_config : EdgeClusteringConfig, optional
        Configuration for edge clustering.
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
    
    # Stage 1: Core clustering
    if verbose:
        print("\n[Stage 1] Core region clustering...")
    
    df, eps_core, n_core = auto_dbscan_clustering(
        df, x_col=x_col, y_col=y_col,
        config=core_config, verbose=verbose
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
    
    return df, params, n_total


def auto_hdbscan_clustering(
    df: pd.DataFrame,
    x_col: str = 'target_x',
    y_col: str = 'target_y',
    min_cluster_size_range: Tuple[int, int] = (30, 100),
    min_samples_range: Optional[Tuple[int, int]] = (30, 80),
    target_clusters: Tuple[int, int] = (50, 70),
    max_iterations: int = 10,
    distance_threshold: float = 1.0,
    max_cluster_size: float = 2.2,
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
        Column name for x-coordinate. Default is 'target_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'target_y'.
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
    
    # Safety check
    if best_params is None:
        if verbose:
            print("⚠️ Warning: No valid parameters found, using defaults")
        best_params = {
            'min_cluster_size': min_cluster_sizes[0],
            'min_samples': min_samples_candidates[0]
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
    
    return df, best_params, best_n_clusters


def cluster_by_foil_position(
    df: pd.DataFrame,
    method: str = 'dbscan',
    foil_col: str = 'foil_position',
    x_col: str = 'target_x',
    y_col: str = 'target_y',
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
        Column name for x-coordinate. Default is 'target_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'target_y'.
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
