"""
Evaluation module for SHMS Optics Calibration.

This module provides functions for evaluating clustering results,
calculating benchmark metrics (efficiency, purity), and computing
cluster separability in focal plane phase space.
"""

from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd

try:
    from sklearn.metrics import (
        silhouette_score, silhouette_samples,
        davies_bouldin_score, calinski_harabasz_score
    )
    from sklearn.preprocessing import StandardScaler
except ImportError:
    silhouette_score = None
    silhouette_samples = None
    davies_bouldin_score = None
    calinski_harabasz_score = None
    StandardScaler = None

try:
    from scipy.spatial.distance import cdist, pdist, squareform
except ImportError:
    cdist = None
    pdist = None
    squareform = None

from .config import (
    BenchmarkConfig,
    SeparabilityConfig,
    DEFAULT_BENCHMARK_CONFIG,
    DEFAULT_SEPARABILITY_CONFIG,
)


def calculate_cluster_metrics(
    df: pd.DataFrame,
    truth_col: str = 'truth_hole_id',
    cluster_col: str = 'cluster',
    config: Optional[BenchmarkConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Calculate Efficiency and Purity metrics for clustering results.
    
    This function compares clustering results against ground truth labels
    to compute per-cluster and per-truth-hole metrics.
    
    Definitions:
    - **Efficiency**: Fraction of events from a truth hole that are correctly
      classified to a cluster dominated by that hole.
    - **Purity**: Fraction of events in a cluster that belong to the
      dominant truth hole for that cluster.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering results (cluster column) and ground
        truth labels (truth_col).
    truth_col : str, optional
        Column name for ground truth hole IDs. Default is 'truth_hole_id'.
    cluster_col : str, optional
        Column name for predicted cluster IDs. Default is 'cluster'.
    config : BenchmarkConfig, optional
        Configuration object. If provided, overrides individual parameters.
    verbose : bool, optional
        If True, prints summary information. Default is True.
    
    Returns
    -------
    tuple
        (cluster_metrics_df, truth_metrics_df, overall_metrics) where:
        - cluster_metrics_df: Per-cluster metrics (purity, size, etc.)
        - truth_metrics_df: Per-truth-hole metrics (efficiency, etc.)
        - overall_metrics: Dictionary with global metrics
    
    Examples
    --------
    >>> df_sim = load_simulation_data("simulation.root")
    >>> df_clustered, _, _ = auto_dbscan_clustering(df_sim)
    >>> cluster_metrics, truth_metrics, overall = calculate_cluster_metrics(
    ...     df_clustered, truth_col='truth_hole_id'
    ... )
    >>> print(f"Mean Purity: {overall['mean_purity']:.4f}")
    >>> print(f"Mean Efficiency: {overall['mean_efficiency']:.4f}")
    
    See Also
    --------
    calculate_separability_metrics : Cluster separability in 4D phase space
    visualize_benchmark_results : Visualization of benchmark results
    """
    # Use config if provided
    if config is not None:
        truth_col = config.truth_col
        cluster_col = config.cluster_col
    
    # Filter out noise points
    df_valid = df[df[cluster_col] != -1].copy()
    
    clusters = df_valid[cluster_col].unique()
    truth_holes = df[truth_col].unique()
    
    # For each cluster, find dominant truth hole
    cluster_to_truth = {}
    cluster_metrics_list = []
    
    for cluster_id in clusters:
        cluster_mask = df_valid[cluster_col] == cluster_id
        cluster_data = df_valid[cluster_mask]
        
        # Find dominant truth hole
        truth_counts = cluster_data[truth_col].value_counts()
        dominant_truth = truth_counts.index[0]
        dominant_count = truth_counts.iloc[0]
        
        cluster_to_truth[cluster_id] = dominant_truth
        
        # Calculate purity
        n_cluster = len(cluster_data)
        n_correct = dominant_count
        purity = n_correct / n_cluster if n_cluster > 0 else 0
        
        cluster_metrics_list.append({
            'cluster_id': cluster_id,
            'dominant_truth_hole': dominant_truth,
            'cluster_size': n_cluster,
            'correct_hits': n_correct,
            'purity': purity
        })
    
    cluster_metrics_df = pd.DataFrame(cluster_metrics_list)
    
    # Calculate efficiency for each truth hole
    truth_metrics_list = []
    
    for truth_hole in truth_holes:
        # All hits from this truth hole
        truth_mask = df[truth_col] == truth_hole
        n_initial = truth_mask.sum()
        
        # Hits that were clustered (not noise)
        truth_clustered = df_valid[df_valid[truth_col] == truth_hole]
        n_clustered = len(truth_clustered)
        
        # Find clusters dominated by this truth hole
        assigned_clusters = [c for c, t in cluster_to_truth.items() if t == truth_hole]
        
        # Correctly assigned hits
        if len(assigned_clusters) > 0:
            correct_mask = truth_clustered[cluster_col].isin(assigned_clusters)
            n_correct = correct_mask.sum()
        else:
            n_correct = 0
        
        efficiency = n_correct / n_initial if n_initial > 0 else 0
        clustering_rate = n_clustered / n_initial if n_initial > 0 else 0
        
        truth_metrics_list.append({
            'truth_hole_id': truth_hole,
            'initial_hits': n_initial,
            'clustered_hits': n_clustered,
            'correct_hits': n_correct,
            'efficiency': efficiency,
            'clustering_rate': clustering_rate
        })
    
    truth_metrics_df = pd.DataFrame(truth_metrics_list)
    
    # Calculate overall metrics
    n_total = len(df)
    n_noise = (df[cluster_col] == -1).sum()
    n_clustered = n_total - n_noise
    
    overall_metrics = {
        'n_total': n_total,
        'n_clustered': n_clustered,
        'n_noise': n_noise,
        'noise_rate': n_noise / n_total if n_total > 0 else 0,
        'n_clusters': len(clusters),
        'n_truth_holes': len(truth_holes),
        
        # Purity metrics
        'mean_purity': cluster_metrics_df['purity'].mean(),
        'weighted_purity': (
            (cluster_metrics_df['purity'] * cluster_metrics_df['cluster_size']).sum() /
            cluster_metrics_df['cluster_size'].sum()
            if cluster_metrics_df['cluster_size'].sum() > 0 else 0
        ),
        'min_purity': cluster_metrics_df['purity'].min(),
        'max_purity': cluster_metrics_df['purity'].max(),
        
        # Efficiency metrics
        'mean_efficiency': truth_metrics_df['efficiency'].mean(),
        'weighted_efficiency': (
            (truth_metrics_df['efficiency'] * truth_metrics_df['initial_hits']).sum() /
            truth_metrics_df['initial_hits'].sum()
            if truth_metrics_df['initial_hits'].sum() > 0 else 0
        ),
        'min_efficiency': truth_metrics_df['efficiency'].min(),
        'max_efficiency': truth_metrics_df['efficiency'].max(),
    }
    
    if verbose:
        print("\n=== Benchmark Metrics ===")
        print(f"Clusters: {overall_metrics['n_clusters']} "
              f"(Truth holes: {overall_metrics['n_truth_holes']})")
        print(f"Noise rate: {overall_metrics['noise_rate']*100:.2f}%")
        print(f"Mean Purity: {overall_metrics['mean_purity']:.4f}")
        print(f"Weighted Purity: {overall_metrics['weighted_purity']:.4f}")
        print(f"Mean Efficiency: {overall_metrics['mean_efficiency']:.4f}")
        print(f"Weighted Efficiency: {overall_metrics['weighted_efficiency']:.4f}")
    
    return cluster_metrics_df, truth_metrics_df, overall_metrics


def calculate_separability_metrics(
    df: pd.DataFrame,
    fp_cols: Optional[List[str]] = None,
    normalize: bool = True,
    config: Optional[SeparabilityConfig] = None,
    verbose: bool = True
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """
    Calculate cluster separability metrics in 4D focal plane phase space.
    
    This function evaluates how well-separated clusters are in the
    4-dimensional focal plane space (x_fp, y_fp, xp_fp, yp_fp).
    
    Metrics computed:
    - **Silhouette Score**: Overall cluster quality (-1 to 1, higher is better)
    - **Davies-Bouldin Index**: Cluster compactness (lower is better)
    - **Calinski-Harabasz Score**: Between/within cluster variance ratio
    - **Separability Ratio**: Inter-cluster / intra-cluster distance
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering results and focal plane columns.
    fp_cols : list of str, optional
        Focal plane column names [x_fp, y_fp, xp_fp, yp_fp].
        Default is ['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp'].
    normalize : bool, optional
        If True, normalize features before computing distances.
        Recommended due to different units. Default is True.
    config : SeparabilityConfig, optional
        Configuration object. If provided, overrides individual parameters.
    verbose : bool, optional
        If True, prints summary information. Default is True.
    
    Returns
    -------
    tuple
        (global_metrics, per_cluster_metrics) where:
        - global_metrics: Dictionary with overall separability metrics
        - per_cluster_metrics: DataFrame with per-cluster metrics
        Returns (None, None) if insufficient data.
    
    Examples
    --------
    >>> df, _, _ = auto_dbscan_clustering(df)
    >>> metrics, cluster_df = calculate_separability_metrics(df)
    >>> if metrics:
    ...     print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    ...     print(f"Mean Separability Ratio: {metrics['mean_separability_ratio']:.3f}")
    
    See Also
    --------
    calculate_cluster_metrics : Efficiency and purity metrics
    visualize_separability : Separability visualization
    """
    if silhouette_score is None or StandardScaler is None:
        raise ImportError(
            "sklearn is required for separability metrics. "
            "Install it with: pip install scikit-learn"
        )
    
    if pdist is None or squareform is None:
        raise ImportError(
            "scipy is required for separability metrics. "
            "Install it with: pip install scipy"
        )
    
    # Use config if provided
    if config is not None:
        fp_cols = config.fp_cols
        normalize = config.normalize
    
    if fp_cols is None:
        fp_cols = ['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']
    
    # Check columns exist
    missing = [c for c in fp_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing focal plane columns: {missing}")
    
    # Filter out noise
    df_signal = df[~df['is_noise']].copy()
    
    if len(df_signal) < 2:
        return None, None
    
    # Extract 4D features
    X = df_signal[fp_cols].values
    labels = df_signal['cluster'].values
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    unique_clusters = np.unique(labels)
    n_clusters = len(unique_clusters)
    
    # Global metrics
    global_metrics = {}
    
    if n_clusters > 1:
        global_metrics['silhouette_score'] = silhouette_score(X_scaled, labels)
        global_metrics['davies_bouldin_index'] = davies_bouldin_score(X_scaled, labels)
        global_metrics['calinski_harabasz_score'] = calinski_harabasz_score(X_scaled, labels)
    else:
        global_metrics['silhouette_score'] = np.nan
        global_metrics['davies_bouldin_index'] = np.nan
        global_metrics['calinski_harabasz_score'] = np.nan
    
    # Per-cluster metrics
    cluster_metrics_list = []
    
    # Calculate cluster centers
    centers = {}
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        centers[cluster_id] = X_scaled[mask].mean(axis=0)
    
    # Inter-cluster distances
    center_array = np.array([centers[c] for c in unique_clusters])
    if len(center_array) > 1:
        inter_cluster_dists = squareform(pdist(center_array))
    else:
        inter_cluster_dists = np.array([[0]])
    
    # Per-cluster silhouette
    if n_clusters > 1:
        silhouette_vals = silhouette_samples(X_scaled, labels)
    else:
        silhouette_vals = np.zeros(len(labels))
    
    for i, cluster_id in enumerate(unique_clusters):
        mask = labels == cluster_id
        cluster_points = X_scaled[mask]
        n_points = mask.sum()
        center = centers[cluster_id]
        
        # Intra-cluster metrics
        dist_to_center = np.linalg.norm(cluster_points - center, axis=1)
        intra_cluster_dist = dist_to_center.mean()
        intra_cluster_std = dist_to_center.std()
        max_radius = dist_to_center.max()
        
        # Inter-cluster metrics
        if n_clusters > 1:
            other_dists = inter_cluster_dists[i, :]
            other_dists = other_dists[other_dists > 0]
            min_inter_dist = other_dists.min() if len(other_dists) > 0 else np.nan
            mean_inter_dist = other_dists.mean() if len(other_dists) > 0 else np.nan
        else:
            min_inter_dist = np.nan
            mean_inter_dist = np.nan
        
        # Separability ratio
        if intra_cluster_dist > 0 and not np.isnan(min_inter_dist):
            separability_ratio = min_inter_dist / intra_cluster_dist
        else:
            separability_ratio = np.nan
        
        # Per-cluster silhouette
        cluster_silhouette = silhouette_vals[mask].mean()
        
        cluster_metrics_list.append({
            'cluster_id': cluster_id,
            'n_points': n_points,
            'intra_cluster_dist': intra_cluster_dist,
            'intra_cluster_std': intra_cluster_std,
            'max_radius': max_radius,
            'min_inter_cluster_dist': min_inter_dist,
            'mean_inter_cluster_dist': mean_inter_dist,
            'separability_ratio': separability_ratio,
            'silhouette': cluster_silhouette
        })
    
    cluster_metrics_df = pd.DataFrame(cluster_metrics_list)
    
    # Add summary statistics to global metrics
    global_metrics['n_clusters'] = n_clusters
    global_metrics['n_points'] = len(df_signal)
    global_metrics['mean_intra_cluster_dist'] = cluster_metrics_df['intra_cluster_dist'].mean()
    global_metrics['mean_inter_cluster_dist'] = cluster_metrics_df['min_inter_cluster_dist'].mean()
    global_metrics['mean_separability_ratio'] = cluster_metrics_df['separability_ratio'].mean()
    global_metrics['min_separability_ratio'] = cluster_metrics_df['separability_ratio'].min()
    global_metrics['max_separability_ratio'] = cluster_metrics_df['separability_ratio'].max()
    
    if verbose:
        print("\n=== 4D Focal Plane Separability ===")
        print(f"Clusters: {n_clusters}")
        print(f"Silhouette Score: {global_metrics['silhouette_score']:.4f}")
        print(f"Davies-Bouldin Index: {global_metrics['davies_bouldin_index']:.4f}")
        print(f"Mean Separability Ratio: {global_metrics['mean_separability_ratio']:.3f}")
        
        well_separated = (cluster_metrics_df['separability_ratio'] > 1.0).sum()
        print(f"Well-separated clusters (ratio > 1): {well_separated}/{n_clusters}")
    
    return global_metrics, cluster_metrics_df


def compare_algorithms(
    results: Dict[str, Dict[str, Any]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create comparison table for multiple clustering algorithms.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping algorithm names to their results.
        Expected structure: {
            'algorithm_name': {
                'cluster_metrics': DataFrame,
                'truth_metrics': DataFrame,
                'overall': Dict
            }
        }
    verbose : bool, optional
        If True, prints comparison table. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Comparison table with a 'Metric' column and one column per
        algorithm, containing the following rows (in order):

        - Clusters Detected
        - Truth Holes
        - Noise Rate (%)
        - Mean Purity
        - Weighted Purity
        - Mean Efficiency
        - Weighted Efficiency

    Notes
    -----
    The ``overall`` dictionary for each algorithm is expected to be the
    third element returned by :func:`calculate_cluster_metrics`.  Keys
    used are ``'n_clusters'``, ``'n_truth_holes'``, ``'noise_rate'``,
    ``'mean_purity'``, ``'weighted_purity'``, ``'mean_efficiency'``, and
    ``'weighted_efficiency'``.  Missing keys default to 0 or ``'N/A'``.
    
    Examples
    --------
    >>> _, _, dbscan_overall = calculate_cluster_metrics(df_dbscan)
    >>> _, _, hdbscan_overall = calculate_cluster_metrics(df_hdbscan)
    >>> comparison = compare_algorithms({
    ...     'DBSCAN': {
    ...         'cluster_metrics': dbscan_cluster_df,
    ...         'truth_metrics': dbscan_truth_df,
    ...         'overall': dbscan_overall
    ...     },
    ...     'HDBSCAN': {
    ...         'cluster_metrics': hdbscan_cluster_df,
    ...         'truth_metrics': hdbscan_truth_df,
    ...         'overall': hdbscan_overall
    ...     }
    ... })
    >>> print(comparison)
    
    See Also
    --------
    calculate_cluster_metrics : Compute per-algorithm overall metrics
    visualize_benchmark_comparison : Visualize comparison as bar charts
    """
    metrics_names = [
        'Clusters Detected',
        'Truth Holes',
        'Noise Rate (%)',
        'Mean Purity',
        'Weighted Purity',
        'Mean Efficiency',
        'Weighted Efficiency'
    ]
    
    comparison_data = {'Metric': metrics_names}
    
    for alg_name, result in results.items():
        overall = result['overall']
        comparison_data[alg_name] = [
            overall.get('n_clusters', 'N/A'),
            overall.get('n_truth_holes', 'N/A'),
            f"{overall.get('noise_rate', 0)*100:.2f}",
            f"{overall.get('mean_purity', 0):.4f}",
            f"{overall.get('weighted_purity', 0):.4f}",
            f"{overall.get('mean_efficiency', 0):.4f}",
            f"{overall.get('weighted_efficiency', 0):.4f}"
        ]
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if verbose:
        print("\n=== Algorithm Comparison ===")
        print(comparison_df.to_string(index=False))
    
    return comparison_df


def get_low_performance_holes(
    truth_metrics: pd.DataFrame,
    efficiency_threshold: float = 0.5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Identify truth holes with low efficiency.
    
    Parameters
    ----------
    truth_metrics : pd.DataFrame
        Per-truth-hole metrics from calculate_cluster_metrics.
    efficiency_threshold : float, optional
        Threshold below which a hole is considered low-efficiency.
        Default is 0.5.
    verbose : bool, optional
        If True, prints information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Subset of truth_metrics for low-efficiency holes.
    
    Examples
    --------
    >>> _, truth_metrics, _ = calculate_cluster_metrics(df)
    >>> low_eff = get_low_performance_holes(truth_metrics, efficiency_threshold=0.5)
    >>> print(f"Low-efficiency holes: {len(low_eff)}")
    """
    low_eff = truth_metrics[truth_metrics['efficiency'] < efficiency_threshold].copy()
    low_eff = low_eff.sort_values('efficiency')
    
    if verbose:
        print(f"\nHoles with efficiency < {efficiency_threshold}:")
        print(f"Count: {len(low_eff)} / {len(truth_metrics)}")
        if len(low_eff) > 0:
            print("\nLowest efficiency holes:")
            print(low_eff.head(10).to_string(index=False))
    
    return low_eff


def get_low_purity_clusters(
    cluster_metrics: pd.DataFrame,
    purity_threshold: float = 0.9,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Identify clusters with low purity.
    
    Parameters
    ----------
    cluster_metrics : pd.DataFrame
        Per-cluster metrics from calculate_cluster_metrics.
    purity_threshold : float, optional
        Threshold below which a cluster is considered low-purity.
        Default is 0.9.
    verbose : bool, optional
        If True, prints information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Subset of cluster_metrics for low-purity clusters.
    
    Examples
    --------
    >>> cluster_metrics, _, _ = calculate_cluster_metrics(df)
    >>> low_purity = get_low_purity_clusters(cluster_metrics, purity_threshold=0.9)
    >>> print(f"Low-purity clusters: {len(low_purity)}")
    """
    low_purity = cluster_metrics[cluster_metrics['purity'] < purity_threshold].copy()
    low_purity = low_purity.sort_values('purity')
    
    if verbose:
        print(f"\nClusters with purity < {purity_threshold}:")
        print(f"Count: {len(low_purity)} / {len(cluster_metrics)}")
        if len(low_purity) > 0:
            print("\nLowest purity clusters:")
            print(low_purity.head(10).to_string(index=False))
    
    return low_purity
