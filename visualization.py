"""
Visualization module for SHMS Optics Calibration.

This module provides functions for visualizing clustering results,
target plane patterns, and focal plane distributions.
"""

from typing import Optional, Tuple, Dict, List, Any
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None
    cm = None

from .config import (
    VisualizationConfig,
    DEFAULT_VISUALIZATION_CONFIG,
)


def _check_matplotlib():
    """Check if matplotlib is available and raise an informative error if not.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    if plt is None:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def visualize_dbscan_results(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    best_eps: Optional[float] = None,
    n_clusters: int = 0,
    figsize: Tuple[int, int] = (10, 10),
    xlim: Tuple[float, float] = (-20, 20),
    ylim: Tuple[float, float] = (-20, 20),
    title_prefix: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize DBSCAN/HDBSCAN clustering results.
    
    Creates a scatter plot showing clustered points with different colors
    for each cluster, noise points in black, and cluster centers marked.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering results (must have 'cluster' and 
        'is_noise' columns).
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    best_eps : float, optional
        Eps parameter to display in title.
    n_clusters : int, optional
        Number of clusters to display in title.
    figsize : tuple of int, optional
        Figure size (width, height). Default is (10, 10).
    xlim : tuple of float, optional
        X-axis limits. Default is (-20, 20).
    ylim : tuple of float, optional
        Y-axis limits. Default is (-20, 20).
    title_prefix : str, optional
        Prefix for the plot title (e.g., "Foil 0").
    config : VisualizationConfig, optional
        Configuration object. If provided, overrides individual parameters.
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> df, eps, n = auto_dbscan_clustering(df)
    >>> fig = visualize_dbscan_results(
    ...     df, best_eps=eps, n_clusters=n, title_prefix="Foil 0"
    ... )
    
    See Also
    --------
    visualize_clustering_summary : Multi-panel clustering visualization
    visualize_clusters_in_focal_plane : Focal plane projection
    """
    _check_matplotlib()
    
    # Use config if provided
    if config is not None:
        figsize = config.figsize
        xlim = config.xlim
        ylim = config.ylim
    
    fig = plt.figure(figsize=figsize)
    
    # Separate signal and noise points
    mask_signal = ~df['is_noise']
    
    # Plot clustered points
    scatter = plt.scatter(
        df.loc[mask_signal, x_col],
        df.loc[mask_signal, y_col],
        s=0.2,
        c=df.loc[mask_signal, 'cluster'],
        cmap='nipy_spectral',
        alpha=0.6,
        label='Clustered points'
    )
    
    # Plot noise points
    if df['is_noise'].sum() > 0:
        plt.scatter(
            df.loc[df['is_noise'], x_col],
            df.loc[df['is_noise'], y_col],
            s=0.1,
            c='black',
            alpha=0.3,
            label='Noise'
        )
    
    # Plot cluster centers
    centroids = df[mask_signal].groupby('cluster')[[x_col, y_col]].mean()
    plt.plot(
        centroids[x_col], centroids[y_col],
        'w+', markersize=10, markeredgewidth=1.5,
        label='Cluster centers'
    )
    
    # Set figure properties
    title = "DBSCAN Clustering Results"
    if title_prefix:
        title = f"{title_prefix} | {title}"
    if best_eps is not None and n_clusters is not None:
        if isinstance(best_eps, dict):
            title += f"\nClusters={n_clusters}"
        else:
            title += f"\neps={best_eps:.4f}, Clusters={n_clusters}"
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    if show:
        plt.show()
    
    # Print statistics
    print("\n=== Clustering Statistics ===")
    cluster_stats = df[~df['is_noise']].groupby('cluster').size()
    print(f"Total clusters: {len(cluster_stats)}")
    print(f"Average points per cluster: {cluster_stats.mean():.1f}")
    print(f"Max cluster size: {cluster_stats.max()}")
    print(f"Min cluster size: {cluster_stats.min()}")
    
    return fig


def visualize_clustering_summary(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    n_clusters: int = 0,
    figsize: Tuple[int, int] = (14, 12),
    xlim: Tuple[float, float] = (-20, 20),
    ylim: Tuple[float, float] = (-20, 20),
    title_prefix: str = "Clustering",
    show: bool = True
) -> plt.Figure:
    """
    Create a multi-panel summary visualization of clustering results.
    
    Creates a 2x2 figure with:
    - Clustered scatter plot
    - Cluster centers distribution
    - Cluster size histogram
    - Cluster statistics text
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering results.
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    n_clusters : int, optional
        Number of clusters for title.
    figsize : tuple of int, optional
        Figure size. Default is (14, 12).
    xlim : tuple of float, optional
        X-axis limits. Default is (-20, 20).
    ylim : tuple of float, optional
        Y-axis limits. Default is (-20, 20).
    title_prefix : str, optional
        Prefix for figure title. Default is "Clustering".
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> df, _, n = auto_dbscan_clustering(df)
    >>> fig = visualize_clustering_summary(df, n_clusters=n)
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Clustering results with color coding
    ax = axes[0, 0]
    for cluster_id in sorted(df[~df['is_noise']]['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        ax.scatter(
            cluster_data[x_col], cluster_data[y_col],
            s=0.5, alpha=0.6,
            label=f'Cluster {cluster_id}' if cluster_id < 10 else ''
        )
    
    noise_data = df[df['is_noise']]
    if len(noise_data) > 0:
        ax.scatter(
            noise_data[x_col], noise_data[y_col],
            s=0.3, c='gray', alpha=0.3, label='Noise'
        )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'{title_prefix} - Results ({n_clusters} clusters)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    if n_clusters <= 10:
        ax.legend(fontsize=8, loc='best')
    
    # 2. Cluster centers
    ax = axes[0, 1]
    centroids = df[~df['is_noise']].groupby('cluster')[[x_col, y_col]].mean()
    
    ax.scatter(
        centroids[x_col], centroids[y_col],
        s=100, c='red', marker='x', linewidths=2, label='Cluster Centers'
    )
    
    for idx, (cluster_id, center) in enumerate(centroids.iterrows()):
        if idx < 50:
            ax.annotate(
                f'{cluster_id}', (center[x_col], center[y_col]),
                fontsize=6, alpha=0.7, ha='center'
            )
    
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f'Cluster Centers ({len(centroids)} centers)')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Cluster size histogram
    ax = axes[1, 0]
    cluster_sizes = df[~df['is_noise']].groupby('cluster').size()
    ax.hist(cluster_sizes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Cluster Size (points)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Cluster Size Distribution (mean={cluster_sizes.mean():.1f})')
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics text
    ax = axes[1, 1]
    ax.axis('off')
    
    n_signal = (~df['is_noise']).sum()
    n_noise = df['is_noise'].sum()
    
    stats_text = f"""
    Clustering Statistics
    {'=' * 40}
    
    Total Events: {len(df):,}
    Signal Points: {n_signal:,} ({n_signal/len(df)*100:.1f}%)
    Noise Points: {n_noise:,} ({n_noise/len(df)*100:.1f}%)
    
    Number of Clusters: {n_clusters}
    
    Cluster Size:
      Mean: {cluster_sizes.mean():.1f}
      Std: {cluster_sizes.std():.1f}
      Min: {cluster_sizes.min()}
      Max: {cluster_sizes.max()}
    
    Coordinate Ranges:
      X: [{df[x_col].min():.2f}, {df[x_col].max():.2f}]
      Y: [{df[y_col].min():.2f}, {df[y_col].max():.2f}]
    """
    
    ax.text(
        0.1, 0.5, stats_text, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def visualize_clusters_in_focal_plane(
    df: pd.DataFrame,
    foil_pos: Optional[int] = None,
    fp_cols: List[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    show: bool = True
) -> plt.Figure:
    """
    Visualize clustering results projected onto focal plane variables.
    
    Creates a 2x3 figure showing cluster distributions in different
    focal plane variable combinations and a cluster size histogram.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering results and focal plane columns.
    foil_pos : int, optional
        Foil position for title.
    fp_cols : list of str, optional
        Focal plane column names [x_fp, y_fp, xp_fp, yp_fp].
        Default is ['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp'].
    figsize : tuple of int, optional
        Figure size. Default is (16, 12).
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> df, _, n = auto_dbscan_clustering(df)
    >>> fig = visualize_clusters_in_focal_plane(df, foil_pos=0)
    """
    _check_matplotlib()
    
    if fp_cols is None:
        fp_cols = ['P_dc_x_fp', 'P_dc_y_fp', 'P_dc_xp_fp', 'P_dc_yp_fp']
    
    x_fp, y_fp, xp_fp, yp_fp = fp_cols
    
    # Check columns exist
    missing = [c for c in fp_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing focal plane columns: {missing}")
    
    # Separate signal and noise
    mask_signal = ~df['is_noise']
    df_signal = df[mask_signal]
    df_noise = df[~mask_signal]
    
    n_clusters = df_signal['cluster'].nunique()
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    title = f'Focal Plane Clustering Visualization'
    if foil_pos is not None:
        title += f' | foil_position = {foil_pos}'
    title += f'\nClusters: {n_clusters}, Signal: {len(df_signal):,}, Noise: {len(df_noise):,}'
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    cmap = plt.cm.nipy_spectral
    
    # 1. x_fp vs y_fp
    ax = axes[0, 0]
    if len(df_noise) > 0:
        ax.scatter(df_noise[x_fp], df_noise[y_fp], s=0.3, c='lightgray', alpha=0.3)
    scatter = ax.scatter(
        df_signal[x_fp], df_signal[y_fp], s=0.5,
        c=df_signal['cluster'], cmap=cmap, alpha=0.6
    )
    ax.set_xlabel('P_dc_x_fp [cm]')
    ax.set_ylabel('P_dc_y_fp [cm]')
    ax.set_title('x_fp vs y_fp')
    ax.grid(True, alpha=0.3)
    
    # 2. xp_fp vs yp_fp
    ax = axes[0, 1]
    if len(df_noise) > 0:
        ax.scatter(df_noise[xp_fp], df_noise[yp_fp], s=0.3, c='lightgray', alpha=0.3)
    scatter = ax.scatter(
        df_signal[xp_fp], df_signal[yp_fp], s=0.5,
        c=df_signal['cluster'], cmap=cmap, alpha=0.6
    )
    ax.set_xlabel("P_dc_xp_fp [rad]")
    ax.set_ylabel("P_dc_yp_fp [rad]")
    ax.set_title("xp_fp vs yp_fp (Angular)")
    ax.grid(True, alpha=0.3)
    
    # 3. x_fp vs xp_fp
    ax = axes[0, 2]
    if len(df_noise) > 0:
        ax.scatter(df_noise[x_fp], df_noise[xp_fp], s=0.3, c='lightgray', alpha=0.3)
    scatter = ax.scatter(
        df_signal[x_fp], df_signal[xp_fp], s=0.5,
        c=df_signal['cluster'], cmap=cmap, alpha=0.6
    )
    ax.set_xlabel('P_dc_x_fp [cm]')
    ax.set_ylabel("P_dc_xp_fp [rad]")
    ax.set_title("x_fp vs xp_fp (X Phase Space)")
    ax.grid(True, alpha=0.3)
    
    # 4. y_fp vs yp_fp
    ax = axes[1, 0]
    if len(df_noise) > 0:
        ax.scatter(df_noise[y_fp], df_noise[yp_fp], s=0.3, c='lightgray', alpha=0.3)
    scatter = ax.scatter(
        df_signal[y_fp], df_signal[yp_fp], s=0.5,
        c=df_signal['cluster'], cmap=cmap, alpha=0.6
    )
    ax.set_xlabel('P_dc_y_fp [cm]')
    ax.set_ylabel("P_dc_yp_fp [rad]")
    ax.set_title("y_fp vs yp_fp (Y Phase Space)")
    ax.grid(True, alpha=0.3)
    
    # 5. x_fp vs y_fp (2D histogram)
    ax = axes[1, 1]
    h = ax.hist2d(df_signal[x_fp], df_signal[y_fp], bins=100, cmap='hot')
    ax.set_xlabel('P_dc_x_fp [cm]')
    ax.set_ylabel('P_dc_y_fp [cm]')
    ax.set_title('x_fp vs y_fp (Density)')
    plt.colorbar(h[3], ax=ax, label='Counts')
    
    # 6. Cluster size distribution
    ax = axes[1, 2]
    cluster_sizes = df_signal.groupby('cluster').size()
    ax.hist(cluster_sizes, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(
        cluster_sizes.mean(), color='red', linestyle='--',
        label=f'Mean: {cluster_sizes.mean():.1f}'
    )
    ax.axvline(
        cluster_sizes.median(), color='green', linestyle='--',
        label=f'Median: {cluster_sizes.median():.1f}'
    )
    ax.set_xlabel('Cluster Size (events)')
    ax.set_ylabel('Frequency')
    ax.set_title('Cluster Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def visualize_foil_classification(
    df: pd.DataFrame,
    col_name: str = 'P_gtr_y',
    y_range: Tuple[float, float] = (-5, 5),
    bins: int = 100,
    figsize: Tuple[int, int] = (12, 6),
    show: bool = True
) -> plt.Figure:
    """
    Visualize foil position classification results.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'foil_position' column.
    col_name : str, optional
        Column used for classification. Default is 'P_gtr_y'.
    y_range : tuple of float, optional
        Range for histogram. Default is (-5, 5).
    bins : int, optional
        Number of histogram bins. Default is 100.
    figsize : tuple of int, optional
        Figure size. Default is (12, 6).
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> # load_and_prepare_data and classify_foils_with_range are from
    >>> # shms_optics_calibration.data_io and .preprocessing respectively
    >>> from shms_optics_calibration import (
    ...     load_and_prepare_data, classify_foils_with_range
    ... )
    >>> df = load_and_prepare_data("data.root")
    >>> df = classify_foils_with_range(df)
    >>> fig = visualize_foil_classification(df, col_name='P_gtr_y')
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get foil positions
    foil_positions = sorted([
        int(p) for p in df['foil_position'].unique() if p != -1
    ])
    
    colors = plt.cm.tab10.colors
    
    # Plot each foil
    for i, foil_pos in enumerate(foil_positions):
        mask = df['foil_position'] == foil_pos
        data = df.loc[mask, col_name].dropna()
        ax.hist(
            data, bins=bins, range=y_range,
            alpha=0.5, color=colors[i % len(colors)],
            label=f'Foil {foil_pos} (n={len(data):,})'
        )
    
    # Plot unclassified
    unclassified = df.loc[df['foil_position'] == -1, col_name].dropna()
    if len(unclassified) > 0:
        ax.hist(
            unclassified, bins=bins, range=y_range,
            alpha=0.3, color='gray',
            label=f'Unclassified (n={len(unclassified):,})'
        )
    
    ax.set_xlabel(col_name)
    ax.set_ylabel('Counts')
    ax.set_title(f'Foil Position Classification | {col_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def visualize_sieve_plane(
    df: pd.DataFrame,
    x_col: str = 'sieve_x',
    y_col: str = 'sieve_y',
    bins: int = 200,
    xlim: Tuple[float, float] = (-20, 20),
    ylim: Tuple[float, float] = (-20, 20),
    figsize: Tuple[int, int] = (10, 10),
    title: str = 'Sieve Plane Reconstruction',
    show: bool = True
) -> plt.Figure:
    """
    Visualize sieve-plane reconstruction as 2D histogram.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with sieve-plane coordinates.
    x_col : str, optional
        Column name for x-coordinate. Default is 'sieve_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'sieve_y'.
    bins : int, optional
        Number of bins for histogram. Default is 200.
    xlim : tuple of float, optional
        X-axis limits. Default is (-20, 20).
    ylim : tuple of float, optional
        Y-axis limits. Default is (-20, 20).
    figsize : tuple of int, optional
        Figure size. Default is (10, 10).
    title : str, optional
        Plot title. Default is 'Sieve Plane Reconstruction'.
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> fig = visualize_sieve_plane(df)
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    h = ax.hist2d(
        df[x_col], df[y_col],
        bins=bins, range=[xlim, ylim],
        cmap='viridis'
    )
    
    plt.colorbar(h[3], ax=ax, label='Counts')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def visualize_benchmark_comparison(
    comparison_data: Dict[str, Dict[str, Any]],
    figsize: Tuple[int, int] = (18, 6),
    show: bool = True
) -> plt.Figure:
    """
    Visualize benchmark comparison across multiple algorithms.
    
    Parameters
    ----------
    comparison_data : dict
        Dictionary mapping algorithm names to their metrics.
        Expected keys in each metric dict: 'n_clusters', 'mean_purity',
        'weighted_purity', 'mean_efficiency', 'weighted_efficiency'.
    figsize : tuple of int, optional
        Figure size. Default is (18, 6).
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.
    
    Examples
    --------
    >>> comparison = {
    ...     'DBSCAN': {'n_clusters': 85, 'mean_purity': 0.95, ...},
    ...     'HDBSCAN': {'n_clusters': 82, 'mean_purity': 0.97, ...}
    ... }
    >>> fig = visualize_benchmark_comparison(comparison)
    """
    _check_matplotlib()
    
    algorithms = list(comparison_data.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # 1. Cluster count comparison
    ax = axes[0]
    n_clusters = [comparison_data[alg]['n_clusters'] for alg in algorithms]
    colors = plt.cm.tab10.colors[:len(algorithms)]
    bars = ax.bar(algorithms, n_clusters, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Cluster Detection')
    for bar, val in zip(bars, n_clusters):
        ax.text(
            bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            str(val), ha='center', va='bottom', fontweight='bold'
        )
    ax.grid(True, alpha=0.3, axis='y')
    
    # 2. Purity and Efficiency comparison
    ax = axes[1]
    metrics = ['Mean\nPurity', 'Weighted\nPurity', 'Mean\nEfficiency', 'Weighted\nEfficiency']
    
    x_pos = np.arange(len(metrics))
    width = 0.8 / len(algorithms)
    
    for i, alg in enumerate(algorithms):
        data = comparison_data[alg]
        values = [
            data.get('mean_purity', 0),
            data.get('weighted_purity', 0),
            data.get('mean_efficiency', 0),
            data.get('weighted_efficiency', 0)
        ]
        offset = (i - len(algorithms)/2 + 0.5) * width
        ax.bar(x_pos + offset, values, width, label=alg, color=colors[i])
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Summary table
    ax = axes[2]
    ax.axis('off')
    
    summary = "Algorithm Comparison Summary\n" + "=" * 40 + "\n\n"
    for alg in algorithms:
        data = comparison_data[alg]
        summary += f"{alg}:\n"
        summary += f"  Clusters: {data.get('n_clusters', 'N/A')}\n"
        summary += f"  Mean Purity: {data.get('mean_purity', 0):.4f}\n"
        summary += f"  Mean Efficiency: {data.get('mean_efficiency', 0):.4f}\n\n"
    
    ax.text(
        0.1, 0.5, summary, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_efficiency_map(
    df: pd.DataFrame,
    efficiency_col: str = 'efficiency',
    x_col: str = 'xsieve_truth',
    y_col: str = 'ysieve_truth',
    xlim: Tuple[float, float] = (-15, 15),
    ylim: Tuple[float, float] = (-10, 10),
    figsize: Tuple[int, int] = (10, 8),
    title: str = 'Efficiency Map',
    show: bool = True
) -> plt.Figure:
    """
    Plot efficiency map for simulation benchmark.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with efficiency values and truth positions.
    efficiency_col : str, optional
        Column name for efficiency values. Default is 'efficiency'.
    x_col : str, optional
        Column name for x position. Default is 'xsieve_truth'.
    y_col : str, optional
        Column name for y position. Default is 'ysieve_truth'.
    xlim : tuple of float, optional
        X-axis limits. Default is (-15, 15).
    ylim : tuple of float, optional
        Y-axis limits. Default is (-10, 10).
    figsize : tuple of int, optional
        Figure size. Default is (10, 8).
    title : str, optional
        Plot title. Default is 'Efficiency Map'.
    show : bool, optional
        If True, displays the figure. Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure object.

    Examples
    --------
    >>> _, truth_metrics, _ = calculate_cluster_metrics(df_sim)
    >>> # Merge truth positions into metrics
    >>> truth_pos = df_sim.groupby('truth_hole_id')[
    ...     ['xsieve_truth', 'ysieve_truth']
    ... ].first().reset_index()
    >>> df_eff = truth_metrics.merge(truth_pos, on='truth_hole_id')
    >>> fig = plot_efficiency_map(df_eff, title='DBSCAN Efficiency Map')
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(
        df[x_col], df[y_col],
        c=df[efficiency_col], cmap='RdYlGn', s=100,
        edgecolors='black', vmin=0, vmax=1
    )
    
    plt.colorbar(scatter, ax=ax, label='Efficiency')
    ax.set_xlabel(f'{x_col} [cm]')
    ax.set_ylabel(f'{y_col} [cm]')
    ax.set_title(title)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig
