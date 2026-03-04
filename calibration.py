"""
Calibration module for SHMS Optics Calibration.

This module provides functions for building grid indices from cluster
centers and performing calibration alignment.
"""

from typing import Optional, Tuple, Dict, List, Any
import warnings
import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
except ImportError:
    PCA = None
    NearestNeighbors = None

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial import ConvexHull, Delaunay
except ImportError:
    fcluster = None
    linkage = None
    ConvexHull = None
    Delaunay = None

from .config import (
    GridIndexConfig,
    DEFAULT_GRID_INDEX_CONFIG,
)

# Warning thresholds for unusual sieve pattern shapes
_MAX_MISSING_RATIO = 0.15
_MAX_ROW_COUNT_CV = 0.3
_MAX_SPACING_CV = 0.3
_SPACING_CHECK_NEIGHBORS = 3


def build_grid_index_from_centers(
    df: pd.DataFrame,
    x_col: str = 'cluster_center_x',
    y_col: str = 'cluster_center_y',
    cluster_col: str = 'cluster',
    use_pca_alignment: bool = False,
    merge_threshold: float = 0.5,
    config: Optional[GridIndexConfig] = None,
    verbose: bool = True
) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """
    Build row/column grid indices from cluster centers.
    
    This function takes cluster centers and assigns row and column indices
    based on their spatial arrangement, assuming an approximately rectangular
    grid pattern (sieve hole pattern).
    
    The algorithm:
    1. Extract unique cluster centers
    2. Optionally merge centers that are too close together
    3. Optionally align to principal axes using PCA
    4. Estimate grid spacing from nearest neighbor distances
    5. Assign row/column indices based on grid spacing
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with clustering results (must have cluster_center_x,
        cluster_center_y, cluster, and is_noise columns).
    x_col : str, optional
        Column name for cluster center x. Default is 'cluster_center_x'.
    y_col : str, optional
        Column name for cluster center y. Default is 'cluster_center_y'.
    cluster_col : str, optional
        Column name for cluster labels. Default is 'cluster'.
    use_pca_alignment : bool, optional
        If True, align to principal axes using PCA. Default is False.
    merge_threshold : float, optional
        Distance threshold for merging close centers. Default is 0.5 cm.
    config : GridIndexConfig, optional
        Configuration object. If provided, overrides individual parameters.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    tuple
        (centers_df, grid_params) where:
        - centers_df: DataFrame with cluster, center_x, center_y, row, col
        - grid_params: Dictionary with grid parameters (spacing, origin, etc.)
        Returns (None, None) if insufficient centers.
    
    Examples
    --------
    >>> df, _, n = auto_dbscan_clustering(df)
    >>> centers, params = build_grid_index_from_centers(df)
    >>> if centers is not None:
    ...     print(f"Grid spacing: {params['grid_spacing']:.3f} cm")
    ...     print(centers[['cluster', 'row', 'col']].head())
    
    See Also
    --------
    align_grid_to_reference : Align grid to reference pattern
    get_missing_holes : Identify missing sieve holes
    
    Raises
    ------
    ImportError
        If sklearn or scipy is not installed.
    ValueError
        If the DataFrame is missing required columns ('is_noise',
        x_col, or y_col).
    """
    if PCA is None or NearestNeighbors is None:
        raise ImportError(
            "sklearn is required for grid indexing. "
            "Install it with: pip install scikit-learn"
        )
    
    if fcluster is None or linkage is None:
        raise ImportError(
            "scipy is required for grid indexing. "
            "Install it with: pip install scipy"
        )
    
    # Use config if provided
    if config is not None:
        x_col = config.x_col
        y_col = config.y_col
        cluster_col = config.cluster_col
        use_pca_alignment = config.use_pca_alignment
        merge_threshold = config.merge_threshold
    
    # Extract unique cluster centers
    valid_mask = ~df['is_noise'] & df[x_col].notna() & df[y_col].notna()
    centers = df.loc[valid_mask, [cluster_col, x_col, y_col]].drop_duplicates(
        subset=[cluster_col]
    )
    centers = centers.rename(columns={x_col: 'center_x', y_col: 'center_y'})
    centers = centers.reset_index(drop=True)
    
    if len(centers) < 2:
        if verbose:
            print("Insufficient cluster centers for grid indexing.")
        return None, None
    
    if verbose:
        print(f"Original cluster centers: {len(centers)}")
    
    # Merge close centers
    coords = centers[['center_x', 'center_y']].values
    if len(coords) > 1 and merge_threshold > 0:
        linkage_matrix = linkage(coords, method='average')
        cluster_ids = fcluster(linkage_matrix, t=merge_threshold, criterion='distance')
        
        centers['merge_id'] = cluster_ids
        merged_centers = centers.groupby('merge_id').agg({
            cluster_col: 'first',
            'center_x': 'mean',
            'center_y': 'mean'
        }).reset_index(drop=True)
        
        if len(merged_centers) < len(centers) and verbose:
            print(f"Merged close centers: {len(centers)} -> {len(merged_centers)}")
        
        centers = merged_centers
        coords = centers[['center_x', 'center_y']].values
    
    if verbose:
        print(f"Centers for indexing: {len(centers)}")
    
    # PCA alignment
    rotation_angle = 0.0
    if use_pca_alignment and len(coords) >= 3:
        pca = PCA(n_components=2)
        coords_aligned = pca.fit_transform(coords)
        rotation_angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
        if verbose:
            print(f"PCA rotation angle: {np.degrees(rotation_angle):.2f}°")
    else:
        coords_aligned = coords.copy()
    
    centers['aligned_x'] = coords_aligned[:, 0]
    centers['aligned_y'] = coords_aligned[:, 1]
    
    # Estimate grid spacing using nearest neighbors
    if len(coords_aligned) >= 2:
        nn = NearestNeighbors(n_neighbors=min(5, len(coords_aligned)))
        nn.fit(coords_aligned)
        distances, _ = nn.kneighbors(coords_aligned)
        
        # Use median of nearest neighbor distances as grid spacing
        nn_dists = distances[:, 1] if distances.shape[1] > 1 else distances[:, 0]
        grid_spacing = np.median(nn_dists)
        
        if verbose:
            print(f"Estimated grid spacing: {grid_spacing:.3f} cm")
    else:
        grid_spacing = 1.0
        if verbose:
            print("Insufficient points for spacing estimation, using default 1.0 cm")
    
    # Find origin (closest to centroid)
    centroid = coords_aligned.mean(axis=0)
    dists_to_centroid = np.linalg.norm(coords_aligned - centroid, axis=1)
    origin_idx = np.argmin(dists_to_centroid)
    origin_x, origin_y = coords_aligned[origin_idx]
    origin_cluster = centers.iloc[origin_idx][cluster_col]
    
    if verbose:
        print(f"Origin: cluster {origin_cluster} at ({origin_x:.2f}, {origin_y:.2f})")
    
    # Assign row/column indices
    centers['row'] = np.round((centers['aligned_y'] - origin_y) / grid_spacing).astype(int)
    centers['col'] = np.round((centers['aligned_x'] - origin_x) / grid_spacing).astype(int)
    
    # Calculate grid extent
    row_range = (centers['row'].min(), centers['row'].max())
    col_range = (centers['col'].min(), centers['col'].max())
    
    # Identify expected positions based on range
    expected_positions = set()
    for r in range(row_range[0], row_range[1] + 1):
        for c in range(col_range[0], col_range[1] + 1):
            expected_positions.add((r, c))
    
    # Identify detected positions
    detected_positions = set(
        zip(centers['row'].values, centers['col'].values)
    )
    
    # Find missing positions
    missing_positions = expected_positions - detected_positions
    
    if verbose:
        print(f"Grid range: rows [{row_range[0]}, {row_range[1]}], "
              f"cols [{col_range[0]}, {col_range[1]}]")
        print(f"Expected positions: {len(expected_positions)}")
        print(f"Detected positions: {len(detected_positions)}")
        print(f"Missing positions: {len(missing_positions)}")
    
    # Warn about high fraction of missing sieve holes
    if len(expected_positions) > 0:
        missing_ratio = len(missing_positions) / len(expected_positions)
        if missing_ratio > _MAX_MISSING_RATIO:
            warnings.warn(
                f"Unusual sieve pattern: {missing_ratio:.1%} of expected grid "
                f"positions ({len(missing_positions)}/{len(expected_positions)}) "
                "are missing. The pattern may be nonrectangular or poorly arranged.",
                UserWarning, stacklevel=2
            )
    
    # Warn about irregular row/column counts (non-rectangular pattern)
    row_col_counts = centers.groupby('row')['col'].count()
    if len(row_col_counts) > 1 and row_col_counts.mean() > 0:
        cv = row_col_counts.std() / row_col_counts.mean()
        if cv > _MAX_ROW_COUNT_CV:
            warnings.warn(
                f"Unusual sieve pattern: rows have inconsistent hole counts "
                f"(coefficient of variation: {cv:.2f}). "
                "The sieve pattern may be poorly arranged or nonrectangular.",
                UserWarning, stacklevel=2
            )
    
    # Warn about inconsistent grid spacing
    if len(coords_aligned) >= 4:
        nn_check = NearestNeighbors(
            n_neighbors=min(_SPACING_CHECK_NEIGHBORS, len(coords_aligned))
        )
        nn_check.fit(coords_aligned)
        dists_check, _ = nn_check.kneighbors(coords_aligned)
        nn_dists_all = dists_check[:, 1:].flatten()
        nn_dists_all = nn_dists_all[nn_dists_all > 0]
        spacing_mean = np.mean(nn_dists_all) if len(nn_dists_all) > 0 else 0
        if spacing_mean > 0:
            spacing_cv = np.std(nn_dists_all) / spacing_mean
            if spacing_cv > _MAX_SPACING_CV:
                warnings.warn(
                    f"Unusual sieve pattern: grid spacing is highly inconsistent "
                    f"(coefficient of variation: {spacing_cv:.2f}). "
                    "The sieve pattern may be poorly arranged.",
                    UserWarning, stacklevel=2
                )
    
    grid_params = {
        'grid_spacing': grid_spacing,
        'rotation_angle': rotation_angle,
        'origin_x': origin_x,
        'origin_y': origin_y,
        'origin_cluster': origin_cluster,
        'row_range': row_range,
        'col_range': col_range,
        'missing_positions': list(missing_positions)
    }
    
    # Select output columns
    output_cols = [cluster_col, 'center_x', 'center_y', 'row', 'col']
    
    return centers[output_cols], grid_params


def get_grid_occupancy_table(
    centers: pd.DataFrame,
    cluster_col: str = 'cluster'
) -> pd.DataFrame:
    """
    Create a pivot table showing grid occupancy (cluster IDs).
    
    Parameters
    ----------
    centers : pd.DataFrame
        DataFrame with row, col, and cluster columns from build_grid_index.
    cluster_col : str, optional
        Column name for cluster labels. Default is 'cluster'.
    
    Returns
    -------
    pd.DataFrame
        Pivot table with rows as row index, columns as col index,
        and values as cluster IDs. NaN indicates empty positions.

    Notes
    -----
    The returned table is sorted by descending row index so that higher
    rows appear at the top (matching the physical geometry of the sieve
    plate where row 0 is the bottom row). Column indices increase from
    left to right.  Use ``pd.isnull(occupancy)`` to get a boolean mask
    of missing positions.
    
    Examples
    --------
    >>> centers, params = build_grid_index_from_centers(df)
    >>> occupancy = get_grid_occupancy_table(centers)
    >>> print(occupancy)
    >>> # Identify empty cells in the grid
    >>> empty_mask = pd.isnull(occupancy)
    >>> print(f"Empty grid positions: {empty_mask.values.sum()}")
    
    See Also
    --------
    build_grid_index_from_centers : Generate the centers DataFrame
    get_missing_holes : Identify missing holes as (row, col) tuples
    """
    grid_table = centers.pivot_table(
        values=cluster_col,
        index='row',
        columns='col',
        aggfunc='first'
    ).sort_index(ascending=False)
    
    return grid_table


def get_missing_holes(
    centers: pd.DataFrame,
    grid_params: Dict[str, Any],
    only_internal: bool = True,
    verbose: bool = True
) -> List[Tuple[int, int]]:
    """
    Identify missing sieve holes in the grid.
    
    Parameters
    ----------
    centers : pd.DataFrame
        DataFrame with row, col columns from build_grid_index.
    grid_params : dict
        Grid parameters from build_grid_index.
    only_internal : bool, optional
        If True, only return holes inside the convex hull of detected
        holes (true missing holes, not edge effects). Default is True.
    verbose : bool, optional
        If True, prints information. Default is True.
    
    Returns
    -------
    list of tuple
        List of (row, col) tuples for missing holes.
    
    Examples
    --------
    >>> centers, params = build_grid_index_from_centers(df)
    >>> missing = get_missing_holes(centers, params)
    >>> print(f"Missing internal holes: {missing}")
    """
    missing = grid_params.get('missing_positions', [])
    
    if not only_internal or len(missing) == 0:
        return missing
    
    if Delaunay is None:
        raise ImportError(
            "scipy is required for internal hole detection. "
            "Install it with: pip install scipy"
        )
    
    # Get detected points in (row, col) space
    detected_points = centers[['row', 'col']].values
    
    if len(detected_points) < 3:
        return missing
    
    try:
        hull = Delaunay(detected_points)
        internal_missing = []
        
        for (r, c) in missing:
            if hull.find_simplex([r, c]) >= 0:
                internal_missing.append((r, c))
        
        if verbose:
            print(f"Total missing positions: {len(missing)}")
            print(f"Internal missing (inside convex hull): {len(internal_missing)}")
        
        return internal_missing
    
    except Exception as e:
        if verbose:
            print(f"Could not compute internal missing holes: {e}")
        return missing


def estimate_hole_positions(
    centers: pd.DataFrame,
    grid_params: Dict[str, Any],
    missing_positions: Optional[List[Tuple[int, int]]] = None
) -> pd.DataFrame:
    """
    Estimate positions of missing holes based on grid parameters.
    
    Parameters
    ----------
    centers : pd.DataFrame
        DataFrame with row, col, center_x, center_y columns.
    grid_params : dict
        Grid parameters from build_grid_index.
    missing_positions : list of tuple, optional
        List of (row, col) for missing positions. If None, uses
        grid_params['missing_positions'].
    
    Returns
    -------
    pd.DataFrame
        DataFrame with estimated positions for missing holes.
        Columns: row, col, estimated_x, estimated_y

    Notes
    -----
    The estimated positions are derived from the regular grid model fitted
    by ``build_grid_index_from_centers``.  They reflect where a sieve hole
    *should* be in target-plane coordinates given the observed grid spacing
    and origin.  If PCA alignment was used during grid construction, the
    inverse rotation is applied automatically so that the returned
    ``estimated_x`` / ``estimated_y`` values are in the original (unaligned)
    coordinate system.

    These estimates are useful for flagging detector regions where the
    clustering algorithm failed to detect a physically present hole, or for
    seeding manual inspection of the raw data around the predicted location.
    
    Examples
    --------
    >>> centers, params = build_grid_index_from_centers(df)
    >>> missing = get_missing_holes(centers, params)
    >>> estimated = estimate_hole_positions(centers, params, missing)
    >>> print(estimated)
    >>> # Overlay estimated positions on the target-plane scatter plot
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.scatter(centers['center_x'], centers['center_y'],
    ...            marker='o', s=40, label='Detected holes')
    >>> ax.scatter(estimated['estimated_x'], estimated['estimated_y'],
    ...            marker='x', s=80, color='red', label='Estimated missing')
    >>> ax.legend()
    >>> plt.show()
    
    See Also
    --------
    get_missing_holes : Identify the (row, col) positions that are missing
    build_grid_index_from_centers : Fit the grid model used for estimation
    """
    if missing_positions is None:
        missing_positions = grid_params.get('missing_positions', [])
    
    if len(missing_positions) == 0:
        return pd.DataFrame(columns=['row', 'col', 'estimated_x', 'estimated_y'])
    
    grid_spacing = grid_params['grid_spacing']
    origin_x = grid_params['origin_x']
    origin_y = grid_params['origin_y']
    rotation_angle = grid_params.get('rotation_angle', 0.0)
    
    estimated = []
    
    for r, c in missing_positions:
        # Calculate position in aligned coordinates
        aligned_x = origin_x + c * grid_spacing
        aligned_y = origin_y + r * grid_spacing
        
        # Rotate back if PCA was used
        if rotation_angle != 0:
            cos_a = np.cos(-rotation_angle)
            sin_a = np.sin(-rotation_angle)
            est_x = aligned_x * cos_a - aligned_y * sin_a
            est_y = aligned_x * sin_a + aligned_y * cos_a
        else:
            est_x = aligned_x
            est_y = aligned_y
        
        estimated.append({
            'row': r,
            'col': c,
            'estimated_x': est_x,
            'estimated_y': est_y
        })
    
    return pd.DataFrame(estimated)


def build_full_grid_index(
    clustering_results: Dict[int, Dict[str, Any]],
    x_col: str = 'cluster_center_x',
    y_col: str = 'cluster_center_y',
    cluster_col: str = 'cluster',
    config: Optional[GridIndexConfig] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]]]:
    """
    Build grid indices for all foil positions.
    
    This is a convenience function that applies build_grid_index_from_centers
    to clustering results from all foil positions.
    
    Parameters
    ----------
    clustering_results : dict
        Dictionary from cluster_by_foil_position with structure:
        {foil_position: {'df': DataFrame, 'params': dict, 'n_clusters': int}}
    x_col : str, optional
        Column name for cluster center x. Default is 'cluster_center_x'.
    y_col : str, optional
        Column name for cluster center y. Default is 'cluster_center_y'.
    cluster_col : str, optional
        Column name for cluster labels. Default is 'cluster'.
    config : GridIndexConfig, optional
        Configuration object.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    tuple
        (full_index_df, grid_params_dict) where:
        - full_index_df: Combined DataFrame with foil_position, cluster,
          center_x, center_y, row, col columns
        - grid_params_dict: Dictionary mapping foil_position to grid_params
    
    Examples
    --------
    >>> results = cluster_by_foil_position(df)
    >>> full_index, params_dict = build_full_grid_index(results)
    >>> print(full_index.head())
    >>> for foil, params in params_dict.items():
    ...     print(f"Foil {foil}: spacing={params['grid_spacing']:.3f} cm")
    
    See Also
    --------
    cluster_by_foil_position : Generate clustering results
    build_grid_index_from_centers : Single foil grid indexing
    """
    all_centers = []
    grid_params_dict = {}
    
    for foil_pos, result in clustering_results.items():
        df_clustered = result['df']
        
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Building grid index for foil_position = {foil_pos}")
            print("=" * 50)
        
        centers, grid_params = build_grid_index_from_centers(
            df_clustered,
            x_col=x_col,
            y_col=y_col,
            cluster_col=cluster_col,
            config=config,
            verbose=verbose
        )
        
        if centers is not None:
            centers['foil_position'] = foil_pos
            all_centers.append(centers)
            grid_params_dict[foil_pos] = grid_params
    
    if len(all_centers) == 0:
        return pd.DataFrame(), {}
    
    full_index = pd.concat(all_centers, ignore_index=True)
    
    # Reorder columns
    cols = ['foil_position', 'cluster', 'row', 'col', 'center_x', 'center_y']
    full_index = full_index[[c for c in cols if c in full_index.columns]]
    
    if verbose:
        print(f"\n{'=' * 50}")
        print("Grid indexing complete!")
        print(f"Total indexed clusters: {len(full_index)}")
        print("=" * 50)
    
    return full_index, grid_params_dict


def get_row_statistics(
    centers: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate statistics for each row in the grid.
    
    Parameters
    ----------
    centers : pd.DataFrame
        DataFrame with row, col columns from build_grid_index.
    verbose : bool, optional
        If True, prints information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with row statistics sorted by descending row index.
        Columns:

        - **row** : Row index in the grid.
        - **count** : Number of detected holes in this row.
        - **min_col** : Smallest column index in this row.
        - **max_col** : Largest column index in this row.
        - **span** : ``max_col - min_col + 1`` (total expected holes in row).
    
    Examples
    --------
    >>> centers, params = build_grid_index_from_centers(df)
    >>> row_stats = get_row_statistics(centers)
    >>> # Identify rows with fewer detected holes than expected
    >>> incomplete = row_stats[row_stats['count'] < row_stats['span']]
    >>> print(f"Rows with missing holes: {len(incomplete)}")
    
    See Also
    --------
    build_grid_index_from_centers : Generate the centers DataFrame
    get_missing_holes : Identify individual missing (row, col) positions
    """
    row_stats = centers.groupby('row').agg({
        'col': ['count', 'min', 'max']
    }).reset_index()
    
    row_stats.columns = ['row', 'count', 'min_col', 'max_col']
    row_stats['span'] = row_stats['max_col'] - row_stats['min_col'] + 1
    row_stats = row_stats.sort_values('row', ascending=False).reset_index(drop=True)
    
    if verbose:
        print("Row Statistics:")
        print(row_stats.to_string(index=False))
    
    return row_stats
