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
    MechanicalGridConfig,
    DEFAULT_GRID_INDEX_CONFIG,
    DEFAULT_MECHANICAL_GRID_CONFIG,
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


# ================================================================
# Mechanical grid matching (from build_stage2_labels)
# ================================================================

def infer_lattice_origin_cm(
    values_cm: np.ndarray,
    spacing_cm: float,
    initial_origin_cm: float = 0.0,
) -> float:
    """Infer the origin of a regular 1-D lattice from observed values.

    The origin is refined by computing the median residual between
    observed values and their nearest integer grid point:

        residual = values - (initial_origin + round((values - initial_origin) / spacing) * spacing)
        origin = initial_origin + median(residual)

    Parameters
    ----------
    values_cm : np.ndarray
        1-D array of observed positions in cm.
    spacing_cm : float
        Known grid spacing in cm.
    initial_origin_cm : float
        Initial guess for the origin in cm.

    Returns
    -------
    float
        Refined origin in cm.

    Examples
    --------
    >>> observed = np.array([2.03, 4.52, 7.01, 9.49])
    >>> origin = infer_lattice_origin_cm(observed, spacing_cm=2.5, initial_origin_cm=0.0)
    >>> print(f"{origin:.3f}")  # ~0.02
    """
    values = np.asarray(values_cm, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float(initial_origin_cm)
    nearest_index = np.rint((values - float(initial_origin_cm)) / float(spacing_cm))
    residual = values - (float(initial_origin_cm) + nearest_index * float(spacing_cm))
    return float(initial_origin_cm + float(np.median(residual)))


def _estimate_grid_spacing_from_centers(
    coords: np.ndarray,
) -> Tuple[float, float]:
    """Estimate x and y grid spacing from cluster centre coordinates.

    Uses the median of the k-nearest-neighbour distances (k=3) projected
    onto the x and y axes to estimate the characteristic spacing in each
    direction independently.

    Parameters
    ----------
    coords : np.ndarray
        (N, 2) array of (x, y) cluster centre positions in cm.

    Returns
    -------
    tuple[float, float]
        (x_spacing_cm, y_spacing_cm)

    Raises
    ------
    ImportError
        If sklearn is not installed.
    ValueError
        If fewer than 4 centres are available.
    """
    if NearestNeighbors is None:
        raise ImportError(
            "sklearn is required for spacing estimation. "
            "Install it with: pip install scikit-learn"
        )
    if len(coords) < 4:
        raise ValueError(
            f"Need at least 4 cluster centres to estimate spacing; got {len(coords)}."
        )

    k = min(4, len(coords))
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Project neighbour offsets onto x and y axes
    dx_all: list[float] = []
    dy_all: list[float] = []
    for i in range(len(coords)):
        for j_idx in range(1, k):
            j = int(indices[i, j_idx])
            dx = abs(float(coords[i, 0] - coords[j, 0]))
            dy = abs(float(coords[i, 1] - coords[j, 1]))
            if dx > 1e-6:
                dx_all.append(dx)
            if dy > 1e-6:
                dy_all.append(dy)

    if not dx_all or not dy_all:
        raise ValueError("Could not estimate spacing: insufficient distinct neighbours.")

    x_spacing_cm = float(np.median(dx_all))
    y_spacing_cm = float(np.median(dy_all))
    return x_spacing_cm, y_spacing_cm


def build_candidate_mechanical_grid(
    cluster_centers: pd.DataFrame,
    config: Optional[MechanicalGridConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build a candidate mechanical hole grid from cluster centres.

    This generates candidate sieve-hole positions based on either
    explicitly provided or auto-detected grid spacing.  The resulting
    table can be used with ``match_clusters_to_mechanical_grid``.

    Parameters
    ----------
    cluster_centers : pd.DataFrame
        DataFrame with columns ``[foil_position, cluster_center_x,
        cluster_center_y]`` (or the column names specified in *config*).
    config : MechanicalGridConfig, optional
        Configuration.  If None, uses ``DEFAULT_MECHANICAL_GRID_CONFIG``.
    verbose : bool
        If True, prints spacing and origin information.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (candidate_design, meta) where:
        - **candidate_design** has columns:
          ``foil_position, hole_row, hole_col,
          candidate_sieve_x_cm, candidate_sieve_y_cm,
          weak_hole_xptar_center, weak_hole_yptar_center,
          weak_hole_xptar_tol, weak_hole_yptar_tol``
        - **meta** contains spacing, origin, and per-foil range info.

    Examples
    --------
    >>> centers = extract_cluster_centers(clustering_results)
    >>> design, meta = build_candidate_mechanical_grid(centers)
    >>> print(f"Candidates per foil: {meta['per_foil']}")
    """
    if config is None:
        config = DEFAULT_MECHANICAL_GRID_CONFIG

    x_col = config.x_col
    y_col = config.y_col

    if cluster_centers.empty:
        raise RuntimeError("No cluster centres available; cannot build mechanical grid candidates.")

    required_cols = ["foil_position", x_col, y_col]
    missing = [c for c in required_cols if c not in cluster_centers.columns]
    if missing:
        raise ValueError(f"cluster_centers missing required columns: {missing}")

    # ---- resolve spacing ----
    coords_all = cluster_centers[[x_col, y_col]].to_numpy(dtype=np.float64)
    if config.x_spacing_mm is not None and config.y_spacing_mm is not None:
        x_spacing_cm = float(config.x_spacing_mm) / 10.0
        y_spacing_cm = float(config.y_spacing_mm) / 10.0
        spacing_source = "explicit"
    elif config.auto_spacing:
        x_spacing_cm, y_spacing_cm = _estimate_grid_spacing_from_centers(coords_all)
        spacing_source = "auto"
    else:
        raise ValueError(
            "Grid spacing not provided and auto_spacing is disabled. "
            "Set x_spacing_mm / y_spacing_mm or enable auto_spacing."
        )

    tolerance_rad = (float(config.tolerance_mm) / 10.0) / float(config.sieve_distance_cm)
    init_x_origin_cm = float(config.hole_origin_xptar) * float(config.sieve_distance_cm)
    init_y_origin_cm = float(config.hole_origin_yptar) * float(config.sieve_distance_cm)

    x_origin_cm = infer_lattice_origin_cm(
        cluster_centers[x_col].to_numpy(dtype=np.float64),
        x_spacing_cm, init_x_origin_cm,
    )
    y_origin_cm = infer_lattice_origin_cm(
        cluster_centers[y_col].to_numpy(dtype=np.float64),
        y_spacing_cm, init_y_origin_cm,
    )

    if verbose:
        print(f"  Grid spacing: x={x_spacing_cm:.3f} cm ({x_spacing_cm*10:.1f} mm), "
              f"y={y_spacing_cm:.3f} cm ({y_spacing_cm*10:.1f} mm) [{spacing_source}]")
        print(f"  Grid origin:  x={x_origin_cm:.4f} cm, y={y_origin_cm:.4f} cm")
        print(f"  Hole tolerance: {config.tolerance_mm:.1f} mm → {tolerance_rad:.6f} rad")

    # ---- build per-foil candidate grid ----
    cols_est = cluster_centers.copy()
    cols_est["hole_col_est"] = np.rint(
        (cols_est[x_col] - x_origin_cm) / x_spacing_cm
    ).astype(int)
    cols_est["hole_row_est"] = np.rint(
        (cols_est[y_col] - y_origin_cm) / y_spacing_cm
    ).astype(int)

    design_frames: list[pd.DataFrame] = []
    per_foil_summary: dict[str, Any] = {}
    for foil_pos, df_foil in cols_est.groupby("foil_position"):
        foil_pos_int = int(foil_pos)
        min_col = int(df_foil["hole_col_est"].min())
        max_col = int(df_foil["hole_col_est"].max())
        min_row = int(df_foil["hole_row_est"].min())
        max_row = int(df_foil["hole_row_est"].max())

        candidate = pd.MultiIndex.from_product(
            [[foil_pos_int], range(min_row, max_row + 1), range(min_col, max_col + 1)],
            names=["foil_position", "hole_row", "hole_col"],
        ).to_frame(index=False)
        candidate["candidate_sieve_x_cm"] = (
            x_origin_cm + candidate["hole_col"].to_numpy(dtype=np.float64) * x_spacing_cm
        )
        candidate["candidate_sieve_y_cm"] = (
            y_origin_cm + candidate["hole_row"].to_numpy(dtype=np.float64) * y_spacing_cm
        )
        candidate["weak_hole_xptar_center"] = (
            candidate["candidate_sieve_x_cm"] / float(config.sieve_distance_cm)
        )
        candidate["weak_hole_yptar_center"] = (
            candidate["candidate_sieve_y_cm"] / float(config.sieve_distance_cm)
        )
        candidate["weak_hole_xptar_tol"] = float(tolerance_rad)
        candidate["weak_hole_yptar_tol"] = float(tolerance_rad)
        design_frames.append(candidate)
        per_foil_summary[str(foil_pos_int)] = {
            "row_range": [min_row, max_row],
            "col_range": [min_col, max_col],
            "candidate_count": int(len(candidate)),
        }

    design = pd.concat(design_frames, ignore_index=True)
    meta = {
        "x_spacing_cm": float(x_spacing_cm),
        "y_spacing_cm": float(y_spacing_cm),
        "x_origin_cm": float(x_origin_cm),
        "y_origin_cm": float(y_origin_cm),
        "tolerance_mm": float(config.tolerance_mm),
        "tolerance_rad": float(tolerance_rad),
        "sieve_distance_cm": float(config.sieve_distance_cm),
        "spacing_source": spacing_source,
        "per_foil": per_foil_summary,
    }
    return design.sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True), meta


def match_clusters_to_mechanical_grid(
    cluster_centers: pd.DataFrame,
    candidate_design: pd.DataFrame,
    config: Optional[MechanicalGridConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Match cluster centres to a known mechanical hole grid.

    Supports two matching strategies:

    * **nearest** — independent nearest-neighbour match for each cluster.
    * **center_out_penalized** — clusters are sorted by distance from the
      grid centre (inner first); holes already assigned to a previous
      cluster receive an occupancy penalty, discouraging duplicate
      assignments.

    Parameters
    ----------
    cluster_centers : pd.DataFrame
        Columns: ``foil_position, cluster, cluster_center_x,
        cluster_center_y``.
    candidate_design : pd.DataFrame
        Output of ``build_candidate_mechanical_grid``.  Must contain
        ``candidate_sieve_x_cm``, ``candidate_sieve_y_cm``,
        ``hole_row``, ``hole_col``, and ``foil_position``.
    config : MechanicalGridConfig, optional
    verbose : bool

    Returns
    -------
    tuple[pd.DataFrame, dict]
        (assignment_df, summary) where assignment_df has columns:
        ``foil_position, cluster, cluster_center_x, cluster_center_y,
        hole_row, hole_col, matched_sieve_x_cm, matched_sieve_y_cm,
        match_dx_cm, match_dy_cm, match_distance_cm,
        nearest_match_distance_cm, effective_match_cost_cm,
        hole_occupancy_before_assignment, assignment_mode``.
    """
    if config is None:
        config = DEFAULT_MECHANICAL_GRID_CONFIG

    required_centers_cols = ["foil_position", "cluster", config.x_col, config.y_col]
    missing_center = [c for c in required_centers_cols if c not in cluster_centers.columns]
    if missing_center:
        raise ValueError(f"cluster_centers missing required columns: {missing_center}")

    required_design_cols = [
        "foil_position", "hole_row", "hole_col",
        "candidate_sieve_x_cm", "candidate_sieve_y_cm",
    ]
    missing_design = [c for c in required_design_cols if c not in candidate_design.columns]
    if missing_design:
        raise ValueError(f"candidate_design missing required columns: {missing_design}")

    assignment_mode = config.assignment_mode
    occupancy_penalty_cm = float(max(config.occupancy_penalty_cm, 0.0))

    assignments: list[dict[str, Any]] = []
    per_foil_stats: dict[str, Any] = {}

    for foil_pos, df_foil in cluster_centers.groupby("foil_position"):
        foil_pos_int = int(foil_pos)
        candidates = candidate_design.loc[
            candidate_design["foil_position"] == foil_pos_int
        ].copy()
        if candidates.empty:
            continue

        candidate_xy = candidates[["candidate_sieve_x_cm", "candidate_sieve_y_cm"]].to_numpy(
            dtype=np.float64
        )
        candidate_rowcol = candidates[["hole_row", "hole_col"]].to_numpy(dtype=np.int64)

        # Grid centre (hole closest to (0,0) in sieve coordinates)
        center_idx = int(np.argmin(np.sqrt(candidate_xy[:, 0] ** 2 + candidate_xy[:, 1] ** 2)))
        center_ref_x = float(candidate_xy[center_idx, 0])
        center_ref_y = float(candidate_xy[center_idx, 1])

        cluster_frame = df_foil.copy()
        cluster_x = cluster_frame[config.x_col].to_numpy(dtype=np.float64)
        cluster_y = cluster_frame[config.y_col].to_numpy(dtype=np.float64)
        cluster_radius = np.sqrt(
            (cluster_x - center_ref_x) ** 2 + (cluster_y - center_ref_y) ** 2
        )
        cluster_frame["__center_out_radius"] = cluster_radius

        if assignment_mode == "center_out_penalized":
            cluster_frame = cluster_frame.sort_values(
                ["__center_out_radius", config.x_col, config.y_col, "cluster"],
                ascending=[True, True, True, True],
            ).reset_index(drop=True)

        distances: list[float] = []
        effective_costs: list[float] = []
        occupancies_before: list[int] = []
        reassigned_from_nearest = 0
        hole_usage_counts: dict[tuple[int, int], int] = {}

        for row in cluster_frame.itertuples(index=False):
            cluster_id = int(getattr(row, "cluster"))
            cx = float(getattr(row, config.x_col))
            cy = float(getattr(row, config.y_col))
            delta = candidate_xy - np.array([cx, cy], dtype=np.float64)
            dist = np.sqrt(np.sum(delta * delta, axis=1))
            nearest_idx = int(np.argmin(dist))

            if assignment_mode == "center_out_penalized":
                occupancy_counts = np.array(
                    [
                        hole_usage_counts.get((int(rc[0]), int(rc[1])), 0)
                        for rc in candidate_rowcol
                    ],
                    dtype=np.float64,
                )
                effective_cost = dist + occupancy_penalty_cm * occupancy_counts
                best_idx = int(np.argmin(effective_cost))
                occupancies_before.append(int(occupancy_counts[best_idx]))
                effective_costs.append(float(effective_cost[best_idx]))
            else:
                best_idx = nearest_idx
                occupancies_before.append(
                    int(hole_usage_counts.get(
                        (int(candidate_rowcol[best_idx, 0]), int(candidate_rowcol[best_idx, 1])), 0
                    ))
                )
                effective_costs.append(float(dist[best_idx]))

            if best_idx != nearest_idx:
                reassigned_from_nearest += 1

            distances.append(float(dist[best_idx]))
            chosen_key = (int(candidate_rowcol[best_idx, 0]), int(candidate_rowcol[best_idx, 1]))
            hole_usage_counts[chosen_key] = hole_usage_counts.get(chosen_key, 0) + 1

            assignments.append({
                "foil_position": foil_pos_int,
                "cluster": cluster_id,
                "cluster_center_x": cx,
                "cluster_center_y": cy,
                "hole_row": int(candidate_rowcol[best_idx, 0]),
                "hole_col": int(candidate_rowcol[best_idx, 1]),
                "matched_sieve_x_cm": float(candidate_xy[best_idx, 0]),
                "matched_sieve_y_cm": float(candidate_xy[best_idx, 1]),
                "match_dx_cm": float(cx - candidate_xy[best_idx, 0]),
                "match_dy_cm": float(cy - candidate_xy[best_idx, 1]),
                "match_distance_cm": float(dist[best_idx]),
                "nearest_match_distance_cm": float(dist[nearest_idx]),
                "effective_match_cost_cm": float(effective_costs[-1]),
                "hole_occupancy_before_assignment": int(occupancies_before[-1]),
                "assignment_mode": assignment_mode,
            })

        per_foil_stats[str(foil_pos_int)] = {
            "n_clusters": int(len(df_foil)),
            "median_match_distance_cm": float(np.median(distances)) if distances else None,
            "max_match_distance_cm": float(np.max(distances)) if distances else None,
            "min_match_distance_cm": float(np.min(distances)) if distances else None,
            "median_effective_match_cost_cm": (
                float(np.median(effective_costs)) if effective_costs else None
            ),
            "reassigned_from_nearest_count": int(reassigned_from_nearest),
            "max_hole_occupancy": int(max(hole_usage_counts.values())) if hole_usage_counts else 0,
        }

    assignment_df = pd.DataFrame(assignments).sort_values(
        ["foil_position", "cluster"]
    ).reset_index(drop=True)

    duplicate_assignments = (
        assignment_df.groupby(["foil_position", "hole_row", "hole_col"])
        .size()
        .reset_index(name="assigned_clusters")
    )
    summary = {
        "assignment_mode": assignment_mode,
        "occupancy_penalty_cm": float(occupancy_penalty_cm),
        "per_foil": per_foil_stats,
        "duplicate_mechanical_holes": int(
            (duplicate_assignments["assigned_clusters"] > 1).sum()
        ),
    }

    if verbose:
        print(f"  Assignment mode: {assignment_mode}")
        print(f"  Duplicate mechanical holes: {summary['duplicate_mechanical_holes']}")
        for foil_key, stats in sorted(per_foil_stats.items()):
            print(
                f"  Foil {foil_key}: median match={stats['median_match_distance_cm']:.4f} cm, "
                f"max occ={stats['max_hole_occupancy']}, "
                f"reassigned={stats['reassigned_from_nearest_count']}"
            )

    return assignment_df, summary


def mm_to_target_angle(mm_value: float, sieve_distance_cm: float) -> float:
    """Convert a millimetre offset at the sieve plane to a target angle in rad.

    Parameters
    ----------
    mm_value : float
        Displacement in mm at the sieve plane.
    sieve_distance_cm : float
        Target-to-sieve distance in cm.

    Returns
    -------
    float
        Equivalent angle in rad.
    """
    if float(sieve_distance_cm) <= 0:
        raise ValueError(f"sieve_distance_cm must be positive, got {sieve_distance_cm}")
    return float((float(mm_value) / 10.0) / float(sieve_distance_cm))


def build_mechanical_grid_index(
    clustering_results: Dict[int, Dict[str, Any]],
    config: Optional[MechanicalGridConfig] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """Full pipeline: cluster centres → mechanical hole grid match.

    Combines data-driven spacing detection (or explicit spacing) with
    configurable matching strategies to assign HDBSCAN/DBSCAN clusters
    to known sieve-hole positions on a mechanical design grid.

    This is the recommended entry point for mechanical grid matching.
    It wraps ``build_candidate_mechanical_grid`` and
    ``match_clusters_to_mechanical_grid``.

    Parameters
    ----------
    clustering_results : dict
        Dictionary from ``cluster_by_foil_position`` (or equivalent),
        with structure ``{foil_position: {'df': DataFrame, ...}}``.
        Each per-foil DataFrame must contain ``cluster``,
        ``cluster_center_x``, ``cluster_center_y``, and ``is_noise``
        columns.
    config : MechanicalGridConfig, optional
    verbose : bool

    Returns
    -------
    tuple
        (hole_design, design_meta, cluster_hole_map, match_summary)
        - **hole_design** — candidate mechanical hole design table
          (columns: foil_position, hole_row, hole_col,
          candidate_sieve_x_cm, candidate_sieve_y_cm,
          weak_hole_xptar_center, weak_hole_yptar_center,
          weak_hole_xptar_tol, weak_hole_yptar_tol)
        - **design_meta** — grid spacing, origin, per-foil info
        - **cluster_hole_map** — one row per matched cluster with
          match quality columns
        - **match_summary** — per-foil match statistics and duplicate
          hole count

    Examples
    --------
    >>> from shms_optics_calibration import (
    ...     cluster_by_foil_position,
    ...     build_mechanical_grid_index,
    ...     MechanicalGridConfig,
    ... )
    >>> results = cluster_by_foil_position(df, method='hdbscan')
    >>> hole_design, meta, hole_map, summary = build_mechanical_grid_index(results)
    >>> print(f"Duplicate holes: {summary['duplicate_mechanical_holes']}")
    >>> print(hole_map[['foil_position', 'cluster', 'hole_row', 'hole_col']].head())

    See Also
    --------
    build_grid_index_from_centers : Data-driven grid indexing (no mechanical design).
    build_candidate_mechanical_grid : Generate candidate grid from spacing.
    match_clusters_to_mechanical_grid : Perform the cluster→hole matching.
    """
    if config is None:
        config = DEFAULT_MECHANICAL_GRID_CONFIG

    # ---- extract cluster centres ----
    frames: list[pd.DataFrame] = []
    for foil_pos, result in sorted(clustering_results.items()):
        df_foil = result["df"].copy()
        df_foil = df_foil.loc[df_foil.get("cluster", -1) != -1].copy()
        if df_foil.empty:
            continue
        centers = (
            df_foil.groupby("cluster", as_index=False)
            .agg(
                cluster_center_x=("cluster_center_x", "median"),
                cluster_center_y=("cluster_center_y", "median"),
            )
        )
        centers["foil_position"] = int(foil_pos)
        frames.append(
            centers[["foil_position", "cluster", "cluster_center_x", "cluster_center_y"]]
        )

    if not frames:
        raise RuntimeError("No valid cluster centres found in clustering_results.")

    cluster_centers = pd.concat(frames, ignore_index=True)

    # ---- build candidate mechanical grid ----
    hole_design, design_meta = build_candidate_mechanical_grid(
        cluster_centers, config=config, verbose=verbose,
    )

    # ---- match clusters to grid ----
    cluster_hole_map, match_summary = match_clusters_to_mechanical_grid(
        cluster_centers, hole_design, config=config, verbose=verbose,
    )

    return hole_design, design_meta, cluster_hole_map, match_summary
