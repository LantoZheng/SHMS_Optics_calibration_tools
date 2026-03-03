"""
Data preprocessing module for SHMS Optics Calibration.

This module provides functions for preprocessing SHMS optics data,
including foil position classification based on P_gtr_y distribution.
"""

from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

try:
    from scipy.signal import find_peaks, peak_widths
except ImportError:
    find_peaks = None
    peak_widths = None

from .config import (
    FoilClassificationConfig,
    DEFAULT_FOIL_CLASSIFICATION_CONFIG,
)


def classify_foils_with_range(
    df: pd.DataFrame,
    col_name: str = 'P_gtr_y',
    bins: int = 50,
    sigma_factor: float = 2.5,
    y_range: Optional[Tuple[float, float]] = (-5.0, 5.0),
    peak_height_fraction: float = 0.05,
    peak_distance: int = 10,
    drop_unclassified: bool = True,
    plot: bool = False,
    config: Optional[FoilClassificationConfig] = None
) -> pd.DataFrame:
    """
    Classify foil positions based on peaks in the P_gtr_y distribution.
    
    This function analyzes the distribution of a specified column (typically
    P_gtr_y) to identify multiple foil positions. Events are classified into
    foil groups based on their proximity to detected peaks.
    
    The algorithm:
    1. Creates a histogram of the specified column
    2. Finds peaks in the histogram (foil positions)
    3. Calculates FWHM for each peak to determine classification ranges
    4. Assigns each event to a foil position based on its value
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the column to analyze.
    col_name : str, optional
        Column name to analyze for foil classification.
        Default is 'P_gtr_y'.
    bins : int, optional
        Number of histogram bins. Default is 50.
    sigma_factor : float, optional
        Classification range tolerance. Events within
        peak ± sigma_factor * sigma are assigned to that foil.
        Default is 2.5.
    y_range : tuple of float, optional
        (min, max) range for valid values. Events outside this range
        are not used for peak finding and are classified as -1.
        Default is (-5.0, 5.0).
    peak_height_fraction : float, optional
        Minimum peak height as fraction of maximum peak.
        Default is 0.05.
    peak_distance : int, optional
        Minimum distance between peaks in bins. Default is 10.
    drop_unclassified : bool, optional
        If True (default), events that are not assigned to any foil
        (foil_position == -1) are removed from the returned DataFrame.
        Set to False to retain them with foil_position == -1.
    plot : bool, optional
        If True, creates a visualization plot. Default is False.
    config : FoilClassificationConfig, optional
        Configuration object. If provided, overrides individual parameters.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'foil_position' column.
        Values are integers starting from 0, or -1 for unclassified events.
        When ``drop_unclassified=True`` (default), rows with
        foil_position == -1 are removed before returning.
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> df = classify_foils_with_range(df, y_range=(-5, 5))
    >>> print(df['foil_position'].value_counts())
    
    >>> # Using configuration object
    >>> config = FoilClassificationConfig(bins=100, sigma_factor=3.0)
    >>> df = classify_foils_with_range(df, config=config)
    """
    if find_peaks is None or peak_widths is None:
        raise ImportError(
            "scipy is required for foil classification. "
            "Install it with: pip install scipy"
        )
    
    # Use config if provided
    if config is not None:
        col_name = config.col_name
        bins = config.bins
        sigma_factor = config.sigma_factor
        y_range = config.y_range
        peak_height_fraction = config.peak_height_fraction
        peak_distance = config.peak_distance
        drop_unclassified = config.drop_unclassified
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Data preprocessing and range filtering
    if y_range is not None:
        y_min, y_max = y_range
        data_hist = df.loc[
            (df[col_name] >= y_min) & (df[col_name] <= y_max), 
            col_name
        ].dropna()
        print(f"Applied range limit: [{y_min}, {y_max}]")
        print(f"  - Original data count: {len(df)}")
        print(f"  - In-range data count: {len(data_hist)} "
              f"(removed {len(df) - len(data_hist)} outliers)")
    else:
        data_hist = df[col_name].dropna()
        y_min, y_max = data_hist.min(), data_hist.max()
    
    if len(data_hist) == 0:
        print("Warning: No data in range!")
        df['foil_position'] = -1
        if drop_unclassified:
            return df.iloc[0:0].copy()
        return df
    
    # Generate histogram
    counts, bin_edges = np.histogram(data_hist, bins=bins, range=(y_min, y_max))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find peaks
    peaks, _ = find_peaks(
        counts, 
        height=np.max(counts) * peak_height_fraction,
        distance=peak_distance
    )
    
    if len(peaks) == 0:
        print("No significant peaks found.")
        df['foil_position'] = -1
        if drop_unclassified:
            return df.iloc[0:0].copy()
        return df
    
    # Calculate FWHM for each peak
    widths_results = peak_widths(counts, peaks, rel_height=0.5)
    fwhms = widths_results[0] * (bin_edges[1] - bin_edges[0])
    peak_locs = bin_centers[peaks]
    
    print(f"Detected {len(peaks)} foil peak(s).")
    
    # Build classification conditions
    conditions = []
    choices = []
    ranges_info = []
    
    df['foil_position'] = -1
    
    for i, (center, fwhm) in enumerate(zip(peak_locs, fwhms)):
        sigma = fwhm / 2.355  # Convert FWHM to sigma
        half_width = sigma_factor * sigma
        
        lower_bound = center - half_width
        upper_bound = center + half_width
        
        ranges_info.append((lower_bound, upper_bound))
        
        conditions.append(
            (df[col_name] >= lower_bound) & (df[col_name] <= upper_bound)
        )
        choices.append(i)
        
        print(f"  Foil {i}: center={center:.3f}, range=[{lower_bound:.3f}, {upper_bound:.3f}]")
    
    # Apply classification
    df['foil_position'] = np.select(conditions, choices, default=-1)
    
    # Print classification statistics
    for i in range(len(peaks)):
        count = (df['foil_position'] == i).sum()
        print(f"  Foil {i}: {count:,} events")
    
    unclassified = (df['foil_position'] == -1).sum()
    print(f"  Unclassified: {unclassified:,} events")
    
    # Visualization
    if plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available for plotting")
            return df
        
        plt.figure(figsize=(12, 6))
        
        plt.hist(
            data_hist, bins=bins, range=(y_min, y_max),
            color='lightgray', label='Filtered Data'
        )
        
        plt.plot(
            bin_centers[peaks], counts[peaks], "x",
            label='Peaks', color='red', markersize=8
        )
        
        for i, (lb, ub) in enumerate(ranges_info):
            plt.axvspan(
                lb, ub, color=plt.cm.tab10(i % 10),
                alpha=0.3, label=f'Foil {i}'
            )
        
        plt.title(f"Foil Classification ({col_name}) | Range: [{y_min}, {y_max}]")
        plt.xlabel(col_name)
        plt.ylabel("Counts")
        plt.legend()
        plt.xlim(y_min, y_max)
        plt.show()
    
    # Drop unclassified events if requested
    if drop_unclassified:
        n_before = len(df)
        df = df[df['foil_position'] != -1].copy()
        print(f"Dropped {n_before - len(df):,} unclassified events "
              f"({len(df):,} remaining)")
    
    return df


def get_foil_positions(df: pd.DataFrame) -> List[int]:
    """
    Get list of valid foil position values from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'foil_position' column.
    
    Returns
    -------
    list of int
        Sorted list of valid foil position values (excluding -1).
    
    Examples
    --------
    >>> df = classify_foils_with_range(df)
    >>> foil_positions = get_foil_positions(df)
    >>> print(foil_positions)  # [0, 1, 2]
    """
    if 'foil_position' not in df.columns:
        raise ValueError("DataFrame must have 'foil_position' column. "
                        "Run classify_foils_with_range first.")
    
    positions = df['foil_position'].dropna().unique()
    valid_positions = sorted([int(p) for p in positions if p != -1])
    
    return valid_positions


def get_foil_subset(
    df: pd.DataFrame,
    foil_position: int,
    copy: bool = True
) -> pd.DataFrame:
    """
    Get subset of DataFrame for a specific foil position.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'foil_position' column.
    foil_position : int
        Foil position to extract.
    copy : bool, optional
        If True, returns a copy. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Subset of data for the specified foil position.
    
    Examples
    --------
    >>> df = classify_foils_with_range(df)
    >>> df_foil0 = get_foil_subset(df, foil_position=0)
    """
    if 'foil_position' not in df.columns:
        raise ValueError("DataFrame must have 'foil_position' column. "
                        "Run classify_foils_with_range first.")
    
    mask = df['foil_position'] == foil_position
    
    if copy:
        return df[mask].copy()
    return df[mask]


def prepare_clustering_data(
    df: pd.DataFrame,
    x_col: str = 'target_x',
    y_col: str = 'target_y',
    required_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Prepare data array for clustering algorithms.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    x_col : str, optional
        Column name for x-coordinate. Default is 'target_x'.
    y_col : str, optional
        Column name for y-coordinate. Default is 'target_y'.
    required_cols : list of str, optional
        Additional columns that must be present.
    
    Returns
    -------
    tuple
        (data_array, df) where data_array is shape (n_samples, 2)
        and df is the validated DataFrame.
    
    Raises
    ------
    ValueError
        If required columns are missing.
    
    Examples
    --------
    >>> data, df = prepare_clustering_data(df)
    >>> print(data.shape)  # (n_samples, 2)
    """
    # Check required columns
    missing = []
    for col in [x_col, y_col]:
        if col not in df.columns:
            missing.append(col)
    
    if required_cols:
        for col in required_cols:
            if col not in df.columns:
                missing.append(col)
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Extract data array
    data = df[[x_col, y_col]].values
    
    return data, df


def initialize_clustering_columns(
    df: pd.DataFrame,
    cluster_col: str = 'cluster',
    noise_col: str = 'is_noise',
    center_x_col: str = 'cluster_center_x',
    center_y_col: str = 'cluster_center_y',
    inplace: bool = False
) -> pd.DataFrame:
    """
    Initialize clustering result columns in DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    cluster_col : str, optional
        Column name for cluster labels. Default is 'cluster'.
    noise_col : str, optional
        Column name for noise flag. Default is 'is_noise'.
    center_x_col : str, optional
        Column name for cluster center x. Default is 'cluster_center_x'.
    center_y_col : str, optional
        Column name for cluster center y. Default is 'cluster_center_y'.
    inplace : bool, optional
        If True, modifies DataFrame in place. Default is False.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with initialized clustering columns.
    
    Examples
    --------
    >>> df = initialize_clustering_columns(df)
    >>> print(df[['cluster', 'is_noise']].head())
    """
    if not inplace:
        df = df.copy()
    
    if cluster_col not in df.columns:
        df[cluster_col] = -1
    
    if noise_col not in df.columns:
        df[noise_col] = True
    
    if center_x_col not in df.columns:
        df[center_x_col] = np.nan
    
    if center_y_col not in df.columns:
        df[center_y_col] = np.nan
    
    return df
