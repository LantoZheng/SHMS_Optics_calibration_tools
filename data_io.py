"""
Data I/O module for SHMS Optics Calibration.

This module provides functions for loading ROOT files and reading
SHMS optics calibration data.
"""

from typing import Optional, List, Tuple, Union
import pandas as pd
import numpy as np

try:
    import uproot
except ImportError:
    uproot = None

from .config import (
    DataLoadingConfig,
    TargetProjectionConfig,
    DEFAULT_DATA_LOADING_CONFIG,
    DEFAULT_TARGET_PROJECTION_CONFIG,
)


def load_root_file(
    file_path: str,
    tree_name: str = "T",
    branches: Optional[List[str]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load data from a ROOT file into a pandas DataFrame.
    
    This function reads a ROOT file containing SHMS optics data and returns
    it as a pandas DataFrame. It supports reading specific branches or all
    branches from the specified TTree.
    
    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str, optional
        Name of the TTree to read from. Default is "T".
    branches : list of str, optional
        List of branch names to read. If None, reads all branches.
    verbose : bool, optional
        If True, prints loading information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame containing the loaded data.
    
    Raises
    ------
    ImportError
        If uproot is not installed.
    FileNotFoundError
        If the specified file does not exist.
    KeyError
        If the specified tree or branches are not found.
    
    Examples
    --------
    >>> df = load_root_file("data.root")
    >>> df = load_root_file("data.root", branches=["P_gtr_x", "P_gtr_y"])
    >>> df = load_root_file("data.root", tree_name="T", verbose=False)
    """
    if uproot is None:
        raise ImportError(
            "uproot is required for loading ROOT files. "
            "Install it with: pip install uproot"
        )
    
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        
        if verbose:
            print(f"Data File: {file_path}")
            print(f"Total Events: {tree.num_entries:,}")
        
        if branches is not None:
            df = tree.arrays(expressions=branches, library="pd")
        else:
            df = tree.arrays(library="pd")
    
    if verbose:
        print(f"Data loaded successfully! DataFrame shape: {df.shape}")
    
    return df


def get_root_file_info(file_path: str, tree_name: str = "T") -> dict:
    """
    Get information about a ROOT file without loading all data.
    
    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str, optional
        Name of the TTree to examine. Default is "T".
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'file_path': Path to the file
        - 'tree_name': Name of the tree
        - 'num_entries': Number of events in the tree
        - 'branches': List of branch names
    
    Examples
    --------
    >>> info = get_root_file_info("data.root")
    >>> print(f"Events: {info['num_entries']}")
    >>> print(f"Branches: {info['branches']}")
    """
    if uproot is None:
        raise ImportError(
            "uproot is required for loading ROOT files. "
            "Install it with: pip install uproot"
        )
    
    with uproot.open(file_path) as file:
        tree = file[tree_name]
        
        info = {
            'file_path': file_path,
            'tree_name': tree_name,
            'num_entries': tree.num_entries,
            'branches': list(tree.keys())
        }
    
    return info


def project_to_target(
    df: pd.DataFrame,
    x_col: str = 'P_gtr_x',
    y_col: str = 'P_gtr_y',
    th_col: str = 'P_gtr_th',
    ph_col: str = 'P_gtr_ph',
    dp_col: str = 'P_gtr_dp',
    config: Optional[TargetProjectionConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate target plane projection from reconstructed variables.
    
    This function projects the reconstructed tracking variables to the
    target plane using the SHMS optics transformation formulas.
    
    The projection formulas are:
    
    - Target_x = x + th * z_coefficient
    - Target_y = (-0.019 * dp + 0.00019 * dp² + 213 * ph + y) 
                 + 40.0 * (-0.00052 * dp + 0.0000052 * dp² + ph)
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the tracking variables.
    x_col : str, optional
        Column name for x position. Default is 'P_gtr_x'.
    y_col : str, optional
        Column name for y position. Default is 'P_gtr_y'.
    th_col : str, optional
        Column name for in-plane angle θ. Default is 'P_gtr_th'.
    ph_col : str, optional
        Column name for out-of-plane angle φ. Default is 'P_gtr_ph'.
    dp_col : str, optional
        Column name for momentum deviation δ. Default is 'P_gtr_dp'.
    config : TargetProjectionConfig, optional
        Configuration object containing projection coefficients.
        If None, uses default configuration.
    
    Returns
    -------
    tuple of np.ndarray
        (target_x, target_y) arrays containing the projected coordinates.
    
    Examples
    --------
    >>> df = load_root_file("data.root")
    >>> target_x, target_y = project_to_target(df)
    >>> df['target_x'] = target_x
    >>> df['target_y'] = target_y
    """
    if config is None:
        config = DEFAULT_TARGET_PROJECTION_CONFIG
    
    x = df[x_col].values
    y = df[y_col].values
    th = df[th_col].values
    ph = df[ph_col].values
    dp = df[dp_col].values
    
    # Target_x projection
    target_x = x + th * config.x_z_coefficient
    
    # Target_y projection
    target_y = (
        config.y_dp_linear * dp + 
        config.y_dp_quadratic * dp**2 + 
        config.y_ph_coefficient * ph + 
        y
    ) + config.y_offset_multiplier * (
        config.y_offset_dp_linear * dp + 
        config.y_offset_dp_quadratic * dp**2 + 
        ph
    )
    
    return target_x, target_y


def add_target_projection(
    df: pd.DataFrame,
    config: Optional[TargetProjectionConfig] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Add target plane projection columns to DataFrame.
    
    This is a convenience function that calculates the target plane
    projection and adds the results as new columns 'target_x' and 'target_y'.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the tracking variables.
    config : TargetProjectionConfig, optional
        Configuration object containing projection coefficients.
        If None, uses default configuration.
    inplace : bool, optional
        If True, modifies the DataFrame in place. Default is False.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'target_x' and 'target_y' columns.
    
    Examples
    --------
    >>> df = load_root_file("data.root")
    >>> df = add_target_projection(df)
    >>> print(df[['target_x', 'target_y']].head())
    """
    if not inplace:
        df = df.copy()
    
    target_x, target_y = project_to_target(df, config=config)
    df['target_x'] = target_x
    df['target_y'] = target_y
    
    return df


def filter_target_range(
    df: pd.DataFrame,
    x_range: Tuple[float, float] = (-20.0, 20.0),
    y_range: Tuple[float, float] = (-20.0, 20.0),
    target_x_col: str = 'target_x',
    target_y_col: str = 'target_y',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filter DataFrame to keep only events within the target plane range.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target_x and target_y columns.
    x_range : tuple of float, optional
        (min, max) range for target_x. Default is (-20.0, 20.0).
    y_range : tuple of float, optional
        (min, max) range for target_y. Default is (-20.0, 20.0).
    target_x_col : str, optional
        Column name for target x. Default is 'target_x'.
    target_y_col : str, optional
        Column name for target y. Default is 'target_y'.
    verbose : bool, optional
        If True, prints filtering statistics. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    
    Examples
    --------
    >>> df = add_target_projection(load_root_file("data.root"))
    >>> df_filtered = filter_target_range(df, x_range=(-15, 15))
    """
    original_count = len(df)
    
    mask = (
        (df[target_x_col] >= x_range[0]) & (df[target_x_col] <= x_range[1]) &
        (df[target_y_col] >= y_range[0]) & (df[target_y_col] <= y_range[1])
    )
    
    df_filtered = df[mask].copy()
    
    if verbose:
        print(f"Data filtering complete: {original_count:,} -> {len(df_filtered):,} events")
        print(f"Removed {original_count - len(df_filtered):,} events outside range")
    
    return df_filtered


def load_and_prepare_data(
    file_path: str,
    tree_name: str = "T",
    add_projection: bool = True,
    filter_range: bool = True,
    data_config: Optional[DataLoadingConfig] = None,
    projection_config: Optional[TargetProjectionConfig] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load ROOT file and prepare data for clustering analysis.
    
    This is a convenience function that combines loading, projection,
    and filtering into a single step.
    
    By default, only a subset of branches is read from the ROOT file
    (focal plane, target reconstruction, and reaction vertex variables),
    as defined by ``DataLoadingConfig.branches``. To read all branches,
    pass ``data_config=DataLoadingConfig(branches=None)``.
    
    Parameters
    ----------
    file_path : str
        Path to the ROOT file.
    tree_name : str, optional
        Name of the TTree to read from. Default is "T".
    add_projection : bool, optional
        If True, adds target plane projection. Default is True.
    filter_range : bool, optional
        If True, filters data to target plane range. Default is True.
    data_config : DataLoadingConfig, optional
        Configuration for data loading. If None, uses defaults.
        The ``branches`` field controls which branches are read.
    projection_config : TargetProjectionConfig, optional
        Configuration for target projection. If None, uses defaults.
    verbose : bool, optional
        If True, prints progress information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        Prepared DataFrame ready for clustering analysis.
    
    Examples
    --------
    >>> df = load_and_prepare_data("data.root")
    >>> print(df[['target_x', 'target_y', 'P_gtr_y']].head())
    
    >>> # Read all branches
    >>> from shms_optics_calibration import DataLoadingConfig
    >>> df = load_and_prepare_data("data.root", data_config=DataLoadingConfig(branches=None))
    """
    if data_config is None:
        data_config = DEFAULT_DATA_LOADING_CONFIG
    
    # Load data
    df = load_root_file(
        file_path, tree_name=tree_name, branches=data_config.branches, verbose=verbose
    )
    
    # Add target projection
    if add_projection:
        df = add_target_projection(df, config=projection_config)
    
    # Filter to target range
    if filter_range and add_projection:
        df = filter_target_range(
            df,
            x_range=data_config.target_x_range,
            y_range=data_config.target_y_range,
            verbose=verbose
        )
    
    return df


def load_simulation_data(
    file_path: str,
    tree_name: Optional[str] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Load simulation data for benchmark evaluation.
    
    This function loads simulation ROOT files and calculates the
    reconstructed sieve pattern along with truth information.
    
    Parameters
    ----------
    file_path : str
        Path to the simulation ROOT file.
    tree_name : str, optional
        Name of the TTree. If None, auto-detects common names.
    verbose : bool, optional
        If True, prints loading information. Default is True.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with reconstructed and truth columns including:
        - target_x, target_y: Reconstructed sieve pattern
        - xsieve_truth, ysieve_truth: Truth sieve positions
        - xsnum, ysnum: Truth sieve hole numbers
        - truth_hole_id: Combined truth hole identifier
    
    Examples
    --------
    >>> df_sim = load_simulation_data("simulation.root")
    >>> print(df_sim[['target_x', 'target_y', 'truth_hole_id']].head())
    """
    if uproot is None:
        raise ImportError(
            "uproot is required for loading ROOT files. "
            "Install it with: pip install uproot"
        )
    
    sim_file = uproot.open(file_path)
    
    if verbose:
        print(f"File: {file_path}")
        print(f"Available trees: {list(sim_file.keys())}")
    
    # Auto-detect tree name
    if tree_name is None:
        for name in sim_file.keys():
            if 'h10' in name.lower() or 'tree' in name.lower() or 't' == name.lower().strip(';1'):
                tree_name = name
                break
        if tree_name is None:
            tree_name = list(sim_file.keys())[0]
    
    if verbose:
        print(f"Using tree: {tree_name}")
    
    sim_tree = sim_file[tree_name]
    
    # Read required branches
    psdelta = sim_tree['psdelta'].array(library='np')
    psyptar = sim_tree['psyptar'].array(library='np')
    psxptar = sim_tree['psxptar'].array(library='np')
    psytar = sim_tree['psytar'].array(library='np')
    fry = sim_tree['fry'].array(library='np')
    xsieve = sim_tree['xsieve'].array(library='np')
    ysieve = sim_tree['ysieve'].array(library='np')
    xsnum = sim_tree['xsnum'].array(library='np')
    ysnum = sim_tree['ysnum'].array(library='np')
    
    if verbose:
        print(f"Loaded {len(psdelta)} events from simulation")
    
    # Calculate reconstructed sieve pattern
    sv_h = (
        -0.019 * psdelta + 0.00019 * psdelta**2 +
        (138.0 + 75.0) * psyptar + psytar +
        40.0 * (-0.00052 * psdelta + 0.0000052 * psdelta**2 + psyptar)
    )
    sv_v = fry + psxptar * 253.0
    
    # Flatten arrays
    sv_h = sv_h.flatten() if hasattr(sv_h, 'flatten') else sv_h
    sv_v = sv_v.flatten() if hasattr(sv_v, 'flatten') else sv_v
    xsieve = xsieve.flatten() if hasattr(xsieve, 'flatten') else xsieve
    ysieve = ysieve.flatten() if hasattr(ysieve, 'flatten') else ysieve
    xsnum = xsnum.flatten() if hasattr(xsnum, 'flatten') else xsnum
    ysnum = ysnum.flatten() if hasattr(ysnum, 'flatten') else ysnum
    
    # Create DataFrame
    df_sim = pd.DataFrame({
        'target_x': sv_v,
        'target_y': sv_h,
        'xsieve_truth': xsieve,
        'ysieve_truth': ysieve,
        'xsnum': xsnum,
        'ysnum': ysnum,
        'psdelta': psdelta.flatten() if hasattr(psdelta, 'flatten') else psdelta,
        'psyptar': psyptar.flatten() if hasattr(psyptar, 'flatten') else psyptar,
        'psxptar': psxptar.flatten() if hasattr(psxptar, 'flatten') else psxptar,
        'psytar': psytar.flatten() if hasattr(psytar, 'flatten') else psytar
    })
    
    # Create combined truth hole ID
    # Handle negative ysnum by encoding
    df_sim['truth_hole_id'] = df_sim['xsnum'].astype(int) * 100 + (
        df_sim['ysnum'].astype(int) % 100
    )
    
    if verbose:
        print(f"Reconstructed sieve pattern calculated")
        print(f"  sv_h range: [{sv_h.min():.2f}, {sv_h.max():.2f}] cm")
        print(f"  sv_v range: [{sv_v.min():.2f}, {sv_v.max():.2f}] cm")
    
    sim_file.close()
    
    return df_sim
