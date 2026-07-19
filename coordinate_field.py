"""FP5D coordinate-field construction for cluster diagnostics.

The global coordinates are whitened PCA components of robust-scaled focal-plane
measurements.  ``local_z3`` is deliberately different: it is a Fisher/LDA
decision coordinate fitted only inside a local group of neighbouring sieve
clusters.  It is useful for resolving an overlap, but is not a coordinate that
can be compared across separate local neighbourhoods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import RobustScaler


FP5D_CANDIDATES = (
    ("P.dc.x_fp", "P_dc_x_fp", "x_fp"),
    ("P.dc.y_fp", "P_dc_y_fp", "y_fp"),
    ("P.dc.xp_fp", "P_dc_xp_fp", "xp_fp"),
    ("P.dc.yp_fp", "P_dc_yp_fp", "yp_fp"),
    ("P.rb.raster.frybRawAdc", "P_rb_raster_frybRawAdc", "raster", "raster_fryb"),
)


@dataclass
class CoordinateFieldResult:
    """Coordinate table plus concise, GUI-ready provenance."""

    df: pd.DataFrame
    feature_columns: list[str]
    summary: dict


def resolve_fp5d_columns(columns: Iterable[str]) -> list[str]:
    """Return the five available FP5D columns, accepting ROOT and GUI aliases."""
    available = set(columns)
    resolved = [next((name for name in names if name in available), None) for names in FP5D_CANDIDATES]
    missing = [" / ".join(names) for names, name in zip(FP5D_CANDIDATES, resolved) if name is None]
    if missing:
        raise ValueError("FP5D columns unavailable: " + "; ".join(missing))
    return [name for name in resolved if name is not None]


def _neighbour_components(centers: pd.DataFrame, radius_cm: float) -> list[list[int]]:
    """Connected components of cluster centres within *radius_cm*."""
    ids = centers.index.to_list()
    parent = {cluster: cluster for cluster in ids}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    xy = centers[["sieve_x", "sieve_y"]].to_numpy(float)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if np.linalg.norm(xy[i] - xy[j]) <= radius_cm:
                union(ids[i], ids[j])
    groups: dict[int, list[int]] = {}
    for cluster in ids:
        groups.setdefault(find(cluster), []).append(cluster)
    return [group for group in groups.values() if len(group) > 1]


def build_coordinate_field(
    events: pd.DataFrame,
    *,
    cluster_column: str = "cluster",
    neighbour_radius_cm: float = 0.90,
    seed_probability: float = 0.80,
    assignment_probability: float = 0.90,
    assignment_margin: float = 0.20,
) -> CoordinateFieldResult:
    """Build global FP5D PCA coordinates and conservative local LDA coordinates.

    Cluster labels are optional.  Without them the function still provides the
    globally comparable ``fp5d_z1`` through ``fp5d_z5`` coordinates; local
    refinement remains unavailable until clustering has been run.
    """
    feature_columns = resolve_fp5d_columns(events.columns)
    df = events.copy()
    values = df[feature_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.isfinite(values).all(axis=1)
    valid_by_index = pd.Series(valid, index=df.index)
    if valid.sum() < 5:
        raise ValueError("Need at least five finite FP5D events to construct coordinates.")

    scaled = RobustScaler(quantile_range=(5, 95)).fit_transform(values[valid])
    pca = PCA(n_components=5, whiten=True, random_state=42)
    z = pca.fit_transform(scaled)
    for index in range(5):
        column = f"fp5d_z{index + 1}"
        df[column] = np.nan
        df.loc[valid, column] = z[:, index]

    df["local_component"] = -1
    df["local_z3"] = np.nan
    df["local_probability"] = np.nan
    df["local_margin"] = np.nan
    df["local_refined_cluster"] = df[cluster_column] if cluster_column in df else -1

    components = 0
    reassigned = 0
    if cluster_column in df and {"sieve_x", "sieve_y"}.issubset(df.columns):
        group_column = "foil_position" if "foil_position" in df.columns else None
        groups = df.groupby(group_column, dropna=False) if group_column else [(None, df)]
        component_id = 0
        for _, group in groups:
            active = group[group[cluster_column].notna() & (group[cluster_column] >= 0)]
            if active.empty:
                continue
            centers = active.groupby(cluster_column)[["sieve_x", "sieve_y"]].median().dropna()
            for cluster_ids in _neighbour_components(centers, neighbour_radius_cm):
                local = group[group[cluster_column].isin(cluster_ids) & valid_by_index.loc[group.index]]
                if local[cluster_column].nunique() < 2:
                    continue
                seed = local
                if "hdbscan_probability" in local:
                    seed = local[local["hdbscan_probability"] >= seed_probability]
                seed_counts = seed[cluster_column].value_counts()
                if len(seed_counts) < 2 or int(seed_counts.min()) < 2 or len(seed) < 8:
                    continue
                seed_x = df.loc[seed.index, [f"fp5d_z{i}" for i in range(1, 6)]].to_numpy(float)
                local_x = df.loc[local.index, [f"fp5d_z{i}" for i in range(1, 6)]].to_numpy(float)
                lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                lda.fit(seed_x, seed[cluster_column].to_numpy())
                probabilities = lda.predict_proba(local_x)
                prediction = lda.classes_[probabilities.argmax(axis=1)]
                ordered = np.sort(probabilities, axis=1)
                confidence, margin = ordered[:, -1], ordered[:, -1] - ordered[:, -2]
                df.loc[local.index, "local_component"] = component_id
                df.loc[local.index, "local_probability"] = confidence
                df.loc[local.index, "local_margin"] = margin
                if len(lda.classes_) == 2:
                    # The signed binary score is the local third coordinate.
                    score = lda.decision_function(local_x)
                    df.loc[local.index, "local_z3"] = np.asarray(score).reshape(-1)
                change = (confidence >= assignment_probability) & (margin >= assignment_margin)
                old = df.loc[local.index, cluster_column].to_numpy()
                df.loc[local.index[change], "local_refined_cluster"] = prediction[change]
                reassigned += int((prediction[change] != old[change]).sum())
                components += 1
                component_id += 1

    return CoordinateFieldResult(
        df=df,
        feature_columns=feature_columns,
        summary={
            "events": int(len(df)),
            "finite_fp5d_events": int(valid.sum()),
            "features": feature_columns,
            "global_coordinate": "whitened PCA(robust-scaled FP5D)",
            "local_coordinate": "local_z3 = shrinkage-LDA decision score in neighbouring sieve-cluster cells",
            "local_components": components,
            "conservative_reassignments": reassigned,
            "neighbour_radius_cm": neighbour_radius_cm,
        },
    )
