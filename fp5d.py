"""FP5D clustering and coordinate-field utilities.

This module is deliberately independent of ROOT and of the GUI.  It turns the
four focal-plane measurements plus the raster coordinate into a reproducible
five-dimensional chart, clusters it, and optionally builds weakly supervised
``z3`` coordinates.  The latter use foil-like labels only as a *weak ordering*
signal; they never require a sieve-hole label or a hard sieve-plane cut.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence
import warnings

import numpy as np
import pandas as pd

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import RobustScaler, SplineTransformer
except ImportError:  # pragma: no cover - gives a useful error at call time
    KMeans = PCA = LinearDiscriminantAnalysis = Ridge = RobustScaler = SplineTransformer = None


_FP_ALIASES = (
    ("P.dc.x_fp", "P_dc_x_fp", "P.gtr.x_fp", "xfp", "x_fp"),
    ("P.dc.y_fp", "P_dc_y_fp", "P.gtr.y_fp", "yfp", "y_fp"),
    ("P.dc.xp_fp", "P_dc_xp_fp", "P.gtr.xp_fp", "xpfp", "xp_fp"),
    ("P.dc.yp_fp", "P_dc_yp_fp", "P.gtr.yp_fp", "ypfp", "yp_fp"),
)
_RASTER_ALIASES = ("P.rb.raster.frybRawAdc", "P_rb_raster_frybRawAdc", "fr_ybpm", "fryb", "raster_y")
_WEAK_LABEL_ALIASES = ("final_relative_foil", "foil_position", "foil", "inferred_foil", "cluster_assigned_foil")


@dataclass(frozen=True)
class FP5DColumns:
    """Resolved input columns, in canonical FP5D order."""
    fp: tuple[str, str, str, str]
    raster: str

    @property
    def all(self) -> tuple[str, str, str, str, str]:
        return (*self.fp, self.raster)


@dataclass
class FP5DResult:
    """Tabular output plus fitted lightweight models and audit metadata."""
    table: pd.DataFrame
    columns: FP5DColumns
    valid_mask: pd.Series
    metadata: dict[str, Any] = field(default_factory=dict)
    raster_model: Any = None
    scaler: Any = None
    pca: Any = None


@dataclass
class FP5DClusterResult(FP5DResult):
    clusterer: Any = None


def _require_sklearn() -> None:
    if PCA is None:
        raise ImportError("FP5D tools require scikit-learn. Install it with `pip install scikit-learn`.")


def _first_present(columns: Iterable[str], candidates: Sequence[str]) -> Optional[str]:
    available = set(columns)
    for name in candidates:
        if name in available:
            return name
    # ROOT branch spelling is commonly changed only by dot/underscore conversion.
    normalised = {str(c).replace(".", "_").lower(): c for c in columns}
    for name in candidates:
        hit = normalised.get(name.replace(".", "_").lower())
        if hit is not None:
            return str(hit)
    return None


def resolve_fp5d_columns(
    data: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    raster_column: Optional[str] = None,
) -> FP5DColumns:
    """Resolve canonical FP coordinates from ROOT-style or GUI-friendly names."""
    if feature_columns is not None:
        if len(feature_columns) != 4:
            raise ValueError("feature_columns must contain exactly xfp, yfp, xpfp, ypfp.")
        missing = [c for c in feature_columns if c not in data]
        if missing:
            raise KeyError(f"FP5D feature columns are absent: {missing}")
        fp = tuple(feature_columns)
    else:
        resolved = [_first_present(data.columns, aliases) for aliases in _FP_ALIASES]
        if any(c is None for c in resolved):
            raise KeyError("Could not resolve the four FP columns. Provide feature_columns explicitly.")
        fp = tuple(resolved)  # type: ignore[assignment]
    raster = raster_column or _first_present(data.columns, _RASTER_ALIASES)
    if raster is None or raster not in data:
        raise KeyError("Could not resolve raster coordinate. Provide raster_column explicitly.")
    return FP5DColumns(fp=fp, raster=raster)


def build_fp5d_latent(
    data: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    raster_column: Optional[str] = None,
    condition_on_raster: bool = True,
    n_spline_knots: int = 8,
    ridge_alpha: float = 2.0,
    random_state: int = 25521,
) -> FP5DResult:
    """Raster-condition FP coordinates then create whitened PCA ``flow_z1..z5``.

    This is a dependable lightweight replacement for the exploratory neural
    flow: it preserves five dimensions and has no torch dependency.  The PCA
    coordinates are intentionally named ``flow_z*`` so downstream GUI plots
    can use the same coordinate selector for both versions.
    """
    _require_sklearn()
    cols = resolve_fp5d_columns(data, feature_columns, raster_column)
    values = data.loc[:, list(cols.all)].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=data.index, name="fp5d_valid")
    if int(valid.sum()) < 12:
        raise ValueError("FP5D requires at least 12 finite events.")
    raw_fp = values.loc[valid, list(cols.fp)].to_numpy(float)
    raster = values.loc[valid, cols.raster].to_numpy(float).reshape(-1, 1)
    model = None
    corrected = raw_fp.copy()
    if condition_on_raster:
        # The spline captures smooth beam-raster motion without using optics reconstruction.
        knots = int(np.clip(n_spline_knots, 3, max(3, len(raw_fp) // 10)))
        spline = SplineTransformer(n_knots=knots, degree=3, extrapolation="linear")
        basis = spline.fit_transform(raster)
        model = (spline, Ridge(alpha=float(ridge_alpha)).fit(basis, raw_fp))
        corrected -= model[1].predict(basis)
    features = np.column_stack([corrected, raster[:, 0]])
    scaler = RobustScaler(quantile_range=(5, 95)).fit(features)
    scaled = scaler.transform(features)
    pca = PCA(n_components=5, whiten=True, random_state=random_state).fit(scaled)
    z = pca.transform(scaled)
    output = data.copy()
    output["fp5d_valid"] = valid
    for i in range(5):
        output[f"fp5d_{i + 1}"] = np.nan
        output[f"flow_z{i + 1}"] = np.nan
        output.loc[valid, f"fp5d_{i + 1}"] = features[:, i]
        output.loc[valid, f"flow_z{i + 1}"] = z[:, i]
    return FP5DResult(output, cols, valid, {
        "method": "raster-conditioned FP5D + RobustScaler + whitened PCA",
        "events_total": int(len(data)), "events_valid": int(valid.sum()),
        "condition_on_raster": bool(condition_on_raster),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "coordinate_columns": [f"flow_z{i}" for i in range(1, 6)],
    }, model, scaler, pca)


def _make_hdbscan(min_cluster_size: int, min_samples: Optional[int], selection_method: str) -> Any:
    try:
        import hdbscan
        return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                               cluster_selection_method=selection_method)
    except ImportError:
        try:
            from sklearn.cluster import HDBSCAN
            return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                           cluster_selection_method=selection_method)
        except (ImportError, AttributeError) as exc:
            raise ImportError("HDBSCAN needs `hdbscan` (recommended) or scikit-learn >= 1.3.") from exc


def cluster_fp5d(
    data: pd.DataFrame,
    min_cluster_size: int = 60,
    min_samples: Optional[int] = 10,
    selection_method: str = "eom",
    **latent_kwargs: Any,
) -> FP5DClusterResult:
    """Build the FP5D chart and cluster all valid events with HDBSCAN."""
    result = build_fp5d_latent(data, **latent_kwargs)
    zcols = [f"flow_z{i}" for i in range(1, 6)]
    x = result.table.loc[result.valid_mask, zcols].to_numpy(float)
    clusterer = _make_hdbscan(int(min_cluster_size), min_samples, selection_method)
    labels = clusterer.fit_predict(x)
    probabilities = getattr(clusterer, "probabilities_", np.full(len(x), np.nan))
    table = result.table.copy()
    table["flow_hdbscan_cluster"] = -1
    table["hdbscan_probability"] = np.nan
    table.loc[result.valid_mask, "flow_hdbscan_cluster"] = labels
    table.loc[result.valid_mask, "hdbscan_probability"] = probabilities
    active = labels >= 0
    meta = dict(result.metadata)
    meta.update({"clustering": "HDBSCAN on flow_z1..flow_z5", "min_cluster_size": int(min_cluster_size),
                 "min_samples": min_samples, "selection_method": selection_method,
                 "clusters": int(len(np.unique(labels[active]))), "noise_fraction": float((~active).mean())})
    return FP5DClusterResult(table, result.columns, result.valid_mask, meta,
                             result.raster_model, result.scaler, result.pca, clusterer)


def cluster_fp5d_research(
    data: pd.DataFrame,
    min_cluster_size: int = 60,
    min_samples: Optional[int] = 10,
    selection_method: str = "eom",
    epochs: int = 80,
    batch_size: int = 2048,
    random_state: int = 25521,
) -> FP5DClusterResult:
    """Exact Stage-1 continuous-prior flow used in the Run-25521 study.

    Unlike :func:`cluster_fp5d`, this is not a PCA preview: it applies the
    study's DC/kinematic/PID quality selection and trains the six-coupling
    invertible map against continuous reconstructed ``(sieve_x, sieve_y,
    ytar)`` weak targets before HDBSCAN.  No sieve, foil, or hole cut enters
    its clustering selection.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise ImportError("Research FP5D flow requires PyTorch. Install `torch` in the GUI Python environment.") from exc

    _require_sklearn()
    cols = resolve_fp5d_columns(data)
    required = list(cols.all) + ["P_gtr_x", "P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_gtr_y", "P_react_z",
                                 "P_ngcer_npeSum", "P_hgcer_npeSum", "P_cal_etottracknorm"]
    missing = [name for name in required if name not in data]
    if missing:
        raise KeyError("Research flow needs these canonical columns: " + ", ".join(missing))
    values = data.loc[:, required].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(np.isfinite(values.to_numpy()).all(axis=1), index=data.index, name="fp5d_valid")
    valid &= (values.loc[:, list(cols.fp)] > -9999).all(axis=1)
    valid &= values.P_gtr_dp.between(-25.0, 22.0) & values.P_gtr_th.between(-.08, .08)
    valid &= values.P_gtr_ph.between(-.06, .06) & values.P_react_z.between(-120.0, 120.0)
    valid &= (values.P_ngcer_npeSum >= 6.0) & (values.P_hgcer_npeSum >= 0.0) & values.P_cal_etottracknorm.between(.8, 1.8)
    if int(valid.sum()) < max(100, int(min_cluster_size) * 2):
        raise ValueError(f"Research quality selection retained only {int(valid.sum())} events.")

    raw_fp = values.loc[valid, list(cols.fp)].to_numpy(float)
    raster = values.loc[valid, cols.raster].to_numpy(float).reshape(-1, 1)
    spline = SplineTransformer(n_knots=8, degree=3, extrapolation="linear")
    basis = spline.fit_transform(raster)
    ridge = Ridge(alpha=2.0).fit(basis, raw_fp)
    features = np.column_stack([raw_fp - ridge.predict(basis), raster[:, 0]])
    scaler = RobustScaler(quantile_range=(5, 95)).fit(features)
    x = scaler.transform(features).astype("float32")
    # Reconstruct only after quality cuts, matching the study and avoiding
    # overflow from rejected tracking/DC sentinel entries.
    q = values.loc[valid]
    dp, th, ph = (q[name].to_numpy(float) for name in ("P_gtr_dp", "P_gtr_th", "P_gtr_ph"))
    sieve_x = q.P_gtr_x.to_numpy(float) + 253.0 * th
    sieve_y = (-.019 * dp + .00019 * dp**2 + 213.0 * ph + q.P_gtr_y.to_numpy(float)) + 40.0 * (-.00052 * dp + .0000052 * dp**2 + ph)
    targets = np.column_stack([sieve_x, sieve_y, q.P_gtr_y.to_numpy(float)])
    target_scaler = __import__("sklearn.preprocessing", fromlist=["StandardScaler"]).StandardScaler().fit(targets)
    y = target_scaler.transform(targets).astype("float32")

    class Coupling(nn.Module):
        def __init__(self, mask):
            super().__init__(); self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
            self.net = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 10))
            nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
        def forward(self, x):
            kept = x * self.mask; log_s, shift = self.net(kept).chunk(2, dim=1)
            log_s = .65 * torch.tanh(log_s) * (1 - self.mask); shift = shift * (1 - self.mask)
            return kept + (1 - self.mask) * (x * torch.exp(log_s) + shift)

    class Flow(nn.Module):
        def __init__(self):
            super().__init__(); masks = ([1,0,1,0,1], [0,1,0,1,0], [1,1,0,0,1], [0,0,1,1,0], [1,0,0,1,1], [0,1,1,0,0])
            self.layers = nn.ModuleList([Coupling(mask) for mask in masks])
        def forward(self, x):
            for layer in self.layers: x = layer(x)
            return x

    torch.manual_seed(int(random_state)); np.random.seed(int(random_state))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Flow().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=int(batch_size), shuffle=True)
    history: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z = model(xb); loss = ((z[:, :3] - yb) ** 2).mean() + .02 * ((z[:, 3:] - xb[:, 3:]) ** 2).mean()
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(xb)
        if epoch in (0, 19, 39, int(epochs) - 1): history.append({"epoch": float(epoch + 1), "loss": total / len(x)})
    with torch.no_grad(): z = model(torch.from_numpy(x).to(device)).cpu().numpy()
    z[:, 3:] *= np.sqrt(.10)
    clusterer = _make_hdbscan(int(min_cluster_size), min_samples, selection_method)
    labels = clusterer.fit_predict(z); probabilities = getattr(clusterer, "probabilities_", np.full(len(z), np.nan))
    table = data.copy(); table["fp5d_valid"] = valid
    table.loc[valid, "sieve_x"] = sieve_x; table.loc[valid, "sieve_y"] = sieve_y
    for i in range(5):
        table[f"fp5d_{i+1}"] = np.nan; table[f"flow_z{i+1}"] = np.nan
        table.loc[valid, f"fp5d_{i+1}"] = features[:, i]; table.loc[valid, f"flow_z{i+1}"] = z[:, i]
    table["flow_hdbscan_cluster"] = -1; table["hdbscan_probability"] = np.nan
    table.loc[valid, "flow_hdbscan_cluster"] = labels; table.loc[valid, "hdbscan_probability"] = probabilities
    active = labels >= 0
    meta = {"method": "research continuous-prior six-coupling flow", "events_total": int(len(data)), "events_quality": int(valid.sum()),
            "weak_continuous_targets": ["sieve_x", "sieve_y", "P_gtr_y"], "epochs": int(epochs), "loss": history,
            "clustering": "HDBSCAN on research flow_z1..flow_z5", "min_cluster_size": int(min_cluster_size),
            "min_samples": min_samples, "selection_method": selection_method, "clusters": int(len(np.unique(labels[active]))),
            "noise_fraction": float((~active).mean()), "device": str(device)}
    return FP5DClusterResult(table, cols, valid, meta, (spline, ridge), scaler, None, clusterer)


def _rbf(u: np.ndarray, centres: np.ndarray, width: float) -> np.ndarray:
    d2 = ((u[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
    phi = np.exp(-0.5 * d2 / max(width * width, 1e-8))
    return phi / (phi.sum(axis=1, keepdims=True) + 1e-12)


def _weak_labels(table: pd.DataFrame, column: Optional[str]) -> tuple[pd.Series, str]:
    chosen = column or _first_present(table.columns, _WEAK_LABEL_ALIASES)
    if chosen is None:
        raise ValueError("A weak foil-like label column is required for Z-coordinate construction.")
    labels = pd.to_numeric(table[chosen], errors="coerce")
    if labels.nunique(dropna=True) < 2:
        raise ValueError(f"Weak label `{chosen}` must contain at least two groups.")
    return labels, chosen


def build_z_coordinate_field(
    data: pd.DataFrame,
    weak_label_column: Optional[str] = None,
    cluster_column: str = "flow_hdbscan_cluster",
    n_experts: int = 8,
    ridge_alpha: float = 0.15,
    random_state: int = 25521,
) -> FP5DResult:
    """Add local-discriminant and global continuous/flattened Z3 coordinates.

    ``z3_local`` is a linear discriminant in original raster-conditioned FP5D.
    ``global_continuous_z3`` is a smooth RBF mixture-of-linear-experts field.
    ``global_flattened_z3`` removes the smooth in-foil z1/z2 trend while
    retaining an ordinal foil offset.  All three outputs remain per-event
    coordinates and are valid for plotting or subsequent clustering.
    """
    _require_sklearn()
    required = [f"flow_z{i}" for i in range(1, 6)] + [f"fp5d_{i}" for i in range(1, 6)] + [cluster_column]
    missing = [c for c in required if c not in data]
    if missing:
        raise KeyError("Run cluster_fp5d/build_fp5d_latent first; missing: " + ", ".join(missing))
    labels, label_source = _weak_labels(data, weak_label_column)
    valid = data["fp5d_valid"].fillna(False) & labels.notna() & (data[cluster_column] >= 0)
    if valid.sum() < 30:
        raise ValueError("Need at least 30 clustered FP5D events with a weak label.")
    table = data.copy()
    f = table.loc[valid, [f"fp5d_{i}" for i in range(1, 6)]].to_numpy(float)
    z = table.loc[valid, ["flow_z1", "flow_z2", "flow_z3"]].to_numpy(float)
    raw_label = labels.loc[valid]
    codes, unique = pd.factorize(raw_label, sort=True)
    # Orient codes using median pre-existing z3, so the result has a stable sign.
    order = pd.DataFrame({"code": codes, "z3": z[:, 2]}).groupby("code").z3.median().sort_values().index.to_numpy()
    remap = np.empty(len(order), dtype=int); remap[order] = np.arange(len(order)); codes = remap[codes]
    # RBF gates live in z1,z2.  They define overlapping local coordinate
    # charts, rather than a hard partition of the focal-plane manifold.
    n_proto = min(max(2, int(n_experts)), max(2, len(f) // 15))
    centres = KMeans(n_clusters=n_proto, n_init=10, random_state=random_state).fit(z[:, :2]).cluster_centers_
    distances = np.sqrt(((centres[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2))
    width = float(np.median(distances[distances > 0]) * 1.3) if np.any(distances > 0) else 1.0
    phi = _rbf(z[:, :2], centres, width)

    # Each local discriminant sees its own smoothly weighted neighbourhood;
    # blending expected class coordinates makes z3_local continuous at chart
    # boundaries.  The global LDA is only a safe fallback for sparse gates.
    global_lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(f, codes)
    global_score = global_lda.predict_proba(f) @ np.arange(len(unique), dtype=float)
    local_score = np.zeros(len(f))
    for expert in range(n_proto):
        support = phi[:, expert] >= 0.35
        if support.sum() >= max(20, 3 * len(unique)) and np.unique(codes[support]).size == len(unique):
            lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(f[support], codes[support])
            score = lda.predict_proba(f) @ np.arange(len(unique), dtype=float)
        else:
            score = global_score
        local_score += phi[:, expert] * score
    local = (local_score - np.mean(local_score)) / (np.std(local_score) + 1e-12)

    # The same gates learn an ordinal guide from FP5D for the global field.
    design = (phi[:, :, None] * np.column_stack([np.ones(len(f)), f])[:, None, :]).reshape(len(f), -1)
    target = codes.astype(float) - np.mean(codes)
    guide_model = Ridge(alpha=float(ridge_alpha)).fit(design, target)
    guide = guide_model.predict(design)
    # Choose a conservative correction scale rather than overwriting flow_z3.
    alpha_grid = np.linspace(-1.0, 1.5, 51)
    scores = [np.corrcoef(z[:, 2] + a * guide, target)[0, 1] for a in alpha_grid]
    alpha = float(alpha_grid[int(np.nanargmax(np.abs(scores)))])
    continuous = z[:, 2] + alpha * guide

    # Smooth within-foil baseline: residual makes a foil layer flatter across z1,z2.
    baseline_design = np.column_stack([phi, phi * target[:, None]])
    baseline = Ridge(alpha=max(float(ridge_alpha), 0.5)).fit(baseline_design, continuous).predict(baseline_design)
    spacing = float(np.median(np.diff(np.sort(pd.Series(continuous).groupby(codes).median().to_numpy())))) if len(unique) > 1 else 1.0
    flattened = continuous - baseline + target * (abs(spacing) if np.isfinite(spacing) and spacing else 1.0)
    for name, values in (("z3_local", local), ("global_ordinal_guide", guide),
                         ("global_continuous_z3", continuous), ("global_flattened_z3", flattened)):
        table[name] = np.nan
        table.loc[valid, name] = values
    columns = resolve_fp5d_columns(table)
    return FP5DResult(table, columns, valid, {
        "method": "linear local discriminant + continuous RBF mixture-of-experts Z3 field",
        "weak_label_column": label_source, "weak_labels": [str(x) for x in unique.tolist()],
        "cluster_column": cluster_column, "experts": n_proto, "rbf_width": width,
        "local_components": n_proto,
        "continuous_correction_alpha": alpha,
        "coordinates": ["z3_local", "global_continuous_z3", "global_flattened_z3"],
    })
