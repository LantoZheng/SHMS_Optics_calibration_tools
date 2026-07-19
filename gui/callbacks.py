"""SOC Labeling GUI — Dash callbacks.

Wires UI events (button clicks, slider changes, graph selections) to
``SHMS_Optics_calibration_tools`` backend functions and updates Plotly
figures in-place via ``patch`` when possible, or regenerates them.
"""

from __future__ import annotations

import colorsys
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any, Optional

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import uproot
from dash import Input, Output, State, callback, dcc, html, no_update, ALL, ctx
from plotly import colors

# The package is a submodule: its parent is the workspace root containing the
# archived FP5D study and the other calibration packages.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import SHMS_Optics_calibration_tools as soc
from SHMS_Optics_calibration_tools.config import MechanicalGridConfig, HDBSCANConfig
from SHMS_Optics_calibration_tools.fp5d import cluster_fp5d, cluster_fp5d_research, build_z_coordinate_field
from SHMS_Optics_calibration_tools.gui.state import get_session

# ── colour palette ────────────────────────────────────────
_FOIL_COLORS = {0: "#2196F3", 1: "#FF9800", 2: "#4CAF50"}
_PLOTLY_TEMPLATE = "plotly_white"
_RESEARCH_OVERLAY_ACTIVE = {"display": "flex", "position": "fixed", "inset": "0", "zIndex": 10000,
                            "background": "rgba(15, 23, 42, 0.72)", "alignItems": "center", "justifyContent": "center"}
_RESEARCH_OVERLAY_HIDDEN = {"display": "none"}

# Branches required for the normal sieve/foil GUI, PID controls and FP5D flow.
# Production replay files can expand to tens of GB when every branch is read.
_OPTIMIZED_ROOT_COLUMNS = (
    "P_gtr_x", "P_gtr_y", "P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_react_z",
    "P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc",
    "P_ngcer_npeSum", "P_hgcer_npeSum", "P_cal_etottracknorm",
)

# ── helpers ───────────────────────────────────────────────


def _root_column_name(name: str) -> str:
    """Convert HCANA's dotted branch spelling into the GUI's canonical form."""
    return str(name).replace(".", "_")


def _resolve_root_branches(
    tree, mode: str, extra_branches: str | None, manual_mapping: Optional[dict[str, str]] = None,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Select physical ROOT branches and map them to canonical GUI columns."""
    available = [str(name).split(";")[0] for name in tree.keys()]
    by_canonical = {_root_column_name(name): name for name in available}
    manual_mapping = manual_mapping or {}
    if manual_mapping:
        absent = [f"{role} → {branch}" for role, branch in manual_mapping.items() if branch not in available]
        if absent:
            raise KeyError("Selected ROOT branches unavailable: " + ", ".join(absent))
        if len(set(manual_mapping.values())) != len(manual_mapping):
            raise ValueError("One ROOT branch cannot be assigned to more than one logical GUI variable.")
        requested = list(manual_mapping.values())
        rename_map = {branch: role for role, branch in manual_mapping.items()}
    else:
        requested = list(available) if mode == "all" else list(_OPTIMIZED_ROOT_COLUMNS)
        rename_map = {}
    requested.extend(item.strip() for item in (extra_branches or "").replace("\n", ",").split(",") if item.strip())

    selected: list[str] = []
    missing_required: list[str] = []
    missing_extra: list[str] = []
    required = set(_OPTIMIZED_ROOT_COLUMNS) if (mode != "all" or manual_mapping) else set()
    for requested_name in requested:
        actual = requested_name if requested_name in available else by_canonical.get(_root_column_name(requested_name))
        if actual is None:
            (missing_required if _root_column_name(requested_name) in required else missing_extra).append(requested_name)
            continue
        if actual not in selected:
            selected.append(actual)
    if missing_required:
        raise KeyError("Required branches unavailable: " + ", ".join(missing_required))
    return selected, missing_extra, rename_map


def _run_exact_run25521_pipeline(root_file: str, force_recompute: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the archived study scripts in their original order and load its final chart.

    This deliberately uses the same code path as the earlier figures, rather
    than an approximate GUI reimplementation.  Its fixed experimental design
    is Run 25521, whose unskimmed ROOT file is the selected GUI reference.
    """
    study = Path(_REPO_ROOT) / "SHMS_Calibration_NN" / "experiments" / "raw_fullroot_fp5d"
    expected = Path(root_file).expanduser().resolve()
    if expected.name.lower() != "shms_coin_replay_production_25521_-1.root":
        raise ValueError("The exact archived pipeline is calibrated to unskimmed Run 25521. Select shms_coin_replay_production_25521_-1.root first.")
    if not expected.is_file():
        raise FileNotFoundError(f"Selected Run-25521 ROOT file does not exist: {expected}")
    scripts = (
        "run_full_linear_local_field_pipeline.py",
        "relative_z_lattice_mapping.py",
        "nonlinear_canonical_z_remapping.py",
        "render_nearest_foil_completed_sieve.py",
        "build_global_continuous_z3.py",
        "flatten_global_z3_foil_layers.py",
    )
    output = study / "results" / "global_foil_flattened_z3_labels.csv"
    # This study has a fixed ROOT input and fixed validated parameters.  Reuse
    # its final artifact when it is newer than both the raw ROOT file and all
    # pipeline scripts; this preserves bit-for-bit prior results while turning
    # routine GUI inspection from minutes into seconds.
    newest_input = max([expected.stat().st_mtime_ns, *[(study / name).stat().st_mtime_ns for name in scripts]])
    cache_hit = output.exists() and output.stat().st_mtime_ns >= newest_input and not force_recompute
    if not cache_hit:
        previous_root = os.environ.get("SHMS_RUN25521_ROOT")
        os.environ["SHMS_RUN25521_ROOT"] = str(expected)
        try:
            for name in scripts:
                runpy.run_path(str(study / name), run_name="__main__")
        finally:
            if previous_root is None:
                os.environ.pop("SHMS_RUN25521_ROOT", None)
            else:
                os.environ["SHMS_RUN25521_ROOT"] = previous_root
    table = pd.read_csv(output).rename(columns=_root_column_name)
    # The viewer's default cluster column follows the conservative local-field
    # assignment, exactly as in the final research figures.
    table["flow_hdbscan_cluster"] = table["linear_local_field_cluster"]
    summary = json.loads((study / "results" / "global_foil_flattened_z3_summary.json").read_text(encoding="utf-8"))
    summary.update({"method": "exact archived Run-25521 research pipeline", "scripts": list(scripts), "cache_hit": cache_hit})
    return table, summary


def _project_to_sieve(df: pd.DataFrame) -> pd.DataFrame:
    """Add sieve_x / sieve_y using the full SHMS optics formula."""
    if "sieve_x" in df.columns and "sieve_y" in df.columns:
        return df
    return soc.add_sieve_projection(df)


@callback(
    Output("manual-root-branch-controls", "style"),
    Input("radio-root-branch-mode", "value"),
)
def toggle_manual_root_branches(mode: str) -> dict[str, Any]:
    return ({"display": "block", "max-height": "360px", "overflow-y": "auto", "padding": "5px", "border": "1px solid #ddd", "border-radius": "4px", "margin-bottom": "8px"}
            if mode == "manual" else {"display": "none"})


@callback(
    Output({"type": "manual-root-branch", "role": ALL}, "options"),
    Output({"type": "manual-root-branch", "role": ALL}, "value"),
    Input("input-root-file", "value"),
    Input("input-tree-name", "value"),
)
def inspect_root_branch_choices(root_file: str, tree_name: str):
    """Populate manual mapping selectors from metadata only, never event arrays."""
    empty = [[] for _ in _OPTIMIZED_ROOT_COLUMNS]
    empty_values = [None for _ in _OPTIMIZED_ROOT_COLUMNS]
    if not root_file or not os.path.isfile(root_file):
        return empty, empty_values
    try:
        tree = uproot.open(root_file)[tree_name or "T"]
        names = [str(name).split(";")[0] for name in tree.keys()]
        options = [{"label": name, "value": name} for name in names]
        canonical = {_root_column_name(name): name for name in names}
        defaults = [canonical.get(role) for role in _OPTIMIZED_ROOT_COLUMNS]
        return [options for _ in _OPTIMIZED_ROOT_COLUMNS], defaults
    except Exception:
        return empty, empty_values


def _classify_foils(df: pd.DataFrame) -> pd.DataFrame:
    """Wrap foil classification with GUI-friendly defaults."""
    return soc.classify_foils_with_range(
        df, col_name="P_gtr_y", bins=50, sigma_factor=3.0,
        y_range=(-5.0, 5.0), drop_unclassified=True,
    )


def _build_empty_figure(message: str = "Load data to begin") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font={"size": 18, "color": "#999"},
    )
    fig.update_layout(template=_PLOTLY_TEMPLATE, margin={"l": 40, "r": 20, "t": 40, "b": 40})
    return fig


def _fp5d_options(df: pd.DataFrame) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return compact axis and categorical colour selectors for the FP5D lab."""
    numeric = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in numeric if c.startswith(("flow_z", "fp5d_", "z3_", "global_"))]
    preferred += [c for c in ("sieve_x", "sieve_y", "P_gtr_y", "foil_position", "flow_hdbscan_cluster") if c in numeric and c not in preferred]
    axes = preferred or numeric
    colour = [c for c in ("linear_local_field_cluster", "flow_hdbscan_cluster", "final_relative_foil", "canonical_foil_layer", "foil_position", "canonical_hole_family", "local_component") if c in df]
    colour += [c for c in axes if c not in colour]
    return ([{"label": c, "value": c} for c in axes], [{"label": c, "value": c} for c in colour])


def _stable_colour(label: object) -> str:
    if str(label) in {"-1", "nan", "None"}:
        return "#9aa3ad"
    hue = (hash(str(label)) & 0xFFFFFFFF) % 360
    return f"hsl({hue}, 63%, 47%)"


def _build_fp5d_3d_figure(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                           colour_col: str, objects: str) -> tuple[go.Figure, str]:
    """Interactive FP5D viewer: translucent events plus opaque cluster centres."""
    required = [x_col, y_col, z_col]
    if any(not c or c not in df for c in required):
        return _build_empty_figure("Run FP5D Flow + HDBSCAN first"), ""
    plot = df.dropna(subset=required).copy()
    if plot.empty:
        return _build_empty_figure("No finite events in this coordinate view"), ""
    cluster_col = "flow_hdbscan_cluster"
    session = get_session()
    manual_matches = session.fp5d_manual_matches
    selected_cluster = session.fp5d_selected_cluster
    if len(plot) > 45_000:
        if cluster_col in plot:
            plot = plot.groupby(cluster_col, group_keys=False).apply(
                lambda g: g.sample(min(len(g), max(15, 45_000 // max(1, plot[cluster_col].nunique()))), random_state=42)
            )
            # pandas 3.x may exclude the grouping column from GroupBy.apply
            # output. Restore it by event index so centroid construction is
            # independent of the installed pandas version.
            if cluster_col not in plot:
                plot[cluster_col] = df.reindex(plot.index)[cluster_col].to_numpy()
        else:
            plot = plot.sample(45_000, random_state=42)
    category = plot[colour_col].fillna(-1).astype(str) if colour_col in plot else pd.Series("events", index=plot.index)
    fig = go.Figure()
    if objects != "centroids":
        for label in sorted(category.unique(), key=str):
            part = plot.loc[category == label]
            fig.add_trace(go.Scatter3d(
                x=part[x_col], y=part[y_col], z=part[z_col], mode="markers", name=f"{colour_col}: {label}",
                marker={"size": 1.35, "opacity": .24, "color": _stable_colour(label)},
                customdata=part[[c for c in (cluster_col, "foil_position", "sieve_x", "sieve_y", "P_gtr_y") if c in part]].to_numpy(),
                hovertemplate="%{x:.4g}, %{y:.4g}, %{z:.4g}<extra></extra>", showlegend=category.nunique() <= 18,
            ))
    n_centroids = 0
    if objects != "points" and cluster_col in plot:
        centers = plot.loc[plot[cluster_col] >= 0].groupby(cluster_col).agg(
            **{x_col: (x_col, "mean"), y_col: (y_col, "mean"), z_col: (z_col, "mean"), "events_displayed": (cluster_col, "size")},
        )
        full_counts = df.loc[df[cluster_col] >= 0, cluster_col].value_counts()
        centers["events_full"] = centers.index.to_series().map(full_counts).fillna(centers["events_displayed"]).to_numpy()
        if colour_col in plot:
            modes = plot.loc[plot[cluster_col] >= 0].groupby(cluster_col)[colour_col].agg(lambda s: s.dropna().mode().iloc[0] if not s.dropna().empty else -1)
            centers["colour"] = centers.index.map(modes).fillna(-1).astype(str)
        else:
            centers["colour"] = centers.index.astype(str)
        max_n = max(1, centers["events_full"].max())
        for label, part in centers.groupby("colour"):
            sizes = 3.0 + 7.0 * np.cbrt(part["events_full"].to_numpy() / max_n)
            fig.add_trace(go.Scatter3d(
                x=part[x_col], y=part[y_col], z=part[z_col], mode="markers", name=f"centroids: {label}",
                marker={"size": sizes, "opacity": 1.0, "color": _stable_colour(label), "line": {"color": "#172033", "width": .8}},
                customdata=[[int(idx)] for idx in part.index],
                text=[
                    f"cluster {idx}<br>N (full) = {int(row.events_full):,}<br>N (shown) = {int(row.events_displayed):,}"
                    + ("<br>Manual: noise" if manual_matches.get(int(idx), {}).get("noise") else
                       f"<br>Manual: foil {manual_matches[int(idx)]['foil']}, grid ({manual_matches[int(idx)]['row']}, {manual_matches[int(idx)]['col']})" if int(idx) in manual_matches else "")
                    for idx, row in part.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>", showlegend=False,
            ))
        if selected_cluster is not None and selected_cluster in centers.index:
            selected = centers.loc[selected_cluster]
            fig.add_trace(go.Scatter3d(
                x=[selected[x_col]], y=[selected[y_col]], z=[selected[z_col]], mode="markers",
                name=f"selected cluster {selected_cluster}",
                marker={"size": 14, "symbol": "circle-open", "color": "#d62728", "line": {"color": "#d62728", "width": 3}},
                customdata=[[int(selected_cluster)]],
                hovertemplate=f"selected cluster {selected_cluster}<extra></extra>", showlegend=True,
            ))
        # Legend-only reference markers give a portable, dynamic size scale.
        for n in sorted(set([int(centers.events_full.min()), int(centers.events_full.median()), int(centers.events_full.max())])):
            fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", name=f"centroid scale: N={n:,}",
                                       marker={"size": 3.0 + 7.0 * (n / max_n) ** (1 / 3), "color": "#64748b"}, showlegend=True))
        n_centroids = len(centers)
    fig.update_layout(template=_PLOTLY_TEMPLATE, title="FP5D latent-space clustering",
                      scene={"xaxis_title": x_col, "yaxis_title": y_col, "zaxis_title": z_col, "aspectmode": "data"},
                      margin={"l": 0, "r": 0, "t": 45, "b": 0}, legend={"itemsizing": "constant"}, uirevision="fp5d-z-space")
    return fig, f"{len(plot):,} displayed events; {n_centroids} opaque centroids; diameter ∝ N^(1/3), so visual volume ∝ N."


@callback(
    Output("fp5d-status", "children"), Output("z-coordinate-status", "children"),
    Output("graph-fp5d-3d", "figure"), Output("fp5d-view-status", "children"),
    Output("fp5d-x-axis", "options"), Output("fp5d-y-axis", "options"), Output("fp5d-z-axis", "options"), Output("fp5d-colour", "options"),
    Output("fp5d-x-axis", "value"), Output("fp5d-y-axis", "value"), Output("fp5d-z-axis", "value"), Output("fp5d-colour", "value"),
    Input("btn-run-fp5d-flow", "n_clicks"), Input("btn-build-z-coordinates", "n_clicks"),
    Input("fp5d-show-objects", "value"), Input("fp5d-x-axis", "value"), Input("fp5d-y-axis", "value"), Input("fp5d-z-axis", "value"), Input("fp5d-colour", "value"),
    State("slider-fp5d-min-cluster-size", "value"), State("slider-fp5d-min-samples", "value"), State("checkbox-force-exact-rerun", "value"),
    prevent_initial_call=True,
    running=[
        (Output("btn-run-fp5d-flow", "disabled"), True, False),
        (Output("fp5d-research-overlay", "style"), _RESEARCH_OVERLAY_ACTIVE, _RESEARCH_OVERLAY_HIDDEN),
    ],
)
def on_fp5d_lab_change(run_clicks, z_clicks, objects, x_col, y_col, z_col, colour_col, min_cluster_size, min_samples, force_exact_rerun):
    """Run the independent FP5D pipeline, construct Z fields, or redraw the viewer."""
    session = get_session()
    trigger = ctx.triggered_id
    fp_status = no_update
    z_status = no_update
    try:
        if trigger == "btn-run-fp5d-flow":
            source = session.fp5d_source_df if session.fp5d_source_df is not None else session.raw_df
            if source is None or source.empty:
                return "⚠️ Load data first.", no_update, _build_empty_figure("Load data first"), "", [], [], [], [], None, None, None, None
            if not session.fp5d_root_path:
                raise ValueError("Reload the ROOT file before running the exact research pipeline.")
            session.fp5d_df, session.fp5d_summary = _run_exact_run25521_pipeline(session.fp5d_root_path, bool(force_exact_rerun))
            # Cluster ids are an output of the run, so annotations cannot be
            # silently carried across a potentially different clustering.
            session.fp5d_manual_matches.clear()
            session.fp5d_selected_cluster = None
            session.z_coordinate_summary = session.fp5d_summary
            cache_text = "validated cache reused" if session.fp5d_summary.get("cache_hit") else "full recomputation complete"
            fp_status = ("✅ Exact Run-25521 research pipeline complete (" + cache_text + "): continuous-prior flow, local fields, "
                         "relative Z mapping, canonical remapping, global Z3 and foil flattening.")
            x_col, y_col, z_col, colour_col = "flow_z1", "flow_z2", "global_foil_flattened_z3", "final_relative_foil"
        elif trigger == "btn-build-z-coordinates":
            if session.fp5d_df is None:
                return no_update, "⚠️ Run FP5D Flow + HDBSCAN first.", _build_empty_figure("Run FP5D Flow + HDBSCAN first"), "", [], [], [], [], None, None, None, None
            if "global_foil_flattened_z3" not in session.fp5d_df:
                return no_update, "⚠️ The exact Z pipeline is run together with the research flow button.", no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update
            z_status = "✅ Exact global continuous and foil-flattened Z3 coordinates are already available."
            z_col = "global_foil_flattened_z3"

        if session.fp5d_df is None:
            return no_update, no_update, _build_empty_figure("Run FP5D Flow + HDBSCAN first"), "", [], [], [], [], None, None, None, None
        df = session.fp5d_df
        axis_options, colour_options = _fp5d_options(df)
        allowed_axes = {item["value"] for item in axis_options}
        allowed_colours = {item["value"] for item in colour_options}
        x_col = x_col if x_col in allowed_axes else "flow_z1"
        y_col = y_col if y_col in allowed_axes else "flow_z2"
        z_col = z_col if z_col in allowed_axes else "flow_z3"
        colour_col = colour_col if colour_col in allowed_colours else "flow_hdbscan_cluster"
        figure, view_status = _build_fp5d_3d_figure(df, x_col, y_col, z_col, colour_col, objects or "both")
        return fp_status, z_status, figure, view_status, axis_options, axis_options, axis_options, colour_options, x_col, y_col, z_col, colour_col
    except Exception as exc:
        message = f"❌ FP5D workflow error: {exc}"
        return message, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update


@callback(
    Output("download-fp5d-html", "data"),
    Input("btn-export-fp5d-html", "n_clicks"),
    State("fp5d-show-objects", "value"), State("fp5d-x-axis", "value"), State("fp5d-y-axis", "value"),
    State("fp5d-z-axis", "value"), State("fp5d-colour", "value"),
    prevent_initial_call=True,
)
def export_fp5d_html(n_clicks, objects, x_col, y_col, z_col, colour_col):
    """Download the same rotatable 3D figure as a self-contained advisor-ready HTML."""
    session = get_session()
    if session.fp5d_df is None:
        return no_update
    fig, _ = _build_fp5d_3d_figure(session.fp5d_df, x_col, y_col, z_col, colour_col, objects or "both")
    html_text = pio.to_html(fig, full_html=True, include_plotlyjs=True, config={"scrollZoom": True, "displaylogo": False})
    return dcc.send_string(html_text, "fp5d-flow-cluster-explorer.html")


def _fp5d_manual_match_table() -> html.Div:
    """Compact, inspectable record of expert assignments in the FP5D lab."""
    matches = get_session().fp5d_manual_matches
    if not matches:
        return html.Span("No manual FP5D assignments recorded.", style={"color": "#777"})
    rows = [
        html.Tr([html.Td(f"C{cluster}"), html.Td("noise" if item.get("noise") else f"foil {item['foil']}"), html.Td("—" if item.get("noise") else f"({item['row']}, {item['col']})")])
        for cluster, item in sorted(matches.items())
    ]
    return html.Div([
        html.Span(f"Recorded {len(rows)} manual assignment(s): ", style={"font-weight": "600"}),
        html.Table([html.Thead(html.Tr([html.Th("Cluster"), html.Th("Foil"), html.Th("Grid (row, col)")])), html.Tbody(rows)],
                   style={"display": "inline-table", "margin-left": "8px", "vertical-align": "middle", "border-collapse": "collapse"}),
    ])


@callback(
    Output("store-fp5d-manual-selection", "data"),
    Output("fp5d-manual-selection", "children"),
    Output("fp5d-manual-match-status", "children"),
    Output("fp5d-manual-match-table", "children"),
    Output("graph-fp5d-3d", "figure", allow_duplicate=True),
    Input("graph-fp5d-3d", "clickData"),
    Input("btn-fp5d-record-match", "n_clicks"),
    Input("btn-fp5d-mark-noise", "n_clicks"),
    State("store-fp5d-manual-selection", "data"),
    State("fp5d-show-objects", "value"), State("fp5d-x-axis", "value"), State("fp5d-y-axis", "value"),
    State("fp5d-z-axis", "value"), State("fp5d-colour", "value"),
    State("fp5d-match-foil", "value"), State("fp5d-match-row", "value"), State("fp5d-match-col", "value"),
    prevent_initial_call=True,
)
def on_fp5d_manual_match(click_data, record_clicks, noise_clicks, selection, objects, x_col, y_col, z_col, colour_col, foil, row, col):
    """Select an FP5D cluster in 3D and explicitly bind it to foil/grid indices."""
    session = get_session()
    if session.fp5d_df is None:
        return {}, "Run FP5D Flow + HDBSCAN first.", "", _fp5d_manual_match_table(), no_update

    selection = selection or {}
    triggered = ctx.triggered_id
    status = ""
    if triggered == "graph-fp5d-3d":
        point = (click_data or {}).get("points", [{}])[0]
        custom = point.get("customdata")
        try:
            cluster = int(custom[0] if isinstance(custom, (list, tuple)) else custom)
        except (TypeError, ValueError, IndexError):
            return selection, "Click a centroid (or an event point) to select its FP5D cluster.", "⚠️ This mark has no cluster id.", _fp5d_manual_match_table(), no_update
        if cluster < 0:
            return selection, "Noise events cannot be assigned to a grid hole.", "⚠️ Select a non-noise cluster.", _fp5d_manual_match_table(), no_update
        subset = session.fp5d_df[session.fp5d_df["flow_hdbscan_cluster"] == cluster]
        if subset.empty:
            return selection, "Cluster not found.", "⚠️ Selected cluster is absent from the active result.", _fp5d_manual_match_table(), no_update
        inferred = subset["final_relative_foil"].dropna().mode()
        inferred_text = f"; inferred foil {int(inferred.iloc[0])}" if not inferred.empty else ""
        selection = {"cluster": cluster}
        session.fp5d_selected_cluster = cluster
        status = f"Selected C{cluster}: {len(subset):,} events{inferred_text}. Choose foil and grid indices, then record."
    elif triggered == "btn-fp5d-mark-noise":
        cluster = selection.get("cluster")
        if cluster is None:
            status = "⚠️ Select a cluster in the 3D view first."
        else:
            cluster = int(cluster)
            session.fp5d_selected_cluster = cluster
            session.fp5d_manual_matches[cluster] = {"noise": True}
            status = f"✅ Marked C{cluster} as manual noise."
    elif triggered == "btn-fp5d-record-match":
        cluster = selection.get("cluster")
        if cluster is None:
            status = "⚠️ Select a cluster in the 3D view first."
        elif any(value is None for value in (foil, row, col)):
            status = "⚠️ Foil, grid row, and grid column are all required."
        else:
            cluster = int(cluster)
            session.fp5d_selected_cluster = cluster
            session.fp5d_manual_matches[cluster] = {"foil": int(foil), "row": int(row), "col": int(col)}
            status = f"✅ Recorded C{cluster} → foil {int(foil)}, grid ({int(row)}, {int(col)})."

    figure, _ = _build_fp5d_3d_figure(session.fp5d_df, x_col, y_col, z_col, colour_col, objects or "both")
    selected_text = (f"Selected FP5D cluster C{session.fp5d_selected_cluster}." if session.fp5d_selected_cluster is not None
                     else "Click a centroid (or an event point) to select its FP5D cluster.")
    return selection, selected_text, status, _fp5d_manual_match_table(), figure


def _filtered_results_for_display(foil_filter: Optional[str] = None):
    """Return (results_dict, foil_positions) respecting the active foil filter."""
    session = get_session()
    if not session.has_clusters():
        return {}, []
    f = foil_filter if foil_filter is not None else session.current_foil_filter
    if f == "all":
        return session.clustered_results, session.foil_positions
    fp = int(f)
    if fp in session.clustered_results:
        return {fp: session.clustered_results[fp]}, [fp]
    return {}, []


def _cluster_display_color(foil_pos: int, cluster_id: int) -> str:
    """Generate a stable high-contrast color for a foil/cluster pair."""
    golden_ratio = 0.6180339887498949
    hue = (foil_pos * 0.17 + cluster_id * golden_ratio) % 1.0
    sat = 0.78
    val = 0.92
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)})"


def _sample_cluster_points_for_display(
    df: pd.DataFrame,
    max_points: int,
    min_points_per_cluster: int,
    random_state: int = 42,
    must_include_indices: Optional[set] = None,
) -> pd.DataFrame:
    """Sample clustered points for display while preserving small clusters.

    A plain random sample over the full foil biases the view toward large clusters.
    This helper guarantees each cluster keeps a visible footprint before allocating
    the remaining point budget proportionally.

    If *must_include_indices* is provided, those DataFrame index values are always
    kept in the returned sample so that user-selected points remain visible.
    """
    if len(df) <= max_points or "cluster" not in df.columns:
        return df

    must = set(must_include_indices or [])
    df_must = df[df.index.isin(must)] if must else df.iloc[:0]
    df_pool = df[~df.index.isin(must)] if must else df

    if len(df_pool) == 0:
        return df_must.sort_index() if len(df_must) > 0 else df

    # Reduce max_points by the already-reserved must-include count
    reserved = len(df_must)
    pool_budget = max(0, max_points - reserved)
    if pool_budget == 0:
        return pd.concat([df_must, df_pool.sample(min(pool_budget + reserved, len(df_pool)), random_state=random_state)]).sort_index() if reserved > 0 else df

    groups = [(cid, g) for cid, g in df_pool.groupby("cluster", sort=True)]
    if not groups:
        return pd.concat([df_must, df_pool]).sort_index() if reserved > 0 else df

    guaranteed = {cid: min(len(g), min_points_per_cluster) for cid, g in groups}
    guaranteed_total = sum(guaranteed.values())

    if guaranteed_total >= pool_budget:
        guaranteed = {cid: 1 for cid, _ in groups}
        guaranteed_total = len(groups)

    remaining_budget = max(0, pool_budget - guaranteed_total)
    leftovers = {cid: max(0, len(g) - guaranteed[cid]) for cid, g in groups}
    leftovers_total = sum(leftovers.values())

    extra = {cid: 0 for cid, _ in groups}
    remainders: list[tuple[float, int]] = []
    if remaining_budget > 0 and leftovers_total > 0:
        for cid, _ in groups:
            share = remaining_budget * leftovers[cid] / leftovers_total
            extra[cid] = min(leftovers[cid], int(share))
            remainders.append((share - int(share), cid))

        assigned = sum(extra.values())
        for _, cid in sorted(remainders, reverse=True):
            if assigned >= remaining_budget:
                break
            if extra[cid] < leftovers[cid]:
                extra[cid] += 1
                assigned += 1

    sampled_parts = [df_must] if reserved > 0 else []
    for cid, group in groups:
        n_take = min(len(group), guaranteed[cid] + extra[cid])
        if n_take >= len(group):
            sampled_parts.append(group)
        else:
            sampled_parts.append(group.sample(n_take, random_state=random_state))

    return pd.concat(sampled_parts, axis=0).sort_index()


def _expand_hole_design_for_manual_binding(
    hole_design: pd.DataFrame,
    design_meta: dict[str, Any],
    extra_rows: int = 1,
    extra_cols: int = 1,
) -> pd.DataFrame:
    """Expand candidate mechanical holes by one outer ring for manual binding.

    The training/grid-match pipeline only builds candidates covering the observed
    row/column extent. For GUI-assisted manual reassignment we want one extra row
    and column on each side so users can bind clusters to nearby not-yet-modeled
    holes directly from the plot.
    """
    if hole_design is None or hole_design.empty:
        return hole_design

    x_spacing_cm = float(design_meta.get("x_spacing_cm", 0.0))
    y_spacing_cm = float(design_meta.get("y_spacing_cm", 0.0))
    x_origin_cm = float(design_meta.get("x_origin_cm", 0.0))
    y_origin_cm = float(design_meta.get("y_origin_cm", 0.0))
    sieve_distance_cm = float(design_meta.get("sieve_distance_cm", 253.0))
    tol_x = float(hole_design.get("weak_hole_xptar_tol", pd.Series([0.0])).iloc[0])
    tol_y = float(hole_design.get("weak_hole_yptar_tol", pd.Series([0.0])).iloc[0])

    expanded_frames: list[pd.DataFrame] = []
    for foil_pos, design_f in hole_design.groupby("foil_position", sort=True):
        foil_pos = int(foil_pos)
        min_row = int(design_f["hole_row"].min()) - int(extra_rows)
        max_row = int(design_f["hole_row"].max()) + int(extra_rows)
        min_col = int(design_f["hole_col"].min()) - int(extra_cols)
        max_col = int(design_f["hole_col"].max()) + int(extra_cols)

        expanded = pd.MultiIndex.from_product(
            [[foil_pos], range(min_row, max_row + 1), range(min_col, max_col + 1)],
            names=["foil_position", "hole_row", "hole_col"],
        ).to_frame(index=False)
        expanded["candidate_sieve_x_cm"] = (
            x_origin_cm + expanded["hole_col"].to_numpy(dtype=np.float64) * x_spacing_cm
        )
        expanded["candidate_sieve_y_cm"] = (
            y_origin_cm + expanded["hole_row"].to_numpy(dtype=np.float64) * y_spacing_cm
        )
        expanded["weak_hole_xptar_center"] = expanded["candidate_sieve_x_cm"] / sieve_distance_cm
        expanded["weak_hole_yptar_center"] = expanded["candidate_sieve_y_cm"] / sieve_distance_cm
        expanded["weak_hole_xptar_tol"] = tol_x
        expanded["weak_hole_yptar_tol"] = tol_y
        expanded_frames.append(expanded)

    expanded_design = pd.concat(expanded_frames, ignore_index=True)
    expanded_design = expanded_design.drop_duplicates(
        subset=["foil_position", "hole_row", "hole_col"]
    ).sort_values(["foil_position", "hole_row", "hole_col"]).reset_index(drop=True)
    return expanded_design


def _init_single_cluster_per_foil(session) -> dict[int, dict[str, Any]]:
    """Assign every event in each foil to cluster 0 (single-cluster mode).

    Returns a ``clustered_results`` dict compatible with the downstream
    plotting and grid-match pipeline.
    """
    results: dict[int, dict[str, Any]] = {}
    for fp in session.foil_positions:
        df_f = session.raw_df[session.raw_df["foil_position"] == fp].copy()
        df_f["cluster"] = 0
        # Median of all points as "cluster center"
        cx = df_f["sieve_x"].median()
        cy = df_f["sieve_y"].median()
        df_f["cluster_center_x"] = cx
        df_f["cluster_center_y"] = cy
        results[fp] = {
            "df": df_f,
            "params": {"method": "manual_single"},
            "n_clusters": 1,
        }
    return results


# ═══════════════════════════════════════════════════════════
# Callback: Load Data
# ═══════════════════════════════════════════════════════════
@callback(
    Output("data-status", "children"),
    Output("graph-sieve-clusters", "figure"),
    Output("cluster-status", "children", allow_duplicate=True),
    Input("btn-load-data", "n_clicks"),
    State("input-root-file", "value"),
    State("input-tree-name", "value"),
    State("radio-root-branch-mode", "value"),
    State("input-extra-root-branches", "value"),
    State({"type": "manual-root-branch", "role": ALL}, "value"),
    State("checkbox-manual-only", "value"),
    prevent_initial_call=True,
)
def on_load_data(
    n_clicks: int,
    root_file: str,
    tree_name: str,
    branch_mode: str,
    extra_branches: str | None,
    manual_branch_values: list[Optional[str]],
    manual_only: bool,
):
    if not root_file:
        return "❌ Please enter a ROOT file path.", _build_empty_figure(), no_update

    session = get_session()
    try:
        if not os.path.exists(root_file):
            return f"❌ File not found: {root_file}", _build_empty_figure(), no_update

        f = uproot.open(root_file)
        tree = f[tree_name]

        mapping = {}
        if branch_mode == "manual":
            if len(manual_branch_values) != len(_OPTIMIZED_ROOT_COLUMNS) or any(value is None for value in manual_branch_values):
                raise ValueError("Manually select one ROOT branch for every required GUI variable.")
            mapping = dict(zip(_OPTIMIZED_ROOT_COLUMNS, manual_branch_values))
        selected_branches, missing_extra, rename_map = _resolve_root_branches(tree, branch_mode, extra_branches, mapping)
        # Read the compact analysis schema by default.  Columns are normalised
        # immediately so the legacy GUI and dotted HCANA ROOT files interoperate.
        df = tree.arrays(expressions=selected_branches, library="pd")
        df = df.rename(columns=lambda name: rename_map.get(str(name), _root_column_name(name)))
        df = df.reset_index(drop=True)

        # Standard filter + project
        df = soc.filter_branch_ranges(
            df,
            {"P_gtr_dp": (-25.0, 22.0), "P_gtr_th": (-0.08, 0.08),
             "P_gtr_ph": (-0.06, 0.06), "P_react_z": (-120.0, 120.0)},
            verbose=False,
        )
        # Preserve the technical/kinematic sample before reconstructed-sieve
        # windows or foil classification.  FP5D flow clustering must not inherit
        # a hard sieve/foil cut from the legacy labeling workflow.
        session.fp5d_source_df = _project_to_sieve(df.copy())
        session.fp5d_root_path = os.path.abspath(root_file)
        session.fp5d_df = None
        session.fp5d_summary = {}
        session.z_coordinate_summary = {}
        session.fp5d_manual_matches.clear()
        session.fp5d_selected_cluster = None
        df = _project_to_sieve(df)
        df = soc.filter_sieve_range(df, x_range=(-20, 20), y_range=(-20, 20), verbose=False)

        # Classify foils
        df = _classify_foils(df)
        session.raw_df = df
        session.foil_positions = sorted(int(v) for v in df["foil_position"].dropna().unique() if v != -1)

        # ── Manual-only: init single cluster per foil ──
        cluster_status = no_update
        if manual_only:
            session.clustered_results = _init_single_cluster_per_foil(session)
            session.hole_design = None
            session.cluster_hole_map = None
            cluster_status = "🔧 Manual mode: each foil = 1 cluster"
            fig = _build_clustered_scatter(session.clustered_results, session.foil_positions)
        else:
            session.clustered_results = None
            fig = _build_sieve_scatter(df, title="Sieve Plane — All Events (colored by foil)")

        load_mode = "manual mapping" if mapping else branch_mode
        load_note = f"; {len(selected_branches)} branches ({load_mode})"
        if missing_extra:
            load_note += "; missing custom: " + ", ".join(missing_extra)
        return (
            f"✅ Loaded {len(df):,} events, {len(session.foil_positions)} foils: {session.foil_positions}{load_note}",
            fig,
            cluster_status,
        )
    except Exception as exc:
        return f"❌ Error: {exc}", _build_empty_figure(), no_update


# ═══════════════════════════════════════════════════════════
# Callback: Apply Filters
# ═══════════════════════════════════════════════════════════
@callback(
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("cluster-status", "children", allow_duplicate=True),
    Input("btn-apply-filters", "n_clicks"),
    State("slider-ngcer-min", "value"),
    State("slider-hgcer-min", "value"),
    State("slider-cal-min", "value"),
    State("slider-cal-max", "value"),
    State("checkbox-manual-only", "value"),
    prevent_initial_call=True,
)
def on_apply_filters(n_clicks, ngcer_min, hgcer_min, cal_min, cal_max, manual_only):
    session = get_session()
    if not session.has_data():
        return no_update, no_update

    df = session.raw_df.copy()
    # Apply PID cuts
    for col, (lo, hi) in [
        ("P_ngcer_npeSum", (ngcer_min, np.inf)),
        ("P_hgcer_npeSum", (hgcer_min, np.inf)),
        ("P_cal_etottracknorm", (cal_min, cal_max)),
    ]:
        if col in df.columns:
            df = df[(df[col] >= lo) & (df[col] <= hi)]

    if len(df) == 0:
        return _build_empty_figure("No events pass filter cuts"), no_update

    # Re-classify foils after filtering
    df = _classify_foils(df)
    session.raw_df = df
    # Filter controls are applied to both paths after data have been loaded.
    session.fp5d_source_df = df.copy()
    session.fp5d_df = None
    session.fp5d_summary = {}
    session.z_coordinate_summary = {}
    session.fp5d_manual_matches.clear()
    session.fp5d_selected_cluster = None
    session.foil_positions = sorted(int(v) for v in df["foil_position"].dropna().unique() if v != -1)

    # ── Manual-only: re-init single cluster per foil ──
    cluster_status = no_update
    if manual_only:
        session.clustered_results = _init_single_cluster_per_foil(session)
        session.hole_design = None
        session.cluster_hole_map = None
        cluster_status = "🔧 Manual mode: each foil = 1 cluster"
        fig = _build_clustered_scatter(session.clustered_results, session.foil_positions)
    else:
        session.clustered_results = None
        session.hole_design = None
        session.cluster_hole_map = None
        fig = _build_sieve_scatter(df, title=f"Sieve Plane — Filtered ({len(df):,} events)")

    return fig, cluster_status


# ═══════════════════════════════════════════════════════════
# Callback: Run Clustering
# ═══════════════════════════════════════════════════════════
@callback(
    Output("cluster-status", "children"),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Input("btn-run-clustering", "n_clicks"),
    State("slider-min-cluster-size", "value"),
    State("slider-min-samples", "value"),
    State("dropdown-cluster-method", "value"),
    State("slider-max-cluster-size", "value"),
    prevent_initial_call=True,
)
def on_run_clustering(n_clicks, min_cluster_size, min_samples, method, max_cluster_size):
    session = get_session()
    if not session.has_data():
        return "❌ No data loaded.", no_update

    try:
        cfg = HDBSCANConfig(
            min_cluster_size_range=(int(min_cluster_size), int(min_cluster_size * 6)),
            min_samples_range=(int(min_samples), int(min_samples * 5)),
            target_clusters=(60, 140),
            cluster_selection_method=method,
            max_cluster_size=float(max_cluster_size),
            distance_threshold=0.6,
            drop_noise=True,
        )
        session.hdbscan_config = cfg

        results: dict[int, dict[str, Any]] = {}
        for foil_pos in session.foil_positions:
            df_foil = session.raw_df[session.raw_df["foil_position"] == foil_pos].copy()
            if len(df_foil) < 10:
                continue
            df_clustered, params, n_clusters = soc.auto_hdbscan_clustering(
                df_foil, config=cfg, verbose=False,
            )
            results[foil_pos] = {
                "df": df_clustered,
                "params": params,
                "n_clusters": n_clusters,
            }

        session.clustered_results = results
        session.hole_design = None
        session.cluster_hole_map = None

        summary_parts = []
        for fp in sorted(results.keys()):
            r = results[fp]
            summary_parts.append(f"Foil {fp}: {r['n_clusters']} clusters")
        status = "✅ " + " | ".join(summary_parts)

        fig = _build_clustered_scatter(results, session.foil_positions)
        return status, fig
    except Exception as exc:
        return f"❌ Clustering error: {exc}", no_update


# ═══════════════════════════════════════════════════════════
# Callback: Run Grid Match
# ═══════════════════════════════════════════════════════════
@callback(
    Output("grid-status", "children"),
    Output("graph-grid-match", "figure"),
    Output("div-grid-match-table", "children", allow_duplicate=True),
    Output("grid-edit-selection", "children", allow_duplicate=True),
    Output("store-grid-edit-selection", "data", allow_duplicate=True),
    Output("btn-grid-apply-match", "disabled", allow_duplicate=True),
    Input("btn-run-grid-match", "n_clicks"),
    State("slider-x-spacing", "value"),
    State("slider-y-spacing", "value"),
    State("slider-tolerance", "value"),
    State("slider-occupancy-penalty", "value"),
    State("dropdown-assignment-mode", "value"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_run_grid_match(n_clicks, x_spacing, y_spacing, tolerance,
                       occupancy_penalty, assignment_mode, foil_filter):
    session = get_session()
    if not session.has_clusters():
        return ("❌ Run clustering first.", _build_empty_figure("Run clustering first"),
                no_update, no_update, no_update, no_update)

    try:
        cfg = MechanicalGridConfig(
            x_spacing_mm=float(x_spacing),
            y_spacing_mm=float(y_spacing),
            tolerance_mm=float(tolerance),
            sieve_distance_cm=253.0,
            assignment_mode=str(assignment_mode),
            occupancy_penalty_cm=float(occupancy_penalty),
            auto_spacing=False,
        )
        session.mechanical_config = cfg

        hole_design, design_meta, cluster_hole_map, match_summary = soc.build_mechanical_grid_index(
            session.clustered_results, config=cfg, verbose=False,
        )
        hole_design = _expand_hole_design_for_manual_binding(hole_design, design_meta)
        session.hole_design = hole_design
        session.cluster_hole_map = cluster_hole_map
        session.summary = match_summary

        dup = match_summary.get("duplicate_mechanical_holes", "?")
        parts = []
        for fk in sorted(match_summary.get("per_foil", {}).keys(), key=int):
            s = match_summary["per_foil"][fk]
            parts.append(
                f"F{fk}: {s['median_match_distance_cm']:.3f}cm med, occ={s['max_hole_occupancy']}"
            )
        status = f"✅ Dup holes: {dup} | " + " | ".join(parts)

        fig = _build_grid_match_figure(
            session.clustered_results, cluster_hole_map, hole_design,
            foil_filter=foil_filter,
        )
        table_html = _build_grid_match_table_html()
        session.store_selection_state = None
        return status, fig, table_html, "No pending reassignment selected.", {}, True
    except Exception as exc:
        return (f"❌ Grid match error: {exc}", _build_empty_figure(),
                no_update, no_update, no_update, no_update)


# ═══════════════════════════════════════════════════════════
# Callback: Toggle manual-only mode → show/hide HDBSCAN controls
# ═══════════════════════════════════════════════════════════
@callback(
    Output("div-hdbscan-controls", "style"),
    Input("checkbox-manual-only", "value"),
    prevent_initial_call=True,
)
def on_toggle_manual_only(manual_only: bool):
    if manual_only:
        return {"display": "none"}
    return {}


# ═══════════════════════════════════════════════════════════
# Callback: Show Edit Clusters checkbox only on Sieve tab
# ═══════════════════════════════════════════════════════════
@callback(
    Output("div-edit-clusters-wrapper", "style"),
    Output("checkbox-edit-clusters", "value"),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def on_tab_toggle_edit_clusters_visible(tab: str):
    if tab == "tab-sieve":
        return {}, no_update
    # Hide checkbox + force-close edit mode when leaving Sieve tab
    return {"display": "none"}, False


# ═══════════════════════════════════════════════════════════
# Callback: Slider ↔ Input bidirectional sync
# ═══════════════════════════════════════════════════════════
_SLIDER_IDS = [
    "slider-ngcer-min", "slider-hgcer-min", "slider-cal-min", "slider-cal-max",
    "slider-min-cluster-size", "slider-min-samples", "slider-max-cluster-size",
    "slider-x-spacing", "slider-y-spacing", "slider-tolerance", "slider-occupancy-penalty",
]

for _sid in _SLIDER_IDS:
    callback(
        Output(_sid, "value"),
        Output(f"{_sid}-input", "value"),
        Input(_sid, "value"),
        Input(f"{_sid}-input", "value"),
        prevent_initial_call=True,
    )(
        lambda sv, iv, _s=_sid: (no_update, sv) if ctx.triggered_id == _s
        else ((iv, no_update) if iv is not None else (no_update, no_update))
    )


# ═══════════════════════════════════════════════════════════
# Callback: Tab switch → update graphs lazily
# ═══════════════════════════════════════════════════════════
@callback(
    Output("graph-grid-match", "figure", allow_duplicate=True),
    Output("graph-ztar", "figure"),
    Input("main-tabs", "value"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_tab_switch(tab, foil_filter):
    session = get_session()
    if tab == "tab-match":
        if session.cluster_hole_map is not None and session.clustered_results:
            fig = _build_grid_match_figure(
                session.clustered_results, session.cluster_hole_map, session.hole_design,
                foil_filter=foil_filter,
            )
            return fig, no_update
        return _build_empty_figure("Run Grid Match first"), no_update
    if tab == "tab-ztar":
        if session.has_data():
            fig = _build_ztar_figure(session.raw_df)
            return no_update, fig
        return no_update, _build_empty_figure("Load data first")
    return no_update, no_update


# ═══════════════════════════════════════════════════════════
# Plot builders
# ═══════════════════════════════════════════════════════════


def _build_sieve_scatter(df: pd.DataFrame, title: str = "Sieve Plane") -> go.Figure:
    """Scatter plot of all events on the sieve plane, coloured by foil."""
    session = get_session()
    fig = go.Figure()
    for foil_pos in sorted(df["foil_position"].dropna().unique()):
        if foil_pos == -1:
            continue
        foil_pos = int(foil_pos)
        sub = df[df["foil_position"] == foil_pos]
        if len(sub) > 8000:
            sub = sub.sample(8000, random_state=42)
        fig.add_trace(go.Scatter(
            x=sub["sieve_x"], y=sub["sieve_y"],
            mode="markers",
            marker={"size": 2, "color": _FOIL_COLORS.get(foil_pos, "#999"),
                    "opacity": 0.5},
            name=f"Foil {foil_pos} ({len(sub):,})",
        ))
    fig.update_layout(
        title=title, template=_PLOTLY_TEMPLATE,
        xaxis_title="Sieve X (cm)", yaxis_title="Sieve Y (cm)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        legend={"font": {"size": 10}},
        dragmode="lasso" if session.edit_mode_active else "zoom",
        clickmode="event+select",
        uirevision=f"sieve-scatter-{int(session.edit_mode_active)}",
    )
    return fig


def _build_clustered_scatter(
    results: dict[int, dict[str, Any]], foil_positions: list[int],
) -> go.Figure:
    """Scatter plot showing clustering results per foil, with noise points dimmed."""
    session = get_session()
    selected_keys = session.selected_event_keys if session.edit_mode_active else set()
    session.trace_point_lookup = {}

    fig = go.Figure()
    signal_marker_size = 4.5 if session.edit_mode_active else 3.0
    signal_marker_opacity = 0.9 if session.edit_mode_active else 0.7
    noise_marker_size = 1.4 if session.edit_mode_active else 1.0
    noise_marker_opacity = 0.12 if session.edit_mode_active else 0.2
    selected_marker_size = 8 if session.edit_mode_active else 7
    unselected_marker_opacity = 0.5 if session.edit_mode_active else 0.7

    for fp in foil_positions:
        if fp not in results:
            continue
        df = results[fp]["df"]

        sig_full = df[df["cluster"] != -1]
        noise_full = df[df["cluster"] == -1]

        # Always keep user-selected points in the display sample
        foil_selected_indices: set = {int(idx) for (idx, f) in selected_keys if int(f) == int(fp)}

        max_signal_points = 100000 if session.edit_mode_active else 8000
        min_cluster_points = 20 if session.edit_mode_active else 8
        sig = _sample_cluster_points_for_display(
            sig_full,
            max_points=max_signal_points,
            min_points_per_cluster=min_cluster_points,
            random_state=42,
            must_include_indices=foil_selected_indices,
        )
        noise = noise_full
        max_noise_points = 2500 if session.edit_mode_active else 4000
        if len(noise) > max_noise_points:
            # Keep selected noise points visible too
            noise_must = noise[noise.index.isin(foil_selected_indices)]
            noise_pool = noise[~noise.index.isin(foil_selected_indices)]
            n_pool = max(0, max_noise_points - len(noise_must))
            if n_pool > 0 and len(noise_pool) > n_pool:
                noise_pool = noise_pool.sample(n_pool, random_state=42)
            noise = pd.concat([noise_must, noise_pool]).sort_index()

        # Signal points
        if len(sig) > 0:
            cluster_ids = sig["cluster"].unique()
            for ci, cid in enumerate(sorted(cluster_ids)):
                sub = sig[sig["cluster"] == cid]
                # Build customdata: [row_index, cluster_id, foil_pos]
                cd = np.column_stack([
                    sub.index.values,
                    np.full(len(sub), cid),
                    np.full(len(sub), fp),
                ])
                # selectedpoints for highlighting
                sel_pts = None
                if selected_keys:
                    sel_mask = np.array([
                        (sub.index.values[i], fp) in selected_keys
                        for i in range(len(sub))
                    ])
                    if sel_mask.any():
                        sel_pts = np.where(sel_mask)[0].tolist()

                trace_idx = len(fig.data)
                session.trace_point_lookup[trace_idx] = [
                    (int(idx), int(fp), int(cid)) for idx in sub.index.values
                ]

                fig.add_trace(go.Scatter(
                    x=sub["sieve_x"], y=sub["sieve_y"],
                    mode="markers",
                    marker={
                        "size": signal_marker_size,
                        "color": _cluster_display_color(int(fp), int(cid)),
                        "opacity": signal_marker_opacity,
                    },
                    selected_marker={"size": selected_marker_size, "color": "#FFD700",
                                     "opacity": 1.0},
                    unselected_marker={"opacity": unselected_marker_opacity},
                    selectedpoints=sel_pts,
                    customdata=cd,
                    name=f"F{fp} C{cid}",
                    legendgroup=f"foil{fp}",
                    legendgrouptitle_text=f"Foil {fp}",
                ))

        # Noise (dimmed)
        if len(noise) > 0:
            cd_noise = np.column_stack([
                noise.index.values,
                np.full(len(noise), -1),
                np.full(len(noise), fp),
            ])
            sel_pts_noise = None
            if selected_keys:
                sel_mask_n = np.array([
                    (noise.index.values[i], fp) in selected_keys
                    for i in range(len(noise))
                ])
                if sel_mask_n.any():
                    sel_pts_noise = np.where(sel_mask_n)[0].tolist()

            trace_idx = len(fig.data)
            session.trace_point_lookup[trace_idx] = [
                (int(idx), int(fp), -1) for idx in noise.index.values
            ]

            fig.add_trace(go.Scatter(
                x=noise["sieve_x"], y=noise["sieve_y"],
                mode="markers",
                marker={"size": noise_marker_size, "color": "#bdbdbd", "opacity": noise_marker_opacity},
                selected_marker={"size": selected_marker_size, "color": "#FFD700",
                                 "opacity": 1.0},
                unselected_marker={"opacity": max(0.08, noise_marker_opacity)},
                selectedpoints=sel_pts_noise,
                customdata=cd_noise,
                name=f"F{fp} noise",
                legendgroup=f"foil{fp}",
                showlegend=True,
            ))

    fig.update_layout(
        title="Sieve Plane — Clustered Events", template=_PLOTLY_TEMPLATE,
        xaxis_title="Sieve X (cm)", yaxis_title="Sieve Y (cm)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        dragmode="lasso" if session.edit_mode_active else "zoom",
        clickmode="event+select",
        uirevision=f"clustered-scatter-{int(session.edit_mode_active)}",
    )
    return fig


def _build_grid_match_figure(
    results: dict[int, dict[str, Any]],
    cluster_hole_map: pd.DataFrame,
    hole_design: pd.DataFrame,
    foil_filter: str = "all",
) -> go.Figure:
    """Overlay cluster centers, matched mechanical holes, and assignment lines.

    Respects *foil_filter* to show all foils or just one."""
    session = get_session()
    session.grid_trace_lookup = {}
    fig = go.Figure()

    foil_list = [int(foil_filter)] if foil_filter != "all" else sorted(results.keys())

    for fp in foil_list:
        if fp not in results:
            continue
        result = results[fp]
        df = result["df"]

        # ── Background: clustered event scatter (dimmed) ──
        sig = df[df["cluster"] != -1]
        if len(sig) > 6000:
            sig = sig.sample(6000, random_state=42)
        if len(sig) > 0:
            fig.add_trace(go.Scatter(
                x=sig["sieve_x"], y=sig["sieve_y"],
                mode="markers",
                marker={"size": 3.0, "color": _FOIL_COLORS.get(fp, "#999"), "opacity": 0.25},
                name=f"F{fp} events",
                legendgroup=f"foil{fp}",
                legendgrouptitle_text=f"Foil {fp}",
                showlegend=True,
            ))
        # Noise (extra dim)
        noise = df[df["cluster"] == -1]
        if len(noise) > 4000:
            noise = noise.sample(4000, random_state=42)
        if len(noise) > 0:
            fig.add_trace(go.Scatter(
                x=noise["sieve_x"], y=noise["sieve_y"],
                mode="markers",
                marker={"size": 1.6, "color": "#bbb", "opacity": 0.12},
                name=f"F{fp} noise",
                legendgroup=f"foil{fp}",
                showlegend=True,
            ))

        # Cluster centers
        centers = df[df["cluster"] != -1].groupby("cluster").agg(
            cx=("cluster_center_x", "median"),
            cy=("cluster_center_y", "median"),
        ).reset_index()

        if len(centers) > 0:
            trace_idx = len(fig.data)
            session.grid_trace_lookup[trace_idx] = [
                {
                    "kind": "cluster",
                    "foil": int(fp),
                    "cluster": int(row["cluster"]),
                    "x": float(row["cx"]),
                    "y": float(row["cy"]),
                }
                for _, row in centers.iterrows()
            ]
            fig.add_trace(go.Scatter(
                x=centers["cx"], y=centers["cy"],
                mode="markers+text",
                marker={"size": 12, "color": _FOIL_COLORS.get(fp, "#999"),
                        "symbol": "circle-open", "line": {"width": 2}},
                customdata=np.column_stack([
                    np.full(len(centers), fp),
                    centers["cluster"].astype(int).values,
                ]),
                text=centers["cluster"].astype(str),
                textposition="top center",
                textfont={"size": 8},
                name=f"F{fp} clusters",
            ))

        # Mechanical hole candidates
        if hole_design is not None:
            design_f = hole_design[hole_design["foil_position"] == fp]
            if len(design_f) > 0:
                trace_idx = len(fig.data)
                session.grid_trace_lookup[trace_idx] = [
                    {
                        "kind": "hole",
                        "foil": int(fp),
                        "hole_row": int(row["hole_row"]),
                        "hole_col": int(row["hole_col"]),
                        "x": float(row["candidate_sieve_x_cm"]),
                        "y": float(row["candidate_sieve_y_cm"]),
                    }
                    for _, row in design_f.iterrows()
                ]
                fig.add_trace(go.Scatter(
                    x=design_f["candidate_sieve_x_cm"],
                    y=design_f["candidate_sieve_y_cm"],
                    mode="markers",
                    marker={
                        "size": 18,
                        "color": "red",
                        "symbol": "cross-thin",
                        "line": {"width": 1},
                    },
                    customdata=np.column_stack([
                        np.full(len(design_f), fp),
                        design_f["hole_row"].astype(int).values,
                        design_f["hole_col"].astype(int).values,
                    ]),
                    name=f"F{fp} mech holes",
                ))

                # ── Center mechanical hole at (row=0, col=0) ──
                center = design_f[
                    (design_f["hole_row"] == 0) & (design_f["hole_col"] == 0)
                ]
                if center.empty:
                    # Fallback to closest to origin if (0,0) is not in the grid
                    design_f["_dist"] = np.sqrt(
                        design_f["hole_row"].astype(float) ** 2
                        + design_f["hole_col"].astype(float) ** 2
                    )
                    center = design_f.loc[[design_f["_dist"].idxmin()]]
                if not center.empty:
                    trace_idx = len(fig.data)
                    session.grid_trace_lookup[trace_idx] = [
                        {
                            "kind": "hole",
                            "foil": int(fp),
                            "hole_row": int(center.iloc[0]["hole_row"]),
                            "hole_col": int(center.iloc[0]["hole_col"]),
                            "x": float(center.iloc[0]["candidate_sieve_x_cm"]),
                            "y": float(center.iloc[0]["candidate_sieve_y_cm"]),
                        }
                    ]
                    fig.add_trace(go.Scatter(
                        x=center["candidate_sieve_x_cm"],
                        y=center["candidate_sieve_y_cm"],
                        mode="markers+text",
                        marker={"size": 22, "color": "#e65100", "symbol": "circle-open",
                                "line": {"width": 2.5}},
                        text=["⚓"],
                        textposition="middle right",
                        textfont={"size": 14, "color": "#e65100"},
                        customdata=np.column_stack([
                            np.full(len(center), fp),
                            center["hole_row"].astype(int).values,
                            center["hole_col"].astype(int).values,
                        ]),
                        name=f"F{fp} center",
                        showlegend=True,
                    ))

        # Assignment lines (cluster → matched hole)
        if cluster_hole_map is not None:
            cmap_f = cluster_hole_map[cluster_hole_map["foil_position"] == fp]
            for _, row in cmap_f.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["cluster_center_x"], row["matched_sieve_x_cm"]],
                    y=[row["cluster_center_y"], row["matched_sieve_y_cm"]],
                    mode="lines",
                    line={"color": "gray", "width": 0.5, "dash": "dot"},
                    showlegend=False,
                    hoverinfo="text",
                    hovertext=f"F{fp} C{int(row['cluster'])} → ({int(row['hole_row'])},{int(row['hole_col'])}) d={row['match_distance_cm']:.3f}cm",
                ))

    fig.update_layout(
        title="Cluster → Mechanical Hole Matching",
        template=_PLOTLY_TEMPLATE,
        xaxis_title="Sieve X (cm)", yaxis_title="Sieve Y (cm)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return fig


def _build_ztar_figure(df: pd.DataFrame) -> go.Figure:
    """Histogram of P_react_z (ztar) coloured by foil."""
    fig = go.Figure()
    for fp in sorted(df["foil_position"].dropna().unique()):
        if fp == -1:
            continue
        fp = int(fp)
        sub = df[df["foil_position"] == fp]
        fig.add_trace(go.Histogram(
            x=sub["P_react_z"], nbinsx=80,
            marker_color=_FOIL_COLORS.get(fp, "#999"),
            opacity=0.6,
            name=f"Foil {fp}",
        ))
    fig.update_layout(
        title="ztar (P_react_z) Distribution by Foil",
        template=_PLOTLY_TEMPLATE,
        xaxis_title="ztar (cm)", yaxis_title="Count",
        barmode="overlay",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
    )
    return fig


# ═══════════════════════════════════════════════════════════
# Callback: Export Labels (Phase 5)
# ═══════════════════════════════════════════════════════════
@callback(
    Output("export-status", "children"),
    Output("download-csv", "data"),
    Input("btn-export", "n_clicks"),
    State("input-output-csv", "value"),
    prevent_initial_call=True,
)
def on_export(n_clicks, output_csv):
    session = get_session()
    if not session.has_clusters() or session.cluster_hole_map is None:
        return "❌ Run clustering + grid match first.", no_update

    try:
        # The stage-2 helpers live in the archived calibration package.  The
        # GUI is launched from the workspace root, so ``training`` is not a
        # top-level package here.
        from SHMS_Calibration_NN.training.scripts.build_stage2_labels_from_25521_fullroot import (
            compute_equal_hole_total_weights,
        )

        hole_design = session.hole_design
        cluster_hole_map = session.cluster_hole_map
        clustered = session.clustered_results

        # Build labels
        labeled_frames = []
        for foil_pos, result in sorted(clustered.items()):
            df_foil = result["df"].copy()
            df_foil = df_foil[df_foil["cluster"] != -1].copy()
            if df_foil.empty:
                continue

            centers = cluster_hole_map[cluster_hole_map["foil_position"] == foil_pos][
                ["foil_position", "cluster", "hole_row", "hole_col",
                 "matched_sieve_x_cm", "matched_sieve_y_cm",
                 "match_dx_cm", "match_dy_cm", "match_distance_cm"]
            ]
            df_foil = df_foil.merge(centers, on=["foil_position", "cluster"], how="inner")
            if df_foil.empty:
                continue

            pop = df_foil.groupby(["foil_position", "hole_row", "hole_col"]).size().reset_index(name="hole_population")
            df_foil = df_foil.merge(pop, on=["foil_position", "hole_row", "hole_col"], how="left")
            df_foil = df_foil.merge(
                hole_design[hole_design["foil_position"] == foil_pos],
                on=["foil_position", "hole_row", "hole_col"], how="left",
            )
            foil_centers = {0: 10.0, 1: 0.0, 2: -10.0}
            df_foil["foil_ytar_center"] = float(foil_centers.get(int(foil_pos), 0.0))
            df_foil["foil_ytar_tol"] = 0.0
            df_foil["foil_population"] = len(df_foil)
            labeled_frames.append(df_foil)

        if not labeled_frames:
            return "❌ No labeled events.", no_update

        labeled = pd.concat(labeled_frames, ignore_index=True)
        labeled["hole_id"] = labeled["foil_position"].astype(int) * 1000 + labeled["cluster"].astype(int)
        labeled["weak_label_weight"] = compute_equal_hole_total_weights(labeled["hole_population"])
        for col in ["hole_row", "hole_col", "foil_position"]:
            if col in labeled.columns:
                labeled[col] = labeled[col].astype(int)

        out_path = output_csv if output_csv else "stage2_soc_gui_labeled.csv"
        labeled.to_csv(out_path, index=False)

        return (
            f"✅ Exported {len(labeled):,} events, {labeled['hole_id'].nunique()} holes → {out_path}",
            dcc.send_file(out_path),
        )
    except Exception as exc:
        return f"❌ Export error: {exc}", no_update


# ═══════════════════════════════════════════════════════════
# Callback: Cluster click → show info in info bar
# ═══════════════════════════════════════════════════════════
@callback(
    Output("cluster-info-bar", "children"),
    Input("graph-sieve-clusters", "clickData"),
    prevent_initial_call=True,
)
def on_cluster_click(clickData):
    if clickData is None:
        return "Click a cluster point to see info"
    pt = clickData.get("points", [{}])[0]
    curve_num = pt.get("curveNumber", 0)
    x = pt.get("x", "?")
    y = pt.get("y", "?")
    label = pt.get("text", "")
    return html.Div([
        html.Span(f"📍 Clicked: {label or f'trace #{curve_num}'}", style={"font-weight": "bold"}),
        html.Span(f" | x={x}, y={y}", style={"color": "#888"}),
        html.Span(
            " | 💡 Run Grid Match to see hole assignments",
            style={"color": "#aaa", "font-style": "italic"},
        ),
    ])


# ═══════════════════════════════════════════════════════════
# Callback: Match Table content
# ═══════════════════════════════════════════════════════════
@callback(
    Output("div-match-table", "children", allow_duplicate=True),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def on_match_table_tab(tab):
    if tab != "tab-table":
        return no_update
    return _build_match_table_html()


def _build_grid_match_table_html() -> html.Div:
    """Inline match table for the Grid Match tab."""
    return html.Div([
        html.H6("Current Cluster → Hole Assignments", style={"margin-bottom": "8px"}),
        _build_match_table_html(),
    ])


def _resolve_grid_click(click_data: dict | None) -> Optional[dict[str, Any]]:
    """Resolve a Grid Match click into a cluster or hole entity."""
    if not click_data or not click_data.get("points"):
        return None

    pt = click_data["points"][0]
    curve_num = pt.get("curveNumber")
    point_num = pt.get("pointNumber", pt.get("pointIndex"))
    session = get_session()

    try:
        lookup = session.grid_trace_lookup.get(int(curve_num), [])
        return lookup[int(point_num)]
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _apply_manual_match_edit(foil: int, cluster: int, new_row: int, new_col: int) -> tuple[bool, str]:
    """Apply or replace a manual cluster→hole assignment edit."""
    session = get_session()
    if session.cluster_hole_map is None:
        return False, "❌ Run Grid Match first"

    cmap = session.cluster_hole_map
    mask = (cmap["foil_position"] == foil) & (cmap["cluster"] == cluster)
    if not mask.any():
        return False, f"❌ Cluster F{foil} C{cluster} not found"

    match_row = cmap[mask].iloc[0]
    original_row = int(match_row["hole_row"])
    original_col = int(match_row["hole_col"])

    replaced = False
    for idx, edit in enumerate(session.manual_edits):
        if edit["foil"] == foil and edit["cluster"] == cluster:
            original_row = int(edit["new_row"])
            original_col = int(edit["new_col"])
            session.manual_edits.pop(idx)
            replaced = True
            break

    session.push_edit(foil, cluster, original_row, original_col, int(new_row), int(new_col))
    verb = "Updated" if replaced else "Assigned"
    return True, f"✅ {verb} F{foil} C{cluster} → (r={new_row}, c={new_col})"


@callback(
    Output("div-grid-match-table", "children"),
    Output("grid-edit-info", "children"),
    Output("grid-edit-selection", "children"),
    Output("store-grid-edit-selection", "data"),
    Output("btn-grid-apply-match", "disabled"),
    Output("btn-grid-undo-edit", "disabled"),
    Output("btn-grid-redo-edit", "disabled"),
    Input("main-tabs", "value"),
    Input("graph-grid-match", "clickData"),
    Input("btn-grid-clear-selection", "n_clicks"),
    State("store-grid-edit-selection", "data"),
    prevent_initial_call=True,
)
def on_grid_match_interaction(tab, click_data, clear_clicks, selection_data):
    """Initialize and update the inline Grid Match editor."""
    session = get_session()
    if tab != "tab-match":
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update

    table_html = _build_grid_match_table_html()
    info = "Click a cluster center, then click a mechanical hole cross to reassign."
    empty_text = "No pending reassignment selected."

    if session.cluster_hole_map is None:
        return html.Div("Run Grid Match first.", style={"color": "#888"}), info, empty_text, {}, True, True, True

    if ctx.triggered_id == "main-tabs" and not click_data:
        return table_html, info, empty_text, {}, True, len(session.undo_stack) == 0, len(session.redo_stack) == 0

    if ctx.triggered_id == "btn-grid-clear-selection":
        session.store_selection_state = None
        return table_html, info, empty_text, {}, True, len(session.undo_stack) == 0, len(session.redo_stack) == 0

    selection = selection_data or {}
    if ctx.triggered_id == "graph-grid-match" and click_data:
        entity = _resolve_grid_click(click_data)
        if entity is None:
            return table_html, "⚠️ Click a cluster center or a mechanical hole cross.", empty_text, selection, not (selection.get("cluster") and selection.get("hole")), len(session.undo_stack) == 0, len(session.redo_stack) == 0
        if entity["kind"] == "cluster":
            selection["cluster"] = entity
            selection.pop("hole", None)  # reset hole when new cluster is picked
        elif entity["kind"] == "hole":
            selection["hole"] = entity
        session.store_selection_state = selection

    cluster_sel = selection.get("cluster") if isinstance(selection, dict) else None
    hole_sel = selection.get("hole") if isinstance(selection, dict) else None
    summary = []
    summary.append(
        f"Cluster: F{cluster_sel['foil']} C{cluster_sel['cluster']}" if cluster_sel else "Cluster: not selected"
    )
    summary.append(
        f"Hole: F{hole_sel['foil']} (r={hole_sel['hole_row']}, c={hole_sel['hole_col']})" if hole_sel else "Hole: not selected"
    )
    can_apply = not (cluster_sel and hole_sel)
    return table_html, info, " | ".join(summary), selection, can_apply, len(session.undo_stack) == 0, len(session.redo_stack) == 0


@callback(
    Output("graph-grid-match", "figure", allow_duplicate=True),
    Output("div-grid-match-table", "children", allow_duplicate=True),
    Output("div-match-table", "children", allow_duplicate=True),
    Output("grid-edit-info", "children", allow_duplicate=True),
    Output("grid-edit-selection", "children", allow_duplicate=True),
    Output("store-grid-edit-selection", "data", allow_duplicate=True),
    Output("btn-grid-apply-match", "disabled", allow_duplicate=True),
    Output("btn-grid-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-grid-redo-edit", "disabled", allow_duplicate=True),
    Output("edit-status-bar", "children", allow_duplicate=True),
    Input("btn-grid-apply-match", "n_clicks"),
    State("store-grid-edit-selection", "data"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_apply_grid_match_edit(n_clicks, selection, foil_filter):
    """Apply interactive cluster→hole reassignment from the Grid Match tab."""
    session = get_session()
    if not selection:
        msg = "⚠️ Nothing selected"
        return no_update, no_update, no_update, msg, no_update, {}, True, len(session.undo_stack) == 0, len(session.redo_stack) == 0, msg

    cluster_sel = selection.get("cluster")
    hole_sel = selection.get("hole")
    if not cluster_sel or not hole_sel:
        msg = "⚠️ Select one cluster center and one mechanical hole first."
        return no_update, no_update, no_update, msg, no_update, selection, True, len(session.undo_stack) == 0, len(session.redo_stack) == 0, msg

    if int(cluster_sel["foil"]) != int(hole_sel["foil"]):
        msg = "⚠️ Cluster and mechanical hole must belong to the same foil."
        return no_update, no_update, no_update, msg, no_update, selection, True, len(session.undo_stack) == 0, len(session.redo_stack) == 0, msg

    ok, msg = _apply_manual_match_edit(
        int(cluster_sel["foil"]),
        int(cluster_sel["cluster"]),
        int(hole_sel["hole_row"]),
        int(hole_sel["hole_col"]),
    )
    if not ok:
        return no_update, no_update, no_update, msg, no_update, selection, True, len(session.undo_stack) == 0, len(session.redo_stack) == 0, msg

    fig = _build_grid_match_figure(
        session.clustered_results,
        session.cluster_hole_map,
        session.hole_design,
        foil_filter=foil_filter,
    )
    return (
        fig,
        _build_grid_match_table_html(),
        _build_match_table_html(),
        msg,
        "No pending reassignment selected.",
        {},
        True,
        len(session.undo_stack) == 0,
        len(session.redo_stack) == 0,
        msg,
    )


# ═══════════════════════════════════════════════════════════
# Callback: Lasso / Box select → dual-mode:
#   Edit mode ON  → show action bar with selection context
#   Edit mode OFF → HDBSCAN re-cluster region (legacy)
# ═══════════════════════════════════════════════════════════
@callback(
    Output("cluster-status", "children", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("div-action-bar", "style", allow_duplicate=True),
    Output("action-bar-info", "children", allow_duplicate=True),
    Output("dropdown-target-cluster", "options", allow_duplicate=True),
    Output("store-selection-state", "data", allow_duplicate=True),
    Input("graph-sieve-clusters", "selectedData"),
    State("slider-min-cluster-size", "value"),
    State("slider-min-samples", "value"),
    State("dropdown-cluster-method", "value"),
    State("slider-max-cluster-size", "value"),
    State("checkbox-edit-clusters", "value"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_lasso_select(selectedData, min_cluster_size, min_samples, method,
                     max_cluster_size, edit_mode, foil_filter):
    """Handle lasso selection in both modes."""
    if selectedData is None or not selectedData.get("points"):
        return no_update, no_update, no_update, no_update, no_update, no_update

    session = get_session()
    if not session.has_data():
        return no_update, no_update, no_update, no_update, no_update, no_update

    pts = selectedData["points"]
    poly_x = np.array([p["x"] for p in pts])
    poly_y = np.array([p["y"] for p in pts])

    # ═══ Edit Mode: Selection → Action Bar ═══
    if edit_mode:
        if not session.has_clusters():
            return ("⚠️ Run clustering first", no_update,
                    {"display": "none"}, no_update, [], {})

        # Extract indices via customdata / trace_point_lookup
        indices_set: set = set()
        clusters_involved: set = set()
        foil_involved: set = set()

        for pt in pts:
            cd = pt.get("customdata")
            resolved = False
            if cd is not None:
                try:
                    if isinstance(cd, dict):
                        idx = int(cd.get(0, cd.get("0")))
                        cid = int(cd.get(1, cd.get("1")))
                        fp = int(cd.get(2, cd.get("2")))
                    else:
                        idx = int(cd[0])
                        cid = int(cd[1])
                        fp = int(cd[2])
                    indices_set.add(idx)
                    clusters_involved.add(cid)
                    foil_involved.add(fp)
                    resolved = True
                except (TypeError, ValueError, KeyError, IndexError):
                    resolved = False

            if not resolved:
                curve_num = pt.get("curveNumber")
                point_num = pt.get("pointNumber", pt.get("pointIndex"))
                try:
                    lookup = session.trace_point_lookup.get(int(curve_num), [])
                    idx, fp, cid = lookup[int(point_num)]
                    indices_set.add(int(idx))
                    clusters_involved.add(int(cid))
                    foil_involved.add(int(fp))
                except (TypeError, ValueError, KeyError, IndexError):
                    continue

        if len(indices_set) < 3:
            session.selected_event_keys = set()
            return (f"⚠️ Only {len(indices_set)} events selected (need ≥3)", no_update,
                    {"display": "none"}, no_update, [], {})

        # Build selected_event_keys for figure highlighting
        selected_keys: set = set()
        for fp in foil_involved:
            if fp in session.clustered_results:
                df_f = session.clustered_results[fp]["df"]
                for idx in indices_set:
                    if idx in df_f.index:
                        selected_keys.add((idx, fp))

        session.selected_event_keys = selected_keys

        # Build target cluster dropdown options
        target_options = []
        if foil_filter != "all":
            fp = int(foil_filter)
            if fp in session.clustered_results:
                df_f = session.clustered_results[fp]["df"]
                valid_clusters = sorted(c for c in df_f["cluster"].unique() if c >= 0)
                # Filter out clusters that ALL selected points already belong to
                # (moving everything to the same cluster is pointless)
                target_options = [
                    {"label": f"F{fp} Cluster {c}", "value": f"{fp}:{c}"}
                    for c in valid_clusters
                ]
        else:
            for fp in sorted(foil_involved):
                if fp in session.clustered_results:
                    df_f = session.clustered_results[fp]["df"]
                    for c in sorted(c for c in df_f["cluster"].unique() if c >= 0):
                        target_options.append(
                            {"label": f"F{fp} Cluster {c}", "value": f"{fp}:{c}"}
                        )

        n_events = len(indices_set)
        n_clusters = len(clusters_involved)
        n_foils = len(foil_involved)
        info_text = f"📊 {n_events} events | {n_clusters} cluster(s) | {n_foils} foil(s)"

        # Store selection state
        sel_state = {
            "indices": list(indices_set),
            "foils": list(foil_involved),
            "clusters_involved": list(clusters_involved),
        }

        # Build figure with highlights
        results_f, positions_f = _filtered_results_for_display(foil_filter=str(foil_filter))
        fig = _build_clustered_scatter(results_f, positions_f)

        return (f"✏️ Edit mode: {info_text}", fig,
                {"display": "flex", "padding": "6px 16px", "border-top": "1px solid #eee",
                 "background": "#fff8e1", "align-items": "center", "flex-wrap": "wrap"},
                info_text, target_options, sel_state)

    # ═══ Legacy Mode: HDBSCAN re-cluster ═══
    session.selected_event_keys = set()
    if len(poly_x) < 10:
        return (f"⚠️ Only {len(poly_x)} points selected (need ≥10)", no_update,
                {"display": "none"}, no_update, [], {})

    df = session.raw_df.copy()
    mask = np.zeros(len(df), dtype=bool)
    for sx, sy in zip(poly_x, poly_y):
        dist = np.sqrt((df["sieve_x"] - sx) ** 2 + (df["sieve_y"] - sy) ** 2)
        mask |= (dist < 0.05)

    selected_df = df[mask].copy()
    if len(selected_df) < 10:
        return (f"⚠️ Only {len(selected_df)} raw events matched (need ≥10)", no_update,
                {"display": "none"}, no_update, [], {})

    cfg = HDBSCANConfig(
        min_cluster_size_range=(max(3, int(min_cluster_size) // 2), int(min_cluster_size) * 3),
        min_samples_range=(max(2, int(min_samples) // 2), int(min_samples) * 3),
        target_clusters=(3, 30),
        cluster_selection_method=method,
        max_cluster_size=float(max_cluster_size),
        distance_threshold=0.3,
        drop_noise=True,
    )
    selected_df_clustered, params, n = soc.auto_hdbscan_clustering(selected_df, config=cfg, verbose=False)

    status = f"🔍 Lasso re-cluster: {len(selected_df)} events → {n} clusters in selected region"
    fig = _build_sieve_scatter(df, title=f"Sieve Plane — Selected region ({len(selected_df)} events, {n} clusters)")

    return status, fig, {"display": "none"}, no_update, [], {}


# ═══════════════════════════════════════════════════════════
# Helper: apply cluster action to selected events
# ═══════════════════════════════════════════════════════════

def _apply_cluster_edit(session, new_cluster: int) -> int:
    """Reassign all selected events to *new_cluster* (-1 = noise).

    Returns number of events reassigned.
    """
    sel_state = session.store_selection_state or {}
    indices = sel_state.get("indices", [])
    foils = sel_state.get("foils", [])
    if not indices:
        return 0

    reassigned = 0
    old_clusters_snapshot = {}
    for fp in foils:
        if fp not in session.clustered_results:
            continue
        df_f = session.clustered_results[fp]["df"]
        for idx in indices:
            if idx in df_f.index:
                old_cid = df_f.at[idx, "cluster"]
                if old_cid != new_cluster:
                    old_clusters_snapshot[(idx, fp)] = old_cid
                    df_f.at[idx, "cluster"] = new_cluster
                    reassigned += 1

    # Recompute cluster centers for affected foils
    for fp in foils:
        if fp not in session.clustered_results:
            continue
        df_f = session.clustered_results[fp]["df"]
        sig = df_f[df_f["cluster"] != -1]
        centers = sig.groupby("cluster").agg(
            cluster_center_x=("sieve_x", "median"),
            cluster_center_y=("sieve_y", "median"),
        ).reset_index()
        for col_name in ["cluster_center_x", "cluster_center_y"]:
            df_f[col_name] = np.nan
            for _, cr in centers.iterrows():
                cid = cr["cluster"]
                df_f.loc[df_f["cluster"] == cid, col_name] = cr[col_name]
        session.clustered_results[fp]["n_clusters"] = int(df_f["cluster"].max()) + 1

    # Record for undo
    if reassigned > 0:
        session.push_undo(
            "cluster_edit",
            {
                "foils": foils,
                "indices": list(indices),
                "old_clusters": old_clusters_snapshot,
                "new_cluster": new_cluster,
            },
        )

    # Invalidate downstream
    session.hole_design = None
    session.cluster_hole_map = None

    return reassigned


# ═══════════════════════════════════════════════════════════
# Callback: ➕ New Cluster action
# ═══════════════════════════════════════════════════════════
@callback(
    Output("cluster-status", "children", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("div-action-bar", "style", allow_duplicate=True),
    Output("store-selection-state", "data", allow_duplicate=True),
    Input("btn-action-new-cluster", "n_clicks"),
    State("store-selection-state", "data"),
    State("checkbox-keep-selection", "value"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_action_new_cluster(n_clicks, sel_state, keep_selection, foil_filter):
    """Create a new cluster from selected events."""
    session = get_session()
    if not sel_state or not sel_state.get("indices"):
        return no_update, no_update, no_update, no_update

    # Determine new cluster ID
    foils = sel_state.get("foils", [])
    new_cid = 0
    for fp in foils:
        if fp in session.clustered_results:
            max_c = int(session.clustered_results[fp]["df"]["cluster"].max())
            new_cid = max(new_cid, max_c + 1)

    # Temporarily store sel_state for _apply_cluster_edit
    session.store_selection_state = sel_state
    n = _apply_cluster_edit(session, new_cid)
    session.store_selection_state = None

    if not keep_selection:
        session.selected_event_keys = set()
        action_bar_style = {"display": "none"}
    else:
        action_bar_style = no_update

    results_f, positions_f = _filtered_results_for_display()
    fig = _build_clustered_scatter(results_f, positions_f) if results_f else no_update
    status = f"➕ New cluster {new_cid}: {n} events reassigned on foil(s) {foils}"
    return status, fig, action_bar_style, {} if not keep_selection else no_update


# ═══════════════════════════════════════════════════════════
# Callback: 📦 Move to Cluster action
# ═══════════════════════════════════════════════════════════
@callback(
    Output("cluster-status", "children", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("div-action-bar", "style", allow_duplicate=True),
    Output("store-selection-state", "data", allow_duplicate=True),
    Input("btn-action-move-cluster", "n_clicks"),
    State("store-selection-state", "data"),
    State("dropdown-target-cluster", "value"),
    State("checkbox-keep-selection", "value"),
    prevent_initial_call=True,
)
def on_action_move_cluster(n_clicks, sel_state, target_value, keep_selection):
    """Move selected events to a target cluster."""
    if not sel_state or not target_value:
        return no_update, no_update, no_update, no_update

    try:
        parts = target_value.split(":")
        target_fp = int(parts[0])
        target_cid = int(parts[1])
    except (ValueError, IndexError):
        return "⚠️ Invalid target cluster", no_update, no_update, no_update

    session = get_session()
    session.store_selection_state = sel_state
    n = _apply_cluster_edit(session, target_cid)
    session.store_selection_state = None

    if not keep_selection:
        session.selected_event_keys = set()
        action_bar_style = {"display": "none"}
    else:
        action_bar_style = no_update

    results_f, positions_f = _filtered_results_for_display()
    fig = _build_clustered_scatter(results_f, positions_f) if results_f else no_update
    status = f"📦 Moved {n} events to Foil {target_fp} Cluster {target_cid}"
    return status, fig, action_bar_style, {} if not keep_selection else no_update


# ═══════════════════════════════════════════════════════════
# Callback: 🗑️ Mark Noise action
# ═══════════════════════════════════════════════════════════
@callback(
    Output("cluster-status", "children", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("div-action-bar", "style", allow_duplicate=True),
    Output("store-selection-state", "data", allow_duplicate=True),
    Input("btn-action-mark-noise", "n_clicks"),
    State("store-selection-state", "data"),
    State("checkbox-keep-selection", "value"),
    prevent_initial_call=True,
)
def on_action_mark_noise(n_clicks, sel_state, keep_selection):
    """Mark selected events as noise (cluster = -1)."""
    session = get_session()
    if not sel_state or not sel_state.get("indices"):
        return no_update, no_update, no_update, no_update

    session.store_selection_state = sel_state
    n = _apply_cluster_edit(session, -1)
    session.store_selection_state = None

    if not keep_selection:
        session.selected_event_keys = set()
        action_bar_style = {"display": "none"}
    else:
        action_bar_style = no_update

    results_f, positions_f = _filtered_results_for_display()
    fig = _build_clustered_scatter(results_f, positions_f) if results_f else no_update
    status = f"🗑️ Marked {n} events as noise"
    return status, fig, action_bar_style, {} if not keep_selection else no_update


# ═══════════════════════════════════════════════════════════
# Callback: ✕ Cancel selection
# ═══════════════════════════════════════════════════════════
@callback(
    Output("div-action-bar", "style", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("cluster-status", "children", allow_duplicate=True),
    Output("store-selection-state", "data", allow_duplicate=True),
    Input("btn-action-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def on_cancel_selection(n_clicks):
    """Clear the current selection and hide action bar."""
    session = get_session()
    session.selected_event_keys = set()
    session.store_selection_state = None

    status = "✕ Selection cancelled"
    fig = no_update
    if session.has_clusters():
        results_f, positions_f = _filtered_results_for_display()
        if results_f:
            fig = _build_clustered_scatter(results_f, positions_f)
    elif session.has_data():
        fig = _build_sieve_scatter(session.raw_df)

    return {"display": "none"}, fig, status, {}


# ═══════════════════════════════════════════════════════════
# Callback: Toggle Edit Clusters → change dragmode
# ═══════════════════════════════════════════════════════════
@callback(
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("div-action-bar", "style", allow_duplicate=True),
    Output("store-selection-state", "data", allow_duplicate=True),
    Input("checkbox-edit-clusters", "value"),
    prevent_initial_call=True,
)
def on_toggle_edit_clusters(edit_mode):
    """Toggle edit mode: change graph dragmode and clear selection."""
    session = get_session()
    session.edit_mode_active = bool(edit_mode)
    session.selected_event_keys = set()
    session.store_selection_state = None

    if session.has_clusters():
        results_f, positions_f = _filtered_results_for_display()
        fig = _build_clustered_scatter(results_f, positions_f) if results_f else _build_empty_figure()
    elif session.has_data():
        fig = _build_sieve_scatter(session.raw_df)
    else:
        fig = no_update

    return fig, {"display": "none"}, {}


# ═══════════════════════════════════════════════════════════
# Helper: filter DataFrame by foil selection
# ═══════════════════════════════════════════════════════════
def _filter_by_foil(df: pd.DataFrame, foil_filter: str) -> pd.DataFrame:
    """Return df subset for the selected foil."""
    if foil_filter == "all" or not foil_filter:
        return df
    return df[df["foil_position"] == int(foil_filter)].copy()


# ═══════════════════════════════════════════════════════════
# Plot builder: Heatmap (2D histogram)
# ═══════════════════════════════════════════════════════════
def _build_sieve_heatmap(df: pd.DataFrame, title: str = "Sieve Plane") -> go.Figure:
    """2D histogram heatmap of sieve plane events, per foil or all."""
    session = get_session()
    fig = go.Figure()
    for foil_pos in sorted(df["foil_position"].dropna().unique()):
        if foil_pos == -1:
            continue
        foil_pos = int(foil_pos)
        sub = df[df["foil_position"] == foil_pos]
        if len(sub) < 10:
            continue
        fig.add_trace(go.Histogram2d(
            x=sub["sieve_x"], y=sub["sieve_y"],
            xbins={"start": -20, "end": 20, "size": 0.25},
            ybins={"start": -20, "end": 20, "size": 0.25},
            colorscale=[[0, "white"], [1, _FOIL_COLORS.get(foil_pos, "#999")]],
            showscale=False,
            name=f"Foil {foil_pos} ({len(sub):,})",
        ))
    fig.update_layout(
        title=title, template=_PLOTLY_TEMPLATE,
        xaxis_title="Sieve X (cm)", yaxis_title="Sieve Y (cm)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        legend={"font": {"size": 10}},
        dragmode="lasso" if session.edit_mode_active else "zoom",
        clickmode="event+select",
        uirevision=f"sieve-heatmap-{int(session.edit_mode_active)}",
    )
    return fig


# ═══════════════════════════════════════════════════════════
# Callback: Viz mode / Foil filter → update Sieve Clusters graph
# ═══════════════════════════════════════════════════════════
@callback(
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Input("dropdown-viz-mode", "value"),
    Input("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_viz_change(viz_mode: str, foil_filter: str):
    """Rebuild sieve plot when viz mode or foil filter changes."""
    session = get_session()
    session.current_foil_filter = str(foil_filter)
    if not session.has_data():
        return no_update

    df = _filter_by_foil(session.raw_df, foil_filter)
    foil_label = f"Foil {foil_filter}" if foil_filter != "all" else "All Foils"

    if session.has_clusters() and viz_mode == "scatter":
        # Show clustered scatter if available
        results_filtered: dict[int, dict[str, Any]] = {}
        if foil_filter == "all":
            results_filtered = session.clustered_results or {}
        else:
            fp = int(foil_filter)
            if fp in (session.clustered_results or {}):
                results_filtered = {fp: session.clustered_results[fp]}
        if results_filtered:
            return _build_clustered_scatter(results_filtered, list(results_filtered.keys()))
        # Fall through to raw scatter
        return _build_sieve_scatter(df, title=f"Sieve Plane — {foil_label} ({len(df):,} events)")

    if viz_mode == "heatmap":
        if session.edit_mode_active and session.has_clusters():
            # Overlay faint scatter on heatmap for lasso selection
            return _build_heatmap_with_selection_clusters(
                df, session.clustered_results, session.foil_positions,
                title=f"Sieve Plane — {foil_label} ({len(df):,} events)",
            )
        return _build_sieve_heatmap(df, title=f"Sieve Plane Heatmap — {foil_label} ({len(df):,} events)")

    # Default: scatter
    return _build_sieve_scatter(df, title=f"Sieve Plane — {foil_label} ({len(df):,} events)")


def _build_heatmap_with_selection_clusters(
    df: pd.DataFrame,
    results: dict[int, dict[str, Any]],
    foil_positions: list[int],
    title: str = "Sieve Plane",
) -> go.Figure:
    """Heatmap with overlaid transparent scatter for lasso selection in edit mode."""
    session = get_session()

    # Build base heatmap
    fig = go.Figure()
    for foil_pos in sorted(df["foil_position"].dropna().unique()):
        if foil_pos == -1:
            continue
        foil_pos = int(foil_pos)
        sub = df[df["foil_position"] == foil_pos]
        if len(sub) < 10:
            continue
        fig.add_trace(go.Histogram2d(
            x=sub["sieve_x"], y=sub["sieve_y"],
            xbins={"start": -20, "end": 20, "size": 0.25},
            ybins={"start": -20, "end": 20, "size": 0.25},
            colorscale=[[0, "white"], [1, _FOIL_COLORS.get(foil_pos, "#999")]],
            showscale=False,
            name=f"Foil {foil_pos} ({len(sub):,})",
        ))

    # Overlay light scatter for selection surface
    selected_keys = session.selected_event_keys if session.edit_mode_active else set()
    for fp in foil_positions:
        if fp not in results:
            continue
        df_r = results[fp]["df"]
        sig_full = df_r[df_r["cluster"] != -1]
        if len(sig_full) == 0:
            continue

        foil_selected_indices = {int(idx) for (idx, f) in selected_keys if int(f) == int(fp)}
        sig = _sample_cluster_points_for_display(
            sig_full,
            max_points=100000,
            min_points_per_cluster=8,
            random_state=42,
            must_include_indices=foil_selected_indices,
        )

        for cid in sorted(sig["cluster"].unique()):
            sub = sig[sig["cluster"] == cid]
            cd = np.column_stack([
                sub.index.values,
                np.full(len(sub), cid),
                np.full(len(sub), fp),
            ])
            sel_pts = None
            if selected_keys:
                sel_mask = np.array([
                    (sub.index.values[i], fp) in selected_keys
                    for i in range(len(sub))
                ])
                if sel_mask.any():
                    sel_pts = np.where(sel_mask)[0].tolist()

            fig.add_trace(go.Scatter(
                x=sub["sieve_x"], y=sub["sieve_y"],
                mode="markers",
                marker={"size": 2.5, "color": _cluster_display_color(int(fp), int(cid)),
                        "opacity": 0.35},
                selected_marker={"size": 9, "color": "#FFD700", "opacity": 1.0},
                unselected_marker={"opacity": 0.18},
                selectedpoints=sel_pts,
                customdata=cd,
                name=f"F{fp} C{cid}",
                legendgroup=f"foil{fp}",
                legendgrouptitle_text=f"Foil {fp}",
                showlegend=False,
            ))

    fig.update_layout(
        title=title, template=_PLOTLY_TEMPLATE,
        xaxis_title="Sieve X (cm)", yaxis_title="Sieve Y (cm)",
        margin={"l": 50, "r": 20, "t": 50, "b": 50},
        legend={"font": {"size": 10}},
        dragmode="lasso" if session.edit_mode_active else "zoom",
        clickmode="event+select",
        uirevision=f"sieve-heatmap-sel-{int(session.edit_mode_active)}",
    )
    return fig


# ═══════════════════════════════════════════════════════════
# Callback: Foil filter → update Grid Match graph (if available)
# ═══════════════════════════════════════════════════════════
@callback(
    Output("graph-grid-match", "figure", allow_duplicate=True),
    Input("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_foil_filter_grid_match(foil_filter: str):
    """Rebuild grid match figure when foil filter changes."""
    session = get_session()
    session.current_foil_filter = str(foil_filter)
    if session.cluster_hole_map is None or session.clustered_results is None:
        return no_update
    return _build_grid_match_figure(
        session.clustered_results, session.cluster_hole_map, session.hole_design,
        foil_filter=foil_filter,
    )


# ═══════════════════════════════════════════════════════════
# Phase 4: Manual Hole Editing
# ═══════════════════════════════════════════════════════════

# ── Shared helper: build match table HTML (used by tab switch
#    and after edit/undo/redo) ──────────────────────────────

def _build_match_table_html() -> html.Div:
    """Build the match table + undo/redo bar HTML from session state."""
    session = get_session()
    if session.cluster_hole_map is None:
        return html.Div("Run Grid Match first to see the match table.",
                        style={"padding": "20px", "color": "#888"})

    cmap = session.cluster_hole_map.copy()
    edited_clusters = {(e["foil"], e["cluster"]) for e in session.manual_edits}

    # Apply manual edits to the display
    for edit in session.manual_edits:
        mask = (cmap["foil_position"] == edit["foil"]) & (cmap["cluster"] == edit["cluster"])
        if mask.any():
            cmap.loc[mask, "hole_row"] = edit["new_row"]
            cmap.loc[mask, "hole_col"] = edit["new_col"]

    rows = []
    for _, row in cmap.sort_values(["foil_position", "cluster"]).iterrows():
        foil = int(row["foil_position"])
        cluster = int(row["cluster"])
        is_edited = (foil, cluster) in edited_clusters
        row_style = {"background": "#fff3e0"} if is_edited else {}
        edit_btn = dbc.Button(
            "✏️" if not is_edited else "🔧",
            id={"type": "edit-btn", "foil": foil, "cluster": cluster},
            size="sm", color="link" if not is_edited else "warning",
            style={"padding": "0 4px"},
        )
        rows.append(html.Tr([
            html.Td(foil),
            html.Td(cluster),
            html.Td(int(row["hole_row"]), style=row_style),
            html.Td(int(row["hole_col"]), style=row_style),
            html.Td(f"{row['match_distance_cm']:.4f}"),
            html.Td(f"{row.get('effective_match_cost_cm', 0):.4f}"),
            html.Td(edit_btn),
        ]))

    table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Foil"), html.Th("Cluster"), html.Th("Row"), html.Th("Col"),
            html.Th("Match cm"), html.Th("Eff. Cost cm"), html.Th("Edit"),
        ])),
        html.Tbody(rows),
    ], bordered=True, striped=True, hover=True, size="sm", style={"font-size": "12px"})

    dup = session.summary.get("duplicate_mechanical_holes", "?")
    n_edits = len(session.manual_edits)
    return html.Div([
        html.H5("Cluster → Mechanical Hole Assignments"),
        html.P([
            f"Duplicate assignments: {dup}",
            html.Span(f"  |  Manual edits: {n_edits}",
                      style={"color": "#e65100" if n_edits > 0 else "#080"}),
        ]),
        table,
    ])


# ── Callback: ✏️ button click → open edit modal ───────────

@callback(
    Output("modal-edit-hole", "is_open", allow_duplicate=True),
    Output("store-edit-context", "data", allow_duplicate=True),
    Output("edit-modal-info", "children", allow_duplicate=True),
    Output("edit-input-row", "value", allow_duplicate=True),
    Output("edit-input-col", "value", allow_duplicate=True),
    Output("edit-modal-warning", "children", allow_duplicate=True),
    Input({"type": "edit-btn", "foil": ALL, "cluster": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_edit_button_click(n_clicks_list):
    """Open the edit modal when an ✏️ button is clicked."""
    if not ctx.triggered_id or not any(c is not None for c in (n_clicks_list or [])):
        return no_update, no_update, no_update, no_update, no_update, no_update

    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or triggered.get("type") != "edit-btn":
        return no_update, no_update, no_update, no_update, no_update, no_update

    foil = int(triggered["foil"])
    cluster = int(triggered["cluster"])

    session = get_session()
    if session.cluster_hole_map is None:
        return no_update, no_update, no_update, no_update, no_update, no_update

    cmap = session.cluster_hole_map
    mask = (cmap["foil_position"] == foil) & (cmap["cluster"] == cluster)
    if not mask.any():
        return no_update, no_update, no_update, no_update, no_update, no_update

    match_row = cmap[mask].iloc[0]
    current_row = int(match_row["hole_row"])
    current_col = int(match_row["hole_col"])
    match_dist = match_row["match_distance_cm"]

    # Check for any manual edits that override this
    for edit in session.manual_edits:
        if edit["foil"] == foil and edit["cluster"] == cluster:
            current_row = edit["new_row"]
            current_col = edit["new_col"]
            break

    info = f"Foil {foil}, Cluster {cluster} — current: (r={current_row}, c={current_col}), "
    info += f"match distance: {match_dist:.4f} cm"

    # Check if target hole already assigned to another cluster
    warning = ""
    for _, r in cmap.iterrows():
        if int(r["foil_position"]) == foil and int(r["cluster"]) != cluster:
            if int(r["hole_row"]) == current_row and int(r["hole_col"]) == current_col:
                warning = f"⚠️ Hole (r={current_row}, c={current_col}) is also "
                warning += f"assigned to cluster {int(r['cluster'])}"
                break

    context = {"foil": foil, "cluster": cluster,
               "original_row": int(match_row["hole_row"]),
               "original_col": int(match_row["hole_col"])}

    return True, context, info, current_row, current_col, warning


# ── Callback: Apply edit ──────────────────────────────────

@callback(
    Output("modal-edit-hole", "is_open", allow_duplicate=True),
    Output("div-match-table", "children", allow_duplicate=True),
    Output("store-edit-context", "data", allow_duplicate=True),
    Output("edit-status-bar", "children", allow_duplicate=True),
    Output("btn-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-redo-edit", "disabled", allow_duplicate=True),
    Input("btn-edit-apply", "n_clicks"),
    State("store-edit-context", "data"),
    State("edit-input-row", "value"),
    State("edit-input-col", "value"),
    prevent_initial_call=True,
)
def on_edit_apply(n_clicks, context, new_row, new_col):
    """Apply the manual hole reassignment."""
    if not context:
        return False, no_update, {}, no_update, no_update, no_update

    session = get_session()
    foil = context["foil"]
    cluster = context["cluster"]
    original_row = context["original_row"]
    original_col = context["original_col"]

    # Validate
    try:
        new_row = int(new_row)
        new_col = int(new_col)
    except (TypeError, ValueError):
        return True, no_update, context, "❌ Invalid row/col values", no_update, no_update

    # Record the edit
    session.push_edit(foil, cluster, original_row, original_col, new_row, new_col)

    # Rebuild table
    table_html = _build_match_table_html()

    has_undo = len(session.undo_stack) > 0
    has_redo = len(session.redo_stack) > 0

    status = f"✅ Edited cluster {cluster} (foil {foil}): "
    status += f"({original_row},{original_col}) → ({new_row},{new_col})"

    return False, table_html, {}, status, not has_undo, not has_redo


# ── Callback: Cancel edit ─────────────────────────────────

@callback(
    Output("modal-edit-hole", "is_open", allow_duplicate=True),
    Output("store-edit-context", "data", allow_duplicate=True),
    Input("btn-edit-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def on_edit_cancel(n_clicks):
    """Close the edit modal without changes."""
    return False, {}


# ── Callback: Undo last edit ──────────────────────────────

@callback(
    Output("div-match-table", "children", allow_duplicate=True),
    Output("edit-status-bar", "children", allow_duplicate=True),
    Output("btn-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-redo-edit", "disabled", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("graph-grid-match", "figure", allow_duplicate=True),
    Output("div-grid-match-table", "children", allow_duplicate=True),
    Output("btn-grid-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-grid-redo-edit", "disabled", allow_duplicate=True),
    Input("btn-undo-edit", "n_clicks"),
    Input("btn-grid-undo-edit", "n_clicks"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_undo_edit(n_clicks, grid_n_clicks, foil_filter):
    """Revert the last manual edit (hole edit or cluster edit)."""
    session = get_session()
    entry = session.undo_last_edit()
    if entry is None:
        return (no_update, "⚠️ Nothing to undo", True, len(session.redo_stack) == 0,
            no_update, no_update, no_update, True, len(session.redo_stack) == 0)

    edit_desc = entry.get("description", "")
    edit_data = entry.get("before", {})

    if edit_desc == "edit" and edit_data:
        # Remove matching edit from manual_edits
        session.manual_edits = [
            e for e in session.manual_edits
            if not (e["foil"] == edit_data.get("foil")
                    and e["cluster"] == edit_data.get("cluster"))
        ]

    # Handle cluster_edit undo: restore old clusters
    if edit_desc == "cluster_edit" and edit_data:
        foils = edit_data.get("foils", [])
        old_clusters = edit_data.get("old_clusters", {})
        indices = edit_data.get("indices", [])
        for fp in foils:
            if fp not in session.clustered_results:
                continue
            df_f = session.clustered_results[fp]["df"]
            for idx in indices:
                key = (idx, int(fp))
                if key in old_clusters and idx in df_f.index:
                    df_f.at[idx, "cluster"] = old_clusters[key]
            sig = df_f[df_f["cluster"] != -1]
            if len(sig) > 0:
                centers = sig.groupby("cluster").agg(
                    cluster_center_x=("sieve_x", "median"),
                    cluster_center_y=("sieve_y", "median"),
                ).reset_index()
                for cn in ["cluster_center_x", "cluster_center_y"]:
                    df_f[cn] = np.nan
                    for _, cr in centers.iterrows():
                        df_f.loc[df_f["cluster"] == cr["cluster"], cn] = cr[cn]
            session.clustered_results[fp]["n_clusters"] = int(df_f["cluster"].max()) + 1
        session.hole_design = None
        session.cluster_hole_map = None

    table_html = _build_match_table_html()
    grid_table_html = _build_grid_match_table_html()
    has_undo = len(session.undo_stack) > 0
    has_redo = len(session.redo_stack) > 0

    # Rebuild sieve figure if cluster edit was undone
    sieve_fig = no_update
    grid_fig = no_update
    if edit_desc == "cluster_edit" and session.has_clusters():
        results_f, positions_f = _filtered_results_for_display()
        if results_f:
            sieve_fig = _build_clustered_scatter(results_f, positions_f)
    if session.cluster_hole_map is not None and session.clustered_results is not None:
        grid_fig = _build_grid_match_figure(
            session.clustered_results, session.cluster_hole_map, session.hole_design,
            foil_filter=foil_filter,
        )

    if edit_desc == "edit":
        ed = edit_data
        status = f"↩ Undo: cluster {ed['cluster']} (foil {ed['foil']}) "
        status += f"reverted to ({ed['old_row']},{ed['old_col']})"
    elif edit_desc == "cluster_edit":
        status = "↩ Undo: cluster edit reverted"
    else:
        status = f"↩ Undo: {edit_desc}"

    return table_html, status, not has_undo, not has_redo, sieve_fig, grid_fig, grid_table_html, not has_undo, not has_redo


# ── Callback: Redo last undone edit ───────────────────────

@callback(
    Output("div-match-table", "children", allow_duplicate=True),
    Output("edit-status-bar", "children", allow_duplicate=True),
    Output("btn-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-redo-edit", "disabled", allow_duplicate=True),
    Output("graph-sieve-clusters", "figure", allow_duplicate=True),
    Output("graph-grid-match", "figure", allow_duplicate=True),
    Output("div-grid-match-table", "children", allow_duplicate=True),
    Output("btn-grid-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-grid-redo-edit", "disabled", allow_duplicate=True),
    Input("btn-redo-edit", "n_clicks"),
    Input("btn-grid-redo-edit", "n_clicks"),
    State("dropdown-foil-filter", "value"),
    prevent_initial_call=True,
)
def on_redo_edit(n_clicks, grid_n_clicks, foil_filter):
    """Re-apply the last undone edit (hole edit or cluster edit)."""
    session = get_session()
    entry = session.redo_last_edit()
    if entry is None:
        return (no_update, "⚠️ Nothing to redo", len(session.undo_stack) == 0, True,
            no_update, no_update, no_update, len(session.undo_stack) == 0, True)

    edit_desc = entry.get("description", "")
    edit_data = entry.get("before", {})

    if edit_desc == "edit" and edit_data:
        session.manual_edits.append(edit_data)

    # Handle cluster_edit redo: re-apply the edit
    if edit_desc == "cluster_edit" and edit_data:
        foils = edit_data.get("foils", [])
        new_cluster = edit_data.get("new_cluster", -1)
        indices = edit_data.get("indices", [])
        for fp in foils:
            if fp not in session.clustered_results:
                continue
            df_f = session.clustered_results[fp]["df"]
            for idx in indices:
                if idx in df_f.index:
                    df_f.at[idx, "cluster"] = new_cluster
            sig = df_f[df_f["cluster"] != -1]
            if len(sig) > 0:
                centers = sig.groupby("cluster").agg(
                    cluster_center_x=("sieve_x", "median"),
                    cluster_center_y=("sieve_y", "median"),
                ).reset_index()
                for cn in ["cluster_center_x", "cluster_center_y"]:
                    df_f[cn] = np.nan
                    for _, cr in centers.iterrows():
                        df_f.loc[df_f["cluster"] == cr["cluster"], cn] = cr[cn]
            session.clustered_results[fp]["n_clusters"] = int(df_f["cluster"].max()) + 1
        session.hole_design = None
        session.cluster_hole_map = None

    table_html = _build_match_table_html()
    grid_table_html = _build_grid_match_table_html()
    has_undo = len(session.undo_stack) > 0
    has_redo = len(session.redo_stack) > 0

    sieve_fig = no_update
    grid_fig = no_update
    if edit_desc == "cluster_edit" and session.has_clusters():
        results_f, positions_f = _filtered_results_for_display()
        if results_f:
            sieve_fig = _build_clustered_scatter(results_f, positions_f)
    if session.cluster_hole_map is not None and session.clustered_results is not None:
        grid_fig = _build_grid_match_figure(
            session.clustered_results, session.cluster_hole_map, session.hole_design,
            foil_filter=foil_filter,
        )

    if edit_desc == "edit":
        ed = edit_data
        status = f"↪ Redo: cluster {ed['cluster']} (foil {ed['foil']}) "
        status += f"→ ({ed['new_row']},{ed['new_col']})"
    elif edit_desc == "cluster_edit":
        status = "↪ Redo: cluster edit re-applied"
    else:
        status = f"↪ Redo: {edit_desc}"

    return table_html, status, not has_undo, not has_redo, sieve_fig, grid_fig, grid_table_html, not has_undo, not has_redo


# ── Callback: Update undo/redo button state on tab switch ─

@callback(
    Output("btn-undo-edit", "disabled", allow_duplicate=True),
    Output("btn-redo-edit", "disabled", allow_duplicate=True),
    Output("edit-status-bar", "children", allow_duplicate=True),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def on_tab_switch_edit_buttons(tab):
    """Update the undo/redo button states when switching to Match Table tab."""
    if tab != "tab-table":
        return no_update, no_update, no_update

    session = get_session()
    has_undo = len(session.undo_stack) > 0
    has_redo = len(session.redo_stack) > 0

    status = ""
    if session.manual_edits:
        status = f"📝 {len(session.manual_edits)} manual edit(s) recorded. "

    return not has_undo, not has_redo, status


# ═══════════════════════════════════════════════════════════
# Data Explorer callbacks
# ═══════════════════════════════════════════════════════════

def _numeric_columns() -> list[str]:
    """Return numeric column names from the loaded DataFrame."""
    session = get_session()
    if not session.has_data():
        return []
    return [c for c in session.raw_df.columns
            if np.issubdtype(session.raw_df[c].dtype, np.number)]


def _apply_explorer_filter(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """Apply user-defined column filter rules to *df*."""
    for f in (filters or []):
        col = f.get("column", "")
        vmin = f.get("min")
        vmax = f.get("max")
        if col and col in df.columns:
            if vmin is not None and vmin != "":
                df = df[df[col] >= float(vmin)]
            if vmax is not None and vmax != "":
                df = df[df[col] <= float(vmax)]
    return df


# ── Populate variable dropdowns ────────────────────────────

@callback(
    Output("explorer-x-var", "options"),
    Output("explorer-y-var", "options"),
    Output("explorer-z-var", "options"),
    Output("explorer-line-varx", "options"),
    Output("explorer-line-vary", "options"),
    Output("explorer-cluster-filter", "options"),
    Input("main-tabs", "value"),
    prevent_initial_call=True,
)
def populate_explorer_dropdowns(tab):
    if tab != "tab-explorer":
        return no_update, no_update, no_update, no_update, no_update, no_update
    nums = _numeric_columns()
    var_opts = [{"label": c, "value": c} for c in nums]

    session = get_session()
    cluster_opts = [{"label": "All clusters", "value": None}]
    if session.has_clusters():
        for fp in session.foil_positions:
            if fp in session.clustered_results:
                df_f = session.clustered_results[fp]["df"]
                for cid in sorted(c for c in df_f["cluster"].unique() if c >= 0):
                    cluster_opts.append({"label": f"F{fp} C{cid}", "value": f"{fp}:{cid}"})
    return var_opts, var_opts, var_opts, var_opts, var_opts, cluster_opts


# ── Mode toggle ↔ show/hide controls ──────────────────────

@callback(
    Output("div-explorer-xy-vars", "style"),
    Output("div-explorer-z-var", "style"),
    Output("div-explorer-line-endpoints", "style"),
    Input("radio-explorer-mode", "value"),
)
def toggle_explorer_mode(mode):
    if mode == "heatmap":
        return {"display": "flex", "margin": "6px 0"}, {"display": "none"}, {"display": "none"}
    if mode == "scatter3d":
        return {"display": "flex", "margin": "6px 0"}, {"display": "flex", "margin": "6px 0"}, {"display": "none"}
    return {"display": "none"}, {"display": "none"}, {"display": "flex", "align-items": "center", "margin": "6px 0", "flex-wrap": "wrap"}


# ── Dynamic filter row management ─────────────────────────

@callback(
    Output("store-explorer-filters", "data"),
    Output("div-explorer-filters", "children"),
    Input("btn-explorer-add-filter", "n_clicks"),
    Input({"type": "explorer-filter-remove", "index": ALL}, "n_clicks"),
    Input({"type": "explorer-filter-col", "index": ALL}, "value"),
    Input({"type": "explorer-filter-min", "index": ALL}, "value"),
    Input({"type": "explorer-filter-max", "index": ALL}, "value"),
    State("store-explorer-filters", "data"),
    State({"type": "explorer-filter-col", "index": ALL}, "id"),
    prevent_initial_call=True,
)
def manage_explorer_filters(add_clicks, remove_clicks_list, column_values, min_values, max_values, filters, column_ids):
    """Maintain filter rows and persist each row's editable values.

    The filter store is the source consumed by ``on_generate_explorer``.  It
    must therefore be synchronised on every dropdown/number edit, not merely
    when rows are added or removed.
    """
    filters = list(filters or [])
    triggered = ctx.triggered_id

    # Remove filter row
    if isinstance(triggered, dict) and triggered.get("type") == "explorer-filter-remove":
        idx = triggered["index"]
        filters = [f for f in filters if f.get("index") != idx]

    # Add filter row
    elif triggered == "btn-explorer-add-filter":
        new_idx = max([f.get("index", -1) for f in filters] + [-1]) + 1
        filters.append({"index": new_idx, "column": "", "min": None, "max": None})

    # Persist fields edited in existing dynamic rows.  Pattern-matching inputs
    # and ids have aligned ordering, so use the row's explicit index rather
    # than relying on insertion order in the backing list.
    else:
        values_by_index = {
            item["index"]: (column_values[pos], min_values[pos], max_values[pos])
            for pos, item in enumerate(column_ids or [])
        }
        for item in filters:
            if item.get("index") in values_by_index:
                column, vmin, vmax = values_by_index[item["index"]]
                item.update({"column": column or "", "min": vmin, "max": vmax})

    nums = _numeric_columns()
    col_opts = [{"label": c, "value": c} for c in nums]

    rows = []
    for f in filters:
        rows.append(html.Div([
            dcc.Dropdown(
                id={"type": "explorer-filter-col", "index": f["index"]},
                options=col_opts, value=f.get("column", ""),
                style={"width": "150px", "font-size": "11px"},
            ),
            dcc.Input(
                id={"type": "explorer-filter-min", "index": f["index"]},
                type="number", value=f.get("min"), placeholder="min",
                style={"width": "65px", "font-size": "11px", "margin-left": "6px"},
            ),
            dcc.Input(
                id={"type": "explorer-filter-max", "index": f["index"]},
                type="number", value=f.get("max"), placeholder="max",
                style={"width": "65px", "font-size": "11px", "margin-left": "4px"},
            ),
            dbc.Button("✕", id={"type": "explorer-filter-remove", "index": f["index"]},
                       size="sm", color="link"),
        ], style={"display": "flex", "align-items": "center", "margin-bottom": "3px"}))

    return filters, rows


# ── Generate button ───────────────────────────────────────

@callback(
    Output("graph-explorer", "figure"),
    Input("btn-explorer-generate", "n_clicks"),
    State("radio-explorer-mode", "value"),
    State("explorer-x-var", "value"), State("explorer-y-var", "value"),
    State("explorer-z-var", "value"),
    State("explorer-line-varx", "value"), State("explorer-line-vary", "value"),
    State("explorer-p1-x", "value"), State("explorer-p1-y", "value"),
    State("explorer-p2-x", "value"), State("explorer-p2-y", "value"),
    State("explorer-bins", "value"),
    State("explorer-xmin", "value"), State("explorer-xmax", "value"),
    State("explorer-ymin", "value"), State("explorer-ymax", "value"),
    State("explorer-zmin", "value"), State("explorer-zmax", "value"),
    State("explorer-cluster-filter", "value"),
    State("store-explorer-filters", "data"),
    prevent_initial_call=True,
)
def on_generate_explorer(n, mode, xvar, yvar, zvar, lxvar, lyvar,
                          p1x, p1y, p2x, p2y,
                          bins, xmin, xmax, ymin, ymax, zmin, zmax,
                          cluster_filter, filters):
    session = get_session()
    if not session.has_data():
        return _build_empty_figure("Load data first")

    df = session.raw_df.copy()

    # Apply custom filters
    df = _apply_explorer_filter(df, filters)

    # Apply cluster filter
    if cluster_filter and session.has_clusters():
        try:
            parts = cluster_filter.split(":")
            fpf, fcid = int(parts[0]), int(parts[1])
            if fpf in session.clustered_results:
                df_c = session.clustered_results[fpf]["df"]
                valid_idx = df_c.index[df_c["cluster"] == fcid]
                df = df[df.index.isin(valid_idx)]
        except (ValueError, IndexError):
            pass

    bins = max(10, min(500, int(bins or 80)))

    # ── Mode A: 2D Heatmap ──
    if mode == "heatmap":
        if not xvar or not yvar or xvar not in df.columns or yvar not in df.columns:
            return _build_empty_figure("Select valid X and Y variables")
        fig = go.Figure(go.Histogram2d(
            x=df[xvar], y=df[yvar],
            nbinsx=bins, nbinsy=bins,
            colorscale="Viridis",
        ))
        fig.update_layout(
            title=f"Heatmap: {xvar} vs {yvar}", template=_PLOTLY_TEMPLATE,
            xaxis_title=xvar, yaxis_title=yvar,
            margin={"l": 60, "r": 20, "t": 50, "b": 60},
        )
        if xmin is not None and xmax is not None:
            fig.update_xaxes(range=[xmin, xmax])
        if ymin is not None and ymax is not None:
            fig.update_yaxes(range=[ymin, ymax])
        return fig

    # ── Mode B: 3D Scatter ──
    if mode == "scatter3d":
        if not all(column and column in df.columns for column in (xvar, yvar, zvar)):
            return _build_empty_figure("Select valid X, Y and Z variables")
        draw = df[[xvar, yvar, zvar] + (["foil_position"] if "foil_position" in df.columns else [])].dropna()
        if draw.empty:
            return _build_empty_figure("No finite events after filtering")
        total = len(draw)
        # Independent z-score normalisation is purely a display transform: it
        # gives all three selected coordinates unit variance after the active
        # filters, so an arbitrary unit scale cannot dominate the 3D view.
        raw_means = draw[[xvar, yvar, zvar]].mean()
        raw_stds = draw[[xvar, yvar, zvar]].std(ddof=0).replace(0, 1.0)
        for column in (xvar, yvar, zvar):
            draw.loc[:, column] = (draw[column] - raw_means[column]) / raw_stds[column]
        if total > 60_000:
            draw = draw.sample(60_000, random_state=25521)
        fig = go.Figure()
        if "foil_position" in draw.columns:
            for foil, part in draw.groupby("foil_position", dropna=False):
                label = f"foil {int(foil)}" if pd.notna(foil) else "unclassified"
                fig.add_trace(go.Scatter3d(x=part[xvar], y=part[yvar], z=part[zvar], mode="markers", name=label,
                    marker={"size": 1.8, "opacity": .42, "color": _FOIL_COLORS.get(int(foil), "#9aa3ad") if pd.notna(foil) else "#9aa3ad"},
                    hovertemplate=f"{xvar}=%{{x:.4g}}<br>{yvar}=%{{y:.4g}}<br>{zvar}=%{{z:.4g}}<extra></extra>"))
        else:
            fig.add_trace(go.Scatter3d(x=draw[xvar], y=draw[yvar], z=draw[zvar], mode="markers", name="events",
                marker={"size": 1.8, "opacity": .42, "color": "#356bb4"},
                hovertemplate=f"{xvar}=%{{x:.4g}}<br>{yvar}=%{{y:.4g}}<br>{zvar}=%{{z:.4g}}<extra></extra>"))
        scene = {"xaxis_title": f"{xvar} (normalized)", "yaxis_title": f"{yvar} (normalized)", "zaxis_title": f"{zvar} (normalized)", "aspectmode": "data"}
        if xmin is not None and xmax is not None: scene["xaxis"] = {"range": [xmin, xmax]}
        if ymin is not None and ymax is not None: scene["yaxis"] = {"range": [ymin, ymax]}
        if zmin is not None and zmax is not None: scene["zaxis"] = {"range": [zmin, zmax]}
        fig.update_layout(title=f"3D scatter (per-axis z-score): {xvar}, {yvar}, {zvar} ({len(draw):,}/{total:,} events)", template=_PLOTLY_TEMPLATE,
            scene=scene, margin={"l": 0, "r": 0, "t": 48, "b": 0}, uirevision="explorer-3d")
        return fig

    # ── Mode C: 1D Line Histogram ──
    if not lxvar or not lyvar or lxvar not in df.columns or lyvar not in df.columns:
        return _build_empty_figure("Select valid line variables")

    p1x = float(p1x or 0)
    p1y = float(p1y or 0)
    p2x = float(p2x or 1)
    p2y = float(p2y or 1)
    dx = p2x - p1x
    dy = p2y - p1y
    seg_len = np.sqrt(dx * dx + dy * dy)
    if seg_len == 0:
        return _build_empty_figure("P1 and P2 must be distinct")

    xvals = df[lxvar].to_numpy(dtype=np.float64)
    yvals = df[lyvar].to_numpy(dtype=np.float64)
    t = ((xvals - p1x) * dx + (yvals - p1y) * dy) / (seg_len * seg_len)
    dist = t * seg_len
    mask = (t >= 0) & (t <= 1)

    fig = go.Figure(go.Histogram(
        x=dist[mask], nbinsx=bins,
    ))
    fig.update_layout(
        title=f"Line Hist: ({p1x:.2f},{p1y:.2f}) → ({p2x:.2f},{p2y:.2f})",
        template=_PLOTLY_TEMPLATE,
        xaxis_title="distance along line",
        margin={"l": 60, "r": 20, "t": 50, "b": 60},
    )
    if xmin is not None and xmax is not None:
        fig.update_xaxes(range=[xmin, xmax])
    else:
        fig.update_xaxes(range=[0, seg_len])
    return fig
