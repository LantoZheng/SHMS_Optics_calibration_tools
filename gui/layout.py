"""SOC Labeling GUI — Dash layout definition.

Composes the full UI from reusable components:
  - Left sidebar: data loading, clustering params, grid matching params
  - Main panel: tabbed views (sieve plot, match plot, ztar, export)
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

# ── reusable style constants ──────────────────────────────
SLIDER_STYLE = {"margin-bottom": "12px"}
SECTION_HEADER = {"font-weight": "bold", "margin-top": "16px", "margin-bottom": "4px"}
LABEL_STYLE = {"font-size": "12px", "margin-bottom": "2px"}


def make_square_graph(graph_id: str, max_size: str = "68vh") -> html.Div:
    """Create a square Plotly graph container that fills horizontal space."""
    return html.Div(
        [
            dcc.Graph(id=graph_id, style={"height": "100%", "width": "100%"}),
        ],
        style={
            "width": "100%",
            "max-width": "100%",
            "aspectRatio": "1 / 1",
            "margin": "0",
        },
    )


def make_slider(
    id_: str,
    label: str,
    min_val: float,
    max_val: float,
    step: float,
    value: float,
    tooltip: str = "",
) -> html.Div:
    """Create a labelled Dash slider with a numeric input for direct entry."""
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            html.Div([
                dcc.Slider(
                    id=id_,
                    min=min_val,
                    max=max_val,
                    step=step,
                    value=value,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                    className="slider-with-input",
                ),
                dcc.Input(
                    id=f"{id_}-input",
                    type="number",
                    min=min_val,
                    max=max_val,
                    step=step,
                    value=value,
                    style={
                        "width": "68px", "margin-left": "8px", "font-size": "11px",
                        "text-align": "right", "padding": "2px 4px",
                        "border": "1px solid #ccc", "border-radius": "3px",
                    },
                ),
            ], style={"display": "flex", "align-items": "center"}),
        ],
        style=SLIDER_STYLE,
    )


def make_dropdown(
    id_: str,
    label: str,
    options: list[dict],
    value: str,
) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Dropdown(id=id_, options=options, value=value, clearable=False, style={"font-size": "12px"}),
        ],
        style=SLIDER_STYLE,
    )


def make_input(id_: str, label: str, value, input_type: str = "number") -> html.Div:
    return html.Div(
        [
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(id=id_, type=input_type, value=value, style={"width": "100%", "font-size": "12px"}),
        ],
        style=SLIDER_STYLE,
    )


# ── Sidebar ───────────────────────────────────────────────
sidebar = html.Div(
    [
        html.H4("SHMS CalibrationTools", style={"text-align": "center", "margin-top": "8px"}),
        html.Hr(),

        # ── Data Loading ──
        html.Div("📂 Data Loading", style=SECTION_HEADER),
        make_input("input-root-file", "ROOT File Path", "", input_type="text"),
        make_input("input-tree-name", "Tree Name", "T", input_type="text"),
        html.Label("Branch loading", style=LABEL_STYLE),
        dcc.RadioItems(
            id="radio-root-branch-mode",
            options=[
                {"label": " Optimized (recommended)", "value": "optimized"},
                {"label": " Read all ROOT branches", "value": "all"},
                {"label": " Manually select branches", "value": "manual"},
            ],
            value="optimized",
            style={"font-size": "11px", "margin-bottom": "5px"},
            inputStyle={"margin-right": "3px", "margin-left": "4px"},
        ),
        html.Div(
            "Optimized reads only reconstruction, FP5D and PID branches; it is suitable for the normal GUI and FP5D workflow.",
            style={"font-size": "10px", "color": "#666", "margin-bottom": "6px"},
        ),
        make_input(
            "input-extra-root-branches",
            "Extra branches (comma-separated, optional)",
            "",
            input_type="text",
        ),
        html.Div(
            id="manual-root-branch-controls",
            children=[
                html.Div("Assign each GUI variable to a ROOT branch. Enter the ROOT path first; choices are then populated without reading event data.",
                         style={"font-size": "10px", "color": "#666", "margin-bottom": "5px"}),
                *[
                    html.Div([
                        html.Label(role, style={"font-size": "10px", "margin-bottom": "1px"}),
                        dcc.Dropdown(id={"type": "manual-root-branch", "role": role}, options=[], value=None,
                                     placeholder="Select ROOT branch", searchable=True, clearable=False,
                                     style={"font-size": "10px"}),
                    ], style={"margin-bottom": "4px"})
                    for role in (
                        "P_gtr_x", "P_gtr_y", "P_gtr_dp", "P_gtr_th", "P_gtr_ph", "P_react_z",
                        "P_dc_x_fp", "P_dc_y_fp", "P_dc_xp_fp", "P_dc_yp_fp", "P_rb_raster_frybRawAdc",
                        "P_ngcer_npeSum", "P_hgcer_npeSum", "P_cal_etottracknorm",
                    )
                ],
            ],
            style={"display": "none", "max-height": "360px", "overflow-y": "auto", "padding": "5px", "border": "1px solid #ddd", "border-radius": "4px", "margin-bottom": "8px"},
        ),
        html.Div(
            [
                dbc.Button("Load Data", id="btn-load-data", color="primary", size="sm", style={"width": "100%"}),
            ],
            style={"margin-bottom": "12px"},
        ),
        dcc.Loading(
            id="loading-data",
            type="dot",
            color="#2196F3",
            children=[html.Div(id="data-status", style={"font-size": "11px", "color": "#888"})],
        ),
        html.Hr(),

        # ── PID / Filter Cuts ──
        html.Div("🔬 Filter Cuts", style=SECTION_HEADER),
        make_slider("slider-ngcer-min", "ngcer_min", 0, 20, 0.5, 2.0),
        make_slider("slider-hgcer-min", "hgcer_min", 0, 5, 0.1, 0.5),
        make_slider("slider-cal-min", "cal_etot_min", 0.2, 1.5, 0.05, 0.6),
        make_slider("slider-cal-max", "cal_etot_max", 1.0, 3.0, 0.05, 1.8),
        dbc.Button("Apply Filters", id="btn-apply-filters", color="secondary", size="sm", style={"width": "100%"}),
        html.Hr(),

        # ── HDBSCAN Params ──
        html.Div("🧩 HDBSCAN Clustering", style=SECTION_HEADER),
        dbc.Checkbox(
            id="checkbox-manual-only",
            label="🔧 Manual Only (single cluster per foil, skip HDBSCAN)",
            value=False,
            style={"font-size": "12px", "margin-bottom": "8px"},
        ),
        html.Div(
            id="div-hdbscan-controls",
            children=[
                make_slider("slider-min-cluster-size", "min_cluster_size", 5, 200, 5, 15),
                make_slider("slider-min-samples", "min_samples", 3, 100, 1, 10),
                make_dropdown("dropdown-cluster-method", "cluster_selection_method",
                              [{"label": "eom", "value": "eom"}, {"label": "leaf", "value": "leaf"}], "eom"),
                make_slider("slider-max-cluster-size", "max_cluster_size (cm)", 0.5, 5.0, 0.1, 1.8),
                dbc.Button("Run Clustering", id="btn-run-clustering", color="info", size="sm", style={"width": "100%"}),
                dcc.Loading(
                    id="loading-cluster-btn",
                    type="dot",
                    color="#2196F3",
                    children=[html.Div(id="cluster-status", style={"font-size": "11px", "color": "#888", "margin-top": "4px"})],
                ),
            ],
        ),
        html.Hr(),

        # ── Mechanical Grid Params ──
        html.Div("🎯 Mechanical Grid", style=SECTION_HEADER),
        make_slider("slider-x-spacing", "x_spacing (mm)", 20, 30, 0.1, 25.0),
        make_slider("slider-y-spacing", "y_spacing (mm)", 12, 22, 0.1, 16.4),
        make_slider("slider-tolerance", "tolerance (mm)", 0.5, 10, 0.1, 3.0),
        make_slider("slider-occupancy-penalty", "occupancy_penalty (cm)", 0, 2.0, 0.05, 0.35),
        make_dropdown("dropdown-assignment-mode", "assignment_mode",
                      [{"label": "Nearest", "value": "nearest"},
                       {"label": "Center-out Penalized", "value": "center_out_penalized"}],
                      "center_out_penalized"),
        dbc.Button("Run Grid Match", id="btn-run-grid-match", color="success", size="sm", style={"width": "100%"}),
        dcc.Loading(
            id="loading-grid-btn",
            type="dot",
            color="#FF9800",
            children=[html.Div(id="grid-status", style={"font-size": "11px", "color": "#888", "margin-top": "4px"})],
        ),
        html.Hr(),

        # ── Export ──
        html.Div("💾 Export", style=SECTION_HEADER),
        make_input("input-output-csv", "Output CSV name", "stage2_soc_gui_labeled.csv", input_type="text"),
        dbc.Button("Export Labels", id="btn-export", color="warning", size="sm", style={"width": "100%"}),
        html.Div(id="export-status", style={"font-size": "11px", "color": "#888", "margin-top": "4px"}),
        dcc.Download(id="download-csv"),

    ],
    style={
        "position": "fixed", "top": 0, "left": 0, "bottom": 0,
        "width": "280px", "padding": "12px", "overflow-y": "auto",
        "background-color": "#f8f9fa", "border-right": "1px solid #ddd",
    },
)

# ── Right sidebar ──────────────────────────────────────────
right_sidebar = html.Div(
    [
        html.Div("Display", style={"font-weight": "bold", "font-size": "11px", "margin-bottom": "8px"}),
        html.Div([
            html.Label("Mode", style={"font-size": "10px", "margin-bottom": "2px", "color": "#888"}),
            dcc.Dropdown(
                id="dropdown-viz-mode",
                options=[{"label": "🔵 Scatter", "value": "scatter"},
                         {"label": "🔥 Heatmap", "value": "heatmap"}],
                value="scatter", clearable=False,
                style={"width": "100%", "font-size": "11px"},
            ),
        ], style={"margin-bottom": "16px"}),
        html.Div([
            html.Label("Foil", style={"font-size": "10px", "margin-bottom": "2px", "color": "#888"}),
            dcc.Dropdown(
                id="dropdown-foil-filter",
                options=[{"label": "All Foils", "value": "all"},
                         {"label": "Foil 0", "value": "0"},
                         {"label": "Foil 1", "value": "1"},
                         {"label": "Foil 2", "value": "2"}],
                value="all", clearable=False,
                style={"width": "100%", "font-size": "11px"},
            ),
        ], style={"margin-bottom": "16px"}),
        html.Hr(style={"margin": "12px 0"}),
        html.Div([
            dbc.Checkbox(
                id="checkbox-edit-clusters",
                label="✏️ Edit Clusters",
                value=False,
                style={"font-size": "11px"},
            ),
        ], id="div-edit-clusters-wrapper"),
    ],
    style={
        "position": "fixed", "top": 0, "right": 0, "bottom": 0,
        "width": "180px", "padding": "12px", "overflow-y": "auto",
        "background-color": "#f8f9fa", "border-left": "1px solid #ddd",
    },
)

# ── Main panel (tabbed) ───────────────────────────────────
main_panel = html.Div(
    [
        dcc.Tabs(
            id="main-tabs",
            value="tab-sieve",
            children=[
                dcc.Tab(label="Sieve Clusters", value="tab-sieve",
                        children=[
                            dcc.Loading(
                                id="loading-sieve",
                                type="cube",
                                color="#2196F3",
                                children=[make_square_graph("graph-sieve-clusters", max_size="68vh")],
                            ),
                        ]),
                dcc.Tab(label="Grid Match", value="tab-match",
                        children=[
                            dcc.Loading(
                                id="loading-match",
                                type="cube",
                                color="#FF9800",
                                children=[
                                    make_square_graph("graph-grid-match", max_size="56vh"),
                                    html.Div([
                                        html.Div([
                                            html.H6("Interactive Match Editor", style={"margin-bottom": "4px"}),
                                            html.Div(
                                                "Click a cluster center, then click a mechanical hole cross to reassign.",
                                                id="grid-edit-info",
                                                style={"font-size": "12px", "color": "#666"},
                                            ),
                                            html.Div(id="grid-edit-selection", style={
                                                "font-size": "12px", "color": "#444", "margin-top": "4px",
                                            }),
                                        ], style={"flex": "1"}),
                                        html.Div([
                                            dbc.Button("Apply Match", id="btn-grid-apply-match", color="primary",
                                                       size="sm", disabled=True, style={"margin-right": "8px"}),
                                            dbc.Button("Clear", id="btn-grid-clear-selection", color="secondary",
                                                       size="sm", style={"margin-right": "8px"}),
                                            dbc.Button("↩ Undo", id="btn-grid-undo-edit", color="secondary",
                                                       size="sm", disabled=True, style={"margin-right": "8px"}),
                                            dbc.Button("↪ Redo", id="btn-grid-redo-edit", color="secondary",
                                                       size="sm", disabled=True),
                                        ]),
                                    ], style={
                                        "display": "flex", "justify-content": "space-between",
                                        "align-items": "center", "gap": "16px", "padding": "8px 16px",
                                        "border-top": "1px solid #eee", "background": "#fffaf2",
                                    }),
                                    html.Div(id="div-grid-match-table", style={"padding": "12px 16px"}),
                                    dcc.Store(id="store-grid-edit-selection", data={}),
                                ],
                            ),
                        ]),
                dcc.Tab(label="ztar Distribution", value="tab-ztar",
                            children=[make_square_graph("graph-ztar", max_size="68vh")]),
                dcc.Tab(label="Match Table", value="tab-table",
                        children=[
                            html.Div(id="div-match-table", style={"padding": "16px"}),
                            # Undo / Redo toolbar for manual edits
                            html.Div([
                                dbc.Button("↩ Undo", id="btn-undo-edit", color="secondary",
                                           size="sm", style={"margin-right": "8px"}, disabled=True),
                                dbc.Button("↪ Redo", id="btn-redo-edit", color="secondary",
                                           size="sm", disabled=True),
                                html.Span(id="edit-status-bar", style={
                                    "font-size": "12px", "color": "#888", "margin-left": "16px",
                                }),
                            ], style={"padding": "8px 16px", "border-top": "1px solid #eee"}),
                        ]),
                dcc.Tab(label="Data Explorer", value="tab-explorer",
                        children=[
                            dcc.Loading(
                                id="loading-explorer",
                                type="cube",
                                color="#9C27B0",
                                children=[
                                    html.Div([
                                        # Mode selector
                                        dcc.RadioItems(
                                            id="radio-explorer-mode",
                                            options=[
                                                {"label": " 2D Heatmap", "value": "heatmap"},
                                                {"label": " 3D Scatter", "value": "scatter3d"},
                                                {"label": " 1D Line Histogram", "value": "linehist"},
                                            ],
                                            value="heatmap", inline=True,
                                            style={"font-size": "12px"},
                                        ),
                                        # Variable selectors (heatmap)
                                        html.Div([
                                            dcc.Dropdown(id="explorer-x-var", options=[],
                                                         placeholder="X variable", clearable=False,
                                                         style={"width": "180px", "font-size": "11px"}),
                                            dcc.Dropdown(id="explorer-y-var", options=[],
                                                         placeholder="Y variable", clearable=False,
                                                         style={"width": "180px", "font-size": "11px",
                                                                "margin-left": "8px"}),
                                        ], id="div-explorer-xy-vars",
                                           style={"display": "flex", "margin": "6px 0"}),
                                        html.Div([
                                            dcc.Dropdown(id="explorer-z-var", options=[],
                                                         placeholder="Z variable", clearable=False,
                                                         style={"width": "180px", "font-size": "11px"}),
                                        ], id="div-explorer-z-var",
                                           style={"display": "none", "margin": "6px 0"}),
                                        # Endpoints (linehist)
                                        html.Div([
                                            html.Label("Var X", style={"font-size": "10px"}),
                                            dcc.Dropdown(id="explorer-line-varx", options=[],
                                                         placeholder="X", clearable=False,
                                                         style={"width": "130px", "font-size": "11px"}),
                                            html.Label("Var Y", style={"font-size": "10px", "margin-left": "8px"}),
                                            dcc.Dropdown(id="explorer-line-vary", options=[],
                                                         placeholder="Y", clearable=False,
                                                         style={"width": "130px", "font-size": "11px",
                                                                "margin-left": "4px"}),
                                            html.Label("P1", style={"font-size": "10px", "margin-left": "12px"}),
                                            dcc.Input(id="explorer-p1-x", type="number", value=0,
                                                      style={"width": "60px", "font-size": "11px", "margin-left": "4px"}),
                                            dcc.Input(id="explorer-p1-y", type="number", value=0,
                                                      style={"width": "60px", "font-size": "11px", "margin-left": "2px"}),
                                            html.Label("P2", style={"font-size": "10px", "margin-left": "8px"}),
                                            dcc.Input(id="explorer-p2-x", type="number", value=1,
                                                      style={"width": "60px", "font-size": "11px", "margin-left": "4px"}),
                                            dcc.Input(id="explorer-p2-y", type="number", value=1,
                                                      style={"width": "60px", "font-size": "11px", "margin-left": "2px"}),
                                        ], id="div-explorer-line-endpoints",
                                           style={"display": "none", "align-items": "center", "margin": "6px 0", "flex-wrap": "wrap"}),
                                        # Cluster filter
                                        html.Div([
                                            html.Label("Cluster", style={"font-size": "11px", "margin-right": "6px"}),
                                            dcc.Dropdown(id="explorer-cluster-filter", options=[],
                                                         placeholder="All clusters", clearable=True,
                                                         style={"width": "200px", "font-size": "11px"}),
                                        ], style={"margin": "6px 0"}),
                                        # Data filters
                                        html.Div([
                                            html.Label("Data Filters", style={"font-size": "11px", "font-weight": "bold", "margin-right": "8px"}),
                                            dbc.Button("+ Add", id="btn-explorer-add-filter", color="link", size="sm"),
                                        ], style={"margin": "4px 0"}),
                                        html.Div(id="div-explorer-filters"),
                                        # Bin & Range
                                        html.Div([
                                            html.Label("Bins", style={"font-size": "10px", "margin-right": "4px"}),
                                            dcc.Input(id="explorer-bins", type="number", value=80, min=10, max=500,
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "8px"}),
                                            html.Label("X", style={"font-size": "10px", "margin-right": "2px"}),
                                            dcc.Input(id="explorer-xmin", type="number", placeholder="min",
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "2px"}),
                                            dcc.Input(id="explorer-xmax", type="number", placeholder="max",
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "8px"}),
                                            html.Label("Y", style={"font-size": "10px", "margin-right": "2px"}),
                                            dcc.Input(id="explorer-ymin", type="number", placeholder="min",
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "2px"}),
                                            dcc.Input(id="explorer-ymax", type="number", placeholder="max",
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "8px"}),
                                            html.Label("Z", style={"font-size": "10px", "margin-right": "2px"}),
                                            dcc.Input(id="explorer-zmin", type="number", placeholder="min",
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "2px"}),
                                            dcc.Input(id="explorer-zmax", type="number", placeholder="max",
                                                      style={"width": "60px", "font-size": "11px", "margin-right": "8px"}),
                                            dbc.Button("Generate", id="btn-explorer-generate", color="primary", size="sm"),
                                        ], style={"margin": "8px 0", "display": "flex", "align-items": "center", "flex-wrap": "wrap"}),
                                    ], style={
                                        "padding": "8px 16px", "background": "#fafafa",
                                        "border-bottom": "1px solid #eee",
                                    }),
                                    make_square_graph("graph-explorer", max_size="54vh"),
                                ],
                            ),
                            dcc.Store(id="store-explorer-filters", data=[]),
                        ]),
                dcc.Tab(label="FP5D / Z-space Lab", value="tab-fp5d-lab",
                        children=[
                            html.Div([
                                html.Div([
                    dbc.Button("Show exact Z coordinates", id="btn-build-z-coordinates", color="secondary", size="sm"),
                                    dbc.Button("Export shareable 3D HTML", id="btn-export-fp5d-html", color="outline-primary", size="sm", style={"margin-left": "8px"}),
                                    html.Span(id="z-coordinate-status", style={"font-size": "12px", "margin-left": "12px", "color": "#555"}),
                                ], style={"margin": "10px 0"}),
                                html.Div("🧭 FP5D Flow Clustering", style={**SECTION_HEADER, "font-size": "15px"}),
                                html.Div(
                                    "Independent of the sieve-plane clustering. Uses x_fp, y_fp, x'_fp, y'_fp and raster.",
                                    style={"font-size": "12px", "color": "#666", "margin-bottom": "8px"},
                                ),
                                html.Div([
                                    html.Div([html.Label("min_cluster_size (validated: 60)", style=LABEL_STYLE), dcc.Slider(id="slider-fp5d-min-cluster-size", min=10, max=300, step=5, value=60, marks=None, tooltip={"placement": "bottom", "always_visible": True})], style={"width": "310px"}),
                                    html.Div([html.Label("min_samples (validated: 10)", style=LABEL_STYLE), dcc.Slider(id="slider-fp5d-min-samples", min=1, max=100, step=1, value=10, marks=None, tooltip={"placement": "bottom", "always_visible": True})], style={"width": "310px"}),
                                    dbc.Checkbox(id="checkbox-force-exact-rerun", label="Force full recompute (ignore validated cache)", value=False, style={"font-size": "12px", "margin-top": "16px"}),
                                    dbc.Button("Run exact FP5D research pipeline (GPU)", id="btn-run-fp5d-flow", color="primary", size="sm", style={"margin-top": "12px"}),
                                ], style={"display": "flex", "align-items": "start", "gap": "18px", "flex-wrap": "wrap", "margin": "6px 0"}),
                                html.Div("Exact mode uses the validated study settings: 80 flow epochs; HDBSCAN (60, 10, eom).", style={"font-size": "11px", "color": "#666", "margin-bottom": "5px"}),
                                dcc.Loading(id="loading-fp5d-research-pipeline", type="circle", color="#2196F3", fullscreen=True,
                                    children=html.Div(id="fp5d-status", style={"font-size": "12px", "color": "#666", "margin-bottom": "8px"})),
                                html.Div([
                                    html.Div([html.Label("Show", style=LABEL_STYLE), dcc.Dropdown(
                                        id="fp5d-show-objects", clearable=False, value="both",
                                        options=[{"label": "Points + centroids", "value": "both"}, {"label": "Points only", "value": "points"}, {"label": "Centroids only", "value": "centroids"}],
                                        style={"width": "180px", "font-size": "12px"})]),
                                    html.Div([html.Label("X", style=LABEL_STYLE), dcc.Dropdown(id="fp5d-x-axis", clearable=False, style={"width": "185px", "font-size": "12px"})]),
                                    html.Div([html.Label("Y", style=LABEL_STYLE), dcc.Dropdown(id="fp5d-y-axis", clearable=False, style={"width": "185px", "font-size": "12px"})]),
                                    html.Div([html.Label("Z", style=LABEL_STYLE), dcc.Dropdown(id="fp5d-z-axis", clearable=False, style={"width": "185px", "font-size": "12px"})]),
                                    html.Div([html.Label("Colour", style=LABEL_STYLE), dcc.Dropdown(id="fp5d-colour", clearable=False, style={"width": "185px", "font-size": "12px"})]),
                                ], style={"display": "flex", "gap": "8px", "flex-wrap": "wrap", "margin": "10px 0"}),
                                html.Div(id="fp5d-view-status", style={"font-size": "12px", "color": "#555", "margin-bottom": "4px"}),
                                html.Div([
                                    html.Span("Manual cluster → foil / grid match", style={"font-weight": "600", "font-size": "12px", "margin-right": "12px"}),
                                    html.Span(id="fp5d-manual-selection", children="Click a centroid (or an event point) to select its FP5D cluster.", style={"font-size": "12px", "color": "#555", "margin-right": "12px"}),
                                    html.Label("Foil", style=LABEL_STYLE),
                                    dcc.Input(id="fp5d-match-foil", type="number", min=0, step=1, value=0, style={"width": "56px", "margin-right": "6px"}),
                                    html.Label("Grid row", style=LABEL_STYLE),
                                    dcc.Input(id="fp5d-match-row", type="number", step=1, value=0, style={"width": "56px", "margin-right": "6px"}),
                                    html.Label("Grid col", style=LABEL_STYLE),
                                    dcc.Input(id="fp5d-match-col", type="number", step=1, value=0, style={"width": "56px", "margin-right": "8px"}),
                                    dbc.Button("Record match", id="btn-fp5d-record-match", color="primary", size="sm"),
                                    dbc.Button("Mark as noise", id="btn-fp5d-mark-noise", color="secondary", size="sm"),
                                ], style={"display": "flex", "align-items": "center", "gap": "5px", "flex-wrap": "wrap", "margin": "7px 0"}),
                                html.Div(id="fp5d-manual-match-status", style={"font-size": "12px", "color": "#555", "margin-bottom": "4px"}),
                                html.Div(id="fp5d-manual-match-table", style={"font-size": "12px", "margin-bottom": "6px"}),
                                dcc.Loading(id="loading-fp5d-graph", type="cube", color="#673AB7", children=[
                                    dcc.Graph(id="graph-fp5d-3d", style={"height": "70vh"}, config={"scrollZoom": True, "displaylogo": False}),
                                ]),
                                dcc.Download(id="download-fp5d-html"),
                                dcc.Store(id="store-fp5d-manual-selection", data={}),
                            ], style={"padding": "8px 16px"}),
                        ]),
            ],
        ),
        # ── Status bar below graph ──
        html.Div([
            html.Span(id="cluster-info-bar", style={
                "font-size": "12px", "color": "#555",
            }),
        ], style={
            "padding": "6px 16px", "border-top": "1px solid #eee",
            "background": "#fafafa",
        }),
        # ── Cluster Edit Action Bar (hidden until selection) ──
        html.Div(
            id="div-action-bar",
            children=[
                html.Span(id="action-bar-info", style={
                    "font-size": "12px", "color": "#555", "margin-right": "12px",
                }),
                dbc.Button("➕ New Cluster", id="btn-action-new-cluster",
                           color="success", size="sm", style={"margin-right": "6px"}),
                dcc.Dropdown(
                    id="dropdown-target-cluster",
                    options=[], value=None, clearable=False,
                    placeholder="Target cluster...",
                    style={"width": "160px", "font-size": "12px", "display": "inline-block",
                           "margin-right": "6px"},
                ),
                dbc.Button("📦 Move to Cluster", id="btn-action-move-cluster",
                           color="primary", size="sm", style={"margin-right": "6px"}),
                dbc.Button("🗑️ Mark Noise", id="btn-action-mark-noise",
                           color="danger", size="sm", style={"margin-right": "6px"}),
                dbc.Checkbox(
                    id="checkbox-keep-selection",
                    label="Keep selection",
                    value=False,
                    style={"font-size": "11px", "display": "inline-block", "margin-right": "6px"},
                ),
                dbc.Button("✕ Cancel", id="btn-action-cancel",
                           color="link", size="sm"),
            ],
            style={
                "padding": "6px 16px", "border-top": "1px solid #eee",
                "background": "#fff8e1", "display": "none",
                "align-items": "center", "flex-wrap": "wrap",
            },
        ),
        # Hidden store for selection state
        dcc.Store(id="store-selection-state", data={}),
    ],
    style={"margin-left": "280px", "margin-right": "180px", "padding": "8px"},
)

# ── Edit Modal (hidden, shown on ✏️ click) ──────────────
edit_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Edit Cluster → Hole Assignment")),
        dbc.ModalBody([
            html.Div(id="edit-modal-body", children=[
                html.P("Loading cluster details...", id="edit-modal-info"),
                html.Div([
                    html.Label("New Row", style=LABEL_STYLE),
                    dcc.Input(id="edit-input-row", type="number", value=0,
                              style={"width": "100%", "font-size": "14px", "margin-bottom": "12px"}),
                ]),
                html.Div([
                    html.Label("New Col", style=LABEL_STYLE),
                    dcc.Input(id="edit-input-col", type="number", value=0,
                              style={"width": "100%", "font-size": "14px", "margin-bottom": "12px"}),
                ]),
                html.Div(id="edit-modal-warning", style={"color": "#e65100", "font-size": "12px"}),
            ]),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="btn-edit-cancel", color="secondary", size="sm"),
            dbc.Button("Apply", id="btn-edit-apply", color="primary", size="sm"),
        ]),
    ],
    id="modal-edit-hole",
    size="sm",
    is_open=False,
)

# ── Hidden stores for edit coordination ──────────────────
edit_store = dcc.Store(id="store-edit-context", data={})  # {foil, cluster, row, col}

# ── Full layout ───────────────────────────────────────────
_RESEARCH_OVERLAY_HIDDEN = {"display": "none"}
_RESEARCH_OVERLAY_ACTIVE = {
    "display": "flex", "position": "fixed", "inset": "0", "zIndex": 10000,
    "background": "rgba(15, 23, 42, 0.72)", "alignItems": "center", "justifyContent": "center",
}

research_overlay = html.Div(
    id="fp5d-research-overlay",
    style=_RESEARCH_OVERLAY_HIDDEN,
    children=html.Div([
        dbc.Spinner(color="light", size="lg"),
        html.Div("Exact FP5D research pipeline is running", style={"fontWeight": "bold", "fontSize": "20px", "marginTop": "18px"}),
        html.Div("Flow training → HDBSCAN → local coordinate fields → relative Z mapping → global flattened Z3",
                 style={"fontSize": "13px", "marginTop": "8px", "textAlign": "center", "maxWidth": "620px"}),
        html.Div("This full Run-25521 analysis can take several minutes; keep this page open.",
                 style={"fontSize": "12px", "marginTop": "14px", "opacity": 0.8}),
    ], style={"color": "white", "textAlign": "center", "padding": "36px", "borderRadius": "12px", "background": "#1e293b", "boxShadow": "0 18px 60px rgba(0,0,0,.45)"}),
)

layout = html.Div([sidebar, main_panel, right_sidebar, edit_modal, edit_store, research_overlay])
