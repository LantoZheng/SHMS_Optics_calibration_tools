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
        html.H4("SOC Labeling GUI", style={"text-align": "center", "margin-top": "8px"}),
        html.Hr(),

        # ── Data Loading ──
        html.Div("📂 Data Loading", style=SECTION_HEADER),
        make_input("input-root-file", "ROOT File Path", "", input_type="text"),
        make_input("input-tree-name", "Tree Name", "T", input_type="text"),
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
layout = html.Div([sidebar, main_panel, right_sidebar, edit_modal, edit_store])
