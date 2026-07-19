"""SOC Labeling GUI — session state management.

Holds in-memory DataFrames, configuration dictionaries, and operation
history so that Dash callbacks are stateless and the server can be
restarted without losing data.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from SHMS_Optics_calibration_tools.config import MechanicalGridConfig, HDBSCANConfig


@dataclass
class SessionState:
    """Mutable session state shared across Dash callbacks.

    Because Dash uses multi-process by default, this object lives inside
    a single-process cache (e.g. ``flask_caching.Cache`` or a module-level
    singleton).  The ``app.py`` entry point wires it up.

    Attributes
    ----------
    raw_df : pd.DataFrame or None
        Full event DataFrame loaded from ROOT / CSV (with sieve_x, sieve_y,
        foil_position, etc.).
    clustered_results : dict or None
        Output of ``cluster_each_foil`` (dict[foil_pos→dict]).
    mechanical_config : MechanicalGridConfig
        Current parameters for grid matching.
    hdbscan_config : HDBSCANConfig
        Current parameters for HDBSCAN clustering.
    hole_design : pd.DataFrame or None
        Candidate mechanical hole design table.
    cluster_hole_map : pd.DataFrame or None
        Cluster → hole assignment DataFrame.
    manual_edits : list[dict]
        Stack of manual cluster reassignments (row, col changes).
    undo_stack : list
        Operation history for undo/redo.
    foil_positions : list[int]
        Detected foil positions.
    """

    raw_df: Optional[pd.DataFrame] = None
    clustered_results: Optional[dict[int, dict[str, Any]]] = None
    mechanical_config: MechanicalGridConfig = field(default_factory=MechanicalGridConfig)
    hdbscan_config: HDBSCANConfig = field(default_factory=HDBSCANConfig)
    hole_design: Optional[pd.DataFrame] = None
    cluster_hole_map: Optional[pd.DataFrame] = None
    manual_edits: list[dict] = field(default_factory=list)
    undo_stack: list[dict] = field(default_factory=list)
    redo_stack: list[dict] = field(default_factory=list)
    foil_positions: list[int] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    cluster_edits: list[dict] = field(default_factory=list)  # undoable cluster edits
    selected_event_keys: set = field(default_factory=set)   # {(evnum, foil_pos)} of selected
    edit_mode_active: bool = False
    store_selection_state: Optional[dict] = None  # transient, for action callbacks
    trace_point_lookup: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)
    grid_trace_lookup: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    current_foil_filter: str = "all"  # tracked so figure rebuilds respect foil choice
    coordinate_df: Optional[pd.DataFrame] = None
    coordinate_summary: dict[str, Any] = field(default_factory=dict)

    def push_undo(self, description: str, before: dict[str, Any]) -> None:
        """Record a snapshot before a mutation for undo support."""
        self.undo_stack.append(
            {"description": description, "before": copy.deepcopy(before)}
        )
        # Clear redo stack on new action
        self.redo_stack.clear()
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def push_edit(self, foil: int, cluster_id: int, old_row: int, old_col: int,
                   new_row: int, new_col: int) -> None:
        """Record a manual edit for undo/redo and tracking."""
        edit = {
            "foil": foil, "cluster": cluster_id,
            "old_row": old_row, "old_col": old_col,
            "new_row": new_row, "new_col": new_col,
        }
        self.manual_edits.append(edit)
        self.push_undo("edit", edit)

    def push_cluster_edit(self, foil: int, action: str, indices: list,
                           old_clusters: list, new_cluster: int) -> None:
        """Record a cluster-level edit for undo."""
        edit = {
            "foil": foil, "action": action,
            "indices": copy.deepcopy(indices),
            "old_clusters": copy.deepcopy(old_clusters),
            "new_cluster": new_cluster,
        }
        self.cluster_edits.append(edit)
        self.push_undo("cluster_edit", edit)

    def undo_last_edit(self) -> Optional[dict]:
        """Pop last undo entry and push to redo stack. Returns the full entry."""
        if not self.undo_stack:
            return None
        entry = self.undo_stack.pop()
        self.redo_stack.append(entry)
        return entry

    def redo_last_edit(self) -> Optional[dict]:
        """Pop last redo entry and push back to undo stack. Returns the full entry."""
        if not self.redo_stack:
            return None
        entry = self.redo_stack.pop()
        self.undo_stack.append(entry)
        return entry

    def has_data(self) -> bool:
        return self.raw_df is not None and len(self.raw_df) > 0

    def has_clusters(self) -> bool:
        return self.clustered_results is not None and len(self.clustered_results) > 0

    def to_config_dict(self) -> dict[str, Any]:
        """Serialize mechanical + hdbscan config for front-end display."""
        mc = self.mechanical_config
        hc = self.hdbscan_config
        return {
            "mechanical": {
                "x_spacing_mm": mc.x_spacing_mm,
                "y_spacing_mm": mc.y_spacing_mm,
                "tolerance_mm": mc.tolerance_mm,
                "sieve_distance_cm": mc.sieve_distance_cm,
                "assignment_mode": mc.assignment_mode,
                "occupancy_penalty_cm": mc.occupancy_penalty_cm,
                "auto_spacing": mc.auto_spacing,
            },
            "hdbscan": {
                "min_cluster_size_range": list(hc.min_cluster_size_range),
                "min_samples_range": list(hc.min_samples_range) if hc.min_samples_range else None,
                "target_clusters": list(hc.target_clusters),
                "cluster_selection_method": hc.cluster_selection_method,
                "max_cluster_size": hc.max_cluster_size,
            },
        }


# Module-level singleton (for single-user desktop use)
_session: Optional[SessionState] = None


def get_session() -> SessionState:
    global _session
    if _session is None:
        _session = SessionState()
    return _session


def reset_session() -> None:
    global _session
    _session = SessionState()
