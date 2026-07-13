"""SOC Labeling GUI — Dash application entry point.

Usage
-----
    python -m SHMS_Optics_calibration_tools.gui

    or

    python SHMS_Optics_calibration_tools/gui/app.py

Opens a browser tab at http://127.0.0.1:8050 with the interactive
SOC labeling dashboard.
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc

from SHMS_Optics_calibration_tools.gui.layout import layout

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="SOC Labeling GUI",
    update_title=None,
    suppress_callback_exceptions=True,
)
app.layout = layout

# Import callbacks so they register with the app
import SHMS_Optics_calibration_tools.gui.callbacks  # noqa: F401, E402


def main() -> None:
    """Launch the Dash development server."""
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
