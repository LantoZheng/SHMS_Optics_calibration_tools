"""SHMS CalibrationTools — interactive dashboard for SHMS optics calibration.

Launch with::

    python -m SHMS_Optics_calibration_tools.gui

or::

    from SHMS_Optics_calibration_tools.gui.app import app
    app.run(debug=True)
"""

from SHMS_Optics_calibration_tools.gui.app import app, main

__all__ = ["app", "main"]
