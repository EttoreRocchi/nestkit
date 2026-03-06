"""Post-hoc probability calibration for classification models.

Provides :class:`PostHocCalibrator` supporting Platt scaling (sigmoid),
isotonic regression, beta calibration, and Venn-ABERS prediction, as well
as :class:`CalibrationDiagnostics` for evaluating calibration quality
via ECE, MCE, Brier score, and reliability diagrams.
"""

from nestkit.calibration.calibrators import PostHocCalibrator
from nestkit.calibration.diagnostics import CalibrationDiagnostics

__all__ = ["CalibrationDiagnostics", "PostHocCalibrator"]
