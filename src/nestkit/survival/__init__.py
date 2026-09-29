"""Survival analysis support for nested cross-validation.

Provides :class:`NestedCVSurvival` for running nested CV with
survival models, :class:`CoxPHWrapper` for wrapping lifelines'
``CoxPHFitter`` in a scikit-learn compatible interface, and
:func:`make_survival_target` for constructing survival targets.
"""

from nestkit.survival._target import make_survival_target
from nestkit.survival._wrapper import CoxPHWrapper
from nestkit.survival.survival import NestedCVSurvival

__all__ = [
    "CoxPHWrapper",
    "NestedCVSurvival",
    "make_survival_target",
]
