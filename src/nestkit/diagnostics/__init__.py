"""Generalization diagnostics for nested cross-validation.

Provides :class:`HyperparameterStability` for analyzing selection
consistency across folds (frequency, Jaccard similarity).
"""

from nestkit.diagnostics.stability import HyperparameterStability

__all__ = ["HyperparameterStability"]
