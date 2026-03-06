"""Cross-fold feature importance aggregation and stability analysis.

Provides :class:`FeatureImportanceAggregator` for computing, ranking,
and assessing the stability of feature importance scores across outer
CV folds, including the Nogueira stability index and consensus feature
selection.
"""

from nestkit.importance.aggregator import FeatureImportanceAggregator

__all__ = ["FeatureImportanceAggregator"]
