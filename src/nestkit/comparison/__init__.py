"""Statistical model comparison using nested CV results.

Provides :class:`NestedCVComparator` for pairwise and multi-model
statistical tests including the Nadeau-Bengio corrected t-test,
Bayesian correlated t-test with ROPE, and Holm-Bonferroni correction.
"""

from nestkit.comparison.comparator import NestedCVComparator

__all__ = ["NestedCVComparator"]
