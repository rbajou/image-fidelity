"""Shared result aggregation and tabular evaluation helpers."""

from .results import EvaluationResult
from .statistics import select_group_columns, summarize_results

__all__ = ["EvaluationResult", "select_group_columns", "summarize_results"]