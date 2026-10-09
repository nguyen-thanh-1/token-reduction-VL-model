"""Structured experiment reports and visualizations."""

from .benchmark import build_benchmark_report
from .failure_gallery import build_failure_gallery

__all__ = ["build_benchmark_report", "build_failure_gallery"]
