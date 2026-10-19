"""Evaluation and metrics modules for parking space detection."""

from .evaluation import (
    calculate_contour_metrics,
    calculate_clustering_metrics,
    calculate_edge_density,
    calculate_processing_time,
    evaluate_threshold_quality,
    calculate_image_quality_metrics,
    compare_methods,
    create_evaluation_report,
    benchmark_methods
)

__all__ = [
    'calculate_contour_metrics',
    'calculate_clustering_metrics',
    'calculate_edge_density',
    'calculate_processing_time',
    'evaluate_threshold_quality',
    'calculate_image_quality_metrics',
    'compare_methods',
    'create_evaluation_report',
    'benchmark_methods'
]
