"""
Evaluation module for MMLCG generated articles.

This module provides comprehensive evaluation of generated articles including:
- Section-level completion scoring
- Article-level quality metrics
- Citation relevance evaluation
- Citation selection precision evaluation
"""

from .evaluator import ArticleEvaluator, load_jsonl, match_generated_to_benchmark
from .citation_evaluator import CitationEvaluator
from .score_calculator import ScoreCalculator
from .citation_precision_evaluator import CitationPrecisionEvaluator, compute_aggregate_stats

__all__ = [
    'ArticleEvaluator',
    'CitationEvaluator',
    'ScoreCalculator',
    'CitationPrecisionEvaluator',
    'compute_aggregate_stats',
    'load_jsonl',
    'match_generated_to_benchmark'
]
