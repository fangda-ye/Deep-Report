"""
Score calculator for final evaluation scores.
Normalizes and combines section, article, and citation scores.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ScoreCalculator:
    """
    Calculates final evaluation scores by normalizing and combining
    section, article, and citation evaluations.

    Final Score = section_score × article_score × citation_score × 100

    Each component is normalized to [0, 1] before multiplication.
    
    Supports two scoring modes:
    - "0-10": Traditional 0-10 continuous scale (divide by 10)
    - "1-5": New 3-level discrete scale (1, 3, 5) -> normalize as (score - 1) / 4
    """

    def __init__(
        self,
        # Weights for section components
        section_description_weight: float = 0.5,
        section_checklist_weight: float = 0.5,
        # Weights for article components
        article_coherence_weight: float = 0.3,
        article_fluency_weight: float = 0.3,
        article_repetition_weight: float = 0.2,
        article_termination_weight: float = 0.2,
        # Score mode: "0-10" for traditional, "1-5" for new 3-level discrete
        score_mode: str = "1-5"
    ):
        """
        Initialize score calculator with component weights.

        Args:
            section_description_weight: Weight for description completion (default 0.5)
            section_checklist_weight: Weight for checklist completion (default 0.5)
            article_coherence_weight: Weight for coherence (default 0.3)
            article_fluency_weight: Weight for fluency (default 0.3)
            article_repetition_weight: Weight for repetition (default 0.2)
            article_termination_weight: Weight for termination (default 0.2)
            score_mode: "0-10" for traditional scale, "1-5" for new 3-level discrete (1, 3, 5)
        """
        # Section weights
        self.section_description_weight = section_description_weight
        self.section_checklist_weight = section_checklist_weight

        # Article weights
        self.article_coherence_weight = article_coherence_weight
        self.article_fluency_weight = article_fluency_weight
        self.article_repetition_weight = article_repetition_weight
        self.article_termination_weight = article_termination_weight
        
        # Score mode
        self.score_mode = score_mode

        # Validate weights sum to 1
        assert abs(section_description_weight + section_checklist_weight - 1.0) < 1e-6, \
            "Section weights must sum to 1"
        assert abs(
            article_coherence_weight + article_fluency_weight +
            article_repetition_weight + article_termination_weight - 1.0
        ) < 1e-6, "Article weights must sum to 1"

        logger.info(f"Initialized ScoreCalculator with score_mode={score_mode}")
    
    def _normalize_score(self, score: float) -> float:
        """
        Normalize a raw score to [0, 1] based on score_mode.
        
        For "0-10" mode: score / 10
        For "1-5" mode: (score - 1) / 4
        """
        if self.score_mode == "1-5":
            # 1 -> 0, 3 -> 0.5, 5 -> 1.0
            return max(0.0, min(1.0, (score - 1) / 4))
        else:
            # Traditional 0-10 scale
            return max(0.0, min(1.0, score / 10))

    def calculate_final_score(
        self,
        article_eval: Dict[str, Any],
        citation_eval: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate final score from article and citation evaluations.

        Args:
            article_eval: Article evaluation results (section + article level)
            citation_eval: Citation evaluation results

        Returns:
            Dictionary with normalized scores and final score
        """
        try:
            # Calculate component scores (normalized to [0, 1])
            section_score = self._calculate_section_score(
                article_eval.get('section_evaluations', [])
            )
            article_score = self._calculate_article_score(
                article_eval.get('article_evaluation', {})
            )
            citation_score = self._calculate_citation_score(
                citation_eval
            )

            # Calculate final score
            final_score = section_score * article_score * citation_score * 100

            result = {
                'section_score': section_score,
                'article_score': article_score,
                'citation_score': citation_score,
                'final_score': final_score,
                'breakdown': {
                    'section': self._get_section_breakdown(article_eval.get('section_evaluations', [])),
                    'article': self._get_article_breakdown(article_eval.get('article_evaluation', {})),
                    'citation': self._get_citation_breakdown(citation_eval)
                }
            }

            logger.info(
                f"Final score calculated: {final_score:.2f} "
                f"(section={section_score:.3f}, article={article_score:.3f}, citation={citation_score:.3f})"
            )
            return result

        except Exception as e:
            logger.error(f"Error calculating final score: {str(e)}", exc_info=True)
            return {
                'error': str(e),
                'section_score': 0,
                'article_score': 0,
                'citation_score': 0,
                'final_score': 0
            }

    def _calculate_section_score(self, section_evaluations: List[Dict[str, Any]]) -> float:
        """
        Calculate normalized section score.

        Returns:
            Normalized score in [0, 1]
        """
        if not section_evaluations:
            logger.warning("No section evaluations found")
            return 0.0

        section_scores = []
        for section_eval in section_evaluations:
            # Description score - normalize based on score_mode
            raw_desc_score = section_eval.get('description_completion_score', 0)
            description_score = self._normalize_score(raw_desc_score)

            # Checklist score - average of normalized item scores
            checklist_evals = section_eval.get('checklist_evaluations', [])
            if checklist_evals:
                normalized_scores = [self._normalize_score(item.get('score', 0)) for item in checklist_evals]
                checklist_score = sum(normalized_scores) / len(normalized_scores)
            else:
                checklist_score = 0.0

            # Weighted combination
            section_score = (
                description_score * self.section_description_weight +
                checklist_score * self.section_checklist_weight
            )
            section_scores.append(section_score)

        # Average across all sections
        avg_section_score = sum(section_scores) / len(section_scores)
        return avg_section_score

    def _calculate_article_score(self, article_evaluation: Dict[str, Any]) -> float:
        """
        Calculate normalized article score.

        Returns:
            Normalized score in [0, 1]
        """
        if not article_evaluation or 'error' in article_evaluation:
            logger.warning("No valid article evaluation found")
            return 0.0

        # Extract and normalize scores based on score_mode
        coherence = self._normalize_score(article_evaluation.get('coherence_score', 0))
        fluency = self._normalize_score(article_evaluation.get('fluency_score', 0))
        repetition = self._normalize_score(article_evaluation.get('repetition_score', 0))
        termination = self._normalize_score(article_evaluation.get('termination_score', 0))

        # Weighted combination
        article_score = (
            coherence * self.article_coherence_weight +
            fluency * self.article_fluency_weight +
            repetition * self.article_repetition_weight +
            termination * self.article_termination_weight
        )

        return article_score

    def _calculate_citation_score(self, citation_evaluation: Dict[str, Any]) -> float:
        """
        Calculate normalized citation score (now: image-text coherence score).

        Returns:
            Normalized score in [0, 1]
        """
        if not citation_evaluation or 'error' in citation_evaluation:
            logger.warning("No valid citation evaluation found")
            return 0.0

        summary = citation_evaluation.get('summary', {})

        # Image-text coherence score (already 0-10, normalize to 0-1)
        overall_score = summary.get('overall_avg_score', 0) / 10.0

        return overall_score

    def _get_section_breakdown(self, section_evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get detailed breakdown of section scores."""
        if not section_evaluations:
            return {}

        section_details = []
        for section_eval in section_evaluations:
            raw_desc = section_eval.get('description_completion_score', 0)
            description_score = self._normalize_score(raw_desc)
            
            checklist_evals = section_eval.get('checklist_evaluations', [])
            if checklist_evals:
                normalized_scores = [self._normalize_score(item.get('score', 0)) for item in checklist_evals]
                checklist_score = sum(normalized_scores) / len(normalized_scores)
                completed_count = sum(1 for s in normalized_scores if s >= 0.5)  # Count scores >= 0.5 as "completed"
            else:
                checklist_score = 0.0
                completed_count = 0

            section_details.append({
                'section_index': section_eval.get('section_index', 0),
                'description_score_normalized': description_score,
                'checklist_score_normalized': checklist_score,
                'checklist_completed': completed_count,
                'checklist_total': len(checklist_evals)
            })

        return {
            'per_section': section_details,
            'avg_description_score': sum(s['description_score_normalized'] for s in section_details) / len(section_details),
            'avg_checklist_score': sum(s['checklist_score_normalized'] for s in section_details) / len(section_details)
        }

    def _get_article_breakdown(self, article_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of article scores."""
        if not article_evaluation:
            return {}

        return {
            'coherence_normalized': self._normalize_score(article_evaluation.get('coherence_score', 0)),
            'fluency_normalized': self._normalize_score(article_evaluation.get('fluency_score', 0)),
            'repetition_normalized': self._normalize_score(article_evaluation.get('repetition_score', 0)),
            'termination_normalized': self._normalize_score(article_evaluation.get('termination_score', 0))
        }

    def _get_citation_breakdown(self, citation_evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed breakdown of citation scores (now: image-text coherence)."""
        if not citation_evaluation:
            return {}

        summary = citation_evaluation.get('summary', {})
        return {
            'overall_score_normalized': summary.get('overall_avg_score', 0) / 10.0,
            'richness_score_normalized': summary.get('richness_avg_score', 0) / 10.0,
            'coherence_score_normalized': summary.get('coherence_avg_score', 0) / 10.0,
            'placement_score_normalized': summary.get('placement_avg_score', 0) / 10.0,
            'clarity_score_normalized': summary.get('clarity_avg_score', 0) / 10.0,
            'section_count': summary.get('section_count', 0),
            'total_images': summary.get('total_images', 0)
        }
