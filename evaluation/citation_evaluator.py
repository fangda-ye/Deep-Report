"""
Image-text coherence evaluator for multimodal articles.
Evaluates the quality of image citations and their integration with text.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI

from .prompts.eval_templates import SECTION_IMAGE_TEXT_EVAL_TEMPLATE

logger = logging.getLogger(__name__)


class CitationEvaluator:
    """
    Evaluates image-text coherence in generated articles.
    Uses VLM to assess image relevance, necessity, and integration quality.
    """

    def __init__(
        self,
        # Image citation VLM (e.g., InternVL)
        image_api_key: str = "EMPTY",
        image_base_url: str = "http://localhost:9000/v1",
        image_model: str = "internvl35-38b",
        # VLM settings
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 300,
        # Legacy parameters (for backward compatibility)
        text_api_key: str = None,
        text_base_url: str = None,
        text_model: str = None
    ):
        """
        Initialize the image-text coherence evaluator.

        Args:
            image_api_key: API key for image citation VLM
            image_base_url: Base URL for image citation VLM
            image_model: Model name for image citations
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            timeout: Request timeout in seconds
            text_api_key: (Deprecated) No longer used
            text_base_url: (Deprecated) No longer used
            text_model: (Deprecated) No longer used
        """
        # Initialize image citation VLM client
        self.image_client = OpenAI(api_key=image_api_key, base_url=image_base_url)
        self.image_model = image_model

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(f"Initialized CitationEvaluator (image-text coherence mode) with VLM: {image_model}")

    def evaluate_citations(
        self,
        generated_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate image-text coherence at section level.

        Args:
            generated_data: Generated article data from session log

        Returns:
            Section-level image-text coherence evaluation results
        """
        try:
            sections = generated_data.get('sections', [])
            if not sections:
                logger.warning("No sections found in generated data")
                return self._create_empty_result()

            section_evaluations = []

            for section in sections:
                section_index = section.get('section_index', 0)
                section_description = section.get('section_description', '')
                content = section.get('content', '')
                image_sources = section.get('image_sources', [])

                # Evaluate entire section's image usage (including sections with no images)
                section_eval = self._evaluate_section_images(
                    section_index,
                    section_description,
                    content,
                    image_sources
                )
                section_evaluations.append(section_eval)

            # Aggregate results
            result = {
                'section_evaluations': section_evaluations,
                'summary': self._compute_summary(section_evaluations)
            }

            logger.info(
                f"Completed section-level image-text coherence evaluation: {len(section_evaluations)} sections"
            )
            return result

        except Exception as e:
            logger.error(f"Error during citation evaluation: {str(e)}", exc_info=True)
            return self._create_empty_result(error=str(e))


    def _evaluate_section_images(
        self,
        section_index: int,
        section_description: str,
        content: str,
        image_sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate all images in a section holistically."""
        import base64
        import os

        try:
            num_images = len(image_sources)

            # Build images_info string and collect image data
            images_info_parts = []
            image_contents = []  # For VLM multimodal message
            successfully_loaded_images = 0

            for idx, source in enumerate(image_sources, 1):
                citation_id = source.get('citation_id', '')
                image_path = source.get('img_path', '')

                # Read and convert image to base64
                if image_path and os.path.exists(image_path):
                    try:
                        with open(image_path, 'rb') as f:
                            image_data = f.read()
                            image_base64 = base64.b64encode(image_data).decode('utf-8')
                            image_contents.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                            })
                            successfully_loaded_images += 1
                            # Only add minimal identification info (NO description to avoid bias)
                            images_info_parts.append(
                                f"**Image {idx}** (Citation ID: {citation_id})"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to read image {image_path}: {e}")
                else:
                    logger.warning(f"Image file not found: {image_path}")

            if successfully_loaded_images == 0:
                images_info = "No images in this section."
            else:
                images_info = f"The section contains {successfully_loaded_images} image(s). Each image is provided below for your evaluation.\n" + "\n".join(images_info_parts)
                images_info += f"\n\n**IMPORTANT**: Evaluate based on what you SEE in the actual images, not on text descriptions."

            # Build prompt
            prompt_text = SECTION_IMAGE_TEXT_EVAL_TEMPLATE.format(
                section_description=section_description,
                section_content=content[:10000],  # Limit content length
                num_images=num_images,
                images_info=images_info
            )

            # Build multimodal message content
            message_content = [{"type": "text", "text": prompt_text}]
            message_content.extend(image_contents)

            # Call VLM
            messages = [
                {"role": "system", "content": "You are an expert evaluator for multimodal content in academic writing. Always respond with valid JSON."},
                {
                    "role": "user",
                    "content": message_content
                }
            ]

            response = self.image_client.chat.completions.create(
                model=self.image_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            # Parse response
            response_text = response.choices[0].message.content
            eval_result = self._parse_citation_response(response_text)

            # Calculate overall score as average of four dimensions
            # Ensure all scores are converted to float (in case VLM returns strings)
            try:
                richness = float(eval_result.get('richness_score', 0))
                coherence = float(eval_result.get('coherence_score', 0))
                placement = float(eval_result.get('placement_score', 0))
                clarity = float(eval_result.get('clarity_score', 0))
                overall_score = (richness + coherence + placement + clarity) / 4.0
            except (ValueError, TypeError) as e:
                logger.warning(f"Error converting scores to float: {e}. Using default values.")
                richness = coherence = placement = clarity = overall_score = 5.0

            # Update eval_result with float scores
            eval_result['richness_score'] = richness
            eval_result['coherence_score'] = coherence
            eval_result['placement_score'] = placement
            eval_result['clarity_score'] = clarity
            eval_result['overall_score'] = overall_score

            # Add section info
            eval_result['section_index'] = section_index
            eval_result['num_images'] = num_images

            return eval_result

        except Exception as e:
            logger.error(f"Error evaluating section {section_index}: {str(e)}")
            return {
                'section_index': section_index,
                'num_images': len(image_sources),
                'error': str(e),
                'overall_score': 5.0,
                'richness_score': 5.0,
                'coherence_score': 5.0,
                'placement_score': 5.0,
                'clarity_score': 5.0
            }



    def _parse_citation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from citation evaluation."""
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])

        try:
            eval_data = json.loads(response_text)

            # Validate and clean scores
            required_scores = ['richness_score', 'coherence_score', 'placement_score', 'clarity_score']
            for score_key in required_scores:
                if score_key not in eval_data:
                    logger.warning(f"Missing {score_key} in response, setting to 0")
                    eval_data[score_key] = 0
                else:
                    # Try to convert score to float, handle non-numeric values
                    score_value = eval_data[score_key]
                    try:
                        # If it's already a number, keep it
                        if isinstance(score_value, (int, float)):
                            eval_data[score_key] = float(score_value)
                        # If it's a string, try to convert
                        elif isinstance(score_value, str):
                            # Remove common non-numeric strings
                            if score_value.upper() in ['N/A', 'NA', 'NONE', 'NULL', '']:
                                logger.warning(f"{score_key} is '{score_value}', setting to 0")
                                eval_data[score_key] = 0.0
                            else:
                                # Try to extract number from string
                                eval_data[score_key] = float(score_value)
                        else:
                            logger.warning(f"{score_key} has invalid type {type(score_value)}, setting to 0")
                            eval_data[score_key] = 0.0
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot convert {score_key}='{score_value}' to float, setting to 0")
                        eval_data[score_key] = 0.0

            return eval_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse citation evaluation JSON: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            return {
                'richness_score': 0.0,
                'coherence_score': 0.0,
                'placement_score': 0.0,
                'clarity_score': 0.0,
                'error': f"JSON parse error: {str(e)}"
            }

    def _compute_summary(
        self,
        section_evaluations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute summary statistics for section-level image-text coherence evaluations."""
        def safe_avg(scores):
            if not scores:
                return 0.0
            return sum(scores) / len(scores)

        # Extract scores from evaluations (excluding errors)
        valid_evals = [e for e in section_evaluations if 'error' not in e]

        overall_scores = [e.get('overall_score', 0) for e in valid_evals]
        richness_scores = [e.get('richness_score', 0) for e in valid_evals]
        coherence_scores = [e.get('coherence_score', 0) for e in valid_evals]
        placement_scores = [e.get('placement_score', 0) for e in valid_evals]
        clarity_scores = [e.get('clarity_score', 0) for e in valid_evals]

        # Count total images across all sections
        total_images = sum(e.get('num_images', 0) for e in section_evaluations)

        return {
            'section_count': len(section_evaluations),
            'valid_evaluations': len(valid_evals),
            'total_images': total_images,
            'overall_avg_score': safe_avg(overall_scores),
            'richness_avg_score': safe_avg(richness_scores),
            'coherence_avg_score': safe_avg(coherence_scores),
            'placement_avg_score': safe_avg(placement_scores),
            'clarity_avg_score': safe_avg(clarity_scores),
            'min_overall_score': min(overall_scores) if overall_scores else 0,
            'max_overall_score': max(overall_scores) if overall_scores else 0
        }

    def _create_empty_result(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Create an empty result structure."""
        result = {
            'section_evaluations': [],
            'summary': {
                'section_count': 0,
                'valid_evaluations': 0,
                'total_images': 0,
                'overall_avg_score': 0,
                'richness_avg_score': 0,
                'coherence_avg_score': 0,
                'placement_avg_score': 0,
                'clarity_avg_score': 0,
                'min_overall_score': 0,
                'max_overall_score': 0
            }
        }
        if error:
            result['error'] = error
        return result
