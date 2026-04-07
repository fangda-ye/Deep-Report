"""
Main evaluator for generated articles.
Combines section-level and article-level evaluation in a single LLM call.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from openai import OpenAI

from .prompts.eval_templates import SECTION_AND_ARTICLE_EVAL_TEMPLATE

logger = logging.getLogger(__name__)


class ArticleEvaluator:
    """
    Evaluates generated articles against their benchmark outlines.
    Performs section-level and article-level evaluation in one LLM call.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4.1",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        timeout: int = 300
    ):
        """
        Initialize the evaluator.

        Args:
            api_key: OpenAI API key
            base_url: API base URL (default OpenAI)
            model: Model name (default gpt-4.1)
            temperature: Temperature for generation
            max_tokens: Maximum tokens for response
            timeout: Request timeout in seconds
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        logger.info(f"Initialized ArticleEvaluator with model: {model}")

    def evaluate_article(
        self,
        generated_data: Dict[str, Any],
        benchmark_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate a generated article against its benchmark.

        Args:
            generated_data: Generated article data from session log
            benchmark_data: Benchmark data from deconstruction file

        Returns:
            Evaluation results including section and article scores
        """
        try:
            # Extract necessary information
            query = benchmark_data['deconstruction_result']['query']
            overall_checklist = benchmark_data['deconstruction_result']['overall_checklist']
            detailed_outline = benchmark_data['deconstruction_result']['detailed_outline']
            generated_sections = generated_data.get('sections', [])

            # Validate data
            if not generated_sections:
                logger.warning(f"No sections found in generated data for {generated_data.get('session_id')}")
                return self._create_error_result("No generated sections found")

            if len(generated_sections) != len(detailed_outline):
                logger.warning(
                    f"Section count mismatch: generated={len(generated_sections)}, "
                    f"expected={len(detailed_outline)}"
                )

            # Format sections data for prompt
            sections_data = self._format_sections_for_prompt(
                generated_sections,
                detailed_outline
            )

            # Build prompt
            prompt = SECTION_AND_ARTICLE_EVAL_TEMPLATE.format(
                query=query,
                overall_checklist=json.dumps(overall_checklist, indent=2, ensure_ascii=False),
                num_sections=len(generated_sections),
                sections_data=sections_data
            )

            # Call LLM
            logger.info(f"Calling {self.model} for evaluation...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert article evaluator. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout
            )

            # Parse response
            response_text = response.usage.completion_tokens and response.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from LLM")

            eval_result = self._parse_evaluation_response(response_text)

            # Add metadata
            eval_result['metadata'] = {
                'session_id': generated_data.get('session_id'),
                'user_id': generated_data.get('user_id'),
                'model_used': self.model,
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            logger.info("Successfully completed article evaluation")
            return eval_result

        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
            return self._create_error_result(str(e))

    def _format_sections_for_prompt(
        self,
        generated_sections: List[Dict[str, Any]],
        detailed_outline: List[Dict[str, Any]]
    ) -> str:
        """Format sections data into a string for the prompt."""
        sections_text = []

        for i, gen_section in enumerate(generated_sections):
            section_index = gen_section.get('section_index', i)

            # Find matching outline section
            outline_section = None
            for outline in detailed_outline:
                if i < len(detailed_outline):
                    outline_section = detailed_outline[i]
                    break

            if not outline_section:
                logger.warning(f"No outline found for section {section_index}")
                continue

            section_text = f"""
### Section {section_index}

**Expected Description:**
{outline_section.get('section_description', 'N/A')}

**Sectional Checklist:**
{json.dumps(outline_section.get('sectional_checklist', []), indent=2, ensure_ascii=False)}

**Generated Content:**
{gen_section.get('content', 'N/A')[:5000]}  # Limit to 5000 chars per section

**Content Length:** {gen_section.get('content_length', 0)} characters
**Text Citations:** {gen_section.get('text_citations_count', 0)}
**Image Citations:** {gen_section.get('image_citations_count', 0)}
---
"""
            sections_text.append(section_text)

        return "\n".join(sections_text)

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the JSON response from LLM."""
        # Try to extract JSON from response
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])

        try:
            eval_data = json.loads(response_text)

            # Validate structure
            if 'section_evaluations' not in eval_data:
                raise ValueError("Missing 'section_evaluations' in response")
            if 'article_evaluation' not in eval_data:
                raise ValueError("Missing 'article_evaluation' in response")

            return eval_data

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response text: {response_text[:500]}")
            raise ValueError(f"Invalid JSON response: {str(e)}")

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create an error result structure."""
        return {
            'error': error_message,
            'section_evaluations': [],
            'article_evaluation': {
                'coherence_score': 0,
                'fluency_score': 0,
                'repetition_score': 0,
                'termination_score': 0,
                'error': error_message
            }
        }


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def match_generated_to_benchmark(
    generated_data: List[Dict[str, Any]],
    benchmark_data: List[Dict[str, Any]]
) -> List[tuple]:
    """
    Match generated articles to their benchmark data.

    Returns:
        List of (generated, benchmark) tuples
    """
    # Create benchmark lookup by uid
    benchmark_map = {item['uid']: item for item in benchmark_data}

    matched_pairs = []
    for gen_item in generated_data:
        # Extract uid from user_id (format: "deconstruct_<uid>")
        user_id = gen_item.get('user_id', '')
        if user_id.startswith('deconstruct_'):
            uid = user_id.replace('deconstruct_', '')
            benchmark_item = benchmark_map.get(uid)
            if benchmark_item:
                matched_pairs.append((gen_item, benchmark_item))
            else:
                logger.warning(f"No benchmark found for uid: {uid}")
        else:
            logger.warning(f"Invalid user_id format: {user_id}")

    return matched_pairs
