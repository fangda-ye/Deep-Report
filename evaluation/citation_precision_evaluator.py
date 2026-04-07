"""
Citation Precision Evaluator

Evaluates the precision of citation selection at three stages:
1. Raw retrieval results (from search)
2. After filtering
3. Final selected citations in the generated article

Precision = (Number of GT citations found) / (Number of citations used)
"""

import json
import logging
from typing import Dict, Any, List, Set, Tuple

logger = logging.getLogger(__name__)


class CitationPrecisionEvaluator:
    """
    Evaluates citation selection quality by measuring precision against ground truth.
    """

    def __init__(self):
        """Initialize the evaluator."""
        self.results = []

    def load_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    def extract_gt_citations(
        self,
        enriched_data: Dict[str, Any],
        section_index: int
    ) -> Tuple[Set[str], Set[str]]:
        """
        Extract ground truth citation IDs for a specific section.

        Returns:
            Tuple of (gt_text_ids, gt_image_ids)
        """
        deconstruction = enriched_data.get('deconstruction_result', {})
        detailed_outline = deconstruction.get('detailed_outline', [])

        if section_index >= len(detailed_outline):
            logger.warning(f"Section {section_index} not found in enriched data")
            return set(), set()

        section = detailed_outline[section_index]
        enrichment = section.get('enrichment', {})

        # Extract text GT IDs
        relevant_texts = enrichment.get('relevant_texts', [])
        gt_text_ids = set()
        for text in relevant_texts:
            text_id = text.get('id')
            if text_id:
                gt_text_ids.add(text_id)

        # Extract image GT IDs
        relevant_images = enrichment.get('relevant_images', [])
        gt_image_ids = set()
        for img in relevant_images:
            img_id = img.get('id')
            if img_id:
                gt_image_ids.add(img_id)

        return gt_text_ids, gt_image_ids

    def extract_search_citations(
        self,
        generated_data: Dict[str, Any],
        section_index: int
    ) -> Tuple[Set[str], Set[str]]:
        """
        Extract citation IDs from raw search results.

        Returns:
            Tuple of (text_ids, image_ids)
        """
        search_calls = generated_data.get('search_calls', [])

        # Find the search call for this section (section_index + 1 because call_id starts from 1)
        target_call_id = section_index + 1
        search_result = None
        for call in search_calls:
            if call.get('call_id') == target_call_id:
                search_result = call
                break

        if not search_result:
            logger.warning(f"Search call not found for section {section_index}")
            return set(), set()

        # Extract text IDs
        text_results = search_result.get('text_results', [])
        text_ids = set()
        for result in text_results:
            uid = result.get('uid')
            if uid:
                text_ids.add(uid)

        # Extract image IDs
        image_results = search_result.get('image_results', [])
        image_ids = set()
        for result in image_results:
            uid = result.get('uid')
            if uid:
                image_ids.add(uid)

        return text_ids, image_ids

    def extract_filtered_citations(
        self,
        generated_data: Dict[str, Any],
        section_index: int
    ) -> Tuple[Set[str], Set[str]]:
        """
        Extract citation IDs after filtering.

        Returns:
            Tuple of (text_ids, image_ids)
        """
        search_calls = generated_data.get('search_calls', [])
        filter_calls = generated_data.get('filter_calls', [])

        # Find search and filter calls for this section
        target_call_id = section_index + 1

        search_result = None
        for call in search_calls:
            if call.get('call_id') == target_call_id:
                search_result = call
                break

        filter_result = None
        for call in filter_calls:
            if call.get('section_index') == section_index:
                filter_result = call
                break

        if not search_result or not filter_result:
            logger.warning(f"Search or filter call not found for section {section_index}")
            return set(), set()

        # Extract filtered text IDs
        text_results = search_result.get('text_results', [])
        text_filtering = filter_result.get('text_filtering', {})
        kept_text_indices = text_filtering.get('kept_indices', [])

        # Convert to 0-based indices (kept_indices are 1-based in the data)
        text_ids = set()
        for idx in kept_text_indices:
            # idx is 1-based, convert to 0-based
            actual_idx = idx - 1
            if 0 <= actual_idx < len(text_results):
                uid = text_results[actual_idx].get('uid')
                if uid:
                    text_ids.add(uid)

        # Extract filtered image IDs
        image_results = search_result.get('image_results', [])
        image_filtering = filter_result.get('image_filtering', {})
        decisions = image_filtering.get('decisions', [])

        image_ids = set()
        for decision in decisions:
            if decision.get('kept', False):
                # decision['index'] is 1-based
                idx = decision.get('index', 0) - 1
                if 0 <= idx < len(image_results):
                    uid = image_results[idx].get('uid')
                    if uid:
                        image_ids.add(uid)

        return text_ids, image_ids

    def extract_final_citations(
        self,
        generated_data: Dict[str, Any],
        section_index: int
    ) -> Tuple[Set[str], Set[str]]:
        """
        Extract citation IDs from final generated article.

        Returns:
            Tuple of (text_ids, image_ids)
        """
        sections = generated_data.get('sections', [])

        # Find the section
        target_section = None
        for section in sections:
            if section.get('section_index') == section_index:
                target_section = section
                break

        if not target_section:
            logger.warning(f"Section {section_index} not found in generated data")
            return set(), set()

        # Extract text IDs
        text_sources = target_section.get('text_sources', [])
        text_ids = set()
        for source in text_sources:
            uid = source.get('uid')
            if uid:
                text_ids.add(uid)

        # Extract image IDs
        image_sources = target_section.get('image_sources', [])
        image_ids = set()
        for source in image_sources:
            uid = source.get('uid')
            if uid:
                image_ids.add(uid)

        return text_ids, image_ids

    def calculate_precision(
        self,
        retrieved_ids: Set[str],
        gt_ids: Set[str]
    ) -> Tuple[float, int, int]:
        """
        Calculate precision: (retrieved ∩ GT) / retrieved

        Returns:
            Tuple of (precision, num_found, num_used)
            - precision: Precision score between 0 and 1
            - num_found: Number of GT citations found
            - num_used: Total number of citations used
        """
        if not retrieved_ids:
            return 0.0, 0, 0

        intersection = retrieved_ids & gt_ids
        num_found = len(intersection)
        num_used = len(retrieved_ids)
        precision = num_found / num_used if num_used > 0 else 0.0

        return precision, num_found, num_used

    def evaluate_article(
        self,
        generated_data: Dict[str, Any],
        enriched_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate citation recall for a single article.

        Returns:
            Evaluation results with recall scores for each stage
        """
        uid = enriched_data.get('uid', 'unknown')
        session_id = generated_data.get('session_id', 'unknown')

        logger.info(f"Evaluating article: {uid} (session: {session_id})")

        # Get number of sections
        deconstruction = enriched_data.get('deconstruction_result', {})
        detailed_outline = deconstruction.get('detailed_outline', [])
        num_sections = len(detailed_outline)

        section_results = []
        overall_stats = {
            'text': {'search': [], 'filter': [], 'final': []},
            'image': {'search': [], 'filter': [], 'final': []}
        }

        for section_idx in range(num_sections):
            try:
                # Get GT citations
                gt_text_ids, gt_image_ids = self.extract_gt_citations(enriched_data, section_idx)

                # Get citations at each stage
                search_text_ids, search_image_ids = self.extract_search_citations(generated_data, section_idx)
                filter_text_ids, filter_image_ids = self.extract_filtered_citations(generated_data, section_idx)
                final_text_ids, final_image_ids = self.extract_final_citations(generated_data, section_idx)

                # Calculate precisions
                text_prec_search, text_found_search, text_used_search = self.calculate_precision(search_text_ids, gt_text_ids)
                text_prec_filter, text_found_filter, text_used_filter = self.calculate_precision(filter_text_ids, gt_text_ids)
                text_prec_final, text_found_final, text_used_final = self.calculate_precision(final_text_ids, gt_text_ids)

                image_prec_search, image_found_search, image_used_search = self.calculate_precision(search_image_ids, gt_image_ids)
                image_prec_filter, image_found_filter, image_used_filter = self.calculate_precision(filter_image_ids, gt_image_ids)
                image_prec_final, image_found_final, image_used_final = self.calculate_precision(final_image_ids, gt_image_ids)

                section_result = {
                    'section_index': section_idx,
                    'gt_counts': {
                        'text': len(gt_text_ids),
                        'image': len(gt_image_ids)
                    },
                    'precisions': {
                        'text': {
                            'search': {
                                'precision': text_prec_search,
                                'found': text_found_search,
                                'used': text_used_search
                            },
                            'filter': {
                                'precision': text_prec_filter,
                                'found': text_found_filter,
                                'used': text_used_filter
                            },
                            'final': {
                                'precision': text_prec_final,
                                'found': text_found_final,
                                'used': text_used_final
                            }
                        },
                        'image': {
                            'search': {
                                'precision': image_prec_search,
                                'found': image_found_search,
                                'used': image_used_search
                            },
                            'filter': {
                                'precision': image_prec_filter,
                                'found': image_found_filter,
                                'used': image_used_filter
                            },
                            'final': {
                                'precision': image_prec_final,
                                'found': image_found_final,
                                'used': image_used_final
                            }
                        }
                    }
                }

                section_results.append(section_result)

                # Accumulate stats (only if citations were used)
                if text_used_search > 0:
                    overall_stats['text']['search'].append(text_prec_search)
                if text_used_filter > 0:
                    overall_stats['text']['filter'].append(text_prec_filter)
                if text_used_final > 0:
                    overall_stats['text']['final'].append(text_prec_final)

                if image_used_search > 0:
                    overall_stats['image']['search'].append(image_prec_search)
                if image_used_filter > 0:
                    overall_stats['image']['filter'].append(image_prec_filter)
                if image_used_final > 0:
                    overall_stats['image']['final'].append(image_prec_final)

            except Exception as e:
                logger.error(f"Error evaluating section {section_idx}: {str(e)}", exc_info=True)
                continue

        # Calculate average precisions
        def safe_avg(values):
            return sum(values) / len(values) if values else 0.0

        avg_precisions = {
            'text': {
                'search': safe_avg(overall_stats['text']['search']),
                'filter': safe_avg(overall_stats['text']['filter']),
                'final': safe_avg(overall_stats['text']['final'])
            },
            'image': {
                'search': safe_avg(overall_stats['image']['search']),
                'filter': safe_avg(overall_stats['image']['filter']),
                'final': safe_avg(overall_stats['image']['final'])
            }
        }

        result = {
            'uid': uid,
            'session_id': session_id,
            'num_sections': num_sections,
            'section_results': section_results,
            'average_precisions': avg_precisions
        }

        logger.info(
            f"  Text precisions: search={avg_precisions['text']['search']:.3f}, "
            f"filter={avg_precisions['text']['filter']:.3f}, "
            f"final={avg_precisions['text']['final']:.3f}"
        )
        logger.info(
            f"  Image precisions: search={avg_precisions['image']['search']:.3f}, "
            f"filter={avg_precisions['image']['filter']:.3f}, "
            f"final={avg_precisions['image']['final']:.3f}"
        )

        return result

    def match_data(
        self,
        generated_data: List[Dict[str, Any]],
        enriched_data: List[Dict[str, Any]]
    ) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Match generated data with enriched data by UID.

        Returns:
            List of (generated, enriched) tuples
        """
        # Create enriched lookup by uid
        enriched_map = {item['uid']: item for item in enriched_data}

        matched_pairs = []
        for gen_item in generated_data:
            # Extract uid from user_id (format: "deconstruct_<uid>")
            user_id = gen_item.get('user_id', '')
            if user_id.startswith('deconstruct_'):
                uid = user_id.replace('deconstruct_', '')
                enriched_item = enriched_map.get(uid)
                if enriched_item:
                    matched_pairs.append((gen_item, enriched_item))
                else:
                    logger.warning(f"No enriched data found for uid: {uid}")
            else:
                logger.warning(f"Invalid user_id format: {user_id}")

        return matched_pairs


def compute_aggregate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all articles.
    """
    text_precisions = {'search': [], 'filter': [], 'final': []}
    image_precisions = {'search': [], 'filter': [], 'final': []}

    total_found = {'text': {'search': 0, 'filter': 0, 'final': 0},
                   'image': {'search': 0, 'filter': 0, 'final': 0}}
    total_used = {'text': {'search': 0, 'filter': 0, 'final': 0},
                  'image': {'search': 0, 'filter': 0, 'final': 0}}

    for result in results:
        avg_precisions = result.get('average_precisions', {})

        # Text precisions
        text = avg_precisions.get('text', {})
        if text.get('search') is not None:
            text_precisions['search'].append(text.get('search', 0))
        if text.get('filter') is not None:
            text_precisions['filter'].append(text.get('filter', 0))
        if text.get('final') is not None:
            text_precisions['final'].append(text.get('final', 0))

        # Image precisions
        image = avg_precisions.get('image', {})
        if image.get('search') is not None:
            image_precisions['search'].append(image.get('search', 0))
        if image.get('filter') is not None:
            image_precisions['filter'].append(image.get('filter', 0))
        if image.get('final') is not None:
            image_precisions['final'].append(image.get('final', 0))

        # Accumulate totals
        for section_result in result.get('section_results', []):
            precisions = section_result.get('precisions', {})

            for modality in ['text', 'image']:
                for stage in ['search', 'filter', 'final']:
                    stage_data = precisions.get(modality, {}).get(stage, {})
                    total_found[modality][stage] += stage_data.get('found', 0)
                    total_used[modality][stage] += stage_data.get('used', 0)

    def safe_avg(values):
        return sum(values) / len(values) if values else 0.0

    # Calculate overall precision (total found / total used)
    overall_precision = {}
    for modality in ['text', 'image']:
        overall_precision[modality] = {}
        for stage in ['search', 'filter', 'final']:
            if total_used[modality][stage] > 0:
                overall_precision[modality][stage] = total_found[modality][stage] / total_used[modality][stage]
            else:
                overall_precision[modality][stage] = 0.0

    return {
        'num_articles': len(results),
        'average_precisions': {
            'text': {
                'search': safe_avg(text_precisions['search']),
                'filter': safe_avg(text_precisions['filter']),
                'final': safe_avg(text_precisions['final'])
            },
            'image': {
                'search': safe_avg(image_precisions['search']),
                'filter': safe_avg(image_precisions['filter']),
                'final': safe_avg(image_precisions['final'])
            }
        },
        'overall_precision': overall_precision,
        'total_found': total_found,
        'total_used': total_used
    }
