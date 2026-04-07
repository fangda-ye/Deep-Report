"""
Complete Batch Evaluation Script - Article Quality + Search Precision

Evaluates both:
1. Article Quality (full article, sections, image-text coherence) - saved to data/eval_results/article/
2. Search Precision (3-stage retrieval precision) - saved to data/eval_results/search/

Usage:
    1. Configure the INPUT_FILES list at the top of this script
    2. Configure API keys and settings
    3. Run: python -m eval.batch_evaluate_all
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .evaluator import ArticleEvaluator, load_jsonl, match_generated_to_benchmark
from .citation_evaluator import CitationEvaluator
from .score_calculator import ScoreCalculator
from .citation_precision_evaluator import CitationPrecisionEvaluator, compute_aggregate_stats

INPUT_FILES = [
]

# Benchmark files
BENCHMARK_FILE = "data/article_deconstructions.jsonl"  # For article evaluation
ENRICHED_FILE = "data/article_deconstructions_enriched.jsonl"  # For search evaluation

# Output directories
ARTICLE_OUTPUT_DIR = "data/eval_results/article"  # Article quality results
SEARCH_OUTPUT_DIR = "data/eval_results/search"    # Search precision results

# API Configuration
API_CONFIG = {
    # GPT-4 for article evaluation
    "gpt4_key": os.getenv("GPT4_API_KEY", ""),
    "gpt4_base_url": os.getenv("GPT4_BASE_URL", "https://api.openai.com/v1"),
    "gpt4_model": "gpt-4.1",

    # Qwen for text citation evaluation
    "qwen_key": os.getenv("QWEN_API_KEY", ""),
    "qwen_base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    "qwen_model": "qwen-plus-latest",

    # VLM for image citation evaluation
    "vlm_base_url": os.getenv("VLM_BASE_URL", "http://localhost:9000/v1"),
    "vlm_model": os.getenv("VLM_MODEL", "internvl35-38b"),
    "vlm_key": os.getenv("VLM_API_KEY", "EMPTY"),
}

# Evaluation options
SKIP_CITATIONS = False  # Set to True to skip image-text coherence evaluation
SKIP_SEARCH = False     # Set to True to skip search precision evaluation
LIMIT = 3  # Limit number of articles per file (None = no limit)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments (optional, overrides configuration)."""
    parser = argparse.ArgumentParser(
        description="Complete batch evaluation: article quality + search precision"
    )

    # Optional overrides
    parser.add_argument(
        '--input-files',
        type=str,
        nargs='+',
        help='Override INPUT_FILES: List of generated articles JSONL files'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        help='Override BENCHMARK_FILE: Path to benchmark deconstructions'
    )
    parser.add_argument(
        '--enriched',
        type=str,
        help='Override ENRICHED_FILE: Path to enriched deconstructions'
    )
    parser.add_argument(
        '--article-output',
        type=str,
        help='Override ARTICLE_OUTPUT_DIR: Output directory for article results'
    )
    parser.add_argument(
        '--search-output',
        type=str,
        help='Override SEARCH_OUTPUT_DIR: Output directory for search results'
    )
    parser.add_argument(
        '--skip-citations',
        action='store_true',
        help='Skip image-text coherence evaluation'
    )
    parser.add_argument(
        '--skip-search',
        action='store_true',
        help='Skip search precision evaluation'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of articles to evaluate per file'
    )

    return parser.parse_args()


def get_output_filename(input_file: str, output_dir: str, suffix: str) -> str:
    """
    Generate output filename based on input file.

    Args:
        input_file: Path to input file
        output_dir: Output directory
        suffix: Suffix to add (e.g., 'article', 'search')

    Returns:
        Full path to output file
    """
    input_filename = Path(input_file).stem
    output_filename = f"{input_filename}_{suffix}.jsonl"
    return os.path.join(output_dir, output_filename)


def evaluate_article_quality(
    generated_data: Dict[str, Any],
    benchmark_data: Dict[str, Any],
    article_evaluator: ArticleEvaluator,
    citation_evaluator: CitationEvaluator,
    score_calculator: ScoreCalculator,
    skip_citations: bool = False
) -> Dict[str, Any]:
    """
    Evaluate article quality (section + article + image-text coherence).

    Returns:
        Complete article evaluation result
    """
    session_id = generated_data.get('session_id', 'unknown')
    logger.info(f"  [Article] Evaluating: {session_id}")

    try:
        # Step 1: Article-level evaluation (section + article)
        logger.info("    Step 1/3: Section and article evaluation...")
        article_eval = article_evaluator.evaluate_article(generated_data, benchmark_data)

        # Step 2: Image-text coherence evaluation
        if skip_citations:
            logger.info("    Step 2/3: Skipping image-text coherence evaluation")
            citation_eval = {
                'section_evaluations': [],
                'summary': {
                    'section_count': 0,
                    'valid_evaluations': 0,
                    'total_images': 0,
                    'overall_avg_score': 10,
                    'richness_avg_score': 10,
                    'coherence_avg_score': 10,
                    'placement_avg_score': 10,
                    'clarity_avg_score': 10,
                    'min_overall_score': 10,
                    'max_overall_score': 10
                },
                'skipped': True
            }
        else:
            logger.info("    Step 2/3: Image-text coherence evaluation...")
            citation_eval = citation_evaluator.evaluate_citations(generated_data)

        # Step 3: Calculate final score
        logger.info("    Step 3/3: Calculating final score...")
        final_scores = score_calculator.calculate_final_score(article_eval, citation_eval)

        result = {
            'session_id': session_id,
            'user_id': generated_data.get('user_id', ''),
            'uid': benchmark_data.get('uid', ''),
            'timestamp': datetime.now().isoformat(),
            'article_evaluation': article_eval,
            'citation_evaluation': citation_eval,
            'final_scores': final_scores
        }

        logger.info(f"    ✓ Article eval completed: Final score = {final_scores.get('final_score', 0):.2f}")
        return result

    except Exception as e:
        logger.error(f"    ✗ Error in article evaluation: {str(e)}", exc_info=True)
        return {
            'session_id': session_id,
            'user_id': generated_data.get('user_id', ''),
            'uid': benchmark_data.get('uid', ''),
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'final_scores': {'final_score': 0}
        }


def evaluate_search_precision(
    generated_data: Dict[str, Any],
    enriched_data: Dict[str, Any],
    precision_evaluator: CitationPrecisionEvaluator
) -> Dict[str, Any]:
    """
    Evaluate search precision (search -> filter -> final).

    Returns:
        Search precision evaluation result
    """
    session_id = generated_data.get('session_id', 'unknown')
    logger.info(f"  [Search] Evaluating: {session_id}")

    try:
        result = precision_evaluator.evaluate_article(generated_data, enriched_data)

        avg_prec = result.get('average_precisions', {})
        text_final = avg_prec.get('text', {}).get('final', 0)
        image_final = avg_prec.get('image', {}).get('final', 0)

        logger.info(f"    ✓ Search eval completed: Text={text_final:.3f}, Image={image_final:.3f}")
        return result

    except Exception as e:
        logger.error(f"    ✗ Error in search evaluation: {str(e)}", exc_info=True)
        return {
            'session_id': session_id,
            'uid': enriched_data.get('uid', ''),
            'error': str(e)
        }


def process_single_file(
    input_file: str,
    benchmark_data: List[Dict[str, Any]],
    enriched_data: List[Dict[str, Any]],
    article_evaluator: ArticleEvaluator,
    citation_evaluator: CitationEvaluator,
    score_calculator: ScoreCalculator,
    precision_evaluator: CitationPrecisionEvaluator,
    article_output_file: str,
    search_output_file: str,
    skip_citations: bool = False,
    skip_search: bool = False,
    limit: int = None
) -> Dict[str, Any]:
    """
    Process a single input file with both article and search evaluation.

    Returns:
        Summary statistics for this file
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing file: {input_file}")
    logger.info(f"{'='*80}")

    # Load generated data
    try:
        generated_data = load_jsonl(input_file)
        logger.info(f"  Loaded {len(generated_data)} generated articles")
    except Exception as e:
        logger.error(f"  Error loading file: {str(e)}")
        return {'error': str(e), 'success_count': 0, 'error_count': 0}

    # Match for article evaluation
    logger.info("  Matching for article evaluation...")
    article_pairs = match_generated_to_benchmark(generated_data, benchmark_data)
    logger.info(f"  Matched {len(article_pairs)} article pairs")

    # Match for search evaluation
    search_pairs = []
    if not skip_search:
        logger.info("  Matching for search evaluation...")
        search_pairs = precision_evaluator.match_data(generated_data, enriched_data)
        logger.info(f"  Matched {len(search_pairs)} search pairs")

    if not article_pairs and not search_pairs:
        logger.warning("  No matched pairs found! Skipping this file.")
        return {'success_count': 0, 'error_count': 0}

    # Use the smaller of the two for consistency
    num_to_process = min(len(article_pairs), len(search_pairs)) if search_pairs else len(article_pairs)

    # Apply limit if specified
    if limit:
        num_to_process = min(num_to_process, limit)
        logger.info(f"  Limited to first {limit} pairs for testing")

    # Evaluate all articles
    logger.info(f"\n  Evaluating {num_to_process} articles...")
    logger.info(f"  {'-'*78}")

    article_success = 0
    search_success = 0
    error_count = 0
    final_scores = []
    search_results = []

    for i in range(num_to_process):
        logger.info(f"\n  [{i+1}/{num_to_process}]")

        # Article evaluation
        if i < len(article_pairs):
            gen_data, bench_data = article_pairs[i]
            article_result = evaluate_article_quality(
                gen_data,
                bench_data,
                article_evaluator,
                citation_evaluator,
                score_calculator,
                skip_citations=skip_citations
            )

            # Save article result
            with open(article_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(article_result, ensure_ascii=False) + '\n')

            if 'error' not in article_result:
                article_success += 1
                final_scores.append(article_result['final_scores']['final_score'])
            else:
                error_count += 1

        # Search evaluation
        if not skip_search and i < len(search_pairs):
            gen_data, enrich_data = search_pairs[i]
            search_result = evaluate_search_precision(
                gen_data,
                enrich_data,
                precision_evaluator
            )

            # Save search result
            with open(search_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(search_result, ensure_ascii=False) + '\n')

            if 'error' not in search_result:
                search_success += 1
                search_results.append(search_result)

    # Calculate summary
    summary = {
        'input_file': input_file,
        'total_articles': num_to_process,
        'article_success': article_success,
        'search_success': search_success,
        'error_count': error_count,
        'article_output': article_output_file,
        'search_output': search_output_file
    }

    if final_scores:
        summary['avg_article_score'] = sum(final_scores) / len(final_scores)
        summary['min_article_score'] = min(final_scores)
        summary['max_article_score'] = max(final_scores)

    if search_results:
        # Compute aggregate search stats
        search_stats = compute_aggregate_stats(search_results)
        summary['search_precision'] = search_stats['average_precisions']

    return summary


def main():
    """Main execution function."""
    args = parse_args()

    # Use command-line arguments if provided, otherwise use configuration
    input_files = args.input_files if args.input_files else INPUT_FILES
    benchmark_file = args.benchmark if args.benchmark else BENCHMARK_FILE
    enriched_file = args.enriched if args.enriched else ENRICHED_FILE
    article_output_dir = args.article_output if args.article_output else ARTICLE_OUTPUT_DIR
    search_output_dir = args.search_output if args.search_output else SEARCH_OUTPUT_DIR
    skip_citations = args.skip_citations if args.skip_citations else SKIP_CITATIONS
    skip_search = args.skip_search if args.skip_search else SKIP_SEARCH
    limit = args.limit if args.limit else LIMIT

    # Validate configuration
    if not input_files:
        logger.error("No input files configured! Please edit INPUT_FILES or use --input-files")
        return

    logger.info("="*80)
    logger.info("COMPLETE BATCH EVALUATION - ARTICLE + SEARCH")
    logger.info("="*80)
    logger.info(f"Input files: {len(input_files)}")
    for f in input_files:
        logger.info(f"  - {f}")
    logger.info(f"Benchmark (article): {benchmark_file}")
    logger.info(f"Enriched (search): {enriched_file}")
    logger.info(f"Article output: {article_output_dir}")
    logger.info(f"Search output: {search_output_dir}")
    logger.info(f"Skip citations: {skip_citations}")
    logger.info(f"Skip search: {skip_search}")
    logger.info(f"Limit per file: {limit if limit else 'None'}")
    logger.info("="*80)

    # Create output directories
    Path(article_output_dir).mkdir(parents=True, exist_ok=True)
    Path(search_output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directories")

    # Load benchmark data (for article evaluation)
    logger.info(f"\nLoading benchmark data from: {benchmark_file}")
    try:
        benchmark_data = load_jsonl(benchmark_file)
        logger.info(f"  Loaded {len(benchmark_data)} benchmark articles")
    except Exception as e:
        logger.error(f"  Error loading benchmark: {str(e)}")
        return

    # Load enriched data (for search evaluation)
    enriched_data = []
    if not skip_search:
        logger.info(f"Loading enriched data from: {enriched_file}")
        try:
            precision_evaluator = CitationPrecisionEvaluator()
            enriched_data = precision_evaluator.load_jsonl(enriched_file)
            logger.info(f"  Loaded {len(enriched_data)} enriched articles")
        except Exception as e:
            logger.error(f"  Error loading enriched data: {str(e)}")
            logger.warning("  Search evaluation will be skipped")
            skip_search = True

    # Initialize evaluators
    logger.info("\nInitializing evaluators...")

    article_evaluator = ArticleEvaluator(
        api_key=API_CONFIG['gpt4_key'],
        base_url=API_CONFIG['gpt4_base_url'],
        model=API_CONFIG['gpt4_model']
    )

    citation_evaluator = CitationEvaluator(
        text_api_key=API_CONFIG['qwen_key'],
        text_base_url=API_CONFIG['qwen_base_url'],
        text_model=API_CONFIG['qwen_model'],
        image_api_key=API_CONFIG['vlm_key'],
        image_base_url=API_CONFIG['vlm_base_url'],
        image_model=API_CONFIG['vlm_model']
    )

    score_calculator = ScoreCalculator()

    precision_evaluator = CitationPrecisionEvaluator() if not skip_search else None

    logger.info("Evaluators initialized successfully")

    # Process each input file
    all_summaries = []

    for input_file in input_files:
        # Generate output filenames
        article_output_file = get_output_filename(
            input_file, article_output_dir, 'article'
        )
        search_output_file = get_output_filename(
            input_file, search_output_dir, 'search'
        )

        # Process file
        summary = process_single_file(
            input_file=input_file,
            benchmark_data=benchmark_data,
            enriched_data=enriched_data,
            article_evaluator=article_evaluator,
            citation_evaluator=citation_evaluator,
            score_calculator=score_calculator,
            precision_evaluator=precision_evaluator,
            article_output_file=article_output_file,
            search_output_file=search_output_file,
            skip_citations=skip_citations,
            skip_search=skip_search,
            limit=limit
        )

        all_summaries.append(summary)

    # Print final summary
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)

    total_articles = sum(s.get('total_articles', 0) for s in all_summaries)
    total_article_success = sum(s.get('article_success', 0) for s in all_summaries)
    total_search_success = sum(s.get('search_success', 0) for s in all_summaries)
    total_errors = sum(s.get('error_count', 0) for s in all_summaries)

    logger.info(f"Files processed: {len(input_files)}")
    logger.info(f"Total articles evaluated: {total_articles}")
    logger.info(f"Article evaluations successful: {total_article_success}")
    logger.info(f"Search evaluations successful: {total_search_success}")
    logger.info(f"Errors: {total_errors}")
    logger.info("")

    for i, summary in enumerate(all_summaries, 1):
        logger.info(f"\nFile {i}: {Path(summary.get('input_file', '')).name}")
        logger.info(f"  Articles: {summary.get('total_articles', 0)}")
        logger.info(f"  Article success: {summary.get('article_success', 0)}")
        logger.info(f"  Search success: {summary.get('search_success', 0)}")
        logger.info(f"  Errors: {summary.get('error_count', 0)}")

        if 'avg_article_score' in summary:
            logger.info(f"  Avg article score: {summary['avg_article_score']:.2f}")

        if 'search_precision' in summary:
            text_prec = summary['search_precision']['text']
            image_prec = summary['search_precision']['image']
            logger.info(f"  Search precision (text): final={text_prec['final']:.3f}")
            logger.info(f"  Search precision (image): final={image_prec['final']:.3f}")

        logger.info(f"  Article output: {summary.get('article_output', 'N/A')}")
        logger.info(f"  Search output: {summary.get('search_output', 'N/A')}")

    logger.info("\n" + "="*80)
    logger.info("Batch evaluation completed!")
    logger.info(f"Article results saved to: {article_output_dir}/")
    logger.info(f"Search results saved to: {search_output_dir}/")
    logger.info("="*80)


if __name__ == '__main__':
    main()
