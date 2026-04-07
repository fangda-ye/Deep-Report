"""
Batch evaluation script for generated articles.

Usage:
python -m eval.batch_evaluate \
    --generated session_logs/exp_logs/batch_generation_qwen3_32b_filter.jsonl \
    --benchmark data/article_deconstructions.jsonl \
    --output data/eval_results/qwen3_32b_thinking_eval.jsonl \
    --gpt4-key $GPT4_API_KEY \
    --gpt4-base-url https://api.openai.com/v1 \
    --gpt4-model gpt-4.1 \
    --qwen-key $QWEN_API_KEY \
    --qwen-base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --qwen-model qwen-plus-latest \
    --vlm-base-url http://localhost:9000/v1 \
    --vlm-model internvl35-38b \
    --vlm-key EMPTY \
    --skip-citations \
    --limit 3
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Batch evaluate generated articles against benchmark"
    )

    # Input/Output files
    parser.add_argument(
        '--generated',
        type=str,
        required=True,
        help='Path to generated articles JSONL file'
    )
    parser.add_argument(
        '--benchmark',
        type=str,
        required=True,
        help='Path to benchmark deconstructions JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output evaluation results JSONL file'
    )

    # API configurations for article evaluation (GPT-4)
    parser.add_argument(
        '--gpt4-key',
        type=str,
        required=True,
        help='API key for GPT-4 (article evaluation)'
    )
    parser.add_argument(
        '--gpt4-base-url',
        type=str,
        default='https://api.openai.com/v1',
        help='Base URL for GPT-4 API (default: OpenAI)'
    )
    parser.add_argument(
        '--gpt4-model',
        type=str,
        default='gpt-4.1',
        help='GPT-4 model name (default: gpt-4.1)'
    )

    # API configurations for text citation evaluation (Qwen)
    parser.add_argument(
        '--qwen-key',
        type=str,
        required=True,
        help='API key for Qwen (text citation evaluation)'
    )
    parser.add_argument(
        '--qwen-base-url',
        type=str,
        default='https://dashscope.aliyuncs.com/compatible-mode/v1',
        help='Base URL for Qwen API'
    )
    parser.add_argument(
        '--qwen-model',
        type=str,
        default='qwen-plus-latest',
        help='Qwen model name (default: qwen-plus-latest)'
    )

    # API configurations for image citation evaluation (InternVL)
    parser.add_argument(
        '--vlm-base-url',
        type=str,
        default='http://localhost:9000/v1',
        help='Base URL for VLM API (default: localhost:9000)'
    )
    parser.add_argument(
        '--vlm-model',
        type=str,
        default='internvl35-38b',
        help='VLM model name (default: internvl35-38b)'
    )
    parser.add_argument(
        '--vlm-key',
        type=str,
        default='EMPTY',
        help='API key for VLM (default: EMPTY for local deployment)'
    )

    # Evaluation options
    parser.add_argument(
        '--skip-citations',
        action='store_true',
        help='Skip citation evaluation (faster, but incomplete)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of articles to evaluate (for testing)'
    )

    return parser.parse_args()


def evaluate_single_article(
    generated_data: Dict[str, Any],
    benchmark_data: Dict[str, Any],
    article_evaluator: ArticleEvaluator,
    citation_evaluator: CitationEvaluator,
    score_calculator: ScoreCalculator,
    skip_citations: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single article.

    Returns:
        Complete evaluation result with all scores
    """
    session_id = generated_data.get('session_id', 'unknown')
    logger.info(f"Evaluating article: {session_id}")

    try:
        # Step 1: Article-level evaluation (section + article)
        logger.info("  Step 1/3: Article evaluation...")
        article_eval = article_evaluator.evaluate_article(generated_data, benchmark_data)

        # Step 2: Image-text coherence evaluation
        if skip_citations:
            logger.info("  Step 2/3: Skipping image-text coherence evaluation")
            citation_eval = {
                'section_evaluations': [],
                'summary': {
                    'section_count': 0,
                    'valid_evaluations': 0,
                    'total_images': 0,
                    'overall_avg_score': 10,  # Default perfect score if skipped
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
            logger.info("  Step 2/3: Image-text coherence evaluation...")
            citation_eval = citation_evaluator.evaluate_citations(generated_data)

        # Step 3: Calculate final score
        logger.info("  Step 3/3: Calculating final score...")
        final_scores = score_calculator.calculate_final_score(article_eval, citation_eval)

        # Combine all results
        result = {
            'session_id': session_id,
            'user_id': generated_data.get('user_id', ''),
            'uid': benchmark_data.get('uid', ''),
            'timestamp': datetime.now().isoformat(),
            'article_evaluation': article_eval,
            'citation_evaluation': citation_eval,
            'final_scores': final_scores
        }

        logger.info(f"  ✓ Completed: Final score = {final_scores.get('final_score', 0):.2f}")
        return result

    except Exception as e:
        logger.error(f"  ✗ Error evaluating {session_id}: {str(e)}", exc_info=True)
        return {
            'session_id': session_id,
            'user_id': generated_data.get('user_id', ''),
            'uid': benchmark_data.get('uid', ''),
            'timestamp': datetime.now().isoformat(),
            'error': str(e),
            'final_scores': {'final_score': 0}
        }


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Starting batch evaluation")
    logger.info("=" * 80)

    # Load data
    logger.info(f"Loading generated articles from: {args.generated}")
    generated_data = load_jsonl(args.generated)
    logger.info(f"  Loaded {len(generated_data)} generated articles")

    logger.info(f"Loading benchmark data from: {args.benchmark}")
    benchmark_data = load_jsonl(args.benchmark)
    logger.info(f"  Loaded {len(benchmark_data)} benchmark articles")

    # Match generated to benchmark
    logger.info("Matching generated articles to benchmarks...")
    matched_pairs = match_generated_to_benchmark(generated_data, benchmark_data)
    logger.info(f"  Matched {len(matched_pairs)} article pairs")

    if not matched_pairs:
        logger.error("No matched pairs found! Check your data files.")
        return

    # Apply limit if specified
    if args.limit:
        matched_pairs = matched_pairs[:args.limit]
        logger.info(f"  Limited to first {args.limit} pairs for testing")

    # Initialize evaluators
    logger.info("Initializing evaluators...")
    article_evaluator = ArticleEvaluator(
        api_key=args.gpt4_key,
        base_url=args.gpt4_base_url,
        model=args.gpt4_model
    )

    citation_evaluator = CitationEvaluator(
        text_api_key=args.qwen_key,
        text_base_url=args.qwen_base_url,
        text_model=args.qwen_model,
        image_api_key=args.vlm_key,
        image_base_url=args.vlm_base_url,
        image_model=args.vlm_model
    )

    score_calculator = ScoreCalculator()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate all articles
    logger.info(f"\nEvaluating {len(matched_pairs)} articles...")
    logger.info("-" * 80)

    results = []
    success_count = 0
    error_count = 0

    for i, (gen_data, bench_data) in enumerate(matched_pairs, 1):
        logger.info(f"\n[{i}/{len(matched_pairs)}]")

        result = evaluate_single_article(
            gen_data,
            bench_data,
            article_evaluator,
            citation_evaluator,
            score_calculator,
            skip_citations=args.skip_citations
        )

        results.append(result)

        if 'error' in result:
            error_count += 1
        else:
            success_count += 1

    # Save results
    logger.info("\n" + "=" * 80)
    logger.info(f"Saving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Print summary
    logger.info("=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total articles: {len(matched_pairs)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")

    if success_count > 0:
        final_scores = [r['final_scores']['final_score'] for r in results if 'error' not in r]
        avg_score = sum(final_scores) / len(final_scores)
        logger.info(f"\nAverage final score: {avg_score:.2f}")
        logger.info(f"Min score: {min(final_scores):.2f}")
        logger.info(f"Max score: {max(final_scores):.2f}")

    logger.info("=" * 80)
    logger.info("Batch evaluation completed!")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
