"""
Batch Citation Precision Evaluation Script

Usage:
    python -m eval.batch_citation_precision \
        --generated exp_logs/batch_generation_qwen3_32b_filter_20251107.jsonl \
        --enriched data/article_deconstructions_enriched.jsonl \
        --output eval_results/qwen3_32b_citation_precision.jsonl
python -m eval.batch_citation_precision \
    --generated exp_logs/batch_generation_qwen3_32b_filter_20251107.jsonl \
    --enriched data/article_deconstructions_enriched.jsonl \
    --output eval_results/qwen3_32b_citation_precision.jsonl
python -m eval.batch_citation_precision \
    --generated session_logs/exp_logs/batch_generation_qwen3_32b_filter.jsonl \
    --enriched data/article_deconstructions_enriched.jsonl \
    --output eval/eval_results/qwen3_32b_citation_precision.jsonl
python -m eval.batch_citation_precision \
    --generated session_logs/exp_logs/batch_generation_qwen3_8b_sft_nothink.jsonl \
    --enriched data/article_deconstructions_enriched.jsonl \
    --output eval/eval_results/qwen3_8b_citation_precision_sft.jsonl
python -m eval.batch_citation_precision \
    --generated session_logs/exp_logs/batch_generation_qwen3_8b_filter.jsonl \
    --enriched data/article_deconstructions_enriched.jsonl \
    --output eval/eval_results/qwen3_8b_citation_precision.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

from .citation_precision_evaluator import CitationPrecisionEvaluator, compute_aggregate_stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate citation selection precision"
    )

    parser.add_argument(
        '--generated',
        type=str,
        required=True,
        help='Path to generated articles JSONL file'
    )
    parser.add_argument(
        '--enriched',
        type=str,
        required=True,
        help='Path to enriched deconstructions JSONL file (with GT citations)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output evaluation results JSONL file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of articles to evaluate (for testing)'
    )

    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    logger.info("=" * 80)
    logger.info("Citation Precision Evaluation")
    logger.info("=" * 80)

    # Initialize evaluator
    evaluator = CitationPrecisionEvaluator()

    # Load data
    logger.info(f"Loading generated articles from: {args.generated}")
    generated_data = evaluator.load_jsonl(args.generated)
    logger.info(f"  Loaded {len(generated_data)} generated articles")

    logger.info(f"Loading enriched data from: {args.enriched}")
    enriched_data = evaluator.load_jsonl(args.enriched)
    logger.info(f"  Loaded {len(enriched_data)} enriched articles")

    # Match data
    logger.info("Matching generated articles to enriched data...")
    matched_pairs = evaluator.match_data(generated_data, enriched_data)
    logger.info(f"  Matched {len(matched_pairs)} article pairs")

    if not matched_pairs:
        logger.error("No matched pairs found! Check your data files.")
        return

    # Apply limit if specified
    if args.limit:
        matched_pairs = matched_pairs[:args.limit]
        logger.info(f"  Limited to first {args.limit} pairs for testing")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Evaluate all articles
    logger.info(f"\nEvaluating {len(matched_pairs)} articles...")
    logger.info("-" * 80)

    results = []
    success_count = 0
    error_count = 0

    for i, (gen_data, enrich_data) in enumerate(matched_pairs, 1):
        uid = enrich_data.get('uid', 'unknown')
        logger.info(f"\n[{i}/{len(matched_pairs)}] Evaluating: {uid}")

        try:
            result = evaluator.evaluate_article(gen_data, enrich_data)
            results.append(result)
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ Error: {str(e)}", exc_info=True)
            error_count += 1

    # Compute aggregate statistics
    logger.info("\n" + "=" * 80)
    logger.info("Computing aggregate statistics...")
    aggregate_stats = compute_aggregate_stats(results)

    # Save results
    logger.info(f"\nSaving results to: {args.output}")
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # Save aggregate stats
    aggregate_path = output_path.parent / f"{output_path.stem}_aggregate.json"
    with open(aggregate_path, 'w', encoding='utf-8') as f:
        json.dump(aggregate_stats, f, indent=2, ensure_ascii=False)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total articles: {len(matched_pairs)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Errors: {error_count}")
    logger.info("")

    # Print precision statistics
    logger.info("AVERAGE PRECISION (per article):")
    logger.info("-" * 80)
    avg_prec = aggregate_stats['average_precisions']

    logger.info("Text Citations:")
    logger.info(f"  Search:  {avg_prec['text']['search']:.3f}")
    logger.info(f"  Filter:  {avg_prec['text']['filter']:.3f}")
    logger.info(f"  Final:   {avg_prec['text']['final']:.3f}")
    logger.info("")

    logger.info("Image Citations:")
    logger.info(f"  Search:  {avg_prec['image']['search']:.3f}")
    logger.info(f"  Filter:  {avg_prec['image']['filter']:.3f}")
    logger.info(f"  Final:   {avg_prec['image']['final']:.3f}")
    logger.info("")

    # Print overall precision
    logger.info("OVERALL PRECISION (total found / total used):")
    logger.info("-" * 80)
    overall_prec = aggregate_stats['overall_precision']
    total_found = aggregate_stats['total_found']
    total_used = aggregate_stats['total_used']

    logger.info("Text Citations:")
    logger.info(f"  Search:  {overall_prec['text']['search']:.3f} "
                f"({total_found['text']['search']}/{total_used['text']['search']})")
    logger.info(f"  Filter:  {overall_prec['text']['filter']:.3f} "
                f"({total_found['text']['filter']}/{total_used['text']['filter']})")
    logger.info(f"  Final:   {overall_prec['text']['final']:.3f} "
                f"({total_found['text']['final']}/{total_used['text']['final']})")
    logger.info("")

    logger.info("Image Citations:")
    logger.info(f"  Search:  {overall_prec['image']['search']:.3f} "
                f"({total_found['image']['search']}/{total_used['image']['search']})")
    logger.info(f"  Filter:  {overall_prec['image']['filter']:.3f} "
                f"({total_found['image']['filter']}/{total_used['image']['filter']})")
    logger.info(f"  Final:   {overall_prec['image']['final']:.3f} "
                f"({total_found['image']['final']}/{total_used['image']['final']})")
    logger.info("")

    # Filter effectiveness
    logger.info("FILTER EFFECTIVENESS:")
    logger.info("-" * 80)
    text_improvement = overall_prec['text']['filter'] - overall_prec['text']['search']
    image_improvement = overall_prec['image']['filter'] - overall_prec['image']['search']
    logger.info(f"Text precision improvement:  {text_improvement:+.3f}")
    logger.info(f"Image precision improvement: {image_improvement:+.3f}")
    logger.info("")

    # Model selection effectiveness
    logger.info("MODEL SELECTION EFFECTIVENESS:")
    logger.info("-" * 80)
    text_model_improvement = overall_prec['text']['final'] - overall_prec['text']['filter']
    image_model_improvement = overall_prec['image']['final'] - overall_prec['image']['filter']
    logger.info(f"Text precision change (filter→final):  {text_model_improvement:+.3f}")
    logger.info(f"Image precision change (filter→final): {image_model_improvement:+.3f}")
    logger.info("")

    logger.info("=" * 80)
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to: {args.output}")
    logger.info(f"Aggregate stats saved to: {aggregate_path}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
