"""
Complete Batch Evaluation Script (CONCURRENT VERSION) - Article Quality + Search Precision

Evaluates both:
1. Article Quality (full article, sections, image-text coherence) - saved to data/eval_results/article/
2. Search Precision (3-stage retrieval precision) - saved to data/eval_results/search/

Features:
- Concurrent evaluation of multiple articles using ThreadPoolExecutor
- Configurable worker pool size
- Thread-safe file writing
- Progress tracking

Usage:
    1. Configure the INPUT_FILES list at the top of this script
    2. Configure API keys and settings
    3. Set MAX_WORKERS for concurrency level
    4. Run: python -m eval.batch_evaluate_all_concurrent
    5. python -m eval.batch_evaluate_all_concurrent --fill-citations
    6. python -m eval.batch_evaluate_all_concurrent --skip-citations --max-workers 10

"""

import argparse
import json
import logging
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from .evaluator import ArticleEvaluator, load_jsonl, match_generated_to_benchmark
from .citation_evaluator import CitationEvaluator
from .score_calculator import ScoreCalculator
from .citation_precision_evaluator import CitationPrecisionEvaluator, compute_aggregate_stats

INPUT_FILES = [
    "session_logs/compare_logs/article_deconstructions_long_context_rag_qwen3_8b_nothink.jsonl",
    "session_logs/compare_logs/article_deconstructions_long_context_rag_qwen3_32b_nothink.jsonl",
    "session_logs/compare_logs/article_deconstructions_zero_shot_qwen3_8b_nothink.jsonl",
    "session_logs/compare_logs/article_deconstructions_zero_shot_qwen3_32b_nothink.jsonl",
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
LIMIT = None  # Limit number of articles per file (None = no limit)

# Concurrency settings
MAX_WORKERS = 4  # Number of concurrent workers (adjust based on API rate limits)
FILE_LEVEL_CONCURRENCY = False  # Set to True to process files concurrently (not recommended for large batches)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe file writing lock
file_locks = {}
file_locks_lock = threading.Lock()


def get_file_lock(filepath: str) -> threading.Lock:
    """Get or create a lock for a specific file."""
    with file_locks_lock:
        if filepath not in file_locks:
            file_locks[filepath] = threading.Lock()
        return file_locks[filepath]


def write_result_threadsafe(filepath: str, result: Dict[str, Any]):
    """Thread-safe writing of result to JSONL file."""
    lock = get_file_lock(filepath)
    with lock:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def parse_args():
    """Parse command line arguments (optional, overrides configuration)."""
    parser = argparse.ArgumentParser(
        description="Complete batch evaluation (CONCURRENT): article quality + search precision"
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
    parser.add_argument(
        '--max-workers',
        type=int,
        default=MAX_WORKERS,
        help=f'Maximum number of concurrent workers (default: {MAX_WORKERS})'
    )
    parser.add_argument(
        '--file-concurrency',
        action='store_true',
        help='Enable file-level concurrency (process multiple files at once)'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Disable resume from checkpoint (re-evaluate all articles)'
    )
    parser.add_argument(
        '--fill-citations',
        action='store_true',
        help='Fill in citation evaluations for records that were previously skipped'
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


def load_completed_uids(output_file: str) -> set:
    """
    Load UIDs of already completed evaluations from output file.

    Args:
        output_file: Path to output JSONL file

    Returns:
        Set of completed UIDs
    """
    completed_uids = set()

    if not os.path.exists(output_file):
        return completed_uids

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    uid = result.get('uid')
                    if uid:
                        completed_uids.add(uid)
                except json.JSONDecodeError:
                    continue

        if completed_uids:
            logger.info(f"  Found {len(completed_uids)} completed evaluations in {Path(output_file).name}")
    except Exception as e:
        logger.warning(f"  Error loading completed results from {output_file}: {e}")

    return completed_uids


def load_skipped_citation_records(output_file: str) -> tuple:
    """
    Load records that have skipped citation evaluations.

    Args:
        output_file: Path to article evaluation JSONL file

    Returns:
        Tuple of (skipped_records, non_skipped_records, all_records_ordered)
        - skipped_records: List of records with citation_evaluation.skipped == True
        - non_skipped_records: List of records with complete citation evaluation
        - all_records_ordered: List of all records in original order (for reconstruction)
    """
    skipped_records = []
    non_skipped_records = []
    all_records_ordered = []

    if not os.path.exists(output_file):
        logger.warning(f"Output file does not exist: {output_file}")
        return skipped_records, non_skipped_records, all_records_ordered

    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line.strip())
                    all_records_ordered.append(result)

                    # Check if citation was skipped
                    citation_eval = result.get('citation_evaluation', {})
                    is_skipped = citation_eval.get('skipped', False)

                    if is_skipped:
                        skipped_records.append(result)
                    else:
                        non_skipped_records.append(result)

                except json.JSONDecodeError as e:
                    logger.warning(f"  Line {line_num}: JSON decode error - {e}")
                    continue

        logger.info(f"  Loaded {len(all_records_ordered)} total records")
        logger.info(f"  - Skipped citations (need fill): {len(skipped_records)}")
        logger.info(f"  - Complete citations: {len(non_skipped_records)}")

    except Exception as e:
        logger.error(f"  Error loading records from {output_file}: {e}")

    return skipped_records, non_skipped_records, all_records_ordered


def fill_single_citation(
    record: Dict[str, Any],
    gen_data: Dict[str, Any],
    citation_evaluator: 'CitationEvaluator',
    score_calculator: 'ScoreCalculator'
) -> Dict[str, Any]:
    """
    Fill citation evaluation for a single skipped record.

    Args:
        record: The evaluation record with skipped citation
        gen_data: The original generated article data
        citation_evaluator: CitationEvaluator instance
        score_calculator: ScoreCalculator instance

    Returns:
        Updated record with filled citation evaluation
    """
    uid = record.get('uid', 'unknown')
    thread_id = threading.current_thread().name

    try:
        logger.info(f"[{thread_id}] Filling citation for UID: {uid}")

        # Get existing article evaluation
        article_eval = record.get('article_evaluation', {})

        # Evaluate citations
        citation_eval = citation_evaluator.evaluate_citations(gen_data)

        # Recalculate final scores
        final_scores = score_calculator.calculate_final_score(article_eval, citation_eval)

        # Update record
        updated_record = record.copy()
        updated_record['citation_evaluation'] = citation_eval
        updated_record['final_scores'] = final_scores
        updated_record['timestamp'] = datetime.now().isoformat()
        updated_record['fill_citation_timestamp'] = datetime.now().isoformat()

        logger.info(f"[{thread_id}] ✓ Filled citation for UID: {uid}, "
                   f"citation_score: {final_scores.get('citation_score', 0):.3f}")

        return updated_record

    except Exception as e:
        logger.error(f"[{thread_id}] ✗ Failed to fill citation for UID {uid}: {e}")
        # Return original record on failure
        record['fill_citation_error'] = str(e)
        return record


def process_fill_citations(
    article_output_file: str,
    generated_data: List[Dict[str, Any]],
    citation_evaluator: 'CitationEvaluator',
    score_calculator: 'ScoreCalculator',
    max_workers: int = 4,
    limit: int = None
) -> Dict[str, Any]:
    """
    Process fill-citations mode for a single output file.

    Args:
        article_output_file: Path to article evaluation results file
        generated_data: List of original generated articles (for citation eval)
        citation_evaluator: CitationEvaluator instance
        score_calculator: ScoreCalculator instance
        max_workers: Number of concurrent workers
        limit: Limit number of records to process (for testing)

    Returns:
        Summary statistics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"FILL CITATIONS MODE")
    logger.info(f"Processing: {article_output_file}")
    logger.info(f"{'='*80}")

    # Load existing records
    skipped_records, non_skipped_records, all_records = load_skipped_citation_records(
        article_output_file
    )

    if not skipped_records:
        logger.info("  No skipped citation records found. Nothing to fill.")
        return {'filled_count': 0, 'error_count': 0, 'total_records': len(all_records)}

    # Build UID -> gen_data mapping
    gen_data_map = {}
    for gen_data in generated_data:
        # Try to extract UID from gen_data
        uid = gen_data.get('uid')
        if not uid:
            # Try to extract from user_id (format: deconstruct_XXX)
            user_id = gen_data.get('user_id', '')
            if user_id.startswith('deconstruct_'):
                uid = user_id.replace('deconstruct_', '')
        if uid:
            gen_data_map[uid] = gen_data

    logger.info(f"  Built mapping for {len(gen_data_map)} generated articles")

    # Prepare records to fill
    records_to_fill = []
    for record in skipped_records:
        uid = record.get('uid', '')
        if uid in gen_data_map:
            records_to_fill.append((record, gen_data_map[uid]))
        else:
            logger.warning(f"  Cannot find generated data for UID: {uid}")

    if limit:
        records_to_fill = records_to_fill[:limit]
        logger.info(f"  Limited to first {limit} records for testing")

    if not records_to_fill:
        logger.warning("  No records to fill (missing generated data)")
        return {'filled_count': 0, 'error_count': 0, 'total_records': len(all_records)}

    logger.info(f"\n  Starting to fill {len(records_to_fill)} citation evaluations...")
    logger.info(f"  Using {max_workers} workers")

    # Process with thread pool
    filled_records = {}
    filled_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for record, gen_data in records_to_fill:
            uid = record.get('uid', '')
            future = executor.submit(
                fill_single_citation,
                record=record,
                gen_data=gen_data,
                citation_evaluator=citation_evaluator,
                score_calculator=score_calculator
            )
            futures[future] = uid

        for future in as_completed(futures):
            uid = futures[future]
            try:
                updated_record = future.result()
                filled_records[uid] = updated_record

                # Check if successfully filled
                citation_eval = updated_record.get('citation_evaluation', {})
                if not citation_eval.get('skipped', False) and 'fill_citation_error' not in updated_record:
                    filled_count += 1
                else:
                    error_count += 1

            except Exception as e:
                logger.error(f"  Error processing UID {uid}: {e}")
                error_count += 1

    # Reconstruct file with updated records
    logger.info(f"\n  Reconstructing output file...")

    # Backup original file
    backup_file = article_output_file + '.backup'
    if os.path.exists(article_output_file):
        shutil.copy2(article_output_file, backup_file)
        logger.info(f"  Created backup: {backup_file}")

    # Write updated records
    with open(article_output_file, 'w', encoding='utf-8') as f:
        for record in all_records:
            uid = record.get('uid', '')
            if uid in filled_records:
                # Use updated record
                f.write(json.dumps(filled_records[uid], ensure_ascii=False) + '\n')
            else:
                # Keep original record
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logger.info(f"\n  Fill citations completed!")
    logger.info(f"  Successfully filled: {filled_count}")
    logger.info(f"  Errors: {error_count}")
    logger.info(f"  Total records: {len(all_records)}")

    return {
        'filled_count': filled_count,
        'error_count': error_count,
        'total_records': len(all_records),
        'backup_file': backup_file
    }



def evaluate_single_article(
    article_index: int,
    total_articles: int,
    gen_data: Dict[str, Any],
    bench_data: Dict[str, Any],
    enrich_data: Dict[str, Any],
    article_evaluator: ArticleEvaluator,
    citation_evaluator: CitationEvaluator,
    score_calculator: ScoreCalculator,
    precision_evaluator: CitationPrecisionEvaluator,
    article_output_file: str,
    search_output_file: str,
    skip_citations: bool,
    skip_search: bool
) -> Dict[str, Any]:
    """
    Evaluate a single article (both article quality and search precision).

    Returns:
        Summary of evaluation results
    """
    session_id = gen_data.get('session_id', 'unknown')
    thread_id = threading.current_thread().name

    logger.info(f"[{thread_id}] [{article_index}/{total_articles}] Starting: {session_id}")

    result_summary = {
        'article_index': article_index,
        'session_id': session_id,
        'article_success': False,
        'search_success': False,
        'article_score': 0,
        'error': None
    }

    try:
        # ===== Article Quality Evaluation =====
        try:
            logger.info(f"[{thread_id}]   Article eval...")

            # Step 1: Article-level evaluation
            article_eval = article_evaluator.evaluate_article(gen_data, bench_data)

            # Step 2: Image-text coherence evaluation
            if skip_citations:
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
                citation_eval = citation_evaluator.evaluate_citations(gen_data)

            # Step 3: Calculate final score
            final_scores = score_calculator.calculate_final_score(article_eval, citation_eval)

            article_result = {
                'session_id': session_id,
                'user_id': gen_data.get('user_id', ''),
                'uid': bench_data.get('uid', ''),
                'timestamp': datetime.now().isoformat(),
                'article_evaluation': article_eval,
                'citation_evaluation': citation_eval,
                'final_scores': final_scores
            }

            # Save article result (thread-safe)
            write_result_threadsafe(article_output_file, article_result)

            result_summary['article_success'] = True
            result_summary['article_score'] = final_scores.get('final_score', 0)

            logger.info(f"[{thread_id}]   ✓ Article: {result_summary['article_score']:.2f}")

        except Exception as e:
            logger.error(f"[{thread_id}]   ✗ Article eval error: {str(e)}")
            result_summary['error'] = f"Article: {str(e)}"

        # ===== Search Precision Evaluation =====
        if not skip_search and enrich_data:
            try:
                logger.info(f"[{thread_id}]   Search eval...")

                search_result = precision_evaluator.evaluate_article(gen_data, enrich_data)

                # Save search result (thread-safe)
                write_result_threadsafe(search_output_file, search_result)

                result_summary['search_success'] = True

                avg_prec = search_result.get('average_precisions', {})
                text_final = avg_prec.get('text', {}).get('final', 0)
                image_final = avg_prec.get('image', {}).get('final', 0)

                logger.info(f"[{thread_id}]   ✓ Search: T={text_final:.3f}, I={image_final:.3f}")

            except Exception as e:
                logger.error(f"[{thread_id}]   ✗ Search eval error: {str(e)}")
                if result_summary['error']:
                    result_summary['error'] += f"; Search: {str(e)}"
                else:
                    result_summary['error'] = f"Search: {str(e)}"

        logger.info(f"[{thread_id}] [{article_index}/{total_articles}] Completed: {session_id}")

    except Exception as e:
        logger.error(f"[{thread_id}] [{article_index}/{total_articles}] Fatal error: {str(e)}", exc_info=True)
        result_summary['error'] = str(e)

    return result_summary


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
    limit: int = None,
    max_workers: int = 4,
    resume: bool = True
) -> Dict[str, Any]:
    """
    Process a single input file with concurrent evaluation.

    Args:
        resume: If True, skip already completed evaluations (default: True)

    Returns:
        Summary statistics for this file
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing file: {input_file}")
    logger.info(f"Concurrency: {max_workers} workers")
    logger.info(f"Resume mode: {'Enabled' if resume else 'Disabled'}")
    logger.info(f"{'='*80}")

    # Load generated data
    try:
        generated_data = load_jsonl(input_file)
        logger.info(f"  Loaded {len(generated_data)} generated articles")
    except Exception as e:
        logger.error(f"  Error loading file: {str(e)}")
        return {'error': str(e), 'success_count': 0, 'error_count': 0}

    # Load completed UIDs if resume mode is enabled
    completed_article_uids = set()
    completed_search_uids = set()
    skipped_count = 0

    if resume:
        completed_article_uids = load_completed_uids(article_output_file)
        if not skip_search:
            completed_search_uids = load_completed_uids(search_output_file)

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

    # Filter out completed tasks if resume mode is enabled
    if resume and (completed_article_uids or completed_search_uids):
        original_count = len(article_pairs)
        filtered_article_pairs = []
        filtered_search_pairs = []

        for i, (gen_data, bench_data) in enumerate(article_pairs):
            uid = bench_data.get('uid', '')

            # Check if this UID needs to be evaluated
            needs_article_eval = uid not in completed_article_uids
            needs_search_eval = (not skip_search and
                               i < len(search_pairs) and
                               uid not in completed_search_uids)

            # Keep this pair if either evaluation is needed
            if needs_article_eval or needs_search_eval:
                filtered_article_pairs.append((gen_data, bench_data))
                if i < len(search_pairs):
                    filtered_search_pairs.append(search_pairs[i])
            else:
                skipped_count += 1

        article_pairs = filtered_article_pairs
        search_pairs = filtered_search_pairs

        logger.info(f"  ✓ Skipped {skipped_count} already completed evaluations")
        logger.info(f"  → Remaining to process: {len(article_pairs)} articles")

    if not article_pairs:
        logger.info("  All evaluations already completed!")
        return {
            'success_count': skipped_count,
            'error_count': 0,
            'skipped_count': skipped_count,
            'total_articles': skipped_count
        }

    # Prepare evaluation tasks
    num_to_process = len(article_pairs)
    if limit:
        num_to_process = min(num_to_process, limit)
        logger.info(f"  Limited to first {limit} pairs for testing")

    logger.info(f"\n  Starting concurrent evaluation of {num_to_process} articles...")
    logger.info(f"  {'-'*78}")

    # Submit tasks to thread pool
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(num_to_process):
            gen_data, bench_data = article_pairs[i]
            enrich_data = search_pairs[i][1] if i < len(search_pairs) else None

            future = executor.submit(
                evaluate_single_article,
                article_index=i + 1,
                total_articles=num_to_process,
                gen_data=gen_data,
                bench_data=bench_data,
                enrich_data=enrich_data,
                article_evaluator=article_evaluator,
                citation_evaluator=citation_evaluator,
                score_calculator=score_calculator,
                precision_evaluator=precision_evaluator,
                article_output_file=article_output_file,
                search_output_file=search_output_file,
                skip_citations=skip_citations,
                skip_search=skip_search
            )
            futures.append(future)

    # Collect results
    results = []
    article_success = 0
    search_success = 0
    error_count = 0
    final_scores = []

    for future in as_completed(futures):
        try:
            result = future.result()
            results.append(result)

            if result['article_success']:
                article_success += 1
                final_scores.append(result['article_score'])

            if result['search_success']:
                search_success += 1

            if result['error']:
                error_count += 1

        except Exception as e:
            logger.error(f"  Error collecting result: {str(e)}")
            error_count += 1

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

    # Compute aggregate search stats if available
    if not skip_search:
        try:
            # Load all search results from file
            search_results = load_jsonl(search_output_file)
            if search_results:
                search_stats = compute_aggregate_stats(search_results)
                summary['search_precision'] = search_stats['average_precisions']
        except Exception as e:
            logger.warning(f"  Could not compute search aggregate stats: {str(e)}")

    logger.info(f"\n  File processing completed!")
    logger.info(f"  Article success: {article_success}/{num_to_process}")
    logger.info(f"  Search success: {search_success}/{num_to_process}")
    logger.info(f"  Errors: {error_count}")

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
    max_workers = args.max_workers
    file_concurrency = args.file_concurrency if args.file_concurrency else FILE_LEVEL_CONCURRENCY
    resume = not args.no_resume  # Resume is enabled by default

    # Validate configuration
    if not input_files:
        logger.error("No input files configured! Please edit INPUT_FILES or use --input-files")
        return

    logger.info("="*80)
    logger.info("COMPLETE BATCH EVALUATION (CONCURRENT) - ARTICLE + SEARCH")
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
    logger.info(f"Max workers: {max_workers}")
    logger.info(f"File-level concurrency: {file_concurrency}")
    logger.info(f"Resume mode: {'Enabled (skip completed)' if resume else 'Disabled (re-evaluate all)'}")
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

    if args.fill_citations:
        logger.info("\n" + "="*80)
        logger.info("FILL CITATIONS MODE - Backfilling skipped citation evaluations")
        logger.info("="*80)

        fill_summaries = []

        for input_file in input_files:
            article_output_file = get_output_filename(input_file, article_output_dir, 'article')

            if not os.path.exists(article_output_file):
                logger.warning(f"Article output file not found: {article_output_file}")
                continue

            # Load generated data for this input file
            logger.info(f"\nLoading generated data from: {input_file}")
            try:
                generated_data = load_jsonl(input_file)
                logger.info(f"  Loaded {len(generated_data)} generated articles")
            except Exception as e:
                logger.error(f"  Error loading generated data: {str(e)}")
                continue

            # Process fill citations
            summary = process_fill_citations(
                article_output_file=article_output_file,
                generated_data=generated_data,
                citation_evaluator=citation_evaluator,
                score_calculator=score_calculator,
                max_workers=max_workers,
                limit=limit
            )
            summary['input_file'] = input_file
            summary['article_output_file'] = article_output_file
            fill_summaries.append(summary)

        # Print final summary for fill mode
        logger.info("\n" + "="*80)
        logger.info("FILL CITATIONS SUMMARY")
        logger.info("="*80)

        total_filled = sum(s.get('filled_count', 0) for s in fill_summaries)
        total_errors = sum(s.get('error_count', 0) for s in fill_summaries)
        total_records = sum(s.get('total_records', 0) for s in fill_summaries)

        logger.info(f"Files processed: {len(fill_summaries)}")
        logger.info(f"Total records: {total_records}")
        logger.info(f"Successfully filled: {total_filled}")
        logger.info(f"Errors: {total_errors}")

        for summary in fill_summaries:
            logger.info(f"\n  File: {Path(summary.get('input_file', '')).name}")
            logger.info(f"    Filled: {summary.get('filled_count', 0)}")
            logger.info(f"    Errors: {summary.get('error_count', 0)}")
            logger.info(f"    Backup: {summary.get('backup_file', 'N/A')}")

        logger.info("\n" + "="*80)
        logger.info("Fill citations completed!")
        logger.info("="*80)
        return  # Exit after fill mode


    # Process each input file
    all_summaries = []

    if file_concurrency:
        # Process files concurrently
        logger.info("\nProcessing files concurrently...")
        with ThreadPoolExecutor(max_workers=min(len(input_files), 3)) as executor:
            futures = []
            for input_file in input_files:
                article_output_file = get_output_filename(input_file, article_output_dir, 'article')
                search_output_file = get_output_filename(input_file, search_output_dir, 'search')

                future = executor.submit(
                    process_single_file,
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
                    limit=limit,
                    max_workers=max_workers,
                    resume=resume
                )
                futures.append(future)

            for future in as_completed(futures):
                summary = future.result()
                all_summaries.append(summary)
    else:
        # Process files sequentially
        for input_file in input_files:
            article_output_file = get_output_filename(input_file, article_output_dir, 'article')
            search_output_file = get_output_filename(input_file, search_output_dir, 'search')

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
                limit=limit,
                max_workers=max_workers,
                resume=resume
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
