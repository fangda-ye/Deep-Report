# Evaluation System

Multi-dimensional evaluation for generated multimodal long-form reports.

## Evaluation Dimensions

### 1. Section Anchor
- **Description Adherence**: Alignment between generated content and section description
- **Checklist Satisfaction**: Coverage of required checklist items (true/false + reasoning)

### 2. Section Content (VLM-based)
- **Richness**: Quantity and diversity of visual elements
- **Image-Text Coherence**: Relevance of images to surrounding text
- **Placement**: Logical positioning of images in the text flow
- **Clarity**: Visual quality and readability of images

### 3. Full Report
- **Coherence**: Logical flow and consistency across sections
- **Fluency**: Readability and language quality
- **Repetition**: Absence of redundant content (higher = better)
- **Termination**: Absence of premature summarization (higher = better)

### 4. Citation Precision
- **Text citations**: LLM-evaluated relevance and context match
- **Image citations**: VLM-evaluated visual relevance and information value

## Quick Start

```bash
python batch_evaluate_all.py \
    --input_dir ../gen_articles/your_model/ \
    --deconstruction_file ../data/benchmark/article_deconstructions_enriched.jsonl \
    --output_dir ./eval_results/
```

### Environment Variables

```bash
export GPT4_API_KEY="your-openai-key"
export GPT4_BASE_URL="https://api.openai.com/v1"
export QWEN_API_KEY="your-dashscope-key"
export QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

### Skip Citation Evaluation

```bash
python batch_evaluate.py \
    --generated gen_articles/output.jsonl \
    --benchmark data/article_deconstructions.jsonl \
    --output eval_results/eval.jsonl \
    --gpt4-key $GPT4_API_KEY \
    --qwen-key $QWEN_API_KEY \
    --skip-citations
```

## Output Format

JSONL file with per-article evaluation results:

```json
{
  "uid": "AG001",
  "article_evaluation": {
    "section_evaluations": [...],
    "article_evaluation": {
      "coherence_score": 8.0,
      "fluency_score": 9.0,
      "repetition_score": 7.5,
      "termination_score": 8.5
    }
  },
  "citation_evaluation": {
    "text_citation_evaluations": [...],
    "image_citation_evaluations": [...]
  },
  "final_scores": {
    "section_score": 0.85,
    "article_score": 0.82,
    "citation_score": 0.83,
    "final_score": 57.87
  }
}
```

## Files

| File | Purpose |
|------|---------|
| `evaluator.py` | Section + article quality evaluation |
| `citation_evaluator.py` | Text and image citation evaluation |
| `citation_precision_evaluator.py` | Citation precision against silver annotations |
| `score_calculator.py` | Score normalization and aggregation |
| `batch_evaluate.py` | Single-model batch evaluation |
| `batch_evaluate_all.py` | Multi-model batch evaluation |
| `batch_evaluate_all_concurrent.py` | Parallel multi-model evaluation |
| `prompts/eval_templates.py` | Evaluation prompt templates |
