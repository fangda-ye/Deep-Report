# Deep-Reporter

**Deep Research for Grounded Multimodal Long-Form Generation**

<p align="left">
  <a href="https://arxiv.org/abs/2604.10741"><img src="https://img.shields.io/badge/arXiv-2604.10741-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/Fangda-Ye/Deep-Reporter-Data"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-M2LongBench-yellow" alt="HF Dataset"></a>
  <a href="https://github.com/fangda-ye/Deep-Report"><img src="https://img.shields.io/github/stars/fangda-ye/Deep-Report?style=social" alt="GitHub stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"></a>
</p>

Deep-Reporter is a unified agentic framework for generating comprehensive, evidence-grounded multimodal reports. It orchestrates specialized agents for planning, multimodal retrieval, relevance filtering, and incremental synthesis to produce long-form reports with coherent text-image integration.

## Key Features

- **Agentic Multimodal Search & Filtering** — Dual-stream retrieval (text + images) with VLM-based relevance filtering
- **Checklist-Guided Incremental Synthesis** — Structured outline with semantic anchors ensures content completeness
- **Recurrent Context Management** — Compressed historical context maintains cross-section coherence
- **M2LongBench Benchmark** — 247 tasks, 9 domains, 95K images + 108M text chunks in a stable sandbox

## Project Structure

```
Deep-Report/
├── deep_reporter/          # Core multi-agent framework (LangGraph)
│   ├── core/               # StateGraph & state definitions
│   ├── nodes/              # Agent nodes (planner, search, filter, writer)
│   ├── llms/               # LLM adapters (OpenAI-compatible)
│   ├── config/             # Configuration management
│   ├── prompts/            # Prompt templates for all agents
│   ├── utils/              # Logging, parsing utilities
│   └── api.py              # Main API entry point
├── retriever/              # Multimodal retrieval service (Milvus + Jina)
└── evaluation/             # Multi-dimensional evaluation system
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- GPU with 24GB+ VRAM (for Jina embeddings model)
- ~60GB disk space (for sandbox data)

### Step 1: Clone & Install

```bash
git clone https://github.com/fangda-ye/Deep-Report.git
cd Deep-Report
pip install -r requirements.txt
```

### Step 2: Download Data from HuggingFace

All benchmark data and the pre-built retrieval sandbox are hosted on HuggingFace:
[https://huggingface.co/datasets/Fangda-Ye/Deep-Reporter-Data](https://huggingface.co/datasets/Fangda-Ye/Deep-Reporter-Data)

```bash
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Fangda-Ye/Deep-Reporter-Data',
    repo_type='dataset',
    local_dir='./data'
)
"
```

### Step 3: Extract Image Archives

The sandbox images are distributed as compressed archives for efficient download. Extract them:

```bash
cd data/sandbox
mkdir -p images
for f in TestImages.tar.gz TestRefImages.tar.gz TestAugImages.tar.gz; do
  echo "Extracting $f..."
  tar -xzf "$f" -C images/
done
cd ../..  # back to project root
```

After extraction, your `data/` directory will look like:
```
data/
├── benchmark/
│   ├── article_deconstructions.jsonl          # 247 benchmark tasks
│   └── article_deconstructions_enriched.jsonl # Tasks with retrieved evidence
└── sandbox/
    ├── milvus_MMLF.db                        # Pre-built Milvus vector database (31 GB)
    ├── TestImages.tar.gz                      # (can delete after extraction)
    ├── TestRefImages.tar.gz
    ├── TestAugImages.tar.gz
    └── images/                                # Extracted source images (~22 GB)
        ├── TestImages/       (2,332 images)
        ├── TestRefImages/    (15,960 images)
        └── TestAugImages/    (140,596 images)
```

### Step 4: Set API Keys

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export QWEN_API_KEY="your-dashscope-api-key"   # used by the filter model
```

### Step 5: Start the Retriever Service

Open a **separate terminal** in the project root:

```bash
cd retriever
export DATA_ROOT="../data"
python retriever_app.py
```

The service starts at `http://localhost:5555`. Verify it's working:
```bash
curl -X POST http://localhost:5555/search \
  -H "Content-Type: application/json" \
  -d '{"query_list": ["renewable energy market analysis"], "type": "text", "topk": 5}'
```

### Step 6: Generate a Report

In another terminal, from the **project root**:

```python
from deep_reporter.api import create_openai_api

api = create_openai_api({
    "openai_api_key": "your-openai-api-key",
    "openai_base_url": "https://api.openai.com/v1",
    "primary_model": "gpt-5-mini",
    "auxiliary_model": "gpt-5-mini",
    "retriever_url": "http://localhost:5555/search",
})

result = api.generate_article(
    overall_query="Analyze the impact of large language models on scientific research",
    overall_checklist=[
        "Cover key application areas across disciplines",
        "Discuss methodological shifts in literature review and writing",
        "Include limitations and risks",
    ],
    generation_mode="with_planner",
    text_topk=20,
    image_topk=10,
    enable_filter=True,
)

print(result["final_article"])
```

See `deep_reporter/example.py` for the full set of options.

### Step 7: Evaluate

```bash
cd evaluation

python batch_evaluate_all.py \
    --input-files ../gen_articles/your_model_outputs.jsonl \
    --benchmark ../data/benchmark/article_deconstructions.jsonl \
    --enriched  ../data/benchmark/article_deconstructions_enriched.jsonl \
    --article-output ./eval_results/article \
    --search-output  ./eval_results/search
```

> `article_deconstructions_enriched.jsonl` provides the silver-standard evidence pool
> used to compute retrieval/citation precision against the benchmark; it must not
> be fed back as oracle context to the generator.

**Evaluation dimensions:**

| Level | Metrics |
|-------|---------|
| **Section Anchor** | Description adherence, Checklist coverage |
| **Section Content** | Richness, Image-text coherence, Placement, Clarity |
| **Full Report** | Coherence, Fluency, Repetition control, Termination quality |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
