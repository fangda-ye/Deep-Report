# Deep-Reporter

**Deep Research for Grounded Multimodal Long-Form Generation**

Deep-Reporter is a unified agentic framework for generating comprehensive, evidence-grounded multimodal reports. It orchestrates specialized agents for planning, multimodal retrieval, relevance filtering, and incremental synthesis to produce long-form reports with coherent text-image integration.

## Key Features

- **Agentic Multimodal Search & Filtering** — Dual-stream retrieval (text + images) with VLM-based relevance filtering
- **Checklist-Guided Incremental Synthesis** — Structured outline with semantic anchors ensures content completeness
- **Recurrent Context Management** — Compressed historical context maintains cross-section coherence
- **M2LongBench Benchmark** — 247 tasks, 9 domains, 95K images + 108M text chunks in a stable sandbox

## Project Structure

```
Deep-Reporter/
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
git clone https://github.com/fangda-ye/Deep-Reporter.git
cd Deep-Reporter
pip install -r requirements.txt
```

### Step 2: Download Data from HuggingFace

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

The sandbox images are distributed as compressed archives. Extract them after download:

```bash
cd data/sandbox
mkdir -p images
for f in TestImages.tar.gz TestRefImages.tar.gz TestAugImages.tar.gz; do
  tar -xzf "$f" -C images/
done
```

After extraction, your `data/` directory will contain:
```
data/
├── benchmark/
│   ├── article_deconstructions.jsonl          # 247 benchmark tasks
│   └── article_deconstructions_enriched.jsonl # Tasks with retrieved evidence
└── sandbox/
    ├── milvus_MMLF.db                        # Pre-built vector database (31GB)
    ├── TestImages.tar.gz                      # (can delete after extraction)
    ├── TestRefImages.tar.gz
    ├── TestAugImages.tar.gz
    └── images/                                # Extracted images (~22GB)
        ├── TestImages/
        ├── TestRefImages/
        └── TestAugImages/
```

### Step 4: Set API Keys

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export QWEN_API_KEY="your-dashscope-api-key"   # used by the filter model
```

### Step 5: Start the Retriever Service

```bash
cd retriever
export DATA_ROOT="../data"
python retriever_app.py
```

The service starts at `http://localhost:5555`. Verify:
```bash
curl -X POST http://localhost:5555/search \
  -H "Content-Type: application/json" \
  -d '{"query_list": ["renewable energy market analysis"], "type": "text", "topk": 5}'
```

### Step 6: Generate a Report

```python
from deep_reporter.api import LongGenAPI

api = LongGenAPI(
    primary_model="gpt-5-mini",
    retriever_url="http://localhost:5555/search",
)

result = api.generate_article(
    query="Analyze the impact of large language models on scientific research",
    use_planner=True,
    enable_filter=True,
)

print(result["article"])
```

### Step 7: Evaluate

```bash
cd evaluation

python batch_evaluate_all.py \
    --input_dir ../gen_articles/ \
    --deconstruction_file ../data/benchmark/article_deconstructions_enriched.jsonl \
    --output_dir ./eval_results/
```

**Evaluation dimensions:**

| Level | Metrics |
|-------|---------|
| **Section Anchor** | Description adherence, Checklist coverage |
| **Section Content** | Richness, Image-text coherence, Placement, Clarity |
| **Full Report** | Coherence, Fluency, Repetition control, Termination quality |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
