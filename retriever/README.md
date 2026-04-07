# Retriever Service

Multimodal retrieval service built on Jina Embeddings v4 and Milvus, supporting text and image vector search.

## Quick Start

```bash
# Set data root to the downloaded HuggingFace dataset
export DATA_ROOT="../data"

# Start the service (GPU by default)
python retriever_app.py

# Or use CPU only
USE_CPU=true python retriever_app.py
```

The service starts at `http://localhost:5555`.

## Configuration

All settings are in `config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_ROOT` | `./data` | Root directory of downloaded HF dataset |
| `MILVUS_PATH` | `{DATA_ROOT}/sandbox/milvus_MMLF.db` | Vector database path |
| `IMAGE_BASE_DIR` | `{DATA_ROOT}/sandbox/images` | Image files directory |
| `JINA_MODEL_PATH` | `jinaai/jina-embeddings-v4` | Jina model (local path or HF name) |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device ID |
| `USE_CPU` | `false` | Force CPU mode |
| `FLASK_PORT` | `5555` | Service port |

## API

### Search

```bash
curl -X POST http://localhost:5555/search \
  -H "Content-Type: application/json" \
  -d '{"query_list": ["renewable energy trends"], "type": "text", "topk": 10}'
```

**Parameters:**
- `query_list`: List of query strings
- `type`: `"text"` or `"image"`
- `topk`: Number of results (default: 10)

**Response:**
```json
{
  "status": "success",
  "type": "text",
  "total_hits": 10,
  "results": [
    {"id": "...", "type": "text", "text": "...", "score": 0.95}
  ]
}
```

For image results, each item includes `img_path`, `description`, and `img_base64`.

## Image Path Remapping

The pre-built Milvus database stores image paths from the original build environment. The service automatically remaps these to your local `IMAGE_BASE_DIR` at query time. No manual path editing is needed.
