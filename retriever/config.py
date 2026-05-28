"""
Retriever service configuration.
All paths can be overridden via environment variables.
"""
import os

# ============== Model Configuration ==============
JINA_MODEL_PATH = os.getenv("JINA_MODEL_PATH", "jinaai/jina-embeddings-v4")

# ============== Data Paths ==============
# Root directory of the downloaded HuggingFace dataset
DATA_ROOT = os.getenv("DATA_ROOT", "./data")

# Milvus vector database path (download from HuggingFace)
MILVUS_PATH = os.getenv("MILVUS_PATH", os.path.join(DATA_ROOT, "sandbox", "milvus_MMLF.db"))

IMAGE_BASE_DIR = os.getenv("IMAGE_BASE_DIR", os.path.join(DATA_ROOT, "sandbox", "images"))

# Original image path prefixes stored in the pre-built Milvus DB.
# The retriever strips the matched prefix and joins the remainder with
# IMAGE_BASE_DIR at query time, so users can place the extracted image
# archives anywhere without editing the DB.
ORIGINAL_IMAGE_PREFIXES = [
    "/M2LongBench/SandboxData/SourceImages/v1/",
]

# ============== Processing Parameters ==============
MAX_CHUNK_LENGTH = 350
DEFAULT_TEXT_BATCH_SIZE = 16
DEFAULT_IMAGE_BATCH_SIZE = 16

# ============== Service Configuration ==============
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_PORT = int(os.getenv("FLASK_PORT", "5555"))
FLASK_DEBUG = False

# ============== GPU/CPU Configuration ==============
USE_CPU = os.getenv("USE_CPU", "false").lower() == "true"
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

# ============== Logging ==============
LOG_DIR = "logs"
LOG_FILE = "logs/retriever.log"
LOG_ROTATION = "10 MB"
