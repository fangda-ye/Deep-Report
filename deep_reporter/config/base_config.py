# LongGen/config/base_config.py
"""
Base configuration with support for external config injection.
"""
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class BaseConfig:
    """Base configuration class; all values are injected externally."""
    qwen_api_key: str = field(default_factory=lambda: os.getenv("QWEN_API_KEY", ""))
    qwen_base_url: str = field(default_factory=lambda: os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

    # Model selection
    primary_model: str = "deepseek-r1"
    auxiliary_model: str = "qwen-plus-latest"
    filter_model: str = "qwen-plus-latest"

    # Auxiliary model config (for outline/query generation; independent of primary model)
    auxiliary_api_key: str = field(default_factory=lambda: os.getenv("AUXILIARY_API_KEY", ""))
    auxiliary_base_url: str = field(default_factory=lambda: os.getenv("AUXILIARY_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))

    # Filter model config (defaults to auxiliary settings when None)
    filter_api_key: Optional[str] = None
    filter_base_url: Optional[str] = None

    # Multimodal long-form generation config
    retriever_url: str = "http://localhost:5555/search"
    default_text_topk: int = 20
    default_image_topk: int = 10

    # VLM config (local vllm deployment, OpenAI API compatible)
    vlm_model: str = "internvl35-38b"
    vlm_base_url: str = "http://localhost:9000/v1"
    vlm_api_key: str = "EMPTY"

    http_proxy: Optional[str] = None
    
    # LLM general config
    max_tokens: int = 8192
    timeout: int = 600

    # Logging config
    log_file_prefix: Optional[str] = None
    log_dir: str = "longform_generation_logs"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create a BaseConfig from a dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})