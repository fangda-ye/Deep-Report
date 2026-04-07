# DeepDiver_pipeline/config/runtime_config.py
"""
Runtime configuration management.
"""
from typing import Dict, Any
from .base_config import BaseConfig


class RuntimeConfig:
    """Runtime configuration manager."""

    def __init__(self, base_config: BaseConfig):
        self.implementation = "openai"
        self.base_config = base_config
        self._validate_config()

    def _validate_config(self):
        """Validate that required configuration fields are present."""
        # At least one API key should be configured
        pass

    @property
    def use_streaming(self) -> bool:
        """Whether to use streaming output."""
        return False

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get the LLM configuration dictionary."""
        return {
            "http_proxy": self.base_config.http_proxy,
            "max_tokens": self.base_config.max_tokens,
            "timeout": self.base_config.timeout,
            # Model config
            "primary_model": self.base_config.primary_model,
            "auxiliary_model": self.base_config.auxiliary_model,
            "filter_model": self.base_config.filter_model,
            # API keys and endpoints
            "openai_api_key": self.base_config.openai_api_key,
            "openai_base_url": self.base_config.openai_base_url,
            "deepseek_api_key": self.base_config.deepseek_api_key,
            "deepseek_base_url": self.base_config.deepseek_base_url,
            "qwen_api_key": self.base_config.qwen_api_key,
            "qwen_base_url": self.base_config.qwen_base_url,
            # Auxiliary / filter model config
            "auxiliary_api_key": self.base_config.auxiliary_api_key,
            "auxiliary_base_url": self.base_config.auxiliary_base_url,
            "filter_api_key": self.base_config.filter_api_key,
            "filter_base_url": self.base_config.filter_base_url,
            # VLM config
            "vlm_model": self.base_config.vlm_model,
            "vlm_base_url": self.base_config.vlm_base_url,
            "vlm_api_key": self.base_config.vlm_api_key,
        }


# Global config instance
_global_runtime_config: RuntimeConfig = None

def initialize_config(implementation: str = "openai", config_dict: Dict[str, Any] = None):
    """Initialize the global runtime configuration.

    Args:
        implementation: Kept for backward compatibility; ignored (always OpenAI).
        config_dict: Configuration dictionary.
    """
    global _global_runtime_config
    if config_dict is None:
        config_dict = {}
    base_config = BaseConfig.from_dict(config_dict)
    _global_runtime_config = RuntimeConfig(base_config)

def get_runtime_config() -> RuntimeConfig:
    """Get the global runtime configuration."""
    if _global_runtime_config is None:
        raise RuntimeError("Configuration not initialized. Call initialize_config() first.")
    return _global_runtime_config
