# DeepDiver_pipeline/llms/factory.py
"""
LLM factory: creates different LLM instances based on configuration.
"""
import os
import httpx
from typing import Literal, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from ..config import get_runtime_config

LLMType = Literal["primary", "auxiliary", "filter"]

class LLMFactory:
    """LLM factory class."""

    @staticmethod
    def create_llm(llm_type: LLMType, node_type: str = "answer") -> BaseChatModel:
        """Create an LLM instance."""
        config = get_runtime_config()
        return LLMFactory._create_openai_llm(llm_type, config.llm_config)

    @staticmethod
    def _create_openai_llm(llm_type: LLMType, llm_config: Dict[str, Any]) -> BaseChatModel:
        """Create an OpenAI-series LLM."""
        from .reasoning_openai import ReasoningOpenAI
        
        if llm_type == "primary":
            model_name = llm_config.get("primary_model", "deepseek-r1")

            # Select API config based on model name
            if "deepseek" in model_name.lower():
                api_key = llm_config["deepseek_api_key"]
                base_url = llm_config["deepseek_base_url"]
            elif "gpt" in model_name.lower():
                api_key = llm_config["openai_api_key"]
                base_url = llm_config["openai_base_url"]
            else:
                api_key = llm_config["qwen_api_key"]
                base_url = llm_config["qwen_base_url"]
                
            return ReasoningOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                http_proxy=llm_config.get("http_proxy"),
                max_tokens=llm_config["max_tokens"],
                timeout=llm_config["timeout"],
                enable_reasoning=False
            )
        elif llm_type == "auxiliary":
            model_name = llm_config.get("auxiliary_model", "qwen-plus-latest")
            return ReasoningOpenAI(
                model=model_name,
                api_key=llm_config.get("auxiliary_api_key", os.getenv("AUXILIARY_API_KEY", "")),
                base_url=llm_config.get("auxiliary_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                http_proxy=llm_config.get("http_proxy"),
                max_tokens=llm_config["max_tokens"],
                timeout=llm_config["timeout"],
                enable_reasoning=False
            )
        else:  # filter
            model_name = llm_config.get("filter_model", "qwen-plus-latest")
            filter_api_key = llm_config.get("filter_api_key") or llm_config.get("auxiliary_api_key", os.getenv("AUXILIARY_API_KEY", ""))
            filter_base_url = llm_config.get("filter_base_url") or llm_config.get("auxiliary_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1")

            return ReasoningOpenAI(
                model=model_name,
                api_key=filter_api_key,
                base_url=filter_base_url,
                http_proxy=llm_config.get("http_proxy"),
                max_tokens=llm_config["max_tokens"],
                timeout=llm_config["timeout"],
                enable_reasoning=False
            )

def get_primary_llm(node_type: str = "answer") -> BaseChatModel:
    """Get the primary LLM."""
    return LLMFactory.create_llm("primary", node_type)

def get_auxiliary_llm() -> BaseChatModel:
    """Get the auxiliary LLM (Qwen)."""
    return LLMFactory.create_llm("auxiliary")

def get_filter_llm() -> BaseChatModel:
    """Get the filter LLM (Qwen or DeepSeek)."""
    return LLMFactory.create_llm("filter")

def get_vlm_filter_llm() -> BaseChatModel:
    """Get the VLM filter model (locally deployed InternVL via OpenAI API)."""
    import logging

    config = get_runtime_config()

    # Try creating VLM model (using OpenAI client)
    try:
        vlm = ChatOpenAI(
            model=config.llm_config.get("vlm_model", "internvl35-38b"),
            openai_api_key=config.llm_config.get("vlm_api_key", "EMPTY"),
            openai_api_base=config.llm_config.get("vlm_base_url", "http://localhost:9000/v1"),
            max_tokens=512,
            temperature=0.0,
            timeout=60
        )
        return vlm
    except Exception as e:
        logging.warning(f"Failed to create VLM model: {e}. Image filtering will be skipped.")
        return None
