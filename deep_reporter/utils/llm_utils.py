# DeepDiver_pipeline/utils/llm_utils.py
"""
Unified LLM calling interface.
"""
import json
import time
import logging
from typing import Dict, Any, Tuple, List
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from ..config import get_runtime_config
from .stream_formatter import format_stream_chunk, parse_response_stream

class LLMCaller:
    """Unified LLM caller."""
    
    @staticmethod
    def call_llm(llm: BaseChatModel, messages: List[Dict[str, str]], 
                 stream_writer=None, stream_type: str = "thinking") -> Tuple[str, str]:
        """
        Unified LLM call interface.

        Args:
            llm: LLM instance.
            messages: Message list.
            stream_writer: Stream output writer (streaming mode only).
            stream_type: Stream output type.

        Returns:
            Tuple of (generated_content, reasoning_content).
        """
        config = get_runtime_config()
        
        # Convert to LangChain message format
        lc_messages = [HumanMessage(content=msg["content"]) for msg in messages if msg["role"] == "user"]
        
        if config.use_streaming and stream_writer:
            return LLMCaller._call_streaming(llm, lc_messages, stream_writer, stream_type)
        else:
            return LLMCaller._call_non_streaming(llm, lc_messages)
    
    @staticmethod
    def _call_streaming(llm: BaseChatModel, messages: List, stream_writer, stream_type: str) -> Tuple[str, str]:
        """Streaming call."""
        stream_writer(format_stream_chunk(stream_type, "", "start"))

        all_chunks = []
        
        try:
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    all_chunks.extend(chunk.content)
                    
        except Exception as e:
            logging.insight(f"Streaming call failed: {e}", level="error")
            stream_writer(format_stream_chunk(stream_type, f"Call failed: {str(e)}", "middle"))
        finally:
            # Send end signal
            stream_writer(format_stream_chunk(stream_type, "", "end"))
        
        # Parse the streamed response
        generated_content, generated_reasoning = parse_response_stream(all_chunks)
        return generated_content, generated_reasoning
    
    @staticmethod 
    def _call_non_streaming(llm: BaseChatModel, messages: List) -> Tuple[str, str]:
        """Non-streaming call."""
        try:
            response = llm.invoke(messages)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Check for reasoning content
            reasoning_content = ""
            if hasattr(response, 'additional_kwargs') and response.additional_kwargs.get('reasoning_content'):
                reasoning_content = response.additional_kwargs['reasoning_content']
            
            return content, reasoning_content
            
        except Exception as e:
            logging.insight(f"Non-streaming call failed: {e}", level="error")
            return f"Call failed: {str(e)}", ""
