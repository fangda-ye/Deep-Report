# DeepDiver_pipeline/utils/stream_formatter.py
"""
Stream output formatter.
"""
import json
import time
import re
import logging
import traceback
from typing import Dict, Any, Tuple

def format_stream_chunk(chunk_type: str, content: Any, position: str, search_end: bool = False) -> str:
    """
    Generate a standardized streaming output chunk.

    Args:
        chunk_type: Chunk type (thinking, formal, search, filter, etc.).
        content: Content payload.
        position: Position (start, middle, end).
        search_end: Whether search has ended.

    Returns:
        Formatted JSON string.
    """
    chunk_data = {
        "content": [{
            "type": chunk_type,
            "content": content,
            "chunk_position": position,
            "search_end": search_end
        }],
        "is_deep_research": True,
        "create_time": time.time()
    }
    
    return json.dumps(chunk_data, ensure_ascii=False)

def parse_response_stream(response_stream) -> Tuple[str, str]:
    """
    Parse 'content' and 'reasoning_content' from a model response.

    Handles two input types:
    1. String: direct response content (non-streaming).
    2. Iterable: list of streaming response chunks.

    Returns:
        Tuple of (generated_content, generated_reasoning).
    """
    if isinstance(response_stream, str):
        return response_stream, ""
    
    # Streaming processing logic
    generated_content = ""
    generated_reasoning = ""

    try:
        for formatted_chunk in response_stream:
            # Safely handle chunks that may not match expected structure
            try:
                # Check if chunk is empty
                if not formatted_chunk:
                    continue

                # Safely access nested chunk dict
                inner_chunk = formatted_chunk.get("content", [{}])[0]

                if not inner_chunk:
                    continue
                
                # Extract content
                if inner_chunk.get('content'):
                    if 'thinking' in inner_chunk.get("type", ""):
                        generated_reasoning += inner_chunk['content']
                    elif 'content' in inner_chunk.get("type", ""):
                        generated_content += inner_chunk['content']
                    elif 'formal' in inner_chunk.get("type", ""):
                        generated_content += inner_chunk['content']

            except (IndexError, KeyError, TypeError) as e:
                logging.insight(
                        f"Stream parse warning: skipping malformed chunk - chunk: {formatted_chunk[:100]}..., error: {type(e).__name__}, message: {str(e)}", 
                        level='error'
                    )
                continue
                
    except TypeError:
        logging.insight(
                f"Stream parse error: unsupported response_stream type - {type(response_stream)}, preview: {str(response_stream)[:100]}", 
                level='error'
            )
        return str(response_stream), ""
            
    return generated_content, generated_reasoning


def structured_chunk_format(types, content, chunk_position, search_end=False):
    result = {'content': [{"type": types, "content": content,
                            "chunk_position": chunk_position,
                            "search_end": search_end}],
                "is_deep_research": True,
                "create_time": time.time()}
    return result  
