# LongGen/nodes/search_node.py
"""
Multimodal retrieval node - supports text and image retrieval.
"""
import logging
import requests
import json
import time
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..core.state import MultimodalLongFormState
from ..utils.stream_formatter import format_stream_chunk
from ..utils.session_logger import DetailedSessionLogger


def search_node(state: MultimodalLongFormState, config: RunnableConfig):
    """Multimodal retrieval node."""
    current_idx = state.get("current_section_index", 0)
    outline = state.get("outline", [])

    if current_idx >= len(outline):
        logging.insight(f"Section index {current_idx} out of range, skipping retrieval")
        return {}

    current_section = outline[current_idx]
    text_queries = current_section.get("text_queries", [])
    image_queries = current_section.get("image_queries", [])

    if not text_queries and not image_queries:
        logging.insight("No queries, skipping retrieval")
        return {}

    logging.insight(f"Starting multimodal retrieval - Section {current_idx}, text queries:{len(text_queries)}, image queries:{len(image_queries)}")
    writer = get_stream_writer()

    # Get retrieval config
    retriever_url = config['configurable'].get("retriever_url", "http://localhost:5555/search")
    text_topk = state.get("text_topk", 20)
    image_topk = state.get("image_topk", 10)

    # Retrieve text
    retrieved_texts = []
    if text_queries:
        try:
            if writer:
                writer(format_stream_chunk("thinking", f"Retrieving text... (topk={text_topk})", "start"))

            payload = {
                "query_list": text_queries,
                "type": "text",
                "topk": text_topk
            }

            logging.insight(f"Calling text retrieval service - URL: {retriever_url}, queries: {text_queries}")
            response = requests.post(retriever_url, json=payload, timeout=600)
            response.raise_for_status()

            response_json = response.json()
            if response_json.get("status") == "success":
                retrieved_texts = response_json.get("results", [])
                logging.insight(f"Text retrieval success - {len(retrieved_texts)} results")
            else:
                logging.insight(f"Text retrieval returned non-success status: {response_json.get('status')}", level='warning')

            if writer:
                writer(format_stream_chunk("thinking", f"Retrieved {len(retrieved_texts)} texts", "middle"))

        except Exception as e:
            logging.insight(f"Text retrieval failed: {e}", level="error")
            if writer:
                writer(format_stream_chunk("thinking", f"Text retrieval failed: {str(e)}", "middle"))

    time.sleep(2)
    # Retrieve images
    retrieved_images = []
    if image_queries:
        try:
            if writer:
                writer(format_stream_chunk("thinking", f"Retrieving images... (topk={image_topk})", "middle"))

            payload = {
                "query_list": image_queries,
                "type": "image",
                "topk": image_topk
            }

            logging.insight(f"Calling image retrieval service - URL: {retriever_url}, queries: {image_queries}")
            response = requests.post(retriever_url, json=payload, timeout=600)
            response.raise_for_status()

            response_json = response.json()
            if response_json.get("status") == "success":
                retrieved_images = response_json.get("results", [])
                # Remove base64 data (too large)
                for img in retrieved_images:
                    img.pop('img_base64', None)
                logging.insight(f"Image retrieval success - {len(retrieved_images)} results")
            else:
                logging.insight(f"Image retrieval returned non-success status: {response_json.get('status')}", level='warning')

            if writer:
                writer(format_stream_chunk("thinking", f"Retrieved {len(retrieved_images)} images", "middle"))

        except Exception as e:
            logging.insight(f"Image retrieval failed: {e}", level="error")
            if writer:
                writer(format_stream_chunk("thinking", f"Image retrieval failed: {str(e)}", "middle"))

    if writer:
        writer(format_stream_chunk("thinking", "", "end"))

    # Update current section's retrieval results
    outline[current_idx]["retrieved_texts"] = retrieved_texts
    outline[current_idx]["retrieved_images"] = retrieved_images

    # Log to session
    session_logger: DetailedSessionLogger = state.get('session_logger')
    if session_logger:
        session_logger.log_search_call(
            mode="multimodal",
            text_queries=text_queries,
            image_queries=image_queries,
            text_results=retrieved_texts,
            image_results=retrieved_images,
            success=True,
            iteration_num=current_idx
        )

    logging.insight(f"Retrieval complete - texts:{len(retrieved_texts)}, images:{len(retrieved_images)}")

    return {"outline": outline}
