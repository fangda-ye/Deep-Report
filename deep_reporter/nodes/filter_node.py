# LongGen/nodes/filter_node.py
"""
Optional filter node - filters retrieval results based on enable_filter config.
"""
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..core.state import MultimodalLongFormState
from ..llms.factory import get_filter_llm
from ..utils.llm_utils import LLMCaller
from ..utils.stream_formatter import format_stream_chunk
from ..utils.common_utils import parse_json_output, safe_extract_indices, get_model_name
from ..utils.session_logger import DetailedSessionLogger
from ..prompts.templates import FILTER_TEMPLATE


def filter_node(state: MultimodalLongFormState, config: RunnableConfig):
    """Optional filter node."""
    enable_filter = state.get("enable_filter", False)
    current_idx = state.get("current_section_index", 0)
    outline = state.get("outline", [])

    if current_idx >= len(outline):
        logging.insight(f"Section index {current_idx} out of range, skipping filter")
        return {}

    current_section = outline[current_idx]
    
    if not enable_filter:
        logging.insight("Filter node disabled, using retrieval results directly")
        current_section["filtered_texts"] = current_section.get("retrieved_texts", [])
        current_section["filtered_images"] = current_section.get("retrieved_images", [])
        return {"outline": outline}

    logging.insight(f"Filtering retrieval results for section {current_idx}")
    writer = get_stream_writer()

    retrieved_texts = current_section.get("retrieved_texts", [])
    retrieved_images = current_section.get("retrieved_images", [])

    if not retrieved_texts and not retrieved_images:
        logging.insight("No retrieval results to filter")
        current_section["filtered_texts"] = []
        current_section["filtered_images"] = []
        return {"outline": outline}

    try:
        llm = get_filter_llm()

        # === Filter text ===
        filtered_texts = []
        if retrieved_texts:
            if writer:
                writer(format_stream_chunk("thinking", f"Filtering {len(retrieved_texts)} texts...", "start"))

            # Format text sources as txt1, txt2, ...
            text_sources = []
            for idx, text_item in enumerate(retrieved_texts, start=1):
                content = text_item.get('content', text_item.get('text', ''))
                text_sources.append(f"[Source txt{idx} begin]\n{content}\n[Source txt{idx} end]")

            sources_text = "\n\n".join(text_sources)

            # Build filter prompt
            section_desc = current_section["section_description"]
            sectional_checklist = current_section.get("sectional_checklist", [])
            checklist_text = "\n".join([f"- {item}" for item in sectional_checklist])

            prompt_text = FILTER_TEMPLATE.format(
                section_description=section_desc,
                sectional_checklist=checklist_text,
                sources_list=sources_text
            )

            messages = [{"role": "user", "content": prompt_text}]

            # Call LLM for filtering
            generated_content, _ = LLMCaller.call_llm(llm, messages)

            # Parse filter results
            useful_indices = safe_extract_indices(generated_content)

            logging.insight(f"Text filtering done - original:{len(retrieved_texts)}, filtered:{len(useful_indices)}")

            for idx in useful_indices:
                real_idx = idx - 1  # Convert to 0-based
                if 0 <= real_idx < len(retrieved_texts):
                    filtered_texts.append(retrieved_texts[real_idx])

            if writer:
                writer(format_stream_chunk("thinking", f"Kept {len(filtered_texts)} texts", "middle"))

        # === Filter images (using VLM) ===
        filtered_images = []
        image_filter_decisions = []

        if retrieved_images:
            if writer:
                writer(format_stream_chunk("thinking", f"VLM filtering {len(retrieved_images)} images...", "middle"))

            # Try VLM-based filtering
            try:
                from ..llms.factory import get_vlm_filter_llm
                from ..prompts.templates import IMAGE_FILTER_TEMPLATE
                import concurrent.futures
                from langchain_core.messages import HumanMessage
                import base64
                import os

                vlm = get_vlm_filter_llm()

                if vlm is None:
                    logging.insight("VLM unavailable, using score-based image filtering")
                    if all('score' in img for img in retrieved_images):
                        sorted_images = sorted(retrieved_images, key=lambda x: x.get('score', 0), reverse=True)
                        filtered_images = sorted_images[:5]
                    else:
                        filtered_images = retrieved_images[:5]

                    image_filter_decisions = [
                        {"index": i+1, "decision": "kept_by_score", "reason": "VLM unavailable"}
                        for i in range(len(filtered_images))
                    ]
                else:
                    section_desc = current_section["section_description"]
                    sectional_checklist = current_section.get("sectional_checklist", [])
                    checklist_text = "\n".join([f"- {item}" for item in sectional_checklist])

                    def filter_single_image(img_item, img_idx):
                        """Filter a single image using VLM."""
                        try:
                            img_path = img_item.get("img_path", "")
                            img_desc = img_item.get("caption", img_item.get("description", "No description"))

                            if not os.path.exists(img_path):
                                logging.warning(f"Image file not found: {img_path}")
                                return {
                                    "index": img_idx,
                                    "keep": False,
                                    "decision": "FILE_NOT_FOUND",
                                    "image": img_item
                                }

                            # Read image and convert to base64
                            with open(img_path, 'rb') as f:
                                image_data = f.read()
                                image_base64 = base64.b64encode(image_data).decode('utf-8')

                            # Build prompt
                            prompt_text = IMAGE_FILTER_TEMPLATE.format(
                                section_description=section_desc,
                                sectional_checklist=checklist_text,
                                image_description=img_desc
                            )

                            # Build multimodal message (OpenAI format)
                            message = HumanMessage(
                                content=[
                                    {"type": "text", "text": prompt_text},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_base64}"
                                        }
                                    }
                                ]
                            )

                            # Call VLM
                            response = vlm.invoke([message])
                            decision = response.content.strip().upper()

                            keep = "YES" in decision

                            return {
                                "index": img_idx,
                                "keep": keep,
                                "decision": decision,
                                "image": img_item
                            }
                        except Exception as e:
                            logging.warning(f"VLM image filter failed for image {img_idx}: {e}")
                            return {
                                "index": img_idx,
                                "keep": True,
                                "decision": "ERROR",
                                "error": str(e),
                                "image": img_item
                            }

                    # Process concurrently (max 5 workers)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [
                            executor.submit(filter_single_image, img, i+1)
                            for i, img in enumerate(retrieved_images)
                        ]
                        results = [f.result() for f in concurrent.futures.as_completed(futures)]

                    # Sort by original order
                    results.sort(key=lambda x: x["index"])

                    # Extract kept images and decision records
                    for result in results:
                        image_filter_decisions.append({
                            "index": result["index"],
                            "decision": result["decision"],
                            "kept": result["keep"],
                            "error": result.get("error")
                        })

                        if result["keep"]:
                            filtered_images.append(result["image"])

                    logging.insight(f"VLM image filtering done - original:{len(retrieved_images)}, kept:{len(filtered_images)}")

            except Exception as e:
                logging.warning(f"VLM image filtering failed: {e}, using fallback strategy")
                if all('score' in img for img in retrieved_images):
                    sorted_images = sorted(retrieved_images, key=lambda x: x.get('score', 0), reverse=True)
                    filtered_images = sorted_images[:5]
                else:
                    filtered_images = retrieved_images[:5]

                image_filter_decisions = [
                    {"index": i+1, "decision": "kept_by_fallback", "reason": str(e)}
                    for i in range(len(filtered_images))
                ]

            if writer:
                writer(format_stream_chunk("thinking", f"VLM filtering done, kept {len(filtered_images)} images", "middle"))

        if writer:
            writer(format_stream_chunk("thinking", "", "end"))

        # Update outline
        current_section["filtered_texts"] = filtered_texts
        current_section["filtered_images"] = filtered_images

        # Log to session
        session_logger: DetailedSessionLogger = state.get('session_logger')
        if session_logger:
            session_logger.log_llm_call(
                call_type="filtering",
                function_name="filter_node",
                model_info=get_model_name(llm),
                input_messages=messages if retrieved_texts else [],
                output_content=generated_content if retrieved_texts else "No filtering needed",
                success=True,
                iteration_num=current_idx
            )

            # Log filter details
            session_logger.log_filter_result(
                section_index=current_idx,
                original_text_count=len(retrieved_texts),
                original_image_count=len(retrieved_images),
                filtered_text_count=len(filtered_texts),
                filtered_image_count=len(filtered_images),
                text_filter_mask=[i in useful_indices for i in range(1, len(retrieved_texts) + 1)] if retrieved_texts else [],
                image_filter_strategy="vlm_based" if image_filter_decisions and any(d.get("decision") not in ["kept_by_score", "kept_by_fallback"] for d in image_filter_decisions) else "score_based",
                image_filter_decisions=image_filter_decisions
            )

        return {"outline": outline}

    except Exception as e:
        logging.insight(f"Filter node error: {type(e).__name__}, {str(e)}", level='error')

        if writer:
            writer(format_stream_chunk("thinking", f"Filtering failed, using all results: {str(e)}", "end"))

        # On failure, use all retrieval results
        current_section["filtered_texts"] = retrieved_texts
        current_section["filtered_images"] = retrieved_images

        return {"outline": outline}