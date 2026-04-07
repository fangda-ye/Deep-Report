# LongGen/nodes/planner_node.py
"""
Outline generation node.
Supports two modes:
1. with_planner: auto-generate an outline
2. without_planner: skip this node, use user-provided outline
"""
import logging
import json
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..core.state import MultimodalLongFormState, SectionInfo
from ..llms.factory import get_primary_llm
from ..utils.llm_utils import LLMCaller
from ..utils.common_utils import parse_json_output, get_model_name
from ..utils.stream_formatter import format_stream_chunk
from ..utils.session_logger import DetailedSessionLogger
from ..prompts.templates import OUTLINE_GENERATION_TEMPLATE


def planner_node(state: MultimodalLongFormState, config: RunnableConfig):
    """Outline generation / validation node."""
    generation_mode = state.get("generation_mode", "with_planner")

    # === without_planner mode: validate and use provided outline ===
    if generation_mode == "without_planner":
        outline = state.get("outline", [])
        if not outline or len(outline) == 0:
            logging.insight("without_planner mode but no outline provided, returning default", level='error')
            return {
                "outline": [{
                    "section_description": state["overall_query"],
                    "sectional_checklist": state.get("overall_checklist", []),
                    "content": None,
                    "text_queries": None,
                    "image_queries": None,
                    "retrieved_texts": None,
                    "retrieved_images": None,
                    "filtered_texts": None,
                    "filtered_images": None
                }],
                "current_section_index": 0
            }
        
        logging.insight(f"without_planner mode, using provided outline - {len(outline)} sections")
        for idx, section in enumerate(outline):
            if "section_description" not in section:
                logging.insight(f"Section {idx} missing section_description", level='warning')
            if "sectional_checklist" not in section:
                logging.insight(f"Section {idx} missing sectional_checklist", level='warning')
                section["sectional_checklist"] = []

        # Log to session (without_planner mode)
        session_logger: DetailedSessionLogger = state.get('session_logger')
        if session_logger:
            session_logger.log_outline_generation(
                generation_mode="without_planner",
                outline=outline,
                overall_query=state["overall_query"],
                overall_checklist=state.get("overall_checklist", [])
            )

        return {
            "outline": outline,
            "current_section_index": 0
        }

    # === with_planner mode: auto-generate outline ===
    logging.insight(f"with_planner mode, generating outline - Query: {state['overall_query'][:100]}")
    writer = get_stream_writer()

    try:
        llm = get_primary_llm("planner")
        overall_query = state["overall_query"]
        overall_checklist = state.get("overall_checklist", [])
        checklist_text = "\n".join([f"- {item}" for item in overall_checklist])

        prompt_text = OUTLINE_GENERATION_TEMPLATE.format(
            overall_query=overall_query,
            overall_checklist=checklist_text
        )

        messages = [{"role": "user", "content": prompt_text}]

        logging.insight(f"Calling outline generation LLM - model: {get_model_name(llm)}")
        if writer:
            writer(format_stream_chunk("thinking", "Generating article outline...", "start"))

        generated_content, reasoning_content = LLMCaller.call_llm(
            llm, messages, writer, "thinking"
        )

        if writer:
            writer(format_stream_chunk("thinking", "", "end"))

        logging.insight(f"Outline generation complete - content length: {len(generated_content)}")
        outline_result = parse_json_output(generated_content, ["outline"])

        if not outline_result or "outline" not in outline_result:
            logging.insight("Outline parsing failed, using default single section", level='warning')
            outline = [
                {
                    "section_description": overall_query,
                    "sectional_checklist": overall_checklist,
                    "content": None,
                    "text_queries": None,
                    "image_queries": None,
                    "retrieved_texts": None,
                    "retrieved_images": None,
                    "filtered_texts": None,
                    "filtered_images": None
                }
            ]
        else:
            outline = []
            for section_data in outline_result["outline"]:
                section_info: SectionInfo = {
                    "section_description": section_data.get("section_description", ""),
                    "sectional_checklist": section_data.get("sectional_checklist", []),
                    "content": None,
                    "text_queries": None,
                    "image_queries": None,
                    "retrieved_texts": None,
                    "retrieved_images": None,
                    "filtered_texts": None,
                    "filtered_images": None
                }
                outline.append(section_info)

        logging.insight(f"Successfully generated outline - {len(outline)} sections")
        session_logger: DetailedSessionLogger = state.get('session_logger')
        if session_logger:
            # Log LLM call
            session_logger.log_llm_call(
                call_type="outline_generation",
                function_name="planner_node",
                model_info=get_model_name(llm),
                input_messages=messages,
                output_content=generated_content,
                reasoning_content=reasoning_content,
                success=True
            )
            # Log outline generation result
            session_logger.log_outline_generation(
                generation_mode="with_planner",
                outline=outline,
                overall_query=overall_query,
                overall_checklist=overall_checklist
            )

        return {
            "outline": outline,
            "current_section_index": 0
        }

    except Exception as e:
        logging.insight(f"Outline generation node error: {type(e).__name__}, {str(e)}", level='error')

        if writer:
            writer(format_stream_chunk("thinking", f"Outline generation failed: {str(e)}", "end"))

        # Return default single-section outline
        return {
            "outline": [
                {
                    "section_description": state["overall_query"],
                    "sectional_checklist": state.get("overall_checklist", []),
                    "content": None,
                    "text_queries": None,
                    "image_queries": None,
                    "retrieved_texts": None,
                    "retrieved_images": None,
                    "filtered_texts": None,
                    "filtered_images": None
                }
            ],
            "current_section_index": 0
        }