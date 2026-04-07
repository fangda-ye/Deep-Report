# LongGen/nodes/query_generation_node.py
"""
Query generation node - generates retrieval queries for the current section.
"""
import logging
import json
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..core.state import MultimodalLongFormState
from ..llms.factory import get_primary_llm
from ..utils.llm_utils import LLMCaller
from ..utils.common_utils import parse_json_output, get_model_name
from ..utils.stream_formatter import format_stream_chunk
from ..utils.session_logger import DetailedSessionLogger
from ..prompts.templates import QUERY_GENERATION_TEMPLATE


def query_generation_node(state: MultimodalLongFormState, config: RunnableConfig):
    """Query generation node."""
    current_idx = state.get("current_section_index", 0)
    outline = state.get("outline", [])

    if current_idx >= len(outline):
        logging.insight(f"Section index {current_idx} out of range, skipping query generation")
        return {}

    current_section = outline[current_idx]
    logging.insight(f"Generating queries - Section {current_idx}: {current_section['section_description'][:100]}")

    writer = get_stream_writer()

    try:
        llm = get_primary_llm("planner")
        overall_query = state["overall_query"]
        overall_checklist = state.get("overall_checklist", [])
        section_description = current_section["section_description"]
        sectional_checklist = current_section.get("sectional_checklist", [])
        previous_sections_summary = state.get("previous_sections_summary", "")

        checklist_text = "\n".join([f"- {item}" for item in overall_checklist])
        sectional_checklist_text = "\n".join([f"- {item}" for item in sectional_checklist])

        prompt_text = QUERY_GENERATION_TEMPLATE.format(
            overall_query=overall_query,
            overall_checklist=checklist_text,
            previous_sections_summary=previous_sections_summary or "This is the first section",
            section_description=section_description,
            sectional_checklist=sectional_checklist_text
        )

        messages = [{"role": "user", "content": prompt_text}]

        logging.insight(f"Calling query generation LLM - model: {get_model_name(llm)}")

        if writer:
            writer(format_stream_chunk("thinking", "Generating retrieval queries...", "start"))

        generated_content, reasoning_content = LLMCaller.call_llm(
            llm, messages, writer, "thinking"
        )

        if writer:
            writer(format_stream_chunk("thinking", "", "end"))

        logging.insight(f"Query generation complete - content length: {len(generated_content)}")
        queries_result = parse_json_output(generated_content, ["text_queries", "image_queries"])

        if not queries_result:
            logging.insight("Query parsing failed, using defaults", level='warning')
            text_queries = [section_description]
            image_queries = [section_description]
        else:
            text_queries = queries_result.get("text_queries", [section_description])
            image_queries = queries_result.get("image_queries", [section_description])

        logging.insight(f"Generated {len(text_queries)} text queries, {len(image_queries)} image queries")
        outline[current_idx]["text_queries"] = text_queries
        outline[current_idx]["image_queries"] = image_queries

        # Log to session
        session_logger: DetailedSessionLogger = state.get('session_logger')
        if session_logger:
            session_logger.log_llm_call(
                call_type="query_generation",
                function_name="query_generation_node",
                model_info=get_model_name(llm),
                input_messages=messages,
                output_content=generated_content,
                reasoning_content=reasoning_content,
                success=True
            )

        return {"outline": outline}

    except Exception as e:
        logging.insight(f"Query generation node error: {type(e).__name__}, {str(e)}", level='error')

        if writer:
            writer(format_stream_chunk("thinking", f"Query generation failed: {str(e)}", "end"))

        # Fall back to default queries
        outline[current_idx]["text_queries"] = [current_section["section_description"]]
        outline[current_idx]["image_queries"] = [current_section["section_description"]]

        return {"outline": outline}
