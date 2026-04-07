# LongGen/nodes/section_writing_node.py
"""
Section writing node - generates content for the current section.
Features: position-aware generation, unified citation format,
context continuity, and context updates for subsequent sections.
"""
import logging
from langchain_core.runnables import RunnableConfig
from langgraph.config import get_stream_writer

from ..core.state import MultimodalLongFormState
from ..llms.factory import get_primary_llm
from ..utils.llm_utils import LLMCaller
from ..utils.common_utils import get_model_name
from ..utils.stream_formatter import format_stream_chunk
from ..utils.session_logger import DetailedSessionLogger
from ..prompts.templates import SECTION_WRITING_TEMPLATE


def section_writing_node(state: MultimodalLongFormState, config: RunnableConfig):
    """Section writing node."""
    current_idx = state.get("current_section_index", 0)
    outline = state.get("outline", [])

    if current_idx >= len(outline):
        logging.insight(f"Section index {current_idx} out of range, skipping writing")
        return {}

    current_section = outline[current_idx]
    logging.insight(f"Writing section {current_idx}: {current_section['section_description'][:100]}")

    writer = get_stream_writer()

    try:
        llm = get_primary_llm("answer")
        enable_filter = state.get("enable_filter", False)
        if enable_filter and current_section.get("filtered_texts") is not None:
            text_materials = current_section.get("filtered_texts", [])
            image_materials = current_section.get("filtered_images", [])
            logging.insight(f"Using filtered materials - texts:{len(text_materials)}, images:{len(image_materials)}")
        else:
            text_materials = current_section.get("retrieved_texts", [])
            image_materials = current_section.get("retrieved_images", [])
            logging.insight(f"Using retrieved materials - texts:{len(text_materials)}, images:{len(image_materials)}")

        # Build material list for citation (format: txt1, img1)
        formatted_materials = []

        # Add text materials as txt1, txt2, ...
        for idx, text_item in enumerate(text_materials, start=1):
            content = text_item.get('content', text_item.get('text', ''))
            material_text = f"[Source txt{idx} begin]\n{content}\n[Source txt{idx} end]"
            formatted_materials.append(material_text)

        # Add image materials as img1, img2, ...
        for idx, img_item in enumerate(image_materials, start=1):
            # Image description or caption
            img_desc = img_item.get('caption', img_item.get('description', 'Image'))
            material_text = f"[Source img{idx}: {img_desc}]"
            formatted_materials.append(material_text)

        materials_text = "\n\n".join(formatted_materials) if formatted_materials else "No materials retrieved"

        # Determine section position
        total_sections = len(outline)
        is_first = (current_idx == 0)
        is_last = (current_idx == total_sections - 1)
        
        position_info = {
            "current_index": current_idx + 1,  # 1-based for display
            "total_sections": total_sections,
            "is_first": is_first,
            "is_last": is_last
        }

        # Get context info
        previous_summary = state.get("previous_sections_summary", "")
        previous_tail = state.get("previous_section_tail", "")

        # Format checklists
        overall_checklist = state.get("overall_checklist", [])
        sectional_checklist = current_section.get("sectional_checklist", [])
        
        overall_checklist_text = "\n".join([f"- {item}" for item in overall_checklist])
        sectional_checklist_text = "\n".join([f"- {item}" for item in sectional_checklist])

        # Build prompt
        prompt_text = SECTION_WRITING_TEMPLATE.format(
            overall_query=state["overall_query"],
            overall_checklist=overall_checklist_text,
            section_description=current_section["section_description"],
            sectional_checklist=sectional_checklist_text,
            previous_sections_summary=previous_summary if previous_summary else "This is the first section",
            previous_section_tail=previous_tail if previous_tail else "No previous content",
            retrieved_materials=materials_text,
            position_info=f"Section {position_info['current_index']}/{position_info['total_sections']}",
            is_first_section="yes" if is_first else "no",
            is_last_section="yes" if is_last else "no"
        )

        messages = [{"role": "user", "content": prompt_text}]

        logging.insight(f"Calling section writing LLM - model: {get_model_name(llm)}, position: {position_info}")

        if writer:
            writer(format_stream_chunk("formal", "Generating section content...", "start"))

        generated_content, reasoning_content = LLMCaller.call_llm(
            llm, messages, writer, "formal"
        )

        if writer:
            writer(format_stream_chunk("formal", "", "end"))

        logging.insight(f"Section {current_idx} writing complete - length: {len(generated_content)}")
        outline[current_idx]["content"] = generated_content

        # Update context for subsequent sections
        updated_summary, updated_tail = _update_context(
            state.get("previous_sections_summary", ""),
            generated_content,
            current_section["section_description"]
        )

        # Log to session
        session_logger: DetailedSessionLogger = state.get('session_logger')
        if session_logger:
            session_logger.log_llm_call(
                call_type="section_writing",
                function_name="section_writing_node",
                model_info=get_model_name(llm),
                input_messages=messages,
                output_content=generated_content,
                reasoning_content=reasoning_content,
                success=True,
                iteration_num=current_idx
            )

            # Extract actually used citations
            import re
            text_citations = list(set(re.findall(r'\[citation:(txt\d+)\]', generated_content)))
            image_citations = list(set(re.findall(r'!\[\]\(citation:(img\d+)\)', generated_content)))

            # Get used source info (filtered or retrieved)
            used_text_sources = []
            used_image_sources = []

            # Get material sources
            if enable_filter and current_section.get("filtered_texts") is not None:
                source_texts = current_section.get("filtered_texts", [])
                source_images = current_section.get("filtered_images", [])
            else:
                source_texts = current_section.get("retrieved_texts", [])
                source_images = current_section.get("retrieved_images", [])

            # Extract cited sources
            for citation in text_citations:
                idx = int(citation.replace("txt", "")) - 1  # txt1 -> index 0
                if 0 <= idx < len(source_texts):
                    item = source_texts[idx]
                    used_text_sources.append({
                        "citation_id": citation,
                        "uid": item.get("uid", item.get("id", "")),
                        "score": item.get("score", 0),
                        "content_preview": item.get("content", item.get("text", ""))[:200],
                        "source": item.get("source", "")
                    })

            for citation in image_citations:
                idx = int(citation.replace("img", "")) - 1
                if 0 <= idx < len(source_images):
                    item = source_images[idx]
                    used_image_sources.append({
                        "citation_id": citation,
                        "uid": item.get("uid", item.get("id", "")),
                        "score": item.get("score", 0),
                        "img_path": item.get("img_path", ""),
                        "description": item.get("caption", item.get("description", ""))[:200]
                    })

            # Log section completion
            session_logger.log_section_completion(
                section_index=current_idx,
                section_description=current_section["section_description"],
                content=generated_content,
                text_citations=sorted(text_citations),
                image_citations=sorted(image_citations),
                text_sources=used_text_sources,
                image_sources=used_image_sources
            )

        return {
            "outline": outline,
            "previous_sections_summary": updated_summary,
            "previous_section_tail": updated_tail,
            "current_section_index": current_idx + 1
        }

    except Exception as e:
        logging.insight(f"Section writing node error: {type(e).__name__}, {str(e)}", level='error')

        if writer:
            writer(format_stream_chunk("formal", f"Writing failed: {str(e)}", "end"))

        # Save error info
        outline[current_idx]["content"] = f"[Section generation failed: {str(e)}]"

        return {
            "outline": outline,
            "current_section_index": current_idx + 1
        }


def _update_context(previous_summary: str, new_content: str, section_desc: str) -> tuple:
    """Update context info. Returns (updated_summary, section_tail)."""
    sentences = new_content.split('.')
    if len(sentences) >= 3:
        tail = '. '.join(sentences[-3:]).strip()
    else:
        tail = new_content[-200:].strip() if len(new_content) > 200 else new_content.strip()

    # Generate brief summary (section description + first 100 chars)
    content_preview = new_content[:100].strip() + "..." if len(new_content) > 100 else new_content.strip()
    new_summary_entry = f"Section: {section_desc}\nContent: {content_preview}\n"

    # Update total summary (keep reasonable length)
    if previous_summary:
        updated_summary = previous_summary + "\n" + new_summary_entry
    else:
        updated_summary = new_summary_entry

    # If summary is too long (>2000 chars), compress earlier content
    if len(updated_summary) > 2000:
        summary_parts = updated_summary.split("\nSection:")
        # Keep detail for the most recent 3-4 sections
        if len(summary_parts) > 4:
            early_parts = summary_parts[:len(summary_parts)-3]
            recent_parts = summary_parts[len(summary_parts)-3:]
            
            compressed_early = "\n".join([
                part.split("\n")[0] if "\n" in part else part[:50]
                for part in early_parts[:3]
            ])
            
            updated_summary = compressed_early + "\n\nSection:" + "Section:".join(recent_parts)

    return updated_summary, tail