# LongGen/nodes/combine_node.py
"""
Combine node - merges all sections into the final article.
"""
import logging
from langchain_core.runnables import RunnableConfig

from ..core.state import MultimodalLongFormState
from ..utils.session_logger import DetailedSessionLogger


def combine_node(state: MultimodalLongFormState, config: RunnableConfig):
    """Combine all sections into the final article."""
    outline = state.get("outline", [])
    
    logging.insight(f"Combining article - {len(outline)} sections")
    sections_content = []
    total_citations = 0
    
    for idx, section_info in enumerate(outline):
        content = section_info.get("content", "")
        if content:
            sections_content.append(content)
            # Count citations
            total_citations += content.count("[citation:")
            total_citations += content.count("![](citation:")
        else:
            logging.insight(f"Warning: Section {idx} has no content", level='warning')
            sections_content.append(f"[Section {idx+1} content missing]")
    
    # Merge all content
    final_article = "\n\n".join(sections_content)
    
    logging.insight(f"Article combined - total length: {len(final_article)}, total citations: {total_citations}")
    session_logger: DetailedSessionLogger = state.get('session_logger')
    if session_logger:
        session_logger.log_final_report(
            report_content=final_article
        )
    
    return {"final_article": final_article}