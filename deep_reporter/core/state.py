# LongGen/core/state.py
"""
State definitions for multimodal long-form generation.
"""
from typing import List, Dict, Any, TypedDict, Optional

class SectionInfo(TypedDict):
    """Section info structure."""
    section_description: str
    sectional_checklist: List[str]
    content: Optional[str]
    text_queries: Optional[List[str]]
    image_queries: Optional[List[str]]
    retrieved_texts: Optional[List[Dict[str, Any]]]
    retrieved_images: Optional[List[Dict[str, Any]]]
    filtered_texts: Optional[List[Dict[str, Any]]]
    filtered_images: Optional[List[Dict[str, Any]]]

class MultimodalLongFormState(TypedDict):
    """State for multimodal long-form generation."""
    overall_query: str
    overall_checklist: List[str]
    generation_mode: str  # "with_planner" or "without_planner"

    outline: List[SectionInfo]

    current_section_index: int

    previous_sections_summary: str
    previous_section_tail: str

    text_topk: int
    image_topk: int
    enable_filter: bool

    final_article: str

    session_id: str
    session_logger: Any

class ConfigSchema(TypedDict):
    """Configuration schema."""
    implementation: str
    llm_config: Dict[str, Any]
    retriever_url: str
