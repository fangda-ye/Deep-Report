# LongGen/core/graph.py
"""
Workflow graph for multimodal long-form generation.
Supports two modes:
1. with_planner: auto-generate outline then write per-section.
2. without_planner: use a provided outline and write per-section.
"""
import logging
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.config import RunnableConfig

from .state import MultimodalLongFormState, ConfigSchema
from ..nodes.planner_node import planner_node
from ..nodes.query_generation_node import query_generation_node
from ..nodes.search_node import search_node
from ..nodes.filter_node import filter_node
from ..nodes.section_writing_node import section_writing_node
from ..nodes.combine_node import combine_node


class MultimodalLongFormGraph:
    """Multimodal long-form generation graph."""
    
    def __init__(self):
        self.workflow = self._build_graph()
    
    def _build_graph(self):
        """Build the workflow graph."""
        builder = StateGraph(MultimodalLongFormState, config_schema=ConfigSchema)

        # Add all nodes
        builder.add_node("planner", planner_node)
        builder.add_node("query_generation", query_generation_node)
        builder.add_node("search", search_node)
        builder.add_node("filter", filter_node)
        builder.add_node("section_writing", section_writing_node)
        builder.add_node("combine", combine_node)
        
        # Entry point
        builder.set_entry_point("planner")
        
        # Route after planner
        builder.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "query_generation": "query_generation",
                "combine": "combine"
            }
        )
        
        # Query generation -> Search
        builder.add_edge("query_generation", "search")
        
        # Route after search: depends on enable_filter
        builder.add_conditional_edges(
            "search",
            self._route_after_search,
            {
                "filter": "filter",
                "section_writing": "section_writing"
            }
        )
        
        # Filter -> Section Writing
        builder.add_edge("filter", "section_writing")
        
        # Route after section writing: check for remaining sections
        builder.add_conditional_edges(
            "section_writing",
            self._should_continue_sections,
            {
                "continue": "query_generation",
                "finish": "combine"
            }
        )
        
        # Combine -> END
        builder.add_edge("combine", END)

        return builder.compile()

    def _route_after_planner(self, state: MultimodalLongFormState):
        """Routing decision after planner node."""
        generation_mode = state.get("generation_mode", "with_planner")
        outline = state.get("outline", [])
        
        if not outline or len(outline) == 0:
            logging.insight("Outline is empty, skipping to combine", level='warning')
            return "combine"
        
        logging.insight(f"Planner done, mode: {generation_mode}, sections: {len(outline)}")
        return "query_generation"

    def _route_after_search(self, state: MultimodalLongFormState):
        """Routing decision after search node."""
        enable_filter = state.get("enable_filter", False)
        
        if enable_filter:
            logging.insight("Filter node enabled")
            return "filter"
        else:
            logging.insight("Skipping filter node, proceeding to writing")
            return "section_writing"

    def _should_continue_sections(self, state: MultimodalLongFormState):
        """Decide whether to continue processing the next section."""
        current_idx = state.get("current_section_index", 0)
        outline = state.get("outline", [])
        
        if current_idx < len(outline):
            logging.insight(f"Continuing to section {current_idx}/{len(outline)}")
            return "continue"
        else:
            logging.insight(f"All {len(outline)} sections complete, proceeding to combine")
            return "finish"

    def run(self, initial_state: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Run the graph synchronously."""
        return self.workflow.invoke(initial_state, config)
    
    def stream(self, initial_state: Dict[str, Any], config: Dict[str, Any]):
        """Run the graph in streaming mode."""
        return self.workflow.stream(initial_state, config, stream_mode=["messages", "custom"])