# LongGen/api.py
"""
API interface for multimodal long-form article generation.
"""
import uuid
import time
import logging
from typing import List, Dict, Any, Optional, Iterator

from .config import initialize_config, get_runtime_config
from .core.graph import MultimodalLongFormGraph
from .core.state import SectionInfo
from .utils.session_logger import DetailedSessionLogger


class LongGenAPI:
    """Main API class for long-form article generation."""

    def __init__(self, implementation: str = "openai", config_dict: Dict[str, Any] = None):
        """
        Args:
            implementation: Kept for backward compatibility; ignored (always OpenAI).
            config_dict: Configuration dictionary.
        """
        if config_dict is None:
            config_dict = {}
        initialize_config(implementation, config_dict)
        
        self.config = get_runtime_config()
        self.graph = MultimodalLongFormGraph()
        
        logging.insight(f"LongGen API initialized, implementation: {implementation}")
    
    def generate_article(
        self,
        overall_query: str,
        overall_checklist: List[str],
        generation_mode: str = "with_planner",
        outline: Optional[List[Dict]] = None,
        text_topk: int = 20,
        image_topk: int = 10,
        enable_filter: bool = False,
        retriever_url: str = "http://localhost:5555/search",
        userid: str = "default_user"
    ) -> Dict[str, Any]:
        """
        Generate a long-form article.

        Args:
            overall_query: Main query/topic.
            overall_checklist: List of requirements.
            generation_mode: "with_planner" or "without_planner".
            outline: Required when generation_mode is "without_planner".
            text_topk: Top-k for text retrieval.
            image_topk: Top-k for image retrieval.
            enable_filter: Whether to enable the filter node.
            retriever_url: Retrieval service URL.
            userid: User ID.

        Returns:
            Result dictionary.
        """
        if generation_mode not in ["with_planner", "without_planner"]:
            return {
                "success": False,
                "error": "generation_mode must be 'with_planner' or 'without_planner'"
            }
        
        if generation_mode == "without_planner" and not outline:
            return {
                "success": False,
                "error": "outline is required when generation_mode is 'without_planner'"
            }
        
        session_id = f"{userid}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session_logger = DetailedSessionLogger(
            session_id, 
            userid,
            log_dir=self.config.base_config.log_dir,
            log_file_prefix=self.config.base_config.log_file_prefix
        )
        
        initial_state = {
            "overall_query": overall_query,
            "overall_checklist": overall_checklist,
            "generation_mode": generation_mode,
            "outline": self._prepare_outline(outline) if outline else [],
            "current_section_index": 0,
            "previous_sections_summary": "",
            "previous_section_tail": "",
            "text_topk": text_topk,
            "image_topk": image_topk,
            "enable_filter": enable_filter,
            "final_article": "",
            "session_id": session_id,
            "session_logger": session_logger
        }
        
        config = {
            "configurable": {
                "implementation": self.config.implementation,
                "llm_config": self.config.llm_config,
                "retriever_url": retriever_url
            },
            "recursion_limit": 50
        }
        
        try:
            session_logger.log_user_input(
                query=overall_query,
                document_list=[],
                history=[],
                quotes_left_chars=0,
                userid=userid,
                generation_mode=generation_mode,
                enable_filter=enable_filter,
                text_topk=text_topk,
                image_topk=image_topk
            )
            
            final_state = self.graph.run(initial_state, config)
            
            return {
                "success": True,
                "final_article": final_state.get('final_article', ''),
                "total_sections": len(final_state.get('outline', [])),
                "session_id": session_id,
                "implementation": self.config.implementation,
                "outline": final_state.get('outline', [])
            }
            
        except Exception as e:
            logging.insight(f"Article generation failed: {e}", level="error")
            session_logger.log_error("execution_error", str(e))
            
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "implementation": self.config.implementation
            }
            
        finally:
            session_logger.finalize_session()
    
    def generate_article_stream(
        self,
        overall_query: str,
        overall_checklist: List[str],
        generation_mode: str = "with_planner",
        outline: Optional[List[Dict]] = None,
        text_topk: int = 20,
        image_topk: int = 10,
        enable_filter: bool = False,
        retriever_url: str = "http://localhost:5555/search",
        userid: str = "default_user"
    ) -> Iterator[str]:
        """
        Stream-generate a long-form article.

        Args:
            Same as generate_article.

        Yields:
            Streaming data chunks.
        """
        if generation_mode not in ["with_planner", "without_planner"]:
            yield {"error": "generation_mode must be 'with_planner' or 'without_planner'"}
            return
        
        if generation_mode == "without_planner" and not outline:
            yield {"error": "outline is required when generation_mode is 'without_planner'"}
            return
        
        # Generate session ID and logger
        session_id = f"{userid}_stream_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        session_logger = DetailedSessionLogger(session_id, userid)
        
        initial_state = {
            "overall_query": overall_query,
            "overall_checklist": overall_checklist,
            "generation_mode": generation_mode,
            "outline": self._prepare_outline(outline) if outline else [],
            "current_section_index": 0,
            "previous_sections_summary": "",
            "previous_section_tail": "",
            "text_topk": text_topk,
            "image_topk": image_topk,
            "enable_filter": enable_filter,
            "final_article": "",
            "session_id": session_id,
            "session_logger": session_logger
        }
        
        config = {
            "configurable": {
                "implementation": self.config.implementation,
                "llm_config": self.config.llm_config,
                "retriever_url": retriever_url
            },
            "recursion_limit": 50
        }

        try:
            session_logger.log_user_input(
                query=overall_query,
                document_list=[],
                history=[],
                quotes_left_chars=0,
                userid=userid,
                generation_mode=generation_mode,
                enable_filter=enable_filter,
                text_topk=text_topk,
                image_topk=image_topk
            )
            
            for chunk in self.graph.stream(initial_state, config):
                yield chunk
                
        except Exception as e:
            logging.insight(f"Streaming generation failed: {e}", level="error")
            session_logger.log_error("streaming_error", str(e))
            yield {"error": str(e)}
            
        finally:
            session_logger.finalize_session()
    
    def _prepare_outline(self, outline_dicts: List[Dict]) -> List[SectionInfo]:
        """Convert user-provided outline dicts to SectionInfo format."""
        prepared_outline = []
        for section_dict in outline_dicts:
            section_info: SectionInfo = {
                "section_description": section_dict.get("section_description", ""),
                "sectional_checklist": section_dict.get("sectional_checklist", []),
                "content": None,
                "text_queries": None,
                "image_queries": None,
                "retrieved_texts": None,
                "retrieved_images": None,
                "filtered_texts": None,
                "filtered_images": None
            }
            prepared_outline.append(section_info)
        
        return prepared_outline
    
    def get_config_info(self) -> Dict[str, Any]:
        """Return current configuration info."""
        return {
            "implementation": self.config.implementation,
            "use_streaming": self.config.use_streaming,
            "base_config": self.config.base_config.__dict__
        }


def create_openai_api(config_dict: Dict[str, Any]) -> LongGenAPI:
    """Create an API instance using the OpenAI-compatible backend."""
    return LongGenAPI("openai", config_dict)