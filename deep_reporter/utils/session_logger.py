# DeepDiver_pipeline/utils/session_logger.py
"""
Unified session logger supporting both implementation modes.
"""
import json
import os
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..config import get_runtime_config

class DetailedSessionLogger:
    """Detailed session logger."""
    
    def __init__(self, session_id: str, user_id: str, 
                 log_dir: str = None, log_file_prefix: str = None):
        self.session_id = session_id
        self.user_id = user_id

        # Get config
        try:
            config = get_runtime_config()
            self.log_dir = log_dir or config.base_config.log_dir
            log_prefix = log_file_prefix or config.base_config.log_file_prefix
        except:
            # Use defaults if config not initialized
            self.log_dir = log_dir or "longgen_logs"
            log_prefix = log_file_prefix

        # Generate log file name
        if log_prefix:
            self.log_file = os.path.join(
                self.log_dir,
                f"{log_prefix}_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
        else:
            self.log_file = os.path.join(
                self.log_dir,
                f"longgen_sessions_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Get current config info
        config = get_runtime_config()
        
        # Session record structure
        self.session_record = {
            "session_id": session_id,
            "user_id": user_id,
            "implementation": config.implementation,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "user_input": {},
            "outline_generation": {},
            "sections": [],
            "llm_calls": [],
            "search_calls": [],
            "final_report": {},
            "error_log": [],
            "performance_metrics": {
                "total_llm_calls": 0,
                "total_search_calls": 0,
                "total_processing_time": 0,
                "total_sections_processed": 0
            }
        }
    
    def log_user_input(self, query: str, document_list: List[str], history: List[str],
                      quotes_left_chars: int, userid: str, generation_mode: str = "with_planner",
                      enable_filter: bool = False, text_topk: int = 20, image_topk: int = 10):
        """Log user input."""
        self.session_record["user_input"] = {
            "overall_query": query,
            "generation_mode": generation_mode,
            "enable_filter": enable_filter,
            "text_topk": text_topk,
            "image_topk": image_topk,
            "userid": userid,
            "timestamp": datetime.now().isoformat()
        }

    def log_outline_generation(self, generation_mode: str, outline: List[Dict],
                              overall_query: str, overall_checklist: List[str]):
        """Log outline generation results."""
        outline_data = {
            "generation_mode": generation_mode,
            "overall_query": overall_query,
            "overall_checklist": overall_checklist,
            "total_sections": len(outline),
            "sections_summary": [
                {
                    "section_index": idx,
                    "section_description": section.get("section_description", ""),
                    "sectional_checklist": section.get("sectional_checklist", [])
                }
                for idx, section in enumerate(outline)
            ],
            "timestamp": datetime.now().isoformat()
        }
        self.session_record["outline_generation"] = outline_data
        self.session_record["performance_metrics"]["total_sections_processed"] = len(outline)


    def log_llm_call(self, call_type: str, function_name: str, model_info: str,
                    input_messages: List[Dict], output_content: str,
                    reasoning_content: str = "", success: bool = True,
                    processing_time: float = 0, iteration_num: Optional[int] = None):
        """Log LLM call details."""
        call_record = {
            "call_id": len(self.session_record["llm_calls"]) + 1,
            "call_type": call_type,
            "function_name": function_name,
            "model_info": model_info,
            "input_messages": input_messages,
            "output_content": output_content,
            "reasoning_content": reasoning_content,
            "success": success,
            "processing_time": processing_time,
            "iteration_num": iteration_num,
            "timestamp": datetime.now().isoformat()
        }
        self.session_record["llm_calls"].append(call_record)
        self.session_record["performance_metrics"]["total_llm_calls"] += 1
    
    def log_search_call(self, mode: str, queries: List[str] = None, response_data=None,
                       success: bool = True, processing_time: float = 0,
                       iteration_num: Optional[int] = None,
                       text_queries: Optional[List[str]] = None,
                       image_queries: Optional[List[str]] = None,
                       text_results: Optional[List[Dict]] = None,
                       image_results: Optional[List[Dict]] = None,
                       llm_filter_mask: Optional[List[bool]] = None,
                       deduplication_mask: Optional[List[bool]] = None,
                       status: Optional[List[Dict]] = None):
        """Log search call details."""
        if text_queries is not None or image_queries is not None:
            # Process text results - keep full info
            text_results_summary = []
            if text_results:
                for idx, item in enumerate(text_results, 1):
                    summary_item = {
                        "index": f"txt{idx}",
                        "uid": item.get("uid", item.get("id", "")),
                        "score": item.get("score", 0),
                        "content": item.get("content", item.get("text", "")),
                        "source": item.get("source", ""),
                        "metadata": item.get("metadata", {})
                    }
                    # Preserve additional fields
                    for key in ["doc_id", "chunk_id", "paragraph_id"]:
                        if key in item:
                            summary_item[key] = item[key]
                    text_results_summary.append(summary_item)

            # Process image results - keep full info (exclude base64)
            image_results_summary = []
            if image_results:
                for idx, item in enumerate(image_results, 1):
                    summary_item = {
                        "index": f"img{idx}",
                        "uid": item.get("uid", item.get("id", "")),
                        "score": item.get("score", 0),
                        "img_path": item.get("img_path", ""),
                        "caption": item.get("caption", ""),
                        "description": item.get("description", ""),
                        "source": item.get("source", ""),
                        "metadata": item.get("metadata", {})
                    }
                    # Preserve other fields, exclude base64
                    for key in ["doc_id", "page_num", "figure_id"]:
                        if key in item:
                            summary_item[key] = item[key]
                    image_results_summary.append(summary_item)

            call_record = {
                "call_id": len(self.session_record["search_calls"]) + 1,
                "mode": mode,
                "text_queries": text_queries or [],
                "image_queries": image_queries or [],
                "text_results": text_results_summary,
                "image_results": image_results_summary,
                "text_results_count": len(text_results or []),
                "image_results_count": len(image_results or []),
                "total_results": len(text_results or []) + len(image_results or []),
                "success": success,
                "processing_time": processing_time,
                "section_index": iteration_num,
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Legacy format compatibility
            call_record = {
                "call_id": len(self.session_record["search_calls"]) + 1,
                "mode": mode,
                "input_queries": queries or [],
                "success": success,
                "processing_time": processing_time,
                "iteration_num": iteration_num,
                "timestamp": datetime.now().isoformat()
            }

        if status is not None:
            call_record["status"] = status
            if isinstance(status, list):
                for status_dict in status:
                    if isinstance(status_dict, dict):
                        for code, message in status_dict.items():
                            if code != "200":
                                self.log_error(
                                    error_type=f"search_error_{code}",
                                    error_message=f"Search service error: {message}",
                                    context={
                                        "mode": mode,
                                        "queries": queries,
                                        "iteration_num": iteration_num,
                                        "status_code": code
                                    }
                                )
            elif isinstance(status, str) and status != "success":
                self.log_error(
                    error_type="search_error",
                    error_message=f"Search service error: {status}",
                    context={
                        "mode": mode,
                        "queries": queries,
                        "iteration_num": iteration_num
                    }
                )


        if llm_filter_mask is not None:
            call_record["llm_filter_mask"] = llm_filter_mask

        if deduplication_mask is not None:
            call_record["deduplication_mask"] = deduplication_mask

        self.session_record["search_calls"].append(call_record)
        self.session_record["performance_metrics"]["total_search_calls"] += 1

    def log_filter_result(self, section_index: int, original_text_count: int,
                         original_image_count: int, filtered_text_count: int,
                         filtered_image_count: int, text_filter_mask: List[bool],
                         image_filter_strategy: str, image_filter_decisions: List[Dict] = None):
        """Log filtering results."""
        filter_record = {
            "section_index": section_index,
            "text_filtering": {
                "original_count": original_text_count,
                "filtered_count": filtered_text_count,
                "filter_mask": text_filter_mask,
                "kept_indices": [i+1 for i, keep in enumerate(text_filter_mask) if keep]
            },
            "image_filtering": {
                "original_count": original_image_count,
                "filtered_count": filtered_image_count,
                "strategy": image_filter_strategy,
                "decisions": image_filter_decisions or []
            },
            "timestamp": datetime.now().isoformat()
        }

        # Add to filter records list
        if "filter_calls" not in self.session_record:
            self.session_record["filter_calls"] = []
        self.session_record["filter_calls"].append(filter_record)

    def log_section_completion(self, section_index: int, section_description: str,
                               content: str, text_citations: List[str], image_citations: List[str],
                               text_sources: Optional[List[Dict]] = None,
                               image_sources: Optional[List[Dict]] = None):
        """Log single section completion."""
        section_record = {
            "section_index": section_index,
            "section_description": section_description,
            "content_length": len(content),
            "content": content,
            "text_citations": text_citations,
            "image_citations": image_citations,
            "text_citations_count": len(text_citations),
            "image_citations_count": len(image_citations),
            "total_citations": len(text_citations) + len(image_citations),
            "text_sources": text_sources or [],
            "image_sources": image_sources or [],
            "timestamp": datetime.now().isoformat()
        }
        self.session_record["sections"].append(section_record)

    def log_final_report(self, report_content: str):
        """Log the final report."""
        # Sum citations from each section (each section has independent numbering)
        total_text_citations = 0
        total_image_citations = 0

        for section in self.session_record.get("sections", []):
            total_text_citations += section.get("text_citations_count", 0)
            total_image_citations += section.get("image_citations_count", 0)

        self.session_record["final_report"] = {
            "content": report_content,
            "content_length": len(report_content),
            "total_text_citations": total_text_citations,
            "total_image_citations": total_image_citations,
            "total_citations": total_text_citations + total_image_citations,
            "sections_count": len(self.session_record.get("sections", [])),
            "timestamp": datetime.now().isoformat()
        }
    
    def log_error(self, error_type: str, error_message: str, context: Dict = None):
        """Log an error."""
        error_record = {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
            "timestamp": datetime.now().isoformat()
        }
        self.session_record["error_log"].append(error_record)
    
    def finalize_session(self):
        """Finalize session record and write to file."""
        self.session_record["end_time"] = datetime.now().isoformat()

        # Calculate total processing time
        start_time = datetime.fromisoformat(self.session_record["start_time"])
        end_time = datetime.fromisoformat(self.session_record["end_time"])
        self.session_record["performance_metrics"]["total_processing_time"] = (end_time - start_time).total_seconds()

        # Write to JSONL file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(self.session_record, ensure_ascii=False) + '\n')
            logging.insight(f"Session {self.session_id} logged to {self.log_file}")
        except Exception as e:
            logging.insight(f"Failed to write session log: {e}", level="error")
