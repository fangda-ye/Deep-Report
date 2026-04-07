# DeepDiver_pipeline/utils/common_utils.py
"""
Common utility functions.
"""
import json
import re
from typing import Dict, Any, Optional, List

def parse_json_output(text: str, required_keys: List[str] = None) -> Optional[Dict]:
    """Parse JSON from LLM output text. Returns parsed dict or None."""
    if not isinstance(text, str):
        return None

    try:
        # Strategy 1: Find Markdown code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            parsed_json = json.loads(json_str)
        else:
            # Strategy 2: Find outermost braces/brackets
            start_pos = text.find('{') if text.find('{') != -1 else text.find('[')
            end_pos = text.rfind('}') if text.rfind('}') != -1 else text.rfind(']')
            if start_pos != -1 and end_pos != -1:
                json_str = text[start_pos : end_pos + 1]
                parsed_json = json.loads(json_str)
            else:
                return None
                
        # Key validation
        if parsed_json is not None and required_keys:
            if isinstance(parsed_json, dict):
                if not all(key in parsed_json for key in required_keys):
                    return None
            else:
                return None
                
        return parsed_json
        
    except json.JSONDecodeError:
        return None

def safe_extract_indices(text: str) -> List[int]:
    """Safely extract numeric indices from text."""
    try:
        indices = [int(num) for num in re.findall(r'\d+', str(text))]
        return sorted(list(set(indices)))
    except (ValueError, TypeError):
        return []

def validate_state_keys(state: Dict[str, Any], required_keys: List[str]) -> bool:
    """Validate that the state dict contains all required keys."""
    for key in required_keys:
        if key not in state:
            return False
    return True

def convert_history(history_dicts, max_rounds=5):
    """Convert history dicts to a flat string list."""
    if not history_dicts:
        return []
    
    converted = []
    for item in history_dicts:
        if isinstance(item, dict) and "from_who" in item:
            from_who = item.get("from_who", "")
            content = item.get("content", "")
            
            if from_who == "user":
                converted.append(f"用户：{content.strip()}")
            elif from_who == "ai" or from_who == "assistant":
                converted.append(f"助手：{content.strip()}")
            else:
                converted.append(f"{from_who}：{content.strip()}")
        
        elif isinstance(item, dict) and "role" in item:
            role = item.get("role", "")
            message = item.get("message", "")
            if role == "user":
                converted.append(f"用户：{message}")
            elif role == "assistant":
                converted.append(f"助手：{message}")
            else:
                converted.append(f"{role}：{message}")
        
        elif isinstance(item, str):
            converted.append(item)

    # Limit to max_rounds (each round = user + assistant message)
    if len(converted) > max_rounds * 2:
        converted = converted[-(int(max_rounds) * 2):]
        logging.insight(f"History truncated to last {max_rounds} rounds")
    
    return converted

def get_model_name(llm) -> str:
    """Get the model name from an LLM instance."""
    if hasattr(llm, 'model'):
        return str(llm.model)
    
    elif hasattr(llm, 'model_name'):
        return str(llm.model_name)
    
    # Fallback: return class name
    else:
        return str(type(llm).__name__)
