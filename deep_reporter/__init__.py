# DeepDiver_pipeline/__init__.py
import logging
import sys
import os

# Ensure logging.insight is available
if not hasattr(logging, 'insight'):
    # Try importing logger_config from parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    try:
        from logger_config import configure_logger
        configure_logger()
    except ImportError:
        # Fallback insight method if import fails
        def insight(msg, messageId=None, level='info', **kwargs):
            """Fallback insight logging method."""
            log_func = getattr(logging, level.lower(), logging.info)
            extra_info = f"[messageId: {messageId}]" if messageId else ""
            log_func(f"{extra_info} {msg}")
        
        logging.insight = insight
        logging.warning("Using fallback logging.insight implementation")
