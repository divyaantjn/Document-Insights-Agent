# utils/obs.py

import os
import logging
from typing import Any, Dict, Optional
from .kafka import kafka_logger
from .kafka_base import extract_user_context

logger = logging.getLogger(__name__)


class LLMUsageTracker:
    """
    Unified usage tracker for LiteLLM responses.
    """
    def __init__(self, auth_token: Optional[str] = None):
        self.auth_token = auth_token or ""
        self.agent_name = os.getenv("AGENT_NAME", "DOCUMENT_INSIGHTS")
        self.server_name = os.getenv("SERVER_NAME", "DOCUMENT_INSIGHTS_BACKEND")

    def track_response(self, response: Any, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Track LiteLLM response and send to Kafka."""
        try:
            usage_info = getattr(response, "usage", None)
            if not usage_info:
                logger.debug("No token usage found")
                return {"status": "error", "message": "No usage info"}
            
            if not isinstance(usage_info, dict):
                if hasattr(usage_info, "__dict__"):
                    usage_info = usage_info.__dict__
                else:
                    return {"status": "error", "message": "Cannot parse usage"}
            
            user_context = extract_user_context(self.auth_token)
            
            prompt_tokens = usage_info.get("prompt_tokens", 0)
            completion_tokens = usage_info.get("completion_tokens", 0)
            total_tokens = usage_info.get("total_tokens", 0)
            
            # Extract model name without provider prefix
            if model_name and "/" in model_name:
                clean_model_name = model_name.split("/", 1)[1]
            else:
                clean_model_name = model_name or "UNKNOWN_MODEL"
            
            payload = {
                "encrypted_payload": user_context["encrypted_payload"],
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thoughts_token_count": 0,
                "total_tokens": total_tokens,
                "model_name": clean_model_name,
                "agent_name_constant": self.agent_name,
                "server_name": self.server_name,
            }
            
            if total_tokens > 0:
                kafka_logger.log(payload)
            
            return {"status": "success", "total_tokens": total_tokens}
        except Exception as e:
            logger.error(f"Error in track_response: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


def observe_token_usage(result: Any, auth_token: Optional[str], model_name: Optional[str] = None) -> None:
    """Observe token usage from result object."""
    if auth_token:
        tracker = LLMUsageTracker(auth_token=auth_token)
        tracker.track_response(result, model_name=model_name)