from typing import Optional, Tuple
import re
import logging
from .kafka import create_reasoning_logger

logger = logging.getLogger(__name__)

def extract_and_log_reasoning(
    response_text: str,
    auth_token: Optional[str] = None
) -> Tuple[str, str]:
    """
    Extract reasoning section from LLM response and return cleaned response
    
    Args:
        response_text: Full LLM response text
        
    Returns:
        tuple: (cleaned_response, reasoning_text)
    """
    reasoning_pattern = r'\n\s*REASONING:\s*\n(.*?)$'
    
    match = re.search(reasoning_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if match:
        reasoning_text = match.group(1).strip()

        cleaned_response = re.sub(reasoning_pattern, '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
        
        print("=== LLM REASONING ===")
        print(reasoning_text)
        print("=" * 50)

        if reasoning_text:
            try:
                reasoning_logger = create_reasoning_logger()
                reasoning_logger.log_reasoning(reasoning_text, auth_token)
            except Exception as e:
                logger.debug(f"Failed to log reasoning to Kafka: {e}")

        print(auth_token[:100])
        
        return cleaned_response, reasoning_text
    else:
        print("No REASONING section found in LLM response")
        return response_text, ""


REASONING_SECTION_PROMPT = """

IMPORTANT - Include Reasoning Section:
At the end of your response, add a section titled "REASONING:" (in all caps) that explains:
- Your thought process in formulating this answer
- Which parts of the context were most relevant
- Any assumptions or interpretations you made
- Do not include any personal, sensitive, or confidential information in the reasoning section

Format:
REASONING:
[Your detailed reasoning here in 3-4 sentences]
"""