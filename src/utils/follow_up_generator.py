"""
Generic Follow-Up Question Generator Utility

This utility automatically generates contextual follow-up questions based on 
the user's query, agent's response, and STRICT agent capabilities.
"""

import logging
from typing import Dict, List
from litellm import acompletion

logger = logging.getLogger(__name__)

async def generate_follow_up_questions(
    user_query: str,
    response: str,
    agent_capabilities: List[str],
    context: Dict,
    llm_config: dict,
    num_questions: int = 3
) -> str:
    """
    Generate contextual follow-up questions based on STRICT agent capabilities.
    
    Args:
        user_query: The user's original query/request
        response: The agent's response text
        agent_capabilities: List of EXACT capabilities the agent has
        context: Dictionary with context about the operation performed
        llm_config: LLM configuration for acompletion
        num_questions: Number of questions to generate (default: 3)
    
    Returns:
        Comma-separated string of follow-up questions
    """
    try:
        # Build context string from dictionary
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items() if v])
        
        # Build capabilities string
        capabilities_str = "\n".join([f"- {cap}" for cap in agent_capabilities])
        
        prompt = f"""You are generating follow-up questions for an agent with STRICT, LIMITED capabilities.

AGENT'S EXACT CAPABILITIES (DO NOT SUGGEST ANYTHING OUTSIDE THIS LIST):
{capabilities_str}

USER'S QUERY:
{user_query}

AGENT RESPONSE SUMMARY:
{response}

OPERATION CONTEXT:
{context_str}

Generate exactly {num_questions} follow-up questions that:
1. Are about the CONTENT and INFORMATION present in the response — what was said, what it means, what it implies
2. Are directly grounded in what was returned — do NOT invent facts, assume external knowledge, or speculate
3. Explore a specific detail, clarification, or sub-topic that is visibly present in the response
4. Are ONLY within the agent's capabilities listed above
5. Are phrased as DIRECT COMMANDS starting with action verbs (e.g., "Summarize", "List", "Explain", "Describe", "What is")
6. Are DIFFERENT from what was just done

ABSOLUTELY FORBIDDEN — NEVER ask about any of these:
- Document IDs, chunk IDs, vector IDs, or any alphanumeric identifier strings
- Similarity scores, relevance scores, confidence scores, or any numeric metadata
- Retrieval methods, search methods, or how data was fetched internally
- File types, page numbers, or internal metadata fields
- Milvus, vector databases, embeddings, or any backend infrastructure
- Any technical system internals — treat the response as plain content only

CONTENT RULES:
- Focus ONLY on the subject matter, facts, and information in the response text
- If the response is about a topic (e.g. HR policy, a product, a report), ask about THAT topic
- Questions must make sense to a non-technical end user reading the answer
- DO NOT suggest capabilities NOT listed above
- DO NOT repeat the original operation

FORMATTING RULES:
- Separate each command with a comma
- Each individual command must NOT contain any commas within it
- Rephrase any command that would naturally require a comma (use "and" or restructure the sentence instead)
- No numbering, no extra text

Return ONLY {num_questions} commands separated by commas.
Format: command1, command2, command3"""

        messages = [{"role": "user", "content": prompt}]
        
        response_obj = await acompletion(
            messages=messages,
            **llm_config
        )
        
        questions = response_obj.choices[0].message.content.strip()
        
        print("Generated follow-up questions: ", questions)
        return questions
        
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return ""
