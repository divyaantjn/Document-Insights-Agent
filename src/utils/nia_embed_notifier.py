import os
import httpx
import logging

logger = logging.getLogger(__name__)

AGENT_EMBED_URL = "https://api-nia.dev.aifirstenterprise.ai/api/v1/attachments/agent-embed"


class AgentEmbedAPIError(Exception):
    """Raised when the agent embed API returns a non-success response."""


async def call_agent_embed_api(
    file_url: str,
    auth_token: str,
    db_name: str = "yash_vector_db",
    collection_name: str = "chat_embeddings",
    source: str = "chat",
) -> dict:
    if not auth_token:
        raise ValueError("Missing NIA_BEARER_TOKEN")

    payload = {
        "attachments": [
            {
                "source": source,
                "file_url": file_url,
                "db_name": db_name,
                "collection_name": collection_name,
            }
        ]
    }

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(AGENT_EMBED_URL, headers=headers, json=payload)
            response.raise_for_status()
        logger.info(f"Succesfull Archival Policy API Response{response.status_code}: {response.text}")
        return response.json()

    except httpx.HTTPStatusError as e:
        logger.info(f"Failed for Archival Policy API {response.status_code}: {response.text}")
        raise AgentEmbedAPIError(
            f"Agent embed API failed with status {e.response.status_code}: {e.response.text}"
        ) from e