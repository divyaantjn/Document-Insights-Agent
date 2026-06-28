import boto3
import logging
import os

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "us-east-1"))
    return _client


def get_secret(secret_name: str) -> str:
    response = _get_client().get_secret_value(SecretId=secret_name)
    return response.get("SecretString") or response.get("SecretBinary").decode()
