"""
secrets_loader.py

Fetches pgvector/redis connection secrets from AWS Secrets Manager and
injects them into os.environ so the rest of the code keeps using os.getenv().

Set PGVECTOR_SECRET to the secret ARN or name in the Lambda environment.
If the variable is absent the loader is a no-op (local dev uses env vars directly).

Expected secret JSON keys:
  PG_HOST, PG_PORT, PG_USER, PG_PASSWORD, PG_DATABASE
  REDIS_HOST_PGVECTOR, REDIS_PORT_PGVECTOR,
  REDIS_USERNAME_PGVECTOR, REDIS_PASSWORD_PGVECTOR
"""
import json
import logging
import os

logger = logging.getLogger(__name__)

_loaded = False


def load_pgvector_secrets() -> None:
    """Fetch secret once from AWS Secrets Manager and inject into os.environ.

    Subsequent calls are no-ops. Raises on fetch/parse failure so a
    misconfigured Lambda fails fast at cold-start rather than silently
    connecting to wrong endpoints.
    """
    global _loaded
    if _loaded:
        return
    _loaded = True

    secret_name = os.getenv("PGVECTOR_SECRET")
    if not secret_name:
        return

    try:
        import boto3

        region = os.getenv("AWS_REGION_LAMBDA") or os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secret: dict = json.loads(response["SecretString"])
        for key, value in secret.items():
            os.environ[key] = str(value)
        logger.info("Loaded %d pgvector secrets from Secrets Manager secret '%s'", len(secret), secret_name)
    except Exception as exc:
        logger.error("Failed to load pgvector secrets from '%s': %s", secret_name, exc)
        raise
