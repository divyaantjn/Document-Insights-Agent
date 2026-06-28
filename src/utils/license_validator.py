"""
License validator for backend.

Startup flow:
  Redis cache hit -> use cached APP_LICENSE
  Redis cache miss -> fetch from AWS Secrets Manager -> cache in Redis (TTL 1hr)
  -> RSA-OAEP decrypt AES key -> AES-256-GCM decrypt -> RSA-PSS verify
  -> validate expiry + agent name -> inject PyArmor license
"""
import os
import json
import base64
import logging
import glob
import redis
from datetime import datetime, timezone, timedelta

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.backends import default_backend

from src.utils.secret_manager import get_secret

logger = logging.getLogger(__name__)

SECRET_NAME = os.getenv("LICENSE_SECRET_NAME", "")
REDIS_LICENSE_KEY = "license:app_license"
REDIS_TTL = int(os.getenv("LICENSE_CACHE_TTL", 60))  # 1 hour default

_license_metadata: dict | None = None
_license_expires_at: datetime | None = None

_redis_client = None


def _get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            db=int(os.getenv("REDIS_DB", 0)),
            ssl=os.getenv("REDIS_SSL", "false").strip().lower() == "true",
            decode_responses=True,
            socket_connect_timeout=2,
        )
    return _redis_client


def _fetch_app_license() -> str:
    """Fetch APP_LICENSE from Redis cache, fallback to Secrets Manager."""
    try:
        cached = _get_redis_client().get(REDIS_LICENSE_KEY)
        if cached:
            logger.info("License fetched from Redis cache", extra={"source": "redis", "ttl": REDIS_TTL})
            return cached
    except Exception:
        logger.warning("Redis unavailable, falling back to Secrets Manager")

    try:
        logger.info("Fetching license from AWS Secrets Manager", extra={"secret_name": SECRET_NAME})
        secret = get_secret(SECRET_NAME)
        secret_data = json.loads(secret)
        app_license = secret_data.get("APP_LICENSE", secret_data.get("app_license", ""))

        if not app_license:
            logger.error(
                "APP_LICENSE not found in Secrets Manager secret",
                extra={"secret_name": SECRET_NAME, "keys_found": list(secret_data.keys())},
            )
            raise RuntimeError(f"APP_LICENSE not found in Secrets Manager secret '{SECRET_NAME}'")

        try:
            _get_redis_client().set(REDIS_LICENSE_KEY, app_license, ex=REDIS_TTL)
            logger.info("License cached in Redis", extra={"ttl_seconds": REDIS_TTL, "redis_key": REDIS_LICENSE_KEY})
        except Exception:
            logger.warning("Failed to cache license in Redis")

        return app_license

    except Exception as e:
        logger.error(
            "Failed to fetch license from Secrets Manager",
            extra={"secret_name": SECRET_NAME, "error": str(e)},
            exc_info=True,
        )
        raise RuntimeError(f"Failed to fetch license from Secrets Manager '{SECRET_NAME}': {e}")


def _fetch_keys() -> tuple[str, str]:
    """Fetch ORG_PRIVATE_KEY and YASH_PUBLIC_KEY from env or Secrets Manager."""
    org_private_key_pem = os.getenv("ORG_PRIVATE_KEY", "").strip().replace("\\n", "\n")
    yash_public_key_pem = os.getenv("YASH_PUBLIC_KEY", "").strip().replace("\\n", "\n")

    if not org_private_key_pem or not yash_public_key_pem:
        logger.info("ORG_PRIVATE_KEY/YASH_PUBLIC_KEY not in env, fetching from Secrets Manager",
                    extra={"secret_name": SECRET_NAME})
        try:
            secret = get_secret(SECRET_NAME)
            secret_data = json.loads(secret)
            org_private_key_pem = secret_data.get("ORG_PRIVATE_KEY", "").strip().replace("\\n", "\n")
            yash_public_key_pem = secret_data.get("YASH_PUBLIC_KEY", "").strip().replace("\\n", "\n")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch keys from Secrets Manager: {e}")

    if not org_private_key_pem or not yash_public_key_pem:
        raise RuntimeError("ORG_PRIVATE_KEY and YASH_PUBLIC_KEY not found in env or Secrets Manager")

    return org_private_key_pem, yash_public_key_pem


def _validate_and_store(app_license: str) -> None:
    global _license_metadata, _license_expires_at

    org_private_key_pem, yash_public_key_pem = _fetch_keys()

    logger.info("📦 License middleware: decoding license blob...")
    try:
        license_blob = base64.b64decode(app_license)
        license_package = json.loads(license_blob)
    except Exception as e:
        raise RuntimeError(f"Failed to decode APP_LICENSE: {e}")

    encrypted_aes_key = base64.b64decode(license_package["encrypted_key"])
    iv = base64.b64decode(license_package["iv"])
    encrypted_data = base64.b64decode(license_package["encrypted_data"])
    tag = base64.b64decode(license_package["tag"])
    signature = base64.b64decode(license_package["signature"])

    logger.info("🔑 License middleware: decrypting AES key...")
    try:
        org_private_key = serialization.load_pem_private_key(
            org_private_key_pem.encode(), password=None, backend=default_backend()
        )
        aes_key = org_private_key.decrypt(
            encrypted_aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt AES key: {e}")

    logger.info("🔓 License middleware: decrypting license data...")
    try:
        aesgcm = AESGCM(aes_key)
        decrypted_bytes = aesgcm.decrypt(iv, encrypted_data + tag, None)
        license_data = json.loads(decrypted_bytes.decode())
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt license data: {e}")

    logger.info("✍️  License middleware: verifying signature...")
    try:
        yash_public_key = serialization.load_pem_public_key(
            yash_public_key_pem.encode(), backend=default_backend()
        )
        yash_public_key.verify(
            signature,
            decrypted_bytes,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
    except Exception as e:
        raise RuntimeError(f"License signature verification failed: {e}")

    logger.info("📅 License middleware: validating expiry...")
    metadata = license_data.get("metadata", license_data)

    if "expires_at" in metadata:
        expires_at = datetime.fromisoformat(metadata["expires_at"]).replace(tzinfo=timezone.utc)
    elif "tenure_days" in metadata:
        issued_at = datetime.fromisoformat(metadata["issued_at"]).replace(tzinfo=timezone.utc) if "issued_at" in metadata else datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(days=metadata["tenure_days"])
    elif "tenure_minutes" in metadata:
        issued_at = datetime.fromisoformat(metadata["issued_at"]).replace(tzinfo=timezone.utc) if "issued_at" in metadata else datetime.now(timezone.utc)
        expires_at = issued_at + timedelta(minutes=metadata["tenure_minutes"])
    else:
        raise RuntimeError("License has no expiry information (expires_at, tenure_days, or tenure_minutes)")

    if datetime.now(timezone.utc) > expires_at:
        raise RuntimeError(f"License expired at {expires_at.isoformat()}")

    agent_name = os.getenv("AGENT_NAME", "").strip()
    licensed_agents = metadata.get("agents", [])
    if agent_name not in licensed_agents:
        raise RuntimeError(f"Agent '{agent_name}' is not licensed. Licensed agents: {licensed_agents}")

    pyarmor_license_hex = license_data.get("pyarmor_license", "")
    if pyarmor_license_hex:
        _inject_pyarmor_license(bytes.fromhex(pyarmor_license_hex))

    _license_metadata = metadata
    _license_expires_at = expires_at
    logger.info(f"✅ License validated. Agent: {agent_name}, Expires: {expires_at.isoformat()}")


def initialize_license() -> None:
    global _license_metadata
    if os.getenv("LICENSE_ENFORCE", "true").strip().lower() != "true":
        logger.info("License enforcement disabled (LICENSE_ENFORCE=false), skipping initialization")
        return
    if _license_metadata is not None:
        return
    logger.info("🔐 License middleware: starting validation...")
    app_license = _fetch_app_license()
    _validate_and_store(app_license)


def refresh_license() -> None:
    """Force re-fetch from Secrets Manager (bypasses Redis cache) and re-validate."""
    global _license_metadata, _license_expires_at

    logger.info("Refreshing license from Secrets Manager", extra={"secret_name": SECRET_NAME})

    try:
        secret = get_secret(SECRET_NAME)
        secret_data = json.loads(secret)
        app_license = secret_data.get("APP_LICENSE", secret_data.get("app_license", ""))

        if not app_license:
            logger.error(
                "APP_LICENSE not found in Secrets Manager during refresh",
                extra={"secret_name": SECRET_NAME, "keys_found": list(secret_data.keys())},
            )
            raise RuntimeError(f"APP_LICENSE not found in Secrets Manager secret '{SECRET_NAME}'")

        try:
            _get_redis_client().set(REDIS_LICENSE_KEY, app_license, ex=REDIS_TTL)
            logger.info("License refreshed and cached in Redis",
                        extra={"ttl_seconds": REDIS_TTL, "redis_key": REDIS_LICENSE_KEY})
        except Exception:
            logger.warning("Failed to cache refreshed license in Redis")
    except Exception as e:
        logger.error(
            "Failed to refresh license from Secrets Manager",
            extra={"secret_name": SECRET_NAME, "error": str(e)},
            exc_info=True,
        )
        raise RuntimeError(f"Failed to refresh license from Secrets Manager '{SECRET_NAME}': {e}")

    _license_metadata = None
    _license_expires_at = None
    _validate_and_store(app_license)


def _inject_pyarmor_license(license_bytes: bytes) -> None:
    patterns = [
        "/app/pyarmor_runtime_*/license.lic",
        "./pyarmor_runtime_*/license.lic",
    ]
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, "wb") as f:
                    f.write(license_bytes)
                logger.info(f"✅ PyArmor license injected: {path}")
            except Exception as e:
                logger.warning(f"Failed to inject PyArmor license at {path}: {e}")


def get_license_metadata() -> dict | None:
    return _license_metadata


def get_license_expires_at() -> datetime | None:
    return _license_expires_at


try:
    initialize_license()
except Exception as e:
    logger.error(f"License initialization failed: {e}")
