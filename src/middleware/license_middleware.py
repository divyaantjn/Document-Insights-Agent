"""
License middleware for per-request expiry enforcement.

Skips public paths. On license expiry, attempts refresh from Secrets Manager
before returning 503.
"""
import os
import logging
from datetime import datetime, timezone

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.utils.license_validator import get_license_metadata, get_license_expires_at, refresh_license

logger = logging.getLogger(__name__)


def _is_license_enforced() -> bool:
    """Return True if license enforcement is enabled (default: True)."""
    return os.getenv("LICENSE_ENFORCE", "true").strip().lower() == "true"

PUBLIC_PATHS = {
    "/health",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/.well-known/agent.json",
    "/.well-known/jwks.json",
}

_EXPIRED_RESPONSE = {
    "http_status": 503,
    "details": {
        "mode": "LICENSE_EXPIRED",
        "title": "License Period Expired",
        "message": "Your license has ended. Please upgrade your plan or contact support to continue accessing the workspace.",
    },
}


class LicenseMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        if not _is_license_enforced():
            logger.debug("License enforcement disabled (LICENSE_ENFORCE=false), skipping validation")
            return await call_next(request)

        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        metadata = get_license_metadata()
        if metadata is None:
            try:
                refresh_license()
                metadata = get_license_metadata()
            except Exception as e:
                logger.error(f"License initialization failed: {e}")
            if metadata is None:
                return JSONResponse(status_code=503, content=_EXPIRED_RESPONSE)

        expires_at = get_license_expires_at()
        if expires_at and datetime.now(timezone.utc) > expires_at:
            logger.warning(f"License expired at {expires_at.isoformat()}, attempting refresh...")
            try:
                refresh_license()
                logger.info("✅ License refreshed successfully")
            except Exception as e:
                logger.error(f"License refresh failed: {e}")
                return JSONResponse(status_code=503, content=_EXPIRED_RESPONSE)

        return await call_next(request)
