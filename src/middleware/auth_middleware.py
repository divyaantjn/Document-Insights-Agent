import httpx
import time
import jwt
from jwt import ExpiredSignatureError, InvalidTokenError
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.security.utils import get_authorization_scheme_param
import logging, time
import os
from dotenv import load_dotenv
load_dotenv()
KEYCLOAK_ISSUER = os.getenv("KEYCLOAK_ISSUER")
KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID")

logger = logging.getLogger("main")
# In-memory cache for JWKS per issuer
JWKS_CACHE = {}  # Key: issuer, Value: {"jwks_uri": ..., "fetched_at": timestamp}
JWKS_TTL = 3600  # 1 hour
class KeycloakAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        path = request.url.path
        # Public paths allowed without auth
        if (
            path == "/"
            or path == "/openapi.json"
            or path == "/docs"
            or path == "/docs/"
            or path == "/redoc"
            or path == "/health"
            
        ):
            response = await call_next(request)
            duration = round(time.time() - start_time, 3)
            logger.info("HTTP request",extra={
                "path": path,
                "method": request.method,
                "status_code": response.status_code,
                "response_time": duration,
                "realm": "public route no realm",
                "client_id": "public route no clientId",
                "username": "public route no username",
                "user_id": "public route no user_id",
                "ip": request.client.host
            })
            return response
 
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return JSONResponse(status_code=401, content={"detail": "Missing Authorization Header"})
 
        scheme, token = get_authorization_scheme_param(auth_header)
        token = token.split("$YashUnified2025$")[0]
        if scheme.lower() != "bearer" or not token:
            return JSONResponse(status_code=401, content={"detail": "Invalid Authorization Header"})
 
        try:
            # # Decode unverified token to extract issuer
            # unverified_payload = jwt.decode(token, options={"verify_signature": False})
            parts = token.split(".")
            if len(parts) != 3:
                return JSONResponse(status_code=400, content= {"details":"Token must have 3 parts"})
    
            # Decode payload (no library call that disables verification)
            try:
                payload_bytes = _b64url_decode(parts[1])
                unverified_payload = json.loads(payload_bytes)
            except ValueError as e:
                return JSONResponse(status_code=400, content= {"details":f"{str(e)}"})
            issuer = unverified_payload.get("iss")
            client_id = KEYCLOAK_CLIENT_ID
            if not issuer or not issuer.startswith(KEYCLOAK_ISSUER):
                return JSONResponse(status_code=403, content={"detail": "Untrusted token issuer"})
 
            # Get JWKS URI for issuer (cached or fetched)
            jwks_uri = await get_jwks_uri_for_issuer(issuer)
 
            # Use PyJWKClient with JWKS URI to get signing key
            jwk_client = jwt.PyJWKClient(jwks_uri)
            signing_key = jwk_client.get_signing_key_from_jwt(token).key
 
            # Fully decode and validate token
            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                audience="account",  # Optionally: dynamically read expected audience
                issuer=issuer
            )
            request.state.user = payload
            required_role = f"{client_id}_client"
            resource_access = payload.get("resource_access", {})
           
            # Check if the client exists in resource_access
            if client_id not in resource_access:
                raise PermissionError(f"Client '{client_id}' not found in resource_access.")
           
            # Check if the required role is present
            client_roles = resource_access[client_id].get("roles", [])
            if required_role not in client_roles:
                raise PermissionError(f"Missing required role: {required_role}")
        except ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"detail": "Token has expired"})
        except InvalidTokenError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        except Exception as e:
            return JSONResponse(status_code=400, content={"detail": f"Token processing failed: {str(e)}"})
       
        response = await call_next(request)
        duration = round(time.time() - start_time, 3)
         # Extract useful fields
        realm = payload.get("iss", "").split("/")[-1] if "iss" in payload else "unknown"
        client_id = payload.get("azp", "unknown-client")
        user_id = payload.get("sub", "unknown-user")
        username = payload.get("preferred_username", "anonymous")
        logger.info("HTTP request",
            extra={
                "path": path,
                "method": request.method,
                "status_code": response.status_code,
                "response_time": duration,
                "realm": realm,
                "client_id": client_id,
                "username": username,
                "user_id": user_id,
                "ip": request.client.host
            })
       
        return response
 
# JWKS discovery and caching
async def get_jwks_uri_for_issuer(issuer: str) -> str:
    now = time.time()
    cache = JWKS_CACHE.get(issuer)
 
    if cache and now - cache["fetched_at"] < JWKS_TTL:
        return cache["jwks_uri"]
 
    async with httpx.AsyncClient() as client:
        discovery_url = f"{issuer}/.well-known/openid-configuration"
        discovery_res = await client.get(discovery_url)
        if discovery_res.status_code != 200:
           raise httpx.HTTPStatusError(
                f"Failed to fetch OpenID config for issuer: {issuer}",
                request=None,
                response=discovery_res,
            )
 
        jwks_uri = discovery_res.json()["jwks_uri"]
 
    JWKS_CACHE[issuer] = {"jwks_uri": jwks_uri, "fetched_at": now}
    return jwks_uri

import base64
import json
def _b64url_decode(input_str: str) -> bytes:
    """
    Decode a base64url-encoded string (no padding) to bytes.
    """
    s = input_str.encode() if isinstance(input_str, str) else input_str
    # Add padding if necessary
    rem = len(s) % 4
    if rem:
        s += b"=" * (4 - rem)
    return base64.urlsafe_b64decode(s)