"""
tests/middleware/test_auth_middleware.py
"""

import base64
import json
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response


@pytest.fixture(autouse=True, scope="module")
def _set_module_env(monkeypatch_module=None):
    import os
    os.environ.setdefault("KEYCLOAK_ISSUER", "https://kc.example.com/realms")
    os.environ.setdefault("KEYCLOAK_CLIENT_ID", "test-client")


def _make_jwt(payload: dict) -> str:
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    return f"{header}.{body}.fakesig"


# ===========================================================================
# _b64url_decode
# ===========================================================================

class TestB64urlDecode:

    def test_decodes_string_input(self):
        from src.middleware.auth_middleware import _b64url_decode
        original = b"hello world"
        encoded = base64.urlsafe_b64encode(original).rstrip(b"=").decode()
        assert _b64url_decode(encoded) == original

    def test_decodes_bytes_input(self):
        from src.middleware.auth_middleware import _b64url_decode
        original = b"bytes input"
        encoded = base64.urlsafe_b64encode(original).rstrip(b"=")
        assert _b64url_decode(encoded) == original

    def test_handles_padding_correctly(self):
        from src.middleware.auth_middleware import _b64url_decode
        data = b"test"
        encoded = base64.urlsafe_b64encode(data).rstrip(b"=").decode()
        assert _b64url_decode(encoded) == data

    def test_empty_string_returns_empty_bytes(self):
        from src.middleware.auth_middleware import _b64url_decode
        assert _b64url_decode("") == b""


# ===========================================================================
# get_jwks_uri_for_issuer
# ===========================================================================

class TestGetJwksUriForIssuer:

    @pytest.mark.asyncio
    async def test_returns_cached_uri_when_fresh(self):
        from src.middleware import auth_middleware
        issuer = "https://kc.example.com/realms/test"
        auth_middleware.JWKS_CACHE[issuer] = {
            "jwks_uri": "https://kc.example.com/realms/test/jwks",
            "fetched_at": time.time(),
        }
        result = await auth_middleware.get_jwks_uri_for_issuer(issuer)
        assert result == "https://kc.example.com/realms/test/jwks"
        del auth_middleware.JWKS_CACHE[issuer]

    @pytest.mark.asyncio
    async def test_fetches_when_cache_expired(self):
        import httpx
        from src.middleware import auth_middleware
        issuer = "https://kc.example.com/realms/test"
        auth_middleware.JWKS_CACHE[issuer] = {
            "jwks_uri": "https://old.example.com/jwks",
            "fetched_at": time.time() - auth_middleware.JWKS_TTL - 1,
        }
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "jwks_uri": "https://kc.example.com/realms/test/protocol/openid-connect/certs"
        }
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("src.middleware.auth_middleware.httpx.AsyncClient", return_value=mock_client):
            result = await auth_middleware.get_jwks_uri_for_issuer(issuer)
        assert "certs" in result
        del auth_middleware.JWKS_CACHE[issuer]

    @pytest.mark.asyncio
    async def test_raises_on_non_200_discovery_response(self):
        import httpx
        from src.middleware import auth_middleware
        issuer = "https://kc.example.com/realms/bad"
        auth_middleware.JWKS_CACHE.pop(issuer, None)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        with patch("src.middleware.auth_middleware.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                await auth_middleware.get_jwks_uri_for_issuer(issuer)


# ===========================================================================
# Helpers
# ===========================================================================

def _build_app_with_middleware():
    from src.middleware.auth_middleware import KeycloakAuthMiddleware
    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/protected")
    async def protected():
        return {"status": "protected data"}

    app.add_middleware(KeycloakAuthMiddleware)
    return app


# ===========================================================================
# KeycloakAuthMiddleware — Public Paths
# ===========================================================================

class TestKeycloakAuthMiddlewarePublicPaths:

    @pytest.fixture
    def client(self):
        return TestClient(_build_app_with_middleware(), raise_server_exceptions=False)

    @pytest.mark.parametrize(
        "path",
        ["/docs", "/docs/", "/redoc", "/openapi.json", "/health"],
    )
    def test_public_paths_return_200_without_auth(self, client, path):
        resp = client.get(path)
        assert resp.status_code < 400


# ===========================================================================
# KeycloakAuthMiddleware — Missing / Invalid Token
# ===========================================================================

class TestKeycloakAuthMiddlewareMissingInvalidToken:

    @pytest.fixture
    def client(self):
        return TestClient(_build_app_with_middleware(), raise_server_exceptions=False)

    def test_missing_auth_header_returns_401(self, client):
        resp = client.get("/protected")
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    def test_wrong_scheme_returns_401(self, client):
        resp = client.get("/protected", headers={"Authorization": "Basic dXNlcjpwYXNz"})
        assert resp.status_code == 401

    def test_token_with_wrong_part_count_returns_400(self, client):
        resp = client.get("/protected", headers={"Authorization": "Bearer onlytwoparts.here"})
        assert resp.status_code == 400

    def test_invalid_base64_payload_returns_400(self, client):
        resp = client.get(
            "/protected",
            headers={"Authorization": "Bearer header.!!!invalid!!!.sig"},
        )
        assert resp.status_code == 400


# ===========================================================================
# KeycloakAuthMiddleware — Token Validation
# ===========================================================================

class TestKeycloakAuthMiddlewareTokenValidation:

    @pytest.fixture
    def client(self):
        return TestClient(_build_app_with_middleware(), raise_server_exceptions=False)

    def test_untrusted_issuer_returns_403(self, client):
        payload = {
            "iss": "https://evil.example.com/realms/fake",
            "sub": "user-1",
        }
        token = _make_jwt(payload)
        resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 403

    def test_expired_token_returns_401(self):
        import jwt as pyjwt
        payload = {"iss": "https://kc.example.com/realms/test", "sub": "user-1"}
        token = _make_jwt(payload)

        p1 = patch(
            "src.middleware.auth_middleware.KEYCLOAK_ISSUER",
            new="https://kc.example.com/realms",
        )
        p2 = patch(
            "src.middleware.auth_middleware.get_jwks_uri_for_issuer",
            new=AsyncMock(return_value="https://kc.example.com/jwks"),
        )
        p3 = patch("src.middleware.auth_middleware.jwt.PyJWKClient")
        p4 = patch(
            "src.middleware.auth_middleware.jwt.decode",
            side_effect=pyjwt.ExpiredSignatureError("Token expired"),
        )

        p1.start()
        p2.start()
        mock_jwk_cls = p3.start()
        p4.start()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value.key = "fake-key"
        try:
            client = TestClient(_build_app_with_middleware(), raise_server_exceptions=False)
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 401
        finally:
            p1.stop(); p2.stop(); p3.stop(); p4.stop()

    def test_invalid_token_returns_401(self):
        import jwt as pyjwt
        payload = {"iss": "https://kc.example.com/realms/test", "sub": "user-1"}
        token = _make_jwt(payload)

        p1 = patch(
            "src.middleware.auth_middleware.KEYCLOAK_ISSUER",
            new="https://kc.example.com/realms",
        )
        p2 = patch(
            "src.middleware.auth_middleware.get_jwks_uri_for_issuer",
            new=AsyncMock(return_value="https://kc.example.com/jwks"),
        )
        p3 = patch("src.middleware.auth_middleware.jwt.PyJWKClient")
        p4 = patch(
            "src.middleware.auth_middleware.jwt.decode",
            side_effect=pyjwt.InvalidTokenError("Bad token"),
        )

        p1.start()
        p2.start()
        mock_jwk_cls = p3.start()
        p4.start()
        mock_jwk_cls.return_value.get_signing_key_from_jwt.return_value.key = "fake-key"
        try:
            client = TestClient(_build_app_with_middleware(), raise_server_exceptions=False)
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 401
        finally:
            p1.stop(); p2.stop(); p3.stop(); p4.stop()


# ===========================================================================
# KeycloakAuthMiddleware — Permissions
# ===========================================================================

class TestKeycloakAuthMiddlewarePermissions:

    def _token_with_payload(self, payload):
        return _make_jwt(payload)

    def test_missing_client_in_resource_access_returns_400(self):
        payload = {
            "iss": "https://kc.example.com/realms/test",
            "sub": "user-1",
            "resource_access": {},
        }
        token = self._token_with_payload(payload)

        p1 = patch(
            "src.middleware.auth_middleware.KEYCLOAK_ISSUER",
            new="https://kc.example.com/realms",
        )
        p2 = patch(
            "src.middleware.auth_middleware.get_jwks_uri_for_issuer",
            new=AsyncMock(return_value="https://kc.example.com/jwks"),
        )
        p3 = patch("src.middleware.auth_middleware.jwt.PyJWKClient")
        p4 = patch("src.middleware.auth_middleware.jwt.decode", return_value=payload)

        p1.start()
        p2.start()
        mock_cls = p3.start()
        p4.start()
        mock_cls.return_value.get_signing_key_from_jwt.return_value.key = "k"
        try:
            client = TestClient(_build_app_with_middleware(), raise_server_exceptions=False)
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 400
        finally:
            p1.stop(); p2.stop(); p3.stop(); p4.stop()

    def test_missing_required_role_returns_400(self):
        payload = {
            "iss": "https://kc.example.com/realms/test",
            "sub": "user-1",
            "azp": "test-client",
            "resource_access": {"test-client": {"roles": ["some-other-role"]}},
        }
        token = self._token_with_payload(payload)

        p1 = patch(
            "src.middleware.auth_middleware.KEYCLOAK_ISSUER",
            new="https://kc.example.com/realms",
        )
        p2 = patch(
            "src.middleware.auth_middleware.get_jwks_uri_for_issuer",
            new=AsyncMock(return_value="https://kc.example.com/jwks"),
        )
        p3 = patch("src.middleware.auth_middleware.jwt.PyJWKClient")
        p4 = patch("src.middleware.auth_middleware.jwt.decode", return_value=payload)

        p1.start()
        p2.start()
        mock_cls = p3.start()
        p4.start()
        mock_cls.return_value.get_signing_key_from_jwt.return_value.key = "k"
        try:
            client = TestClient(_build_app_with_middleware(), raise_server_exceptions=False)
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 400
        finally:
            p1.stop(); p2.stop(); p3.stop(); p4.stop()

    def test_valid_token_with_correct_role_passes(self):
        payload = {
            "iss": "https://kc.example.com/realms/test",
            "sub": "user-1",
            "azp": "test-client",
            "preferred_username": "alice",
            "resource_access": {"test-client": {"roles": ["test-client_client"]}},
        }
        token = self._token_with_payload(payload)

        p1 = patch(
            "src.middleware.auth_middleware.KEYCLOAK_ISSUER",
            new="https://kc.example.com/realms",
        )
        p2 = patch(
            "src.middleware.auth_middleware.get_jwks_uri_for_issuer",
            new=AsyncMock(return_value="https://kc.example.com/jwks"),
        )
        p3 = patch("src.middleware.auth_middleware.jwt.PyJWKClient")
        p4 = patch("src.middleware.auth_middleware.jwt.decode", return_value=payload)

        p1.start()
        p2.start()
        mock_cls = p3.start()
        p4.start()
        mock_cls.return_value.get_signing_key_from_jwt.return_value.key = "k"
        try:
            client = TestClient(_build_app_with_middleware(), raise_server_exceptions=False)
            resp = client.get("/protected", headers={"Authorization": f"Bearer {token}"})
            assert resp.status_code == 200
        finally:
            p1.stop(); p2.stop(); p3.stop(); p4.stop()

    def test_token_separator_stripped(self):
        payload = {
            "iss": "https://kc.example.com/realms/test",
            "sub": "user-1",
            "azp": "test-client",
            "preferred_username": "bob",
            "resource_access": {"test-client": {"roles": ["test-client_client"]}},
        }
        real_token = _make_jwt(payload)
        composite_token = f"{real_token}$YashUnified2025$extra_payload"

        p1 = patch(
            "src.middleware.auth_middleware.KEYCLOAK_ISSUER",
            new="https://kc.example.com/realms",
        )
        p2 = patch(
            "src.middleware.auth_middleware.get_jwks_uri_for_issuer",
            new=AsyncMock(return_value="https://kc.example.com/jwks"),
        )
        p3 = patch("src.middleware.auth_middleware.jwt.PyJWKClient")
        p4 = patch("src.middleware.auth_middleware.jwt.decode", return_value=payload)

        p1.start()
        p2.start()
        mock_cls = p3.start()
        p4.start()
        mock_cls.return_value.get_signing_key_from_jwt.return_value.key = "k"
        try:
            client = TestClient(_build_app_with_middleware(), raise_server_exceptions=False)
            resp = client.get("/protected", headers={"Authorization": f"Bearer {composite_token}"})
            assert resp.status_code == 200
        finally:
            p1.stop(); p2.stop(); p3.stop(); p4.stop()