"""Tests for src/middleware/license_middleware.py"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient
from starlette.routing import Route

import src.middleware.license_middleware as lm
from src.middleware.license_middleware import (
    LicenseMiddleware,
    _is_license_enforced,
    PUBLIC_PATHS,
)

_PATCH = "src.middleware.license_middleware"


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_app():
    async def homepage(request):
        return PlainTextResponse("ok")

    async def health(request):
        return PlainTextResponse("healthy")

    app = Starlette(routes=[
        Route("/", homepage, methods=["GET", "POST"]),
        Route("/health", health, methods=["GET"]),
        Route("/docs", health, methods=["GET"]),
    ])
    app.add_middleware(LicenseMiddleware)
    return app


# ── _is_license_enforced ──────────────────────────────────────────────────────

class TestIsLicenseEnforced:
    def test_true_by_default(self, monkeypatch):
        monkeypatch.delenv("LICENSE_ENFORCE", raising=False)
        assert _is_license_enforced() is True

    def test_true_when_set_to_true(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        assert _is_license_enforced() is True

    def test_false_when_set_to_false(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "false")
        assert _is_license_enforced() is False

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "FALSE")
        assert _is_license_enforced() is False


# ── LicenseMiddleware dispatch ────────────────────────────────────────────────

class TestLicenseMiddlewareDispatch:

    # enforcement disabled
    def test_skips_validation_when_enforcement_disabled(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "false")
        with patch(f"{_PATCH}.get_license_metadata") as mock_meta:
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        mock_meta.assert_not_called()
        assert response.status_code == 200

    # public paths
    def test_public_path_health_bypasses_license(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        with patch(f"{_PATCH}.get_license_metadata") as mock_meta:
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/health")
        mock_meta.assert_not_called()
        assert response.status_code == 200

    def test_public_path_docs_bypasses_license(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        with patch(f"{_PATCH}.get_license_metadata") as mock_meta:
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/docs")
        mock_meta.assert_not_called()
        assert response.status_code == 200

    # metadata None -> refresh fails
    def test_returns_503_when_metadata_none_and_refresh_fails(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        with patch(f"{_PATCH}.get_license_metadata", return_value=None), \
             patch(f"{_PATCH}.refresh_license", side_effect=Exception("fail")):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 503

    # metadata None -> refresh succeeds but still None
    def test_returns_503_when_metadata_still_none_after_refresh(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        with patch(f"{_PATCH}.get_license_metadata", return_value=None), \
             patch(f"{_PATCH}.refresh_license"):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 503

    # metadata None -> refresh populates it
    def test_proceeds_when_metadata_none_then_refresh_populates(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        future = datetime.now(timezone.utc) + timedelta(days=1)
        meta = {"agents": ["QA_AGENT"]}
        call_count = {"n": 0}

        def meta_side_effect():
            call_count["n"] += 1
            return None if call_count["n"] == 1 else meta

        with patch(f"{_PATCH}.get_license_metadata", side_effect=meta_side_effect), \
             patch(f"{_PATCH}.get_license_expires_at", return_value=future), \
             patch(f"{_PATCH}.refresh_license"):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 200

    # valid license, not expired
    def test_passes_through_when_license_valid(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        future = datetime.now(timezone.utc) + timedelta(days=1)
        meta = {"agents": ["QA_AGENT"]}

        with patch(f"{_PATCH}.get_license_metadata", return_value=meta), \
             patch(f"{_PATCH}.get_license_expires_at", return_value=future):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 200

    def test_passes_through_when_expires_at_is_none(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        meta = {"agents": ["QA_AGENT"]}

        with patch(f"{_PATCH}.get_license_metadata", return_value=meta), \
             patch(f"{_PATCH}.get_license_expires_at", return_value=None):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 200

    # expired license -> refresh succeeds
    def test_passes_through_after_successful_refresh_on_expiry(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        meta = {"agents": ["QA_AGENT"]}

        with patch(f"{_PATCH}.get_license_metadata", return_value=meta), \
             patch(f"{_PATCH}.get_license_expires_at", return_value=past), \
             patch(f"{_PATCH}.refresh_license"):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 200

    # expired license -> refresh fails
    def test_returns_503_when_refresh_fails_on_expiry(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        meta = {"agents": ["QA_AGENT"]}

        with patch(f"{_PATCH}.get_license_metadata", return_value=meta), \
             patch(f"{_PATCH}.get_license_expires_at", return_value=past), \
             patch(f"{_PATCH}.refresh_license", side_effect=Exception("refresh fail")):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.get("/")
        assert response.status_code == 503

    # expired license -> 503 response body matches _EXPIRED_RESPONSE
    def test_503_response_body_on_expiry(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        past = datetime.now(timezone.utc) - timedelta(seconds=1)
        meta = {"agents": ["QA_AGENT"]}

        with patch(f"{_PATCH}.get_license_metadata", return_value=meta), \
             patch(f"{_PATCH}.get_license_expires_at", return_value=past), \
             patch(f"{_PATCH}.refresh_license", side_effect=Exception("fail")):
            client = TestClient(_make_app(), raise_server_exceptions=False)
            response = client.post(
                "/",
                json={"jsonrpc": "2.0", "method": "test"},
                headers={"content-type": "application/json"},
            )
        assert response.status_code == 503
        body = response.json()
        assert body["http_status"] == 503
        assert body["details"]["mode"] == "LICENSE_EXPIRED"
