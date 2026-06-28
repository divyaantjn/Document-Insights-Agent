"""Tests for src/utils/license_validator.py"""
import json
import base64
import glob
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, mock_open, call
import src.utils.license_validator as lv

_PATCH = "src.utils.license_validator"


# ── helpers ──────────────────────────────────────────────────────────────────

def _reset_globals():
    lv._license_metadata = None
    lv._license_expires_at = None
    lv._redis_client = None


@pytest.fixture(autouse=True)
def reset_state():
    _reset_globals()
    yield
    _reset_globals()


# ── _get_redis_client ─────────────────────────────────────────────────────────

class TestGetRedisClient:
    def test_creates_client_once(self, monkeypatch):
        monkeypatch.setenv("REDIS_HOST", "redis-host")
        monkeypatch.setenv("REDIS_PORT", "6380")
        monkeypatch.setenv("REDIS_SSL", "true")

        fake_redis = MagicMock()
        with patch(f"{_PATCH}.redis.Redis", return_value=fake_redis) as mock_cls:
            c1 = lv._get_redis_client()
            c2 = lv._get_redis_client()

        assert c1 is fake_redis
        assert c2 is fake_redis
        mock_cls.assert_called_once()

    def test_ssl_false_by_default(self, monkeypatch):
        monkeypatch.delenv("REDIS_SSL", raising=False)
        fake_redis = MagicMock()
        with patch(f"{_PATCH}.redis.Redis", return_value=fake_redis) as mock_cls:
            lv._get_redis_client()
        kwargs = mock_cls.call_args[1]
        assert kwargs["ssl"] is False


# ── _fetch_app_license ────────────────────────────────────────────────────────

class TestFetchAppLicense:
    def test_returns_cached_value_from_redis(self):
        fake_redis = MagicMock()
        fake_redis.get.return_value = "cached-license"
        with patch(f"{_PATCH}._get_redis_client", return_value=fake_redis):
            result = lv._fetch_app_license()
        assert result == "cached-license"

    def test_falls_back_to_secrets_manager_on_redis_miss(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        fake_redis = MagicMock()
        fake_redis.get.return_value = None

        secret_payload = json.dumps({"APP_LICENSE": "sm-license"})
        with patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}.get_secret", return_value=secret_payload):
            result = lv._fetch_app_license()

        assert result == "sm-license"
        fake_redis.set.assert_called_once()

    def test_falls_back_to_secrets_manager_on_redis_exception(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        fake_redis = MagicMock()
        fake_redis.get.side_effect = Exception("redis down")

        secret_payload = json.dumps({"app_license": "sm-license-lower"})
        with patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}.get_secret", return_value=secret_payload):
            result = lv._fetch_app_license()

        assert result == "sm-license-lower"

    def test_raises_when_app_license_missing_in_secret(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        fake_redis = MagicMock()
        fake_redis.get.return_value = None

        with patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}.get_secret", return_value=json.dumps({"other_key": "val"})):
            with pytest.raises(RuntimeError, match="APP_LICENSE not found"):
                lv._fetch_app_license()

    def test_redis_cache_set_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        fake_redis.set.side_effect = Exception("redis write fail")

        secret_payload = json.dumps({"APP_LICENSE": "license-val"})
        with patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}.get_secret", return_value=secret_payload):
            result = lv._fetch_app_license()

        assert result == "license-val"

    def test_raises_when_secrets_manager_fails(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        fake_redis = MagicMock()
        fake_redis.get.return_value = None

        with patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}.get_secret", side_effect=Exception("sm error")):
            with pytest.raises(RuntimeError, match="Failed to fetch license from Secrets Manager"):
                lv._fetch_app_license()


# ── _fetch_keys ───────────────────────────────────────────────────────────────

class TestFetchKeys:
    def test_returns_keys_from_env(self, monkeypatch):
        monkeypatch.setenv("ORG_PRIVATE_KEY", "org-key")
        monkeypatch.setenv("YASH_PUBLIC_KEY", "yash-key")
        org, yash = lv._fetch_keys()
        assert org == "org-key"
        assert yash == "yash-key"

    def test_fetches_keys_from_secrets_manager_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("ORG_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("YASH_PUBLIC_KEY", raising=False)
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")

        secret_payload = json.dumps({"ORG_PRIVATE_KEY": "org-from-sm", "YASH_PUBLIC_KEY": "yash-from-sm"})
        with patch(f"{_PATCH}.get_secret", return_value=secret_payload):
            org, yash = lv._fetch_keys()

        assert org == "org-from-sm"
        assert yash == "yash-from-sm"

    def test_raises_when_keys_missing_in_secret(self, monkeypatch):
        monkeypatch.delenv("ORG_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("YASH_PUBLIC_KEY", raising=False)

        with patch(f"{_PATCH}.get_secret", return_value=json.dumps({})):
            with pytest.raises(RuntimeError, match="ORG_PRIVATE_KEY and YASH_PUBLIC_KEY not found"):
                lv._fetch_keys()

    def test_raises_when_secrets_manager_fails(self, monkeypatch):
        monkeypatch.delenv("ORG_PRIVATE_KEY", raising=False)
        monkeypatch.delenv("YASH_PUBLIC_KEY", raising=False)

        with patch(f"{_PATCH}.get_secret", side_effect=Exception("sm down")):
            with pytest.raises(RuntimeError, match="Failed to fetch keys from Secrets Manager"):
                lv._fetch_keys()

    def test_replaces_escaped_newlines_in_keys(self, monkeypatch):
        monkeypatch.setenv("ORG_PRIVATE_KEY", "line1\\nline2")
        monkeypatch.setenv("YASH_PUBLIC_KEY", "pub\\nkey")
        org, yash = lv._fetch_keys()
        assert "\n" in org
        assert "\n" in yash


# ── _validate_and_store ───────────────────────────────────────────────────────

class TestValidateAndStore:
    def _make_valid_license_package(self, metadata: dict) -> tuple:
        from cryptography.hazmat.primitives.asymmetric import rsa, padding
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.backends import default_backend
        import os as _os

        org_private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )
        yash_private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=2048, backend=default_backend()
        )

        org_private_pem = org_private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ).decode()
        yash_public_pem = yash_private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()

        license_data = {"metadata": metadata}
        plaintext = json.dumps(license_data).encode()

        aes_key = _os.urandom(32)
        iv = _os.urandom(12)
        aesgcm = AESGCM(aes_key)
        ciphertext_with_tag = aesgcm.encrypt(iv, plaintext, None)
        encrypted_data = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]

        encrypted_aes_key = org_private_key.public_key().encrypt(
            aes_key,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )

        signature = yash_private_key.sign(
            plaintext,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )

        package = {
            "encrypted_key": base64.b64encode(encrypted_aes_key).decode(),
            "iv": base64.b64encode(iv).decode(),
            "encrypted_data": base64.b64encode(encrypted_data).decode(),
            "tag": base64.b64encode(tag).decode(),
            "signature": base64.b64encode(signature).decode(),
        }
        app_license = base64.b64encode(json.dumps(package).encode()).decode()
        return app_license, org_private_pem, yash_public_pem

    def test_validates_license_with_expires_at(self, monkeypatch):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        metadata = {"expires_at": future, "agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        lv._validate_and_store(app_license)
        assert lv._license_metadata is not None
        assert lv._license_expires_at is not None

    def test_validates_license_with_tenure_days(self, monkeypatch):
        issued = datetime.now(timezone.utc).isoformat()
        metadata = {"tenure_days": 30, "issued_at": issued, "agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        lv._validate_and_store(app_license)
        assert lv._license_metadata is not None

    def test_validates_license_with_tenure_days_no_issued_at(self, monkeypatch):
        metadata = {"tenure_days": 30, "agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        lv._validate_and_store(app_license)
        assert lv._license_metadata is not None

    def test_validates_license_with_tenure_minutes(self, monkeypatch):
        issued = datetime.now(timezone.utc).isoformat()
        metadata = {"tenure_minutes": 60, "issued_at": issued, "agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        lv._validate_and_store(app_license)
        assert lv._license_metadata is not None

    def test_validates_license_with_tenure_minutes_no_issued_at(self, monkeypatch):
        metadata = {"tenure_minutes": 60, "agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        lv._validate_and_store(app_license)
        assert lv._license_metadata is not None

    def test_raises_when_no_expiry_info(self, monkeypatch):
        metadata = {"agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        with pytest.raises(RuntimeError, match="no expiry information"):
            lv._validate_and_store(app_license)

    def test_raises_when_license_expired(self, monkeypatch):
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        metadata = {"expires_at": past, "agents": ["QA_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        with pytest.raises(RuntimeError, match="License expired"):
            lv._validate_and_store(app_license)

    def test_raises_when_agent_not_licensed(self, monkeypatch):
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        metadata = {"expires_at": future, "agents": ["OTHER_AGENT"]}
        monkeypatch.setenv("AGENT_NAME", "QA_AGENT")

        app_license, org_pem, yash_pem = self._make_valid_license_package(metadata)
        monkeypatch.setenv("ORG_PRIVATE_KEY", org_pem)
        monkeypatch.setenv("YASH_PUBLIC_KEY", yash_pem)

        with pytest.raises(RuntimeError, match="not licensed"):
            lv._validate_and_store(app_license)

    def test_raises_on_invalid_base64_blob(self, monkeypatch):
        with patch(f"{_PATCH}._fetch_keys", return_value=("k", "k")):
            with pytest.raises(RuntimeError, match="Failed to decode APP_LICENSE"):
                lv._validate_and_store("!!!not-base64!!!")

    def test_raises_on_bad_aes_key_decryption(self, monkeypatch):
        package = {
            "encrypted_key": base64.b64encode(b"bad").decode(),
            "iv": base64.b64encode(b"x" * 12).decode(),
            "encrypted_data": base64.b64encode(b"data").decode(),
            "tag": base64.b64encode(b"t" * 16).decode(),
            "signature": base64.b64encode(b"sig").decode(),
        }
        app_license = base64.b64encode(json.dumps(package).encode()).decode()

        with patch(f"{_PATCH}._fetch_keys", return_value=("bad-key", "bad-key")):
            with pytest.raises(RuntimeError, match="Failed to decrypt AES key"):
                lv._validate_and_store(app_license)


# ── _inject_pyarmor_license ───────────────────────────────────────────────────

class TestInjectPyarmorLicense:
    def test_writes_to_matched_paths(self, tmp_path):
        license_file = tmp_path / "license.lic"
        license_file.write_bytes(b"old")

        with patch(f"{_PATCH}.glob.glob", return_value=[str(license_file)]):
            lv._inject_pyarmor_license(b"new-license")

        assert license_file.read_bytes() == b"new-license"

    def test_swallows_write_error(self):
        with patch(f"{_PATCH}.glob.glob", return_value=["/nonexistent/path/license.lic"]):
            lv._inject_pyarmor_license(b"data")  # must not raise

    def test_no_paths_matched(self):
        with patch(f"{_PATCH}.glob.glob", return_value=[]):
            lv._inject_pyarmor_license(b"data")  # must not raise


# ── initialize_license ────────────────────────────────────────────────────────

class TestInitializeLicense:
    def test_skips_when_enforcement_disabled(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "false")
        with patch(f"{_PATCH}._fetch_app_license") as mock_fetch:
            lv.initialize_license()
        mock_fetch.assert_not_called()

    def test_skips_when_already_initialized(self):
        lv._license_metadata = {"agents": ["QA_AGENT"]}
        with patch(f"{_PATCH}._fetch_app_license") as mock_fetch:
            lv.initialize_license()
        mock_fetch.assert_not_called()

    def test_calls_fetch_and_validate(self, monkeypatch):
        monkeypatch.setenv("LICENSE_ENFORCE", "true")
        with patch(f"{_PATCH}._fetch_app_license", return_value="license") as mock_fetch, \
             patch(f"{_PATCH}._validate_and_store") as mock_validate:
            lv.initialize_license()
        mock_fetch.assert_called_once()
        mock_validate.assert_called_once_with("license")


# ── refresh_license ───────────────────────────────────────────────────────────

class TestRefreshLicense:
    def test_refreshes_and_revalidates(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        secret_payload = json.dumps({"APP_LICENSE": "new-license"})
        fake_redis = MagicMock()

        with patch(f"{_PATCH}.get_secret", return_value=secret_payload), \
             patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}._validate_and_store") as mock_validate:
            lv.refresh_license()

        mock_validate.assert_called_once_with("new-license")
        assert lv._license_metadata is None

    def test_redis_cache_failure_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        secret_payload = json.dumps({"APP_LICENSE": "new-license"})
        fake_redis = MagicMock()
        fake_redis.set.side_effect = Exception("redis fail")

        with patch(f"{_PATCH}.get_secret", return_value=secret_payload), \
             patch(f"{_PATCH}._get_redis_client", return_value=fake_redis), \
             patch(f"{_PATCH}._validate_and_store"):
            lv.refresh_license()  # must not raise

    def test_raises_when_app_license_missing(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        with patch(f"{_PATCH}.get_secret", return_value=json.dumps({"other": "val"})):
            with pytest.raises(RuntimeError, match="APP_LICENSE not found"):
                lv.refresh_license()

    def test_raises_when_secrets_manager_fails(self, monkeypatch):
        monkeypatch.setenv("LICENSE_SECRET_NAME", "my-secret")
        with patch(f"{_PATCH}.get_secret", side_effect=Exception("sm down")):
            with pytest.raises(RuntimeError, match="Failed to refresh license"):
                lv.refresh_license()


# ── getters ───────────────────────────────────────────────────────────────────

class TestGetters:
    def test_get_license_metadata_returns_none_initially(self):
        assert lv.get_license_metadata() is None

    def test_get_license_metadata_returns_set_value(self):
        lv._license_metadata = {"agents": ["QA_AGENT"]}
        assert lv.get_license_metadata() == {"agents": ["QA_AGENT"]}

    def test_get_license_expires_at_returns_none_initially(self):
        assert lv.get_license_expires_at() is None

    def test_get_license_expires_at_returns_set_value(self):
        dt = datetime.now(timezone.utc)
        lv._license_expires_at = dt
        assert lv.get_license_expires_at() == dt
