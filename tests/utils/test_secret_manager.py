"""Tests for utils/secret_manager.py"""
import pytest
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError


@pytest.fixture(autouse=True)
def reset_client():
    import src.utils.secret_manager as sm
    sm._client = None
    yield
    sm._client = None


class TestGetClient:
    def test_creates_client_once(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        import src.utils.secret_manager as sm

        fake_client = MagicMock()
        with patch("src.utils.secret_manager.boto3.client", return_value=fake_client) as mock_boto:
            c1 = sm._get_client()
            c2 = sm._get_client()

        assert c1 is fake_client
        assert c2 is fake_client
        mock_boto.assert_called_once_with("secretsmanager", region_name="us-west-2")

    def test_uses_default_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        import src.utils.secret_manager as sm

        with patch("src.utils.secret_manager.boto3.client", return_value=MagicMock()) as mock_boto:
            sm._get_client()

        mock_boto.assert_called_once_with("secretsmanager", region_name="us-east-1")


class TestGetSecret:
    def test_returns_secret_string(self):
        import src.utils.secret_manager as sm

        fake_client = MagicMock()
        fake_client.get_secret_value.return_value = {"SecretString": '{"key": "value"}'}
        sm._client = fake_client

        result = sm.get_secret("my-secret")
        assert result == '{"key": "value"}'
        fake_client.get_secret_value.assert_called_once_with(SecretId="my-secret")

    def test_returns_decoded_secret_binary_when_no_string(self):
        import src.utils.secret_manager as sm

        fake_client = MagicMock()
        fake_binary = MagicMock()
        fake_binary.decode.return_value = "binary-secret"
        fake_client.get_secret_value.return_value = {"SecretString": None, "SecretBinary": fake_binary}
        sm._client = fake_client

        result = sm.get_secret("binary-secret")
        assert result == "binary-secret"

    def test_raises_on_client_error(self):
        import src.utils.secret_manager as sm

        fake_client = MagicMock()
        fake_client.get_secret_value.side_effect = ClientError(
            {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
            "GetSecretValue",
        )
        sm._client = fake_client

        with pytest.raises(ClientError):
            sm.get_secret("missing-secret")
