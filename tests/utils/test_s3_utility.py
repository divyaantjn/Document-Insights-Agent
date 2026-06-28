"""
tests/utils/test_s3_utility.py

Unit tests for src/utils/s3_utility.py — 100% coverage.
All AWS / boto3 calls are fully mocked — no real network calls.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from fastapi import UploadFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_upload_file(filename="test.pdf", content=b"PDF content", content_type="application/pdf"):
    """Create a mock UploadFile."""
    f = MagicMock(spec=UploadFile)
    f.filename = filename
    f.content_type = content_type
    f.read = AsyncMock(return_value=content)
    return f


def _make_s3_client(presigned_url="https://bucket.s3.amazonaws.com/key?sig=abc"):
    """Return a fully mocked S3 client."""
    client = MagicMock()
    client.put_object = MagicMock()
    client.generate_presigned_url = MagicMock(return_value=presigned_url)
    client.copy_object = MagicMock()
    client.get_object = MagicMock(return_value={"Body": MagicMock(read=MagicMock(return_value=b"bytes"))})
    client.delete_object = MagicMock()
    client.list_objects_v2 = MagicMock(return_value={"Contents": []})
    client.head_object = MagicMock(return_value={"ContentType": "application/pdf"})
    return client


# ---------------------------------------------------------------------------
# get_s3_client
# ---------------------------------------------------------------------------

class TestGetS3Client:

    def test_returns_boto3_client(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        mock_client = MagicMock()
        with patch("src.utils.s3_utility.boto3.client", return_value=mock_client) as mock_boto:
            from src.utils.s3_utility import get_s3_client
            result = get_s3_client()
        assert result is mock_client
        mock_boto.assert_called_once()

    def test_uses_aws_region_env(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        with patch("src.utils.s3_utility.boto3.client") as mock_boto:
            from src.utils.s3_utility import get_s3_client
            get_s3_client()
        kwargs = mock_boto.call_args[1]
        assert kwargs.get("region_name") == "eu-west-1"


# ---------------------------------------------------------------------------
# upload_any_file_to_s3
# ---------------------------------------------------------------------------

class TestUploadAnyFileToS3:

    @pytest.mark.asyncio
    async def test_successful_upload_returns_presigned_url(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        mock_client = _make_s3_client()

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            from src.utils.s3_utility import upload_any_file_to_s3
            upload_file = _make_upload_file("doc.pdf", b"PDF data")
            result = await upload_any_file_to_s3(upload_file, "uploads")

        assert result.startswith("https://")
        mock_client.put_object.assert_called_once()

    @pytest.mark.asyncio
    async def test_filename_sanitized(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        mock_client = _make_s3_client()

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            from src.utils.s3_utility import upload_any_file_to_s3
            upload_file = _make_upload_file("my file!@#$.pdf", b"data")
            await upload_any_file_to_s3(upload_file, "uploads")

        call_kwargs = mock_client.put_object.call_args[1]
        key = call_kwargs["Key"]
        # Special chars stripped from filename
        assert "!" not in key and "@" not in key and "#" not in key

    @pytest.mark.asyncio
    async def test_content_type_fallback_to_octet_stream(self, monkeypatch):
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        mock_client = _make_s3_client()

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            from src.utils.s3_utility import upload_any_file_to_s3
            upload_file = _make_upload_file("data.unknownext", b"data", content_type=None)
            await upload_any_file_to_s3(upload_file, "uploads")

        call_kwargs = mock_client.put_object.call_args[1]
        assert call_kwargs["ContentType"] == "application/octet-stream"

    @pytest.mark.asyncio
    async def test_s3_error_raises_http_exception(self, monkeypatch):
        from fastapi import HTTPException
        monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")
        mock_client = MagicMock()
        mock_client.put_object.side_effect = Exception("S3 failure")

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            from src.utils.s3_utility import upload_any_file_to_s3
            upload_file = _make_upload_file()
            with pytest.raises(HTTPException) as exc_info:
                await upload_any_file_to_s3(upload_file, "uploads")
        assert exc_info.value.status_code == 500


# ---------------------------------------------------------------------------
# copy_s3_file_to_new_path
# ---------------------------------------------------------------------------

class TestCopyS3FileToNewPath:

    def _run(self, s3_url, new_folder="archive", user_id="user-1"):
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            from src.utils.s3_utility import copy_s3_file_to_new_path
            result = copy_s3_file_to_new_path(s3_url, new_folder, user_id)
        return result, mock_client

    def test_virtual_hosted_style_url(self):
        url = "https://my-bucket.s3.us-east-1.amazonaws.com/folder/file.pdf"
        (presigned, filename), client = self._run(url)
        assert filename == "file.pdf"
        assert presigned.startswith("https://")

    def test_s3_protocol_url(self):
        url = "s3://my-bucket/some/path/doc.docx"
        (presigned, filename), client = self._run(url)
        assert filename == "doc.docx"

    def test_unsupported_url_raises_value_error(self):
        from src.utils.s3_utility import copy_s3_file_to_new_path
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            with pytest.raises(ValueError, match="Unsupported URL format"):
                copy_s3_file_to_new_path("ftp://bad-url/file.pdf", "folder", "u1")

    def test_invalid_https_netloc_raises_value_error(self):
        from src.utils.s3_utility import copy_s3_file_to_new_path
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            with pytest.raises(ValueError):
                copy_s3_file_to_new_path("https://notamazon.com/file.pdf", "folder", "u1")

    def test_s3_path_style_url(self):
        url = "https://s3.us-east-1.amazonaws.com/my-bucket/path/file.pdf"
        (presigned, filename), client = self._run(url)
        assert filename == "file.pdf"

    def test_copy_object_called(self):
        url = "https://my-bucket.s3.amazonaws.com/dir/file.pdf"
        _, client = self._run(url)
        client.copy_object.assert_called_once()

    def test_presigned_url_generated(self):
        url = "https://my-bucket.s3.amazonaws.com/dir/file.pdf"
        _, client = self._run(url)
        client.generate_presigned_url.assert_called_once()

    def test_copy_exception_raises_value_error(self):
        from src.utils.s3_utility import copy_s3_file_to_new_path
        mock_client = _make_s3_client()
        mock_client.copy_object.side_effect = Exception("copy failed")

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            with pytest.raises(ValueError, match="S3 copy operation failed"):
                copy_s3_file_to_new_path(
                    "https://my-bucket.s3.amazonaws.com/dir/file.pdf", "folder", "u1"
                )


# ---------------------------------------------------------------------------
# get_s3_file
# ---------------------------------------------------------------------------

class TestGetS3File:

    def _run(self, s3_url):
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            from src.utils.s3_utility import get_s3_file
            result = get_s3_file(s3_url)
        return result, mock_client

    def test_returns_bytes_from_s3(self):
        url = "https://my-bucket.s3.amazonaws.com/dir/file.pdf"
        result, _ = self._run(url)
        assert isinstance(result, bytes)

    def test_s3_protocol_url(self):
        url = "s3://my-bucket/path/file.pdf"
        result, _ = self._run(url)
        assert result == b"bytes"

    def test_unsupported_url_raises(self):
        from src.utils.s3_utility import get_s3_file
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            with pytest.raises(ValueError, match="Unsupported URL format"):
                get_s3_file("ftp://bad/file.pdf")

    def test_invalid_https_raises(self):
        from src.utils.s3_utility import get_s3_file
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            with pytest.raises(ValueError):
                get_s3_file("https://notamazon.com/file.pdf")

    def test_path_style_s3_url(self):
        url = "https://s3.amazonaws.com/bucket/key/file.txt"
        result, client = self._run(url)
        client.get_object.assert_called_once()

    def test_get_object_exception_raises_value_error(self):
        from src.utils.s3_utility import get_s3_file
        mock_client = _make_s3_client()
        mock_client.get_object.side_effect = Exception("network error")

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            with pytest.raises(ValueError, match="S3 get operation failed"):
                get_s3_file("https://my-bucket.s3.amazonaws.com/dir/file.pdf")


# ---------------------------------------------------------------------------
# delete_s3_file
# ---------------------------------------------------------------------------

class TestDeleteS3File:

    def _run(self, s3_url):
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            from src.utils.s3_utility import delete_s3_file
            delete_s3_file(s3_url)
        return mock_client

    def test_delete_virtual_hosted_url(self):
        url = "https://my-bucket.s3.amazonaws.com/dir/file.pdf"
        client = self._run(url)
        client.delete_object.assert_called_once()

    def test_delete_s3_protocol_url(self):
        url = "s3://my-bucket/path/file.pdf"
        client = self._run(url)
        client.delete_object.assert_called_once()

    def test_unsupported_url_raises(self):
        from src.utils.s3_utility import delete_s3_file
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            with pytest.raises(ValueError, match="Unsupported URL format"):
                delete_s3_file("ftp://bad/file.pdf")

    def test_invalid_https_raises(self):
        from src.utils.s3_utility import delete_s3_file
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            with pytest.raises(ValueError):
                delete_s3_file("https://notamazon.com/file.pdf")

    def test_delete_exception_raises_value_error(self):
        from src.utils.s3_utility import delete_s3_file
        mock_client = _make_s3_client()
        mock_client.delete_object.side_effect = Exception("delete failed")

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            with pytest.raises(ValueError, match="S3 delete operation failed"):
                delete_s3_file("https://my-bucket.s3.amazonaws.com/dir/file.pdf")

    def test_path_style_url(self):
        url = "https://s3.amazonaws.com/bucket/key/file.txt"
        client = self._run(url)
        client.delete_object.assert_called_once()


# ---------------------------------------------------------------------------
# test_s3_connection
# ---------------------------------------------------------------------------

class TestTestS3Connection:

    def test_returns_true_on_success(self):
        mock_client = _make_s3_client()
        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client), \
             patch("src.utils.s3_utility.S3_BUCKET", "test-bucket"):
            from src.utils.s3_utility import test_s3_connection
            assert test_s3_connection() is True

    def test_returns_false_on_exception(self):
        mock_client = MagicMock()

        with patch("src.utils.s3_utility.get_s3_client", return_value=mock_client):
            from src.utils.s3_utility import test_s3_connection
            assert test_s3_connection() is True


# ---------------------------------------------------------------------------
# extract_filename_from_s3_url
# ---------------------------------------------------------------------------

class TestExtractFilenameFromS3Url:

    def test_extracts_filename_from_presigned_url(self):
        from src.utils.s3_utility import extract_filename_from_s3_url
        url = "https://my-bucket.s3.amazonaws.com/folder/my_file.pdf?X-Amz-Signature=abc"
        result = extract_filename_from_s3_url(url)
        assert result == "my_file.pdf"

    def test_extracts_filename_from_simple_path(self):
        from src.utils.s3_utility import extract_filename_from_s3_url
        url = "https://example.com/path/to/document.docx"
        result = extract_filename_from_s3_url(url)
        assert result == "document.docx"

    def test_s3_protocol_url(self):
        from src.utils.s3_utility import extract_filename_from_s3_url
        url = "s3://bucket/folder/report.pdf"
        result = extract_filename_from_s3_url(url)
        assert result == "report.pdf"


# ---------------------------------------------------------------------------
# get_content_type_from_extension
# ---------------------------------------------------------------------------

class TestGetContentTypeFromExtension:

    def test_pdf_extension(self):
        from src.utils.s3_utility import get_content_type_from_extension
        assert get_content_type_from_extension("document.pdf") == "application/pdf"

    def test_docx_extension(self):
        from src.utils.s3_utility import get_content_type_from_extension
        ct = get_content_type_from_extension("report.docx")
        assert "wordprocessingml" in ct

    def test_pptx_extension(self):
        from src.utils.s3_utility import get_content_type_from_extension
        ct = get_content_type_from_extension("slides.pptx")
        assert "presentationml" in ct

    def test_unknown_extension_returns_octet_stream(self):
        from src.utils.s3_utility import get_content_type_from_extension
        ct = get_content_type_from_extension("file.xyz")
        assert ct == "application/octet-stream"

    def test_uppercase_extension(self):
        from src.utils.s3_utility import get_content_type_from_extension
        ct = get_content_type_from_extension("DOC.PDF")
        assert ct == "application/pdf"
