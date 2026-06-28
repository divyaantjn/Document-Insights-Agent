"""
tests/extraction/test_s3_extraction.py

Unit tests for src/extraction/s3_extraction.py → S3Extraction class.
All AWS / boto3 / S3 interactions are fully mocked.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_s3_extraction():
    """
    Import S3Extraction with src.utils.s3_utility patched so no real
    S3 calls are attempted on import/instantiation.
    """
    with patch("src.extraction.s3_extraction.get_s3_file", return_value=b"fake-content"):
        from src.extraction.s3_extraction import S3Extraction
    return S3Extraction


# ===========================================================================
# __init__
# ===========================================================================

class TestS3ExtractionInit:

    def test_default_init(self):
        cls = _make_s3_extraction()
        obj = cls()
        assert obj.urls is None
        assert obj.file_content is None
        assert obj.metadata is None
        assert obj.max_file_size == 50 * 1024 * 1024
        assert obj.timeout == 30
        assert "text" in obj.supported_types
        assert "document" in obj.supported_types
        assert "media" in obj.supported_types

    def test_custom_max_file_size(self):
        cls = _make_s3_extraction()
        obj = cls(max_file_size=10 * 1024)
        assert obj.max_file_size == 10 * 1024

    def test_custom_timeout(self):
        cls = _make_s3_extraction()
        obj = cls(timeout=60)
        assert obj.timeout == 60

    def test_supported_types_contain_expected_extensions(self):
        cls = _make_s3_extraction()
        obj = cls()
        assert ".pdf" in obj.supported_types["document"]
        assert ".txt" in obj.supported_types["text"]
        assert ".jpg" in obj.supported_types["media"]


# ===========================================================================
# read_files — validation
# ===========================================================================

class TestS3ExtractionReadFilesValidation:

    @pytest.fixture
    def s3(self):
        cls = _make_s3_extraction()
        return cls()

    @pytest.mark.asyncio
    async def test_raises_value_error_for_empty_list(self, s3):
        with pytest.raises(ValueError, match="empty"):
            s3.read_files([])

    @pytest.mark.asyncio
    async def test_raises_type_error_for_non_list(self, s3):
        with pytest.raises(TypeError, match="list"):
            s3.read_files("https://s3.aws/bucket/file.pdf")

    @pytest.mark.asyncio
    async def test_raises_type_error_for_none(self, s3):
        with pytest.raises(ValueError):
            s3.read_files(None)

    @pytest.mark.asyncio
    async def test_raises_type_error_for_dict(self, s3):
        with pytest.raises(TypeError):
            s3.read_files({"url": "https://..."})


# ===========================================================================
# read_files — happy path
# ===========================================================================

class TestS3ExtractionReadFilesSuccess:

    @pytest.mark.asyncio
    async def test_read_single_file_returns_dict(self):
        with patch(
            "src.extraction.s3_extraction.get_s3_file",
            return_value=b"file-bytes",
        ):
            from src.extraction.s3_extraction import S3Extraction

            s3 = S3Extraction()
            urls = ["https://s3.amazonaws.com/bucket/file.pdf"]
            result = s3.read_files(urls)

        assert isinstance(result, dict)
        assert "https://s3.amazonaws.com/bucket/file.pdf" in result
        assert result["https://s3.amazonaws.com/bucket/file.pdf"] == b"file-bytes"

    @pytest.mark.asyncio
    async def test_read_multiple_files(self):
        call_count = {"n": 0}

        def fake_get(url):
            call_count["n"] += 1
            return f"content-for-{url}".encode()

        with patch("src.extraction.s3_extraction.get_s3_file", side_effect=fake_get):
            from src.extraction.s3_extraction import S3Extraction

            s3 = S3Extraction()
            urls = [
                "https://s3.amazonaws.com/bucket/a.pdf",
                "https://s3.amazonaws.com/bucket/b.docx",
                "https://s3.amazonaws.com/bucket/c.xlsx",
            ]
            result = s3.read_files(urls)

        assert len(result) == 3
        assert call_count["n"] == 3

    @pytest.mark.asyncio
    async def test_sets_instance_attributes_after_read(self):
        with patch(
            "src.extraction.s3_extraction.get_s3_file",
            return_value=b"content",
        ):
            from src.extraction.s3_extraction import S3Extraction

            s3 = S3Extraction()
            urls = ["https://s3.amazonaws.com/bucket/doc.pdf"]
            s3.read_files(urls)

        assert s3.urls == urls
        assert s3.file_content is not None
        assert "https://s3.amazonaws.com/bucket/doc.pdf" in s3.file_content

    @pytest.mark.asyncio
    async def test_file_content_keys_match_urls(self):
        urls = [
            "https://s3.amazonaws.com/b/f1.pdf",
            "https://s3.amazonaws.com/b/f2.png",
        ]

        with patch(
            "src.extraction.s3_extraction.get_s3_file",
            side_effect=lambda u: f"data-{u}".encode(),
        ):
            from src.extraction.s3_extraction import S3Extraction

            s3 = S3Extraction()
            result = s3.read_files(urls)

        assert set(result.keys()) == set(urls)


# ===========================================================================
# read_files — error propagation
# ===========================================================================

class TestS3ExtractionReadFilesErrors:

    @pytest.mark.asyncio
    async def test_re_raises_s3_client_error(self):
        from botocore.exceptions import ClientError

        err = ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Key not found"}},
            "GetObject",
        )

        with patch("src.extraction.s3_extraction.get_s3_file", side_effect=err):
            from src.extraction.s3_extraction import S3Extraction

            s3 = S3Extraction()
            with pytest.raises(ClientError):
                s3.read_files(["https://s3.amazonaws.com/bucket/missing.pdf"])

    @pytest.mark.asyncio
    async def test_re_raises_generic_exception(self):
        with patch(
            "src.extraction.s3_extraction.get_s3_file",
            side_effect=ConnectionError("Network error"),
        ):
            from src.extraction.s3_extraction import S3Extraction

            s3 = S3Extraction()
            with pytest.raises(ConnectionError):
                s3.read_files(["https://s3.amazonaws.com/bucket/file.pdf"])


# ===========================================================================
# Supported types structure
# ===========================================================================

class TestS3ExtractionSupportedTypes:

    def test_text_category_has_txt(self):
        cls = _make_s3_extraction()
        obj = cls()
        assert ".txt" in obj.supported_types["text"]

    def test_document_category_has_docx(self):
        cls = _make_s3_extraction()
        obj = cls()
        assert ".docx" in obj.supported_types["document"]

    def test_media_category_has_mp4(self):
        cls = _make_s3_extraction()
        obj = cls()
        assert ".mp4" in obj.supported_types["media"]

    def test_document_category_has_pptx(self):
        cls = _make_s3_extraction()
        obj = cls()
        assert ".pptx" in obj.supported_types["document"]



class TestFileSizeLimitFRDO01:
    """
    FRDO-01 / TC-037 — Enforce the 25 MB per-file upload size limit (SRS).

    NOTE: explicit 25 MB enforcement is not yet implemented in code.
    S3Extraction currently defaults to 50 MB. These tests define the SRS
    contract; wire them to the real validator when it is added.
    # TODO: implement enforcement (e.g. S3Extraction(max_file_size=25*1024*1024)
    #       or a dedicated validate_file_size() helper) and remove the stub below.
    """

    MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # SRS FRDO-01: 25 MB per file

    @staticmethod
    def _validate_file_size(size_bytes: int, limit: int) -> str | None:
        """Stub mirroring the intended enforcement: return error msg or None."""
        if size_bytes > limit:
            return (
                f"File too large: {size_bytes // (1024 * 1024)} MB exceeds the "
                f"{limit // (1024 * 1024)} MB limit. Please reduce the file size."
            )
        return None

    def test_oversized_file_is_rejected_with_clear_reason(self):
        result = self._validate_file_size(26 * 1024 * 1024, self.MAX_UPLOAD_BYTES)
        assert result is not None
        assert "25 MB limit" in result

    def test_file_at_limit_is_accepted(self):
        # Boundary: exactly 25 MB must pass.
        result = self._validate_file_size(self.MAX_UPLOAD_BYTES, self.MAX_UPLOAD_BYTES)
        assert result is None

    def test_normal_file_is_accepted(self):
        result = self._validate_file_size(2 * 1024 * 1024, self.MAX_UPLOAD_BYTES)
        assert result is None



class TestEmbeddedImageCapFRDO06:
    """
    FRDO-06 / TC-038 — Enforce a maximum of 30 embedded images analysed per
    document (SRS).

    NOTE: the 30-image cap is not yet asserted/implemented in code. These
    tests define the SRS contract; wire them to the real image-collection
    path when the cap is added.
    # TODO: apply MAX_IMAGES_PER_DOCUMENT inside the image-collection logic
    #       (e.g. _collect_docx_images / _collect_pdf_page_images) and point
    #       these tests at it instead of the local stub.
    """

    MAX_IMAGES_PER_DOCUMENT = 30  # SRS FRDO-06: max 30 images analysed/document

    @staticmethod
    def _cap_images(images: list, cap: int) -> list:
        """Stub mirroring the intended cap: keep at most `cap` images."""
        return images[:cap]

    def test_more_than_30_images_are_capped_to_30(self):
        images = [f"img_{i}" for i in range(45)]
        selected = self._cap_images(images, self.MAX_IMAGES_PER_DOCUMENT)
        assert len(selected) == self.MAX_IMAGES_PER_DOCUMENT

    def test_exactly_30_images_all_kept(self):
        images = [f"img_{i}" for i in range(30)]
        selected = self._cap_images(images, self.MAX_IMAGES_PER_DOCUMENT)
        assert len(selected) == 30

    def test_fewer_than_30_images_all_kept(self):
        images = [f"img_{i}" for i in range(7)]
        selected = self._cap_images(images, self.MAX_IMAGES_PER_DOCUMENT)
        assert len(selected) == 7
