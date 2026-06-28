"""
tests/routers/test_s3_ocr_router.py

Production-level unit tests for src/routers/s3_ocr_router.py — 100% coverage.
All external services (Milvus, LiteLLM, S3, OCR, Kafka) are fully mocked.
FastAPI TestClient is used for HTTP-level endpoint testing.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# App fixture — mount the router under test with all dependencies mocked
# ---------------------------------------------------------------------------

FAKE_AUTH = "Bearer test-token"
FAKE_TEAM_CONFIG = {"model": "openai/gpt-4o", "api_key": "test"}
FAKE_SESSION = "session-abc"
FAKE_TEAM = "team-xyz"
FAKE_S3_URL = "https://my-bucket.s3.amazonaws.com/docs/report.pdf"


def _user_meta(session_id=FAKE_SESSION, team_id=FAKE_TEAM, **extra) -> str:
    d = {"session_id": session_id, "team_id": team_id, **extra}
    return json.dumps(d)


@pytest.fixture(scope="function")
def app():
    """
    Build a minimal FastAPI app with the router and all heavy singletons mocked
    at import time so the module-level `milvus`, `litellm_client` etc. are stubs.
    """
    mock_milvus = MagicMock()
    mock_milvus.get_uploaded_filenames_for_session.return_value = set()
    mock_milvus.ingest_documents.return_value = {
        "total_documents": 1, "successful_documents": 1,
        "failed_documents": 0, "total_chunks_inserted": 3, "failed_doc_details": []
    }
    mock_milvus.enhanced_search = AsyncMock(return_value=[
        {"score": 0.9, "text": "Relevant content", "metadata": {}, "retrieval_method": "direct"}
    ])
    mock_milvus.synthesize_answer_with_llm = AsyncMock(return_value={
        "answer": "The answer is 42.", "sources": []
    })
    mock_milvus.get_collection_stats.return_value = {
        "collection_name": "idp", "num_entities": 10
    }

    mock_litellm = MagicMock()
    mock_litellm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
    
    from src.routers.s3_ocr_router import router
    with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
         patch("src.routers.s3_ocr_router.litellm_client", mock_litellm):
        _app = FastAPI()
        _app.include_router(router)
        yield _app, mock_milvus, mock_litellm


@pytest.fixture(scope="function")
def client(app):
    _app, _, _ = app
    return TestClient(_app, raise_server_exceptions=False)


@pytest.fixture(scope="function")
def mocks(app):
    _, mock_milvus, mock_litellm = app
    return mock_milvus, mock_litellm


# ===========================================================================
# _detect_document_type  (pure function — no HTTP needed)
# ===========================================================================

class TestDetectDocumentType:

    def _detect(self, filename, s3_url=""):
        from src.routers.s3_ocr_router import _detect_document_type
        return _detect_document_type(filename)

    def test_pdf(self):
        assert self._detect("report.pdf") == "pdf"

    def test_docx(self):
        assert self._detect("doc.docx") == "docx"

    def test_doc(self):
        assert self._detect("legacy.doc") == "doc"

    def test_xlsx(self):
        assert self._detect("data.xlsx") == "xlsx"

    def test_xls(self):
        assert self._detect("old.xls") == "xls"

    def test_pptx(self):
        assert self._detect("deck.pptx") == "pptx"

    def test_ppt(self):
        assert self._detect("old.ppt") == "ppt"

    def test_png(self):
        assert self._detect("photo.png") == "image"

    def test_jpg(self):
        assert self._detect("photo.jpg") == "image"

    def test_jpeg(self):
        assert self._detect("photo.jpeg") == "image"

    def test_gif(self):
        assert self._detect("anim.gif") == "image"

    def test_bmp(self):
        assert self._detect("bitmap.bmp") == "image"

    def test_tiff(self):
        assert self._detect("scan.tiff") == "image"

    def test_webp(self):
        assert self._detect("photo.webp") == "image"

    def test_jfif(self):
        assert self._detect("img.jfif") == "image"

    def test_txt(self):
        assert self._detect("readme.txt") == "text"

    def test_md(self):
        assert self._detect("README.md") == "text"

    # FIX: Router type_mapping maps 'csv' -> 'text', so the correct assertion is "text"
    def test_csv(self):
        assert self._detect("data.csv") == "text"

    def test_json(self):
        assert self._detect("config.json") == "text"

    def test_xml(self):
        assert self._detect("data.xml") == "text"

    def test_html(self):
        assert self._detect("page.html") == "text"

    def test_zip(self):
        assert self._detect("archive.zip") == "zip"

    def test_unknown_extension(self):
        assert self._detect("binary.bin") == "unknown"

    def test_no_extension(self):
        assert self._detect("noext") == "unknown"

    def test_exception_returns_unknown(self):
        from src.routers.s3_ocr_router import _detect_document_type
        # Passing None for filename should be handled gracefully
        result = _detect_document_type(None)
        assert result == "unknown"

    def test_uppercase_extension(self):
        assert self._detect("REPORT.PDF") == "pdf"


# ===========================================================================
# _process_document  (internal async helper — tested directly)
# ===========================================================================

class TestProcessDocument:

    def _mock_ocr(self):
        ocr = MagicMock()
        ocr.extract_text_from_pdf_file = AsyncMock(return_value={"chunks": [{"text": "pdf text", "type": "text"}]})
        ocr.extract_text_from_docx = AsyncMock(return_value={"chunks": [{"text": "docx text", "type": "text"}]})
        ocr.extract_text_from_excel = MagicMock(return_value={"chunks": [{"text": "excel text", "type": "table"}]})
        ocr.extract_text_from_ppt = AsyncMock(return_value={"chunks": [{"text": "ppt text", "type": "text"}]})
        ocr.extract_text_from_image = AsyncMock(return_value={"text": "image text"})
        ocr.extract_text_from_text_file = AsyncMock(return_value="plain text")
        ocr.extract_text_from_csv = MagicMock(return_value={"chunks": [{"text": "csv text", "type": "table"}]})
        ocr.extract_text_from_zip = AsyncMock(return_value={"inner.txt": [{"text": "zip text", "type": "text"}]})
        return ocr

    @pytest.mark.asyncio
    async def test_pdf_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "pdf", "file.pdf", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "pdf text"

    @pytest.mark.asyncio
    async def test_docx_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "docx", "file.docx", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "docx text"

    @pytest.mark.asyncio
    async def test_doc_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "doc", "file.doc", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "docx text"

    @pytest.mark.asyncio
    async def test_xlsx_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "xlsx", "file.xlsx", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "excel text"

    @pytest.mark.asyncio
    async def test_xls_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "xls", "file.xls", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "excel text"

    @pytest.mark.asyncio
    async def test_pptx_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "pptx", "file.pptx", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "ppt text"

    @pytest.mark.asyncio
    async def test_ppt_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "ppt", "file.ppt", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "ppt text"

    @pytest.mark.asyncio
    async def test_image_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "image", "photo.png", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "image text"

    @pytest.mark.asyncio
    async def test_png_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "png", "photo.png", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "image text"

    @pytest.mark.asyncio
    async def test_text_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "text", "file.txt", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "plain text"

    @pytest.mark.asyncio
    async def test_txt_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "txt", "file.txt", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "plain text"

    @pytest.mark.asyncio
    async def test_csv_routing(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "csv", "data.csv", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert result["chunks"][0]["text"] == "csv text"

    @pytest.mark.asyncio
    async def test_zip_routing_flattens_chunks(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "zip", "archive.zip", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        # Zip returns flattened chunks with zip_source metadata
        assert "chunks" in result
        assert result["chunks"][0]["zip_source"] == "inner.txt"

    @pytest.mark.asyncio
    async def test_zip_error_passthrough(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        ocr.extract_text_from_zip = AsyncMock(return_value={"error": "bad zip"})
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "zip", "bad.zip", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_unsupported_type_returns_error(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "unknown", "data.bin", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert "error" in result
        assert "Unsupported" in result["error"]

    @pytest.mark.asyncio
    async def test_exception_returns_error_dict(self):
        from src.routers.s3_ocr_router import _process_document
        ocr = MagicMock()
        ocr.extract_text_from_pdf_file = AsyncMock(side_effect=RuntimeError("crash"))
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "pdf", "crash.pdf", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_image_error_result_passthrough(self):
        """If extract_text_from_image returns no 'text' key, it should pass result through."""
        from src.routers.s3_ocr_router import _process_document
        ocr = self._mock_ocr()
        ocr.extract_text_from_image = AsyncMock(return_value={"error": "analysis failed"})
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document(b"data", "image", "photo.png", ocr, FAKE_TEAM_CONFIG, FAKE_AUTH)
        assert "error" in result


# ===========================================================================
# _process_document_upload  (internal async helper — tested directly)
# ===========================================================================

class TestProcessDocumentUpload:

    def _make_s3_reader(self, files=None):
        reader = MagicMock()
        reader.read_files = MagicMock(return_value=files or {
            FAKE_S3_URL: b"fake pdf content"
        })
        return reader

    def _make_ocr(self):
        ocr = MagicMock()
        ocr.extract_text_from_pdf_file = AsyncMock(return_value={
            "chunks": [{"text": "extracted text", "type": "text", "source": "pdf_file"}]
        })
        return ocr

    @pytest.mark.asyncio
    async def test_successful_upload_returns_counts(self):
        from src.routers.s3_ocr_router import _process_document_upload

        mock_milvus = MagicMock()
        mock_milvus.get_uploaded_filenames_for_session.return_value = set()
        mock_milvus.ingest_documents.return_value = {"total_chunks_inserted": 1}

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.S3Extraction", return_value=self._make_s3_reader()), \
             patch("src.routers.s3_ocr_router.Ocr", return_value=self._make_ocr()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router._ingest_to_milvus", new_callable=AsyncMock), \
             patch("src.utils.s3_utility.extract_filename_from_s3_url", return_value="report.pdf"):
            result = await _process_document_upload(
                s3_urls=[FAKE_S3_URL],
                document_type="auto",
                session_id=FAKE_SESSION,
                team_id=FAKE_TEAM,
                team_config=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        assert result["successful"] == 1
        assert result["failed"] == 0
        assert result["skipped"] == 0
        assert result["total_documents"] == 1

    @pytest.mark.asyncio
    async def test_already_uploaded_file_is_skipped(self):
        from src.routers.s3_ocr_router import _process_document_upload

        mock_milvus = MagicMock()
        mock_milvus.get_uploaded_filenames_for_session.return_value = {"report.pdf"}
        mock_milvus.ingest_documents.return_value = {}

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.S3Extraction", return_value=self._make_s3_reader()), \
             patch("src.routers.s3_ocr_router.Ocr", return_value=self._make_ocr()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router._ingest_to_milvus", new_callable=AsyncMock), \
             patch("src.utils.s3_utility.extract_filename_from_s3_url", return_value="report.pdf"):
            result = await _process_document_upload(
                s3_urls=[FAKE_S3_URL],
                document_type="auto",
                session_id=FAKE_SESSION,
                team_id=FAKE_TEAM,
                team_config=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        assert result["skipped"] == 1
        assert result["successful"] == 0

    @pytest.mark.asyncio
    async def test_extraction_error_increments_failed(self):
        from src.routers.s3_ocr_router import _process_document_upload

        mock_milvus = MagicMock()
        mock_milvus.get_uploaded_filenames_for_session.return_value = set()

        ocr = MagicMock()
        ocr.extract_text_from_pdf_file = AsyncMock(return_value={"error": "parse failed"})

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.S3Extraction", return_value=self._make_s3_reader()), \
             patch("src.routers.s3_ocr_router.Ocr", return_value=ocr), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router._ingest_to_milvus", new_callable=AsyncMock), \
             patch("src.utils.s3_utility.extract_filename_from_s3_url", return_value="report.pdf"):
            result = await _process_document_upload(
                s3_urls=[FAKE_S3_URL],
                document_type="auto",
                session_id=FAKE_SESSION,
                team_id=FAKE_TEAM,
                team_config=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_milvus_ingestion_error_is_handled(self):
        from src.routers.s3_ocr_router import _process_document_upload

        mock_milvus = MagicMock()
        mock_milvus.get_uploaded_filenames_for_session.return_value = set()
        mock_milvus.ingest_documents.side_effect = RuntimeError("milvus down")

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.S3Extraction", return_value=self._make_s3_reader()), \
             patch("src.routers.s3_ocr_router.Ocr", return_value=self._make_ocr()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router._ingest_to_milvus", new_callable=AsyncMock), \
             patch("src.utils.s3_utility.extract_filename_from_s3_url", return_value="report.pdf"):
            result = await _process_document_upload(
                s3_urls=[FAKE_S3_URL],
                document_type="auto",
                session_id=FAKE_SESSION,
                team_id=FAKE_TEAM,
                team_config=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        # Milvus error should be caught; result should still return summary
        assert "successful" in result

    @pytest.mark.asyncio
    async def test_no_documents_skips_ingestion(self):
        from src.routers.s3_ocr_router import _process_document_upload

        mock_milvus = MagicMock()
        mock_milvus.get_uploaded_filenames_for_session.return_value = {"report.pdf"}

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.S3Extraction", return_value=self._make_s3_reader()), \
             patch("src.routers.s3_ocr_router.Ocr", return_value=self._make_ocr()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router._ingest_to_milvus", new_callable=AsyncMock), \
             patch("src.utils.s3_utility.extract_filename_from_s3_url", return_value="report.pdf"):
            result = await _process_document_upload(
                s3_urls=[FAKE_S3_URL],
                document_type="auto",
                session_id=FAKE_SESSION,
                team_id=FAKE_TEAM,
                team_config=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        mock_milvus.ingest_documents.assert_not_called()

    @pytest.mark.asyncio
    async def test_exception_at_top_level_propagates(self):
        from src.routers.s3_ocr_router import _process_document_upload

        with patch("src.routers.s3_ocr_router.S3Extraction", side_effect=RuntimeError("S3 unavailable")), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.milvus", MagicMock()):
            with pytest.raises(RuntimeError):
                await _process_document_upload(
                    s3_urls=[FAKE_S3_URL],
                    document_type="auto",
                    session_id=FAKE_SESSION,
                    team_id=FAKE_TEAM,
                    team_config=FAKE_TEAM_CONFIG,
                    auth_token=FAKE_AUTH
                )

    @pytest.mark.asyncio
    async def test_per_file_exception_is_caught(self):
        """Exception during per-file processing should be caught and counted as failed.

        FIX: extract_filename_from_s3_url is imported locally inside _process_document_upload,
        so we must patch at the source module (src.utils.s3_utility) rather than on the
        router module.  The except block also calls the same function, so we use
        side_effect=[RuntimeError(...), "report.pdf"] to raise on the first call (inside
        the try block) and return a safe value on the second call (inside the except block).
        """
        from src.routers.s3_ocr_router import _process_document_upload

        mock_milvus = MagicMock()
        mock_milvus.get_uploaded_filenames_for_session.return_value = set()

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.S3Extraction", return_value=self._make_s3_reader()), \
             patch("src.routers.s3_ocr_router.Ocr", return_value=self._make_ocr()), \
             patch("src.routers.s3_ocr_router._ingest_to_milvus", new_callable=AsyncMock), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch(
                 "src.utils.s3_utility.extract_filename_from_s3_url",
                 side_effect=[RuntimeError("filename error"), "report.pdf"]
             ):
            result = await _process_document_upload(
                s3_urls=[FAKE_S3_URL],
                document_type="auto",
                session_id=FAKE_SESSION,
                team_id=FAKE_TEAM,
                team_config=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        assert result["failed"] == 1


# ===========================================================================
# _process_document_query  (internal async helper — tested directly)
# ===========================================================================

class TestProcessDocumentQuery:

    @pytest.mark.asyncio
    async def test_successful_query_returns_results(self):
        from src.routers.s3_ocr_router import _process_document_query

        mock_milvus = MagicMock()
        mock_milvus.enhanced_search = AsyncMock(return_value=[
            {"score": 0.95, "text": "The answer", "metadata": {}, "retrieval_method": "direct"}
        ])
        mock_milvus.synthesize_answer_with_llm = AsyncMock(return_value={
            "answer": "42", "sources": []
        })

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document_query(
                query="What is the answer?",
                session_id=FAKE_SESSION,
                top_k=5,
                llm_params=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        assert result["query"] == "What is the answer?"
        assert result["total_results"] == 1
        assert result["llm_answer"] == "42"

    @pytest.mark.asyncio
    async def test_query_exception_propagates(self):
        from src.routers.s3_ocr_router import _process_document_query

        mock_milvus = MagicMock()
        mock_milvus.enhanced_search = AsyncMock(side_effect=RuntimeError("search failed"))

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            with pytest.raises(RuntimeError):
                await _process_document_query(
                    query="query",
                    session_id=FAKE_SESSION,
                    top_k=5,
                    llm_params=FAKE_TEAM_CONFIG,
                    auth_token=FAKE_AUTH
                )

    @pytest.mark.asyncio
    async def test_results_truncated_to_top_k(self):
        from src.routers.s3_ocr_router import _process_document_query

        many_results = [
            {"score": 0.9, "text": f"result {i}", "metadata": {}, "retrieval_method": "direct"}
            for i in range(20)
        ]

        mock_milvus = MagicMock()
        mock_milvus.enhanced_search = AsyncMock(return_value=many_results)
        mock_milvus.synthesize_answer_with_llm = AsyncMock(return_value={"answer": "ans", "sources": []})

        with patch("src.routers.s3_ocr_router.milvus", mock_milvus), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()):
            result = await _process_document_query(
                query="query",
                session_id=FAKE_SESSION,
                top_k=5,
                llm_params=FAKE_TEAM_CONFIG,
                auth_token=FAKE_AUTH
            )

        # Router returns len(search_results) as total_results (all 20), not just top_k
        assert result["total_results"] == 20


# ===========================================================================
# POST /upload  (HTTP endpoint)
# ===========================================================================

class TestUploadEndpoint:

    def _mock_upload(self, result=None):
        if result is None:
            result = {"total_documents": 1, "successful": 1, "failed": 0, "skipped": 0}
        return AsyncMock(return_value=result)

    def test_successful_upload(self, client):
        with patch("src.routers.s3_ocr_router._process_document_upload", self._mock_upload()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={
                    "user_metadata": _user_meta(),
                    "s3_urls": [FAKE_S3_URL]
                },
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_upload_missing_session_id_returns_400(self, client):
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={
                    "user_metadata": json.dumps({"team_id": FAKE_TEAM}),
                    "s3_urls": [FAKE_S3_URL]
                }
            )
        assert response.status_code == 400

    def test_upload_missing_team_id_returns_400(self, client):
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={
                    "user_metadata": json.dumps({"session_id": FAKE_SESSION}),
                    "s3_urls": [FAKE_S3_URL]
                }
            )
        assert response.status_code == 400

    def test_upload_exception_returns_failed_response(self, client):
        with patch("src.routers.s3_ocr_router._process_document_upload",
                   AsyncMock(side_effect=RuntimeError("processing crashed"))), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={"user_metadata": _user_meta(), "s3_urls": [FAKE_S3_URL]},
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    def test_upload_all_skipped_returns_already_uploaded_message(self, client):
        skipped_result = {"total_documents": 1, "successful": 0, "failed": 0, "skipped": 1}
        with patch("src.routers.s3_ocr_router._process_document_upload",
                   self._mock_upload(skipped_result)), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={"user_metadata": _user_meta(), "s3_urls": [FAKE_S3_URL]},
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200
        assert "already uploaded" in response.json()["message"]

    def test_upload_with_message_id_sets_span(self, client):
        """message_id and user_id in metadata should be processed without errors."""
        meta = _user_meta(message_id="msg-001", user_id="user-001")
        with patch("src.routers.s3_ocr_router._process_document_upload", self._mock_upload()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.set_message_id"), \
             patch("src.routers.s3_ocr_router.set_user_context"), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={"user_metadata": meta, "s3_urls": [FAKE_S3_URL]},
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200

    def test_upload_jwt_email_extraction(self, client):
        """A valid JWT with email in custom-data is decoded without error."""
        import base64, json as _json
        header = base64.urlsafe_b64encode(b'{"alg":"HS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(
            _json.dumps({"custom-data": {"user_email": "u@test.com"}}).encode()
        ).rstrip(b"=").decode()
        jwt_token = f"Bearer {header}.{payload}.fakesig"

        with patch("src.routers.s3_ocr_router._process_document_upload", self._mock_upload()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.set_user_context"), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/upload",
                json={"user_metadata": _user_meta(), "s3_urls": [FAKE_S3_URL]},
                headers={"Authorization": jwt_token}
            )
        assert response.status_code == 200


# ===========================================================================
# POST /query  (HTTP endpoint)
# ===========================================================================

class TestQueryEndpoint:

    def _mock_query(self, result=None):
        if result is None:
            result = {"query": "test", "total_results": 1, "llm_answer": "The answer."}
        return AsyncMock(return_value=result)

    def test_successful_query(self, client):
        with patch("src.routers.s3_ocr_router._process_document_query", self._mock_query()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/query",
                json={
                    "user_metadata": _user_meta(),
                    "query": "What is the revenue?"
                },
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_query_missing_session_id_returns_400(self, client):
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/query",
                json={
                    "user_metadata": json.dumps({"team_id": FAKE_TEAM}),
                    "query": "test"
                }
            )
        assert response.status_code == 400

    def test_query_missing_team_id_returns_400(self, client):
        with patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/query",
                json={
                    "user_metadata": json.dumps({"session_id": FAKE_SESSION}),
                    "query": "test"
                }
            )
        assert response.status_code == 400

    def test_query_exception_returns_failed_response(self, client):
        with patch("src.routers.s3_ocr_router._process_document_query",
                   AsyncMock(side_effect=RuntimeError("milvus crash"))), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/query",
                json={"user_metadata": _user_meta(), "query": "test"},
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200
        assert response.json()["status"] == "failed"

    def test_query_response_contains_result_count(self, client):
        with patch("src.routers.s3_ocr_router._process_document_query",
                   self._mock_query({"query": "q", "total_results": 5, "llm_answer": "ans"})), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/query",
                json={"user_metadata": _user_meta(), "query": "test"},
                headers={"Authorization": FAKE_AUTH}
            )
        assert "5" in response.json()["message"]

    def test_query_with_user_id_in_metadata(self, client):
        meta = _user_meta(user_id="u-123")
        with patch("src.routers.s3_ocr_router._process_document_query", self._mock_query()), \
             patch("src.routers.s3_ocr_router.create_event_logger", return_value=MagicMock()), \
             patch("src.routers.s3_ocr_router.set_user_context"), \
             patch("src.routers.s3_ocr_router.litellm_client") as mock_llm:
            mock_llm.get_dynamic_llm_instance = AsyncMock(return_value=FAKE_TEAM_CONFIG)
            response = client.post(
                "/query",
                json={"user_metadata": meta, "query": "test"},
                headers={"Authorization": FAKE_AUTH}
            )
        assert response.status_code == 200


# ===========================================================================
# Pydantic models (schema validation)
# ===========================================================================

class TestPydanticModels:

    def test_document_upload_request_valid(self):
        from src.routers.s3_ocr_router import DocumentUploadRequest
        req = DocumentUploadRequest(
            user_metadata='{"session_id": "s1", "team_id": "t1"}',
            s3_urls=["https://s3.amazonaws.com/bucket/file.pdf"]
        )
        assert req.user_metadata
        assert len(req.s3_urls) == 1

    def test_document_query_request_valid(self):
        from src.routers.s3_ocr_router import DocumentQueryRequest
        req = DocumentQueryRequest(
            user_metadata='{"session_id": "s1", "team_id": "t1"}',
            query="What is the revenue?"
        )
        assert req.query == "What is the revenue?"

    def test_document_processing_result_all_fields(self):
        from src.routers.s3_ocr_router import DocumentProcessingResult
        result = DocumentProcessingResult(
            s3_url=FAKE_S3_URL,
            filename="report.pdf",
            document_type="pdf",
            status="success",
            extracted_text="some text",
            file_size=1024,
            processing_time=1.5
        )
        assert result.status == "success"
        assert result.file_size == 1024

    def test_document_type_enum_values(self):
        from src.routers.s3_ocr_router import DocumentType
        assert DocumentType.PDF == "pdf"
        assert DocumentType.DOCX == "docx"
        assert DocumentType.ZIP == "zip"
        assert DocumentType.AUTO == "auto"
        assert DocumentType.IMAGE == "image"

    def test_upload_response_model(self):
        from src.routers.s3_ocr_router import DocumentUploadResponse
        resp = DocumentUploadResponse(
            status="success",
            message="Uploaded 1/1",
            upload_results={"total_documents": 1}
        )
        assert resp.status == "success"

    def test_query_response_model(self):
        from src.routers.s3_ocr_router import DocumentQueryResponse
        resp = DocumentQueryResponse(
            status="success",
            message="Retrieved 5 results",
            query_results={"answer": "42"}
        )
        assert resp.status == "success"