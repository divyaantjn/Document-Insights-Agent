"""
Root conftest.py — shared fixtures, mocks, and pytest configuration
for the entire test suite.
"""

import os
import sys
import json
import base64
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

@pytest.fixture(autouse=True)
def reset_semaphore(request):
    instance = getattr(request.instance, "ocr", None)
    if instance is not None:
        instance._image_semaphore = asyncio.Semaphore(5)

# ---------------------------------------------------------------------------
# Make the project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Pytest asyncio configuration
# ---------------------------------------------------------------------------
pytest_plugins = ["pytest_asyncio"]


# ---------------------------------------------------------------------------
# Environment variable stubs (prevent real network / config look-ups)
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Inject minimal env vars so modules can be imported without real infra."""
    env = {
        "MILVUS_CONFIG_PATH": "config/milvus_config.yaml",
        "KEYCLOAK_ISSUER": "https://keycloak.example.com/realms",
        "KEYCLOAK_CLIENT_ID": "test-client",
        "GOOGLE_API_KEY": "fake-google-api-key",
        "AWS_ACCESS_KEY_ID": "fake-access-key",
        "AWS_SECRET_ACCESS_KEY": "fake-secret-key",
        "AWS_DEFAULT_REGION": "us-east-1",
        "S3_BUCKET_NAME": "test-bucket",
    }
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------------------
# Shared sample data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_jwt_payload():
    return {
        "iss": "https://keycloak.example.com/realms/test",
        "sub": "user-123",
        "preferred_username": "testuser",
        "azp": "test-client",
        "resource_access": {
            "test-client": {"roles": ["test-client_client"]}
        },
        "email": "testuser@example.com",
    }


@pytest.fixture
def sample_jwt_token(sample_jwt_payload):
    """Build a *structurally valid* (3-part) JWT whose payload is base64url-encoded."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "RS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    payload_str = base64.urlsafe_b64encode(
        json.dumps(sample_jwt_payload).encode()
    ).rstrip(b"=").decode()
    signature = "fakesignature"
    return f"{header}.{payload_str}.{signature}"


@pytest.fixture
def valid_auth_header(sample_jwt_token):
    return f"Bearer {sample_jwt_token}"


@pytest.fixture
def sample_s3_urls():
    return [
        "https://s3.amazonaws.com/test-bucket/test-doc.pdf",
        "https://s3.amazonaws.com/test-bucket/test-image.png",
    ]


@pytest.fixture
def sample_file_content():
    """Return minimal bytes that look like a real file."""
    return b"%PDF-1.4 fake pdf content for testing purposes"


@pytest.fixture
def sample_text_bytes():
    return b"Hello, this is sample text content for unit testing."


@pytest.fixture
def sample_csv_bytes():
    return b"Name,Age,City\nAlice,30,NYC\nBob,25,LA\n"


@pytest.fixture
def sample_langchain_documents():
    """Return a list of mock LangchainDocument objects."""
    from unittest.mock import MagicMock
    docs = []
    for i in range(3):
        doc = MagicMock()
        doc.page_content = f"Sample document content {i}"
        doc.metadata = {"source": f"test_file_{i}.txt", "page": i + 1}
        docs.append(doc)
    return docs


@pytest.fixture
def mock_litellm_client():
    client = MagicMock()
    client.generate_response = AsyncMock(return_value="Mocked LLM response")
    client.get_dynamic_llm_instance = AsyncMock(
        return_value={
            "model": "openai/gpt-4o",
            "temperature": 0.0,
            "max_tokens": 1000,
        }
    )
    return client


@pytest.fixture
def mock_milvus_vector_store():
    store = MagicMock()
    store.get_uploaded_filenames_for_session = MagicMock(return_value=set())
    store.ingest_documents = MagicMock(
        return_value={
            "total_documents": 1,
            "successful_documents": 1,
            "failed_documents": 0,
            "total_chunks_inserted": 5,
            "failed_doc_details": [],
        }
    )
    store.enhanced_search = AsyncMock(
        return_value=[
            {
                "score": 0.95,
                "id": "chunk-1",
                "text": "Relevant content",
                "metadata": {"filename": "test.pdf", "page": 1},
                "session_id": "sess-abc",
                "team_id": "team-xyz",
                "created_at": "2025-01-01T00:00:00",
                "retrieval_method": "direct",
            }
        ]
    )
    store.synthesize_answer_with_llm = AsyncMock(
        return_value={
            "answer": "The answer is 42.",
            "sources": [{"source": "test.pdf", "page": 1, "score": 0.95}],
            "num_chunks_used": 1,
            "query": "test query",
        }
    )
    store.get_collection_stats = MagicMock(
        return_value={
            "collection_name": "idp_collection",
            "database_name": "idp_db",
            "num_entities": 100,
            "embedding_model": "gemini-embedding-001",
            "embedding_dim": 768,
        }
    )
    return store


@pytest.fixture
def mock_ocr_processor():
    ocr = MagicMock()
    ocr.extract_text_from_pdf_file = AsyncMock(
        return_value={"chunks": [{"text": "PDF text", "type": "text", "source": "pdf_file", "page": 1}]}
    )
    ocr.extract_text_from_docx = AsyncMock(
        return_value={"chunks": [{"text": "DOCX text", "type": "text", "source": "docx_file", "page": 1}]}
    )
    ocr.extract_text_from_excel = AsyncMock(
        return_value={"chunks": [{"text": "| A | B |\n|---|---|\n| 1 | 2 |", "type": "table", "source": "sheet1"}]}
    )
    ocr.extract_text_from_ppt = AsyncMock(
        return_value={"chunks": [{"text": "PPT text", "type": "text", "source": "ppt_file", "slide_number": 1}]}
    )
    ocr.extract_text_from_csv = AsyncMock(
        return_value={"chunks": [{"text": "| Name | Age |\n|---|---|\n| Alice | 30 |", "type": "table"}]}
    )
    ocr.extract_text_from_text_file = AsyncMock(return_value="Plain text content")
    ocr.extract_text_from_image = AsyncMock(
        return_value={"text": "[Image Content Analysis - Image]:\nVisual Description:\nA chart."}
    )
    ocr.extract_text_from_zip = AsyncMock(
        return_value={"inner.txt": [{"text": "Zip file content", "type": "text"}]}
    )
    return ocr


@pytest.fixture
def mock_s3_extraction():
    s3 = MagicMock()
    s3.read_files = AsyncMock(
        return_value={
            "https://s3.amazonaws.com/test-bucket/test.pdf": b"fake pdf bytes"
        }
    )
    return s3


@pytest.fixture
def mock_event_logger():
    logger = MagicMock()
    logger.log_event = MagicMock()
    logger.log_error = MagicMock()
    return logger

@pytest.fixture(autouse=True)
def _mock_heavy_modules_for_router_tests(request):
    """
    Patches sys.modules for the heavy singletons (MilvusVectorStore, LitellmClient)
    ONLY when running tests inside tests/routers/.
    This prevents the hang caused by real network calls at import time,
    without polluting sys.modules for every other test module.
    """
    # Only activate for tests whose node id starts with "tests/routers/"
    # Adjust the path separator if you're on Windows: "tests\\routers\\"
    node_id = request.node.nodeid
    if not (
        node_id.startswith("tests/routers/") or
        node_id.startswith("tests\\routers\\")
    ):
        yield          # do nothing for non-router tests
        return

    # --- build stubs ---
    mock_milvus_instance = MagicMock()
    mock_milvus_instance.get_uploaded_filenames_for_session.return_value = set()
    mock_milvus_instance.ingest_documents.return_value = {}
    mock_milvus_instance.enhanced_search = AsyncMock(return_value=[])
    mock_milvus_instance.synthesize_answer_with_llm = AsyncMock(
        return_value={"answer": "", "sources": []}
    )
    mock_milvus_instance.get_collection_stats.return_value = {}

    mock_milvus_module = MagicMock()
    mock_milvus_module.MilvusVectorStore.return_value = mock_milvus_instance

    mock_litellm_instance = MagicMock()
    mock_litellm_instance.get_dynamic_llm_instance = AsyncMock(return_value={})

    mock_litellm_module = MagicMock()
    mock_litellm_module.LitellmClient.return_value = mock_litellm_instance

    # --- patch sys.modules BEFORE the router module is imported ---
    original_milvus   = sys.modules.get("src.database.milvus_db")
    original_litellm  = sys.modules.get("src.llm.litellm_client")
    # Also evict any already-imported real router module so a fresh import
    # picks up the mocked dependencies.
    original_router   = sys.modules.pop("src.routers.s3_ocr_router", None)

    sys.modules["src.database.milvus_db"]  = mock_milvus_module
    sys.modules["src.llm.litellm_client"]  = mock_litellm_module

    yield   # run the test

    # --- restore ---
    _restore(sys.modules, "src.database.milvus_db",   original_milvus)
    _restore(sys.modules, "src.llm.litellm_client",   original_litellm)
    _restore(sys.modules, "src.routers.s3_ocr_router", original_router)


def _restore(modules, key, original):
    if original is None:
        modules.pop(key, None)
    else:
        modules[key] = original