import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch, call

# ---------------------------------------------------------------------------
# Shared helpers (mirrors the existing test suite's helpers)
# ---------------------------------------------------------------------------

MINIMAL_YAML_CONFIG = {
    "milvus": {
        "host": "localhost",
        "port": "19530",
        "username": "root",
        "password": "milvus",
    },
    "collection": {
        "name": "test_collection",
        "description": "test collection",
    },
    "database": {"name": "test_db"},
    "embedding": {
        "model": "gemini-embedding-001",
        "dimension": 768,
        "task_type": "SEMANTIC_SIMILARITY",
        "output_dimensionality": 768,
    },
    "chunking": {"chunk_size": 500, "chunk_overlap": 50},
    "vector_search": {
        "metric_type": "IP",
        "index_type": "HNSW",
        "search_params": {"nprobe": 10},
        "index_params": {"nlist": 128},
    },
    "binary_vector_search": {
        "metric_type": "JACCARD",
        "index_type": "BIN_IVF_FLAT",
        "index_params": {"nlist": 128},
    },
    "features": {"enable_logging": True, "enable_error_fallback": True},
    "retrieval": {},
    "ingestion": {
        "batch_size": 10,
        "generate_ids": True,
        "max_chunk_limit": 100,
    },
    "api": {"default_top_k": 5, "max_query_limit": 1000},
    "logging": {"level": "INFO", "format": "%(message)s"},
    "field_schemas": [
        {"name": "id", "dtype": "VARCHAR", "max_length": 128, "is_primary": True},
        {"name": "session_id", "dtype": "VARCHAR", "max_length": 64},
        {"name": "team_id", "dtype": "VARCHAR", "max_length": 64},
        {"name": "text", "dtype": "VARCHAR", "max_length": 4096},
        {"name": "metadata", "dtype": "JSON"},
        {"name": "embedding", "dtype": "FLOAT_VECTOR", "dim": 768},
        {"name": "created_at", "dtype": "VARCHAR", "max_length": 64},
    ],
}


def _deep_get(d, dotted_key, default=None):
    keys = dotted_key.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


def _make_mock_config(extra: dict | None = None):
    cfg = dict(MINIMAL_YAML_CONFIG)
    if extra:
        cfg.update(extra)

    mock_cfg = MagicMock()
    mock_cfg.config = cfg
    mock_cfg.get_all = lambda section: cfg.get(section, {})
    mock_cfg.get = lambda key, default=None: _deep_get(cfg, key, default)
    return mock_cfg


def _build_vector_store():
    """Construct a MilvusVectorStore with all external calls patched out."""
    mock_cfg = _make_mock_config()

    with (
        patch("src.database.milvus_db.MilvusConfigManager", return_value=mock_cfg),
        patch("src.database.milvus_db.milvus_config", mock_cfg),
        patch("src.database.milvus_db.connections"),
        patch("src.database.milvus_db.db") as mock_db,
        patch("src.database.milvus_db.utility") as mock_utility,
        patch("src.database.milvus_db.Collection") as mock_collection_cls,
        patch("src.database.milvus_db.genai") as mock_genai,
        patch("src.database.milvus_db.LitellmClient") as mock_llm_cls,
        patch("src.database.milvus_db.RecursiveCharacterTextSplitter"),
        patch("src.database.milvus_db.FieldSchemaBuilder.build_schemas") as mock_build,
        patch("src.database.milvus_db.CollectionSchema"),
        patch("src.database.milvus_db.logging"),
    ):
        from pymilvus import DataType

        fs_id = MagicMock()
        fs_id.name = "id"
        fs_id.dtype = DataType.VARCHAR
        fs_embed = MagicMock()
        fs_embed.name = "embedding"
        fs_embed.dtype = DataType.FLOAT_VECTOR
        mock_build.return_value = [fs_id, fs_embed]

        mock_db.list_database.return_value = []
        mock_utility.has_collection.return_value = False

        mock_collection = MagicMock()
        mock_collection.num_entities = 42
        mock_collection_cls.return_value = mock_collection

        mock_genai_client = MagicMock()
        mock_genai.Client.return_value = mock_genai_client

        mock_llm = MagicMock()
        mock_llm_cls.return_value = mock_llm

        from src.database.milvus_db import MilvusVectorStore

        store = MilvusVectorStore.__new__(MilvusVectorStore)
        store.logger = MagicMock()
        store.milvus_host = "localhost"
        store.milvus_port = "19530"
        store.username = "root"
        store.password = "milvus"
        store.collection_name = "test_collection"
        store.database_name = "test_db"
        store.embedding_model = "gemini-embedding-001"
        store.embedding_dim = 768
        store.chunk_size = 500
        store.chunk_overlap = 50
        store.metric_type = "IP"
        store.index_type = "HNSW"
        store.nprobe = 10
        store.nlist = 128
        store.enable_logging = True
        store.enable_error_fallback = True
        store.enable_hyde = True
        store.enable_multi_query = True
        store.num_multi_queries = 3
        store.collection = mock_collection
        store.client = mock_genai_client
        store.litellm_client = mock_llm
        store.text_splitter = MagicMock()
        store.enhanced_retrieval = MagicMock()

    return store, mock_collection, mock_genai_client, mock_llm


# ===========================================================================
# 1. MilvusConfigManager._initialize — FileNotFoundError when config absent
# ===========================================================================

class TestMilvusConfigManagerInitialize:

    def test_initialize_raises_file_not_found(self, tmp_path):
        """Covers the `raise FileNotFoundError` branch when config file is missing."""
        import src.database.milvus_db as mod

        missing_path = str(tmp_path / "nonexistent_config.yaml")

        original_instance = mod.MilvusConfigManager._instance
        mod.MilvusConfigManager._instance = None

        try:
            with (
                patch.dict("os.environ", {"MILVUS_CONFIG_PATH": missing_path}),
                patch("src.database.milvus_db.os.path.exists", return_value=False),
            ):
                with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                    mod.MilvusConfigManager()
        finally:
            mod.MilvusConfigManager._instance = original_instance

    def test_initialize_loads_valid_yaml(self, tmp_path):
        """Covers the happy path: file exists, YAML is parsed correctly."""
        import yaml, src.database.milvus_db as mod

        config_data = {
            "milvus": {"host": "remote-host", "port": "19530"},
            "logging": {"level": "DEBUG"},
        }
        config_file = tmp_path / "milvus_config.yaml"
        config_file.write_text(yaml.dump(config_data))

        original_instance = mod.MilvusConfigManager._instance
        mod.MilvusConfigManager._instance = None

        try:
            with patch.dict("os.environ", {"MILVUS_CONFIG_PATH": str(config_file)}):
                mgr = mod.MilvusConfigManager()
                assert mgr.config["milvus"]["host"] == "remote-host"
        finally:
            mod.MilvusConfigManager._instance = original_instance


# ===========================================================================
# 2. MilvusConfigManager.get — non-dict intermediate value returns default
# ===========================================================================

class TestMilvusConfigManagerGet:

    def _get_fresh_manager(self, config_data: dict):
        """Build a MilvusConfigManager with a given config dict."""
        import src.database.milvus_db as mod

        original_instance = mod.MilvusConfigManager._instance
        mod.MilvusConfigManager._instance = None

        import yaml
        import tempfile, os

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            tmp_path = f.name

        try:
            with patch.dict("os.environ", {"MILVUS_CONFIG_PATH": tmp_path}):
                mgr = mod.MilvusConfigManager()
        finally:
            os.unlink(tmp_path)
            mod.MilvusConfigManager._instance = original_instance

        return mgr

    def test_get_returns_default_when_intermediate_is_not_dict(self):
        """
        Covers: `else: return default` when a mid-path value is a non-dict scalar.
        e.g. config = {"a": "scalar"} and key = "a.b" → should return default.
        """
        mgr = self._get_fresh_manager({"a": "scalar_value", "logging": {"level": "INFO"}})
        result = mgr.get("a.b", default="fallback")
        assert result == "fallback"

    def test_get_returns_none_when_key_missing_and_no_default(self):
        """Covers: key not present, default=None returned."""
        mgr = self._get_fresh_manager({"top": {"nested": "val"}, "logging": {"level": "INFO"}})
        result = mgr.get("top.missing_key")
        assert result is None

    def test_get_returns_value_for_existing_dotted_key(self):
        """Covers happy path: dotted key resolves correctly."""
        mgr = self._get_fresh_manager(
            {"section": {"subsection": {"value": 42}}, "logging": {"level": "INFO"}}
        )
        result = mgr.get("section.subsection.value", default=0)
        assert result == 42


# ===========================================================================
# 3. FieldSchemaBuilder.build_schemas — skip field when name or dtype missing
# ===========================================================================

class TestFieldSchemaBuilderMissingNameDtype:

    def test_field_missing_name_is_skipped(self):
        """Covers: `if not name or not dtype_str: continue`"""
        from src.database.milvus_db import FieldSchemaBuilder

        fields_config = [
            {"dtype": "VARCHAR", "max_length": 64, "is_primary": True},  # no name
            {"name": "valid_id", "dtype": "VARCHAR", "max_length": 64, "is_primary": True},
        ]
        schemas = FieldSchemaBuilder.build_schemas(fields_config)
        names = [s.name for s in schemas]
        assert "valid_id" in names
        assert len(schemas) == 1

    def test_field_missing_dtype_is_skipped(self):
        """Covers: `if not name or not dtype_str: continue`"""
        from src.database.milvus_db import FieldSchemaBuilder

        fields_config = [
            {"name": "no_dtype_field"},  # no dtype
            {"name": "pk", "dtype": "INT64", "is_primary": True},
        ]
        schemas = FieldSchemaBuilder.build_schemas(fields_config)
        names = [s.name for s in schemas]
        assert "pk" in names
        assert "no_dtype_field" not in names

    def test_no_primary_key_raises_value_error(self):
        """Covers the `raise ValueError` when no field is marked is_primary."""
        from src.database.milvus_db import FieldSchemaBuilder

        fields_config = [
            {"name": "field_a", "dtype": "VARCHAR", "max_length": 64},
            {"name": "field_b", "dtype": "INT64"},
        ]
        with pytest.raises(ValueError, match="No primary key field found"):
            FieldSchemaBuilder.build_schemas(fields_config)


# ===========================================================================
# 4. EnhancedRetrieval.generate_hypothetical_document
# ===========================================================================

class TestEnhancedRetrievalHyDE:

    def _make_retrieval(self):
        mock_llm = MagicMock()
        mock_logger = MagicMock()
        from src.database.milvus_db import EnhancedRetrieval
        return EnhancedRetrieval(mock_llm, mock_logger), mock_llm, mock_logger

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_happy_path(self):
        """Covers the successful LLM call and return of hypothetical doc."""
        er, mock_llm, mock_logger = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(return_value="A hypothetical document about X.")

        result = await er.generate_hypothetical_document(
            "What is X?", {"model": "gpt-4"}, "auth-token"
        )

        assert result == "A hypothetical document about X."
        mock_llm.generate_response.assert_called_once()
        call_args = mock_llm.generate_response.call_args
        # Verify the prompt contains the query
        messages = call_args[0][1]
        assert any("What is X?" in m["content"] for m in messages)
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_generate_hypothetical_document_falls_back_to_query_on_error(self):
        """Covers: `except Exception` → returns original query as fallback."""
        er, mock_llm, mock_logger = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(side_effect=Exception("LLM timeout"))

        result = await er.generate_hypothetical_document(
            "fallback query", {}, "token"
        )

        assert result == "fallback query"
        mock_logger.error.assert_called()
        assert "LLM timeout" in str(mock_logger.error.call_args)


# ===========================================================================
# 5. EnhancedRetrieval.generate_multi_queries
# ===========================================================================

class TestEnhancedRetrievalMultiQuery:

    def _make_retrieval(self):
        mock_llm = MagicMock()
        mock_logger = MagicMock()
        from src.database.milvus_db import EnhancedRetrieval
        return EnhancedRetrieval(mock_llm, mock_logger), mock_llm, mock_logger

    @pytest.mark.asyncio
    async def test_generate_multi_queries_happy_path(self):
        """Covers successful LLM response parsing into variation list."""
        er, mock_llm, mock_logger = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(
            return_value="1. How does X work?\n2. What is the mechanism of X?\n3. Explain X in detail"
        )

        results = await er.generate_multi_queries("Tell me about X", {}, "token", num_queries=3)

        assert len(results) == 3
        assert "How does X work?" in results
        mock_logger.info.assert_called()

    @pytest.mark.asyncio
    async def test_generate_multi_queries_strips_numbered_prefixes(self):
        """Covers prefix stripping for '1. ', '2. ', etc."""
        er, mock_llm, _ = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(
            return_value="1. First variation\n2. Second variation\n3. Third variation"
        )

        results = await er.generate_multi_queries("Query", {}, "token", num_queries=3)

        assert all(not r.startswith(("1.", "2.", "3.")) for r in results)
        assert "First variation" in results

    @pytest.mark.asyncio
    async def test_generate_multi_queries_strips_dash_prefix(self):
        """Covers prefix stripping for '- '."""
        er, mock_llm, _ = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(
            return_value="- First variation\n- Second variation\n- Third variation"
        )

        results = await er.generate_multi_queries("Query", {}, "token", num_queries=3)

        assert all(not r.startswith("-") for r in results)

    @pytest.mark.asyncio
    async def test_generate_multi_queries_strips_star_prefix(self):
        """Covers prefix stripping for '* '."""
        er, mock_llm, _ = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(
            return_value="* Alpha query\n* Beta query\n* Gamma query"
        )

        results = await er.generate_multi_queries("Query", {}, "token", num_queries=3)

        assert all(not r.startswith("*") for r in results)
        assert "Alpha query" in results

    @pytest.mark.asyncio
    async def test_generate_multi_queries_respects_num_queries_limit(self):
        """Covers `result = clean_variations[:num_queries]`."""
        er, mock_llm, _ = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(
            return_value="1. Q1 variation text here\n2. Q2 variation text here\n3. Q3 variation text here\n4. Q4 variation text here\n5. Q5 variation text here"
        )

        results = await er.generate_multi_queries("Query", {}, "token", num_queries=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_generate_multi_queries_filters_short_lines(self):
        """Covers: `if cleaned and len(cleaned) > 5` — short lines are dropped."""
        er, mock_llm, _ = self._make_retrieval()
        # Lines shorter than 5 chars should be filtered out
        mock_llm.generate_response = AsyncMock(
            return_value="ok\nhi\nA proper long variation that passes the length check"
        )

        results = await er.generate_multi_queries("Query", {}, "token", num_queries=3)

        # Short lines dropped, only the long one survives
        assert len(results) == 1
        assert "A proper long variation" in results[0]

    @pytest.mark.asyncio
    async def test_generate_multi_queries_falls_back_on_error(self):
        """Covers: `except Exception` → returns [query] as fallback."""
        er, mock_llm, mock_logger = self._make_retrieval()
        mock_llm.generate_response = AsyncMock(side_effect=Exception("Network error"))

        results = await er.generate_multi_queries("original query", {}, "token")

        assert results == ["original query"]
        mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_generate_multi_queries_empty_response_falls_back(self):
        """Covers: empty clean_variations → fallback `[query]`."""
        er, mock_llm, _ = self._make_retrieval()
        # Response with only very short lines that get filtered
        mock_llm.generate_response = AsyncMock(return_value="   \n  \n  ")

        results = await er.generate_multi_queries("fallback query", {}, "token")

        assert results == ["fallback query"]


# ===========================================================================
# 6. MilvusVectorStore.generate_embedding — error paths
# ===========================================================================

class TestGenerateEmbeddingErrorPaths:

    def test_embedding_error_returns_zero_vector_when_fallback_enabled(self):
        """Covers: `except Exception` → return `[0.0] * self.embedding_dim`."""
        s, _, mock_genai_client, _ = _build_vector_store()
        mock_genai_client.models.embed_content.side_effect = Exception("API down")
        s.enable_error_fallback = True

        mock_cfg = _make_mock_config()
        with (
            patch("src.database.milvus_db.milvus_config", mock_cfg),
            patch("src.database.milvus_db.types") as mock_types,
        ):
            mock_types.EmbedContentConfig.return_value = MagicMock()
            result = s.generate_embedding("test text")

        assert result == [0.0] * 768
        s.logger.error.assert_called()

    def test_embedding_error_reraises_when_fallback_disabled(self):
        """Covers: fallback disabled → exception propagates."""
        s, _, mock_genai_client, _ = _build_vector_store()
        mock_genai_client.models.embed_content.side_effect = Exception("API down")

        # Override the config to disable error fallback
        no_fallback_cfg = _make_mock_config()
        no_fallback_cfg.get = lambda key, default=None: (
            False if key == "features.enable_error_fallback" else _deep_get(MINIMAL_YAML_CONFIG, key, default)
        )

        with (
            patch("src.database.milvus_db.milvus_config", no_fallback_cfg),
            patch("src.database.milvus_db.types") as mock_types,
        ):
            mock_types.EmbedContentConfig.return_value = MagicMock()
            with pytest.raises(Exception, match="API down"):
                s.generate_embedding("test text")


# ===========================================================================
# 7. MilvusVectorStore.insert_data — error / re-raise path
# ===========================================================================

class TestInsertDataError:

    def test_insert_data_reraises_on_collection_insert_failure(self):
        """Covers: `except Exception as e: raise`."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.insert.side_effect = Exception("Insert failed")

        data = [["id1"], ["sess1"], ["team1"], ["txt"], [{}], [[0.1] * 768], ["ts"]]
        with pytest.raises(Exception, match="Insert failed"):
            s.insert_data(data)

        s.logger.error.assert_called()

    def test_insert_data_reraises_on_flush_failure(self):
        """Covers: flush raises → exception propagates."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.flush.side_effect = Exception("Flush error")

        data = [["id1"], ["sess1"], ["team1"], ["txt"], [{}], [[0.1] * 768], ["ts"]]
        with pytest.raises(Exception, match="Flush error"):
            s.insert_data(data)


# ===========================================================================
# 8. MilvusVectorStore.get_uploaded_filenames_for_session
# ===========================================================================

class TestGetUploadedFilenamesForSession:

    def test_returns_set_of_filenames_for_session(self):
        """Covers the happy path: metadata has filenames."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[
            {"metadata": {"filename": "report.pdf"}},
            {"metadata": {"filename": "invoice.xlsx"}},
            {"metadata": {"filename": "report.pdf"}},  # duplicate
        ])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_uploaded_filenames_for_session("sess-123")

        assert result == {"report.pdf", "invoice.xlsx"}
        s.logger.info.assert_called()

    def test_returns_empty_set_when_no_results(self):
        """Covers: no documents in session → empty set."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_uploaded_filenames_for_session("empty-sess")

        assert result == set()

    def test_skips_metadata_without_filename_key(self):
        """Covers: metadata dict present but no 'filename' key → skipped."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[
            {"metadata": {"source": "web", "page": 1}},  # no filename key
            {"metadata": {"filename": "doc.pdf"}},
        ])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_uploaded_filenames_for_session("sess-abc")

        assert result == {"doc.pdf"}

    def test_skips_non_dict_metadata(self):
        """Covers: metadata is not a dict → `isinstance(metadata, dict)` is False."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[
            {"metadata": "not_a_dict"},
            {"metadata": ["list_metadata"]},
            {"metadata": {"filename": "valid.pdf"}},
        ])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_uploaded_filenames_for_session("sess-xyz")

        assert result == {"valid.pdf"}

    def test_returns_empty_set_on_query_error(self):
        """Covers: `except Exception` → returns empty set as safe fallback."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(side_effect=Exception("Query crashed"))

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_uploaded_filenames_for_session("bad-sess")

        assert result == set()
        s.logger.error.assert_called()

    def test_passes_correct_filter_and_limit_to_query(self):
        """Covers: correct filter expression and limit are passed to self.query."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            s.get_uploaded_filenames_for_session("my-session")

        call_kwargs = s.query.call_args[1]
        assert call_kwargs["filter_expr"] == 'session_id == "my-session"'
        assert call_kwargs["limit"] == 100  # from config ingestion.max_chunk_limit


# ===========================================================================
# 9. MilvusVectorStore.enhanced_search — HyDE, multi-query, dedup, sort
# ===========================================================================

class TestEnhancedSearchBranches:

    def _store_with_mocked_search(self):
        s, _, _, _ = _build_vector_store()
        return s

    def _make_hit(self, hit_id, score, method=None):
        result = {"id": hit_id, "score": score, "text": f"text_{hit_id}", "metadata": {}}
        if method:
            result["retrieval_method"] = method
        return result

    @pytest.mark.asyncio
    async def test_enhanced_search_with_hyde_adds_unique_results(self):
        """Covers: HyDE branch adds new results not in direct search."""
        s = self._store_with_mocked_search()
        s.enhanced_retrieval.generate_hypothetical_document = AsyncMock(
            return_value="A hypothetical document."
        )

        direct_hits = [self._make_hit("d1", 0.9), self._make_hit("d2", 0.7)]
        hyde_hits = [self._make_hit("h1", 0.85), self._make_hit("d1", 0.8)]  # d1 is duplicate

        def mock_search_by_session(query, session_id, top_k=None, output_fields=None):
            if "hypothetical" in query.lower():
                return hyde_hits
            return direct_hits

        s.search_by_session = MagicMock(side_effect=mock_search_by_session)

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            results = await s.enhanced_search(
                "query", "sess", {}, "token",
                use_hyde=True, use_multi_query=False
            )

        ids = [r["id"] for r in results]
        assert "d1" in ids
        assert "d2" in ids
        assert "h1" in ids
        # d1 should appear only once (deduplication)
        assert ids.count("d1") == 1

    @pytest.mark.asyncio
    async def test_enhanced_search_with_multi_query_adds_unique_results(self):
        """Covers: multi-query branch with variation results."""
        s = self._store_with_mocked_search()
        s.enhanced_retrieval.generate_multi_queries = AsyncMock(
            return_value=["variation 1 query text", "variation 2 query text"]
        )

        direct_hits = [self._make_hit("d1", 0.9)]
        var1_hits = [self._make_hit("v1", 0.75)]
        var2_hits = [self._make_hit("v2", 0.65), self._make_hit("d1", 0.6)]

        call_count = {"n": 0}

        def mock_search(query, session_id, top_k=None, output_fields=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return direct_hits
            elif call_count["n"] == 2:
                return var1_hits
            return var2_hits

        s.search_by_session = MagicMock(side_effect=mock_search)

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            results = await s.enhanced_search(
                "query", "sess", {}, "token",
                use_hyde=False, use_multi_query=True
            )

        ids = [r["id"] for r in results]
        assert "d1" in ids
        assert "v1" in ids
        assert "v2" in ids
        assert ids.count("d1") == 1  # deduplication

    @pytest.mark.asyncio
    async def test_enhanced_search_results_sorted_by_score_descending(self):
        """Covers: `all_results.sort(key=lambda x: x['score'], reverse=True)`."""
        s = self._store_with_mocked_search()
        s.enhanced_retrieval.generate_hypothetical_document = AsyncMock(return_value="hypo")

        s.search_by_session = MagicMock(side_effect=[
            [self._make_hit("a", 0.5), self._make_hit("b", 0.9)],
            [self._make_hit("c", 0.7)],
        ])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            results = await s.enhanced_search(
                "q", "s", {}, "t", use_hyde=True, use_multi_query=False
            )

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_enhanced_search_hyde_disabled_by_flag(self):
        """Covers: `use_hyde=False` → HyDE branch is NOT executed."""
        s = self._store_with_mocked_search()
        s.search_by_session = MagicMock(return_value=[self._make_hit("d1", 0.9)])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            await s.enhanced_search(
                "q", "s", {}, "t", use_hyde=False, use_multi_query=False
            )

        s.enhanced_retrieval.generate_hypothetical_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_enhanced_search_multi_query_disabled_by_flag(self):
        """Covers: `use_multi_query=False` → multi-query branch NOT executed."""
        s = self._store_with_mocked_search()
        s.search_by_session = MagicMock(return_value=[self._make_hit("d1", 0.9)])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            await s.enhanced_search(
                "q", "s", {}, "t", use_hyde=False, use_multi_query=False
            )

        s.enhanced_retrieval.generate_multi_queries.assert_not_called()

    @pytest.mark.asyncio
    async def test_enhanced_search_uses_default_output_fields_when_none_provided(self):
        """Covers: `if output_fields is None:` branch."""
        s = self._store_with_mocked_search()
        s.search_by_session = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            await s.enhanced_search(
                "q", "s", {}, "t",
                use_hyde=False, use_multi_query=False,
                output_fields=None
            )

        call_kwargs = s.search_by_session.call_args[1]
        assert "text" in call_kwargs["output_fields"]
        assert "metadata" in call_kwargs["output_fields"]

    @pytest.mark.asyncio
    async def test_enhanced_search_uses_default_top_k_when_none_provided(self):
        """Covers: `if top_k is None: top_k = config.get(...)` branch."""
        s = self._store_with_mocked_search()
        s.search_by_session = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            await s.enhanced_search(
                "q", "s", {}, "t",
                use_hyde=False, use_multi_query=False,
                top_k=None
            )

        call_kwargs = s.search_by_session.call_args[1]
        assert call_kwargs["top_k"] == 5  # from MINIMAL_YAML_CONFIG api.default_top_k

    @pytest.mark.asyncio
    async def test_enhanced_search_returns_top_k_times_two(self):
        """Covers: `final_results = all_results[:top_k * 2]`."""
        s = self._store_with_mocked_search()
        # Return more hits than top_k * 2
        many_hits = [self._make_hit(f"id{i}", float(i) / 20) for i in range(20)]
        s.search_by_session = MagicMock(return_value=many_hits)

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            results = await s.enhanced_search(
                "q", "s", {}, "t",
                use_hyde=False, use_multi_query=False,
                top_k=3
            )

        assert len(results) == 6  # top_k * 2 = 3 * 2


# ===========================================================================
# 10. MilvusVectorStore.query — happy path, default limit, error path
# ===========================================================================

class TestQueryMethod:

    def test_query_happy_path_returns_collection_results(self):
        """Covers: successful collection.query call returns results."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.query.return_value = [{"id": "r1"}, {"id": "r2"}]

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            results = s.query(
                filter_expr='session_id == "sess1"',
                output_fields=["id", "text"],
                limit=50,
            )

        assert len(results) == 2
        mock_collection.query.assert_called_once_with(
            expr='session_id == "sess1"',
            output_fields=["id", "text"],
            limit=50,
        )

    def test_query_uses_default_limit_when_none_provided(self):
        """Covers: `if limit is None: limit = config.get(...)` branch."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.query.return_value = []

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            s.query(filter_expr='session_id == "x"')

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["limit"] == 1000  # from MINIMAL_YAML_CONFIG api.max_query_limit

    def test_query_uses_wildcard_output_fields_when_none_provided(self):
        """Covers: `output_fields=output_fields or ['*']`."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.query.return_value = []

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            s.query(filter_expr='session_id == "x"', output_fields=None)

        call_kwargs = mock_collection.query.call_args[1]
        assert call_kwargs["output_fields"] == ["*"]

    def test_query_reraises_on_collection_error(self):
        """Covers: `except Exception as e: raise`."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.query.side_effect = Exception("Query crashed")

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            with pytest.raises(Exception, match="Query crashed"):
                s.query(filter_expr='session_id == "x"')

        s.logger.error.assert_called()


# ===========================================================================
# 11. MilvusVectorStore.delete — happy path + error path
# ===========================================================================

class TestDeleteMethod:

    def test_delete_happy_path(self):
        """Covers: successful `collection.delete` and logger.info call."""
        s, mock_collection, _, _ = _build_vector_store()

        s.delete('session_id == "sess1"')

        mock_collection.delete.assert_called_once_with('session_id == "sess1"')
        s.logger.info.assert_called()

    def test_delete_reraises_on_collection_error(self):
        """Covers: `except Exception as e: raise`."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.delete.side_effect = Exception("Delete failed")

        with pytest.raises(Exception, match="Delete failed"):
            s.delete('session_id == "sess1"')

        s.logger.error.assert_called()


# ===========================================================================
# 12. MilvusVectorStore.drop_collection — happy path + error path
# ===========================================================================

class TestDropCollectionMethod:

    def test_drop_collection_happy_path(self):
        """Covers: successful `collection.drop` and logger.info call."""
        s, mock_collection, _, _ = _build_vector_store()

        s.drop_collection()

        mock_collection.drop.assert_called_once()
        s.logger.info.assert_called()
        log_msg = str(s.logger.info.call_args)
        assert "test_collection" in log_msg

    def test_drop_collection_reraises_on_error(self):
        """Covers: `except Exception as e: raise`."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.drop.side_effect = Exception("Drop failed")

        with pytest.raises(Exception, match="Drop failed"):
            s.drop_collection()

        s.logger.error.assert_called()


# ===========================================================================
# 13. MilvusVectorStore._determine_file_type — all extension branches
# ===========================================================================

class TestDetermineFileType:

    @pytest.fixture
    def store(self):
        s, _, _, _ = _build_vector_store()
        return s

    @pytest.mark.parametrize("filename,expected", [
        # doc / docx normalization
        ("report.doc", "docx"),
        ("report.docx", "docx"),
        ("REPORT.DOC", "docx"),
        # xls / xlsx normalization
        ("data.xls", "xlsx"),
        ("data.xlsx", "xlsx"),
        # ppt / pptx normalization
        ("slides.ppt", "pptx"),
        ("slides.pptx", "pptx"),
        # pdf
        ("document.pdf", "pdf"),
        ("document.PDF", "pdf"),
        # csv
        ("records.csv", "csv"),
        ("records.CSV", "csv"),
        # unknown extension
        ("archive.zip", "zip"),
        ("image.png", "png"),
        # no extension
        ("filename_no_ext", "unknown"),
    ])
    def test_determine_file_type_parametrized(self, store, filename, expected):
        assert store._determine_file_type(filename) == expected

    def test_determine_file_type_empty_string(self, store):
        """Edge case: empty filename → 'unknown'."""
        result = store._determine_file_type("")
        assert result == "unknown"

    def test_determine_file_type_dotfile_no_base(self, store):
        """Edge case: filename starting with dot like '.env' → 'env'."""
        result = store._determine_file_type(".env")
        assert result == "env"


# ===========================================================================
# 14. MilvusVectorStore.get_session_documents — happy path + metadata extraction
# ===========================================================================

class TestGetSessionDocumentsHappyPath:

    def test_happy_path_extracts_filenames_and_file_types(self):
        """Covers: filenames and file_types set population from metadata."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[
            {"metadata": {"filename": "report.pdf", "file_type": "pdf"}, "team_id": "team-1", "created_at": "ts"},
            {"metadata": {"filename": "data.xlsx", "file_type": "xlsx"}, "team_id": "team-1", "created_at": "ts"},
            {"metadata": {"filename": "report.pdf", "file_type": "pdf"}, "team_id": "team-1", "created_at": "ts"},  # dup
        ])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_session_documents("sess-1")

        assert result["session_id"] == "sess-1"
        assert result["total_chunks"] == 3
        assert result["unique_documents"] == 2
        assert sorted(result["filenames"]) == ["data.xlsx", "report.pdf"]
        assert sorted(result["file_types"]) == ["pdf", "xlsx"]
        assert result["team_id"] == "team-1"

    def test_empty_results_returns_zero_counts(self):
        """Covers: `results[0].get("team_id") if results else None` → None."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_session_documents("empty-sess")

        assert result["total_chunks"] == 0
        assert result["unique_documents"] == 0
        assert result["filenames"] == []
        assert result["file_types"] == []
        assert result["team_id"] is None

    def test_metadata_without_filename_or_file_type_is_skipped(self):
        """Covers: metadata without 'filename' or 'file_type' keys → sets stay empty."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[
            {"metadata": {"source": "web"}, "team_id": "t1", "created_at": "ts"},
            {"metadata": {}, "team_id": "t1", "created_at": "ts"},
        ])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            result = s.get_session_documents("sess-1")

        assert result["filenames"] == []
        assert result["file_types"] == []

    def test_uses_custom_limit(self):
        """Covers: explicit limit is passed through to self.query."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            s.get_session_documents("sess-1", limit=500)

        call_kwargs = s.query.call_args[1]
        assert call_kwargs["limit"] == 500

    def test_uses_default_limit_from_config(self):
        """Covers: `if limit is None: limit = config.get(...)` branch."""
        s, _, _, _ = _build_vector_store()
        s.query = MagicMock(return_value=[])

        with patch("src.database.milvus_db.milvus_config", _make_mock_config()):
            s.get_session_documents("sess-1")  # no limit provided

        call_kwargs = s.query.call_args[1]
        assert call_kwargs["limit"] == 100  # from MINIMAL_YAML_CONFIG ingestion.max_chunk_limit


# ===========================================================================
# 15. MilvusVectorStore.delete_session_documents — filename filter branch
# ===========================================================================

class TestDeleteSessionDocumentsFilenameFilter:

    def test_delete_with_filename_appends_metadata_filter(self):
        """Covers the `if filename:` branch that augments the filter expression."""
        s, _, _, _ = _build_vector_store()
        s.delete = MagicMock()

        s.delete_session_documents("sess-1", filename="invoice.pdf")

        s.delete.assert_called_once()
        filter_expr = s.delete.call_args[0][0]
        assert 'session_id == "sess-1"' in filter_expr
        assert "invoice.pdf" in filter_expr

    def test_delete_without_filename_uses_session_only_filter(self):
        """Covers: no filename → only session_id filter."""
        s, _, _, _ = _build_vector_store()
        s.delete = MagicMock()

        s.delete_session_documents("sess-2")

        s.delete.assert_called_once()
        filter_expr = s.delete.call_args[0][0]
        assert filter_expr == 'session_id == "sess-2"'

    def test_delete_logs_info_on_success(self):
        """Covers the logger.info call after successful delete."""
        s, _, _, _ = _build_vector_store()
        s.delete = MagicMock()

        s.delete_session_documents("sess-1", filename="doc.pdf")

        s.logger.info.assert_called()


# ===========================================================================
# 16. MilvusVectorStore.get_collection_stats — happy path + error path
# ===========================================================================

class TestGetCollectionStats:

    def test_get_collection_stats_happy_path(self):
        """Covers the full happy-path return dict."""
        s, mock_collection, _, _ = _build_vector_store()
        mock_collection.num_entities = 1234

        result = s.get_collection_stats()

        assert result["collection_name"] == "test_collection"
        assert result["database_name"] == "test_db"
        assert result["num_entities"] == 1234
        assert result["embedding_model"] == "gemini-embedding-001"
        assert result["embedding_dim"] == 768
        assert result["chunk_size"] == 500
        assert result["chunk_overlap"] == 50
        assert result["hyde_enabled"] is True
        assert result["multi_query_enabled"] is True

    def test_get_collection_stats_reraises_on_error(self):
        """Covers: `except Exception as e: raise`."""
        s, mock_collection, _, _ = _build_vector_store()
        type(mock_collection).num_entities = property(
            MagicMock(side_effect=Exception("Stats error"))
        )

        with pytest.raises(Exception, match="Stats error"):
            s.get_collection_stats()

        s.logger.error.assert_called()

    def test_get_collection_stats_logs_error_message(self):
        """Verifies the error is logged with the right text."""
        s, mock_collection, _, _ = _build_vector_store()
        type(mock_collection).num_entities = property(
            MagicMock(side_effect=Exception("Timeout"))
        )

        with pytest.raises(Exception):
            s.get_collection_stats()

        error_msg = str(s.logger.error.call_args)
        assert "Timeout" in error_msg or "Stats" in error_msg