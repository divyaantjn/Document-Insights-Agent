import atexit
import asyncio
import os
import json
import uuid
import yaml
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility, db
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from google import genai
from google.genai import types
import logging
from contextlib import contextmanager
from dotenv import load_dotenv
from src.llm.litellm_client import LitellmClient
from src.utils.reasoning_extractor import REASONING_SECTION_PROMPT
from datetime import datetime, timezone

# Thread pool for offloading blocking Gemini embed_content calls.
# Sized to match the embedding concurrency cap so threads are never starved.
# wait=False + cancel_futures=True: prevents Lambda from hanging 6+ minutes
# during Python process shutdown waiting for in-flight embed threads to finish.
_EMBED_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="embed_worker")
atexit.register(lambda: _EMBED_EXECUTOR.shutdown(wait=False, cancel_futures=True))

# Maximum concurrent Gemini embed_content calls during batch embedding.
_EMBED_CONCURRENCY = 8

# Embedding batch size — 30 texts per API call balances latency vs payload size.
_EMBED_BATCH_SIZE = 30

# ── Milvus operation timeouts (seconds) ───────────────────────────────────────
# Chosen to be generous enough for large collections under normal load while
# still bounding Lambda execution time so a single hung gRPC call cannot
# consume the full 15-minute Lambda budget.
#
# _MILVUS_CONNECT_PROBE_TIMEOUT  — has_collection() health-check in _ensure_connected
# _MILVUS_LOAD_TIMEOUT           — collection.load() can take 30-90s on large segments
# _MILVUS_INSERT_TIMEOUT         — collection.insert() per sub-batch (~20 rows)
# _MILVUS_SEARCH_TIMEOUT         — collection.search() per attempt
# _MILVUS_QUERY_TIMEOUT          — collection.query() (filenames pre-flight etc.)
# _EMBED_BATCH_TIMEOUT           — single Gemini embed_content API call per batch
_MILVUS_CONNECT_PROBE_TIMEOUT: int = 10   # cheap no-op; should be fast
_MILVUS_LOAD_TIMEOUT: int = 120           # large collections need time to memory-map
_MILVUS_INSERT_TIMEOUT: int = 30          # per sub-batch insert (~20 rows)
_MILVUS_SEARCH_TIMEOUT: int = 60          # per search attempt including DiskANN scan
_MILVUS_QUERY_TIMEOUT: int = 30           # metadata/filenames query
_EMBED_BATCH_TIMEOUT: int = 60            

load_dotenv()


# ==================== Configuration Manager ====================

class MilvusConfigManager:
    """Centralized configuration management for Milvus Vector Store"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MilvusConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load and parse YAML configuration"""
        config_path = os.getenv("MILVUS_CONFIG_PATH", "config/milvus_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Substitute environment variables
        self._substitute_env_vars()
    
    def _substitute_env_vars(self):
        """Replace ${VAR_NAME} and ${VAR_NAME:default} with environment variables"""
        def substitute_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_expr = value[2:-1]  # Remove ${ and }
                
                # Check if default value is provided
                if ":" in var_expr:
                    var_name, default_value = var_expr.split(":", 1)
                    return os.getenv(var_name, default_value)
                else:
                    return os.getenv(var_expr, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(v) for v in value]
            return value
        
        self.config = substitute_value(self.config)
    
    def get(self, key: str, default=None):
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        
        return value if value is not None else default
    
    def get_all(self, section: str) -> Dict:
        """Get entire configuration section"""
        return self.config.get(section, {})


# Initialize config manager
milvus_config = MilvusConfigManager()


# ==================== Field Schema Builder ====================

class FieldSchemaBuilder:
    """Build FieldSchema objects from YAML configuration"""
    
    @staticmethod
    def build_schemas(fields_config: List[Dict]) -> List[FieldSchema]:
        """
        Build FieldSchema objects from YAML field configuration
        
        Args:
            fields_config: List of field configurations from YAML
            
        Returns:
            List of FieldSchema objects
        """
        schemas = []
        primary_key_found = False
        
        for field_config in fields_config:
            name = field_config.get('name')
            dtype_str = field_config.get('dtype')
            
            if not name or not dtype_str:
                continue
            
            # Map string dtype to DataType enum
            dtype = FieldSchemaBuilder._parse_dtype(dtype_str)
            
            # Build kwargs based on field type
            kwargs = {
                'name': name,
                'dtype': dtype
            }
            
            # Add optional parameters
            is_primary = field_config.get('is_primary', False)
            if is_primary:
                kwargs['is_primary'] = True
                primary_key_found = True
            
            if field_config.get('max_length'):
                kwargs['max_length'] = field_config['max_length']
            
            if dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                if field_config.get('dim'):
                    kwargs['dim'] = field_config['dim']
            
            schemas.append(FieldSchema(**kwargs))
        
        if not primary_key_found:
            raise ValueError("No primary key field found in field_schemas. Mark one field with is_primary: true")
        
        return schemas
    
    @staticmethod
    def _parse_dtype(dtype_str: str) -> DataType:
        """Parse dtype string to DataType enum"""
        dtype_map = {
            'INT8': DataType.INT8,
            'INT16': DataType.INT16,
            'INT32': DataType.INT32,
            'INT64': DataType.INT64,
            'FLOAT': DataType.FLOAT,
            'DOUBLE': DataType.DOUBLE,
            'VARCHAR': DataType.VARCHAR,
            'JSON': DataType.JSON,
            'FLOAT_VECTOR': DataType.FLOAT_VECTOR,
            'BINARY_VECTOR': DataType.BINARY_VECTOR,
        }
        return dtype_map.get(dtype_str.upper(), DataType.VARCHAR)


# ==================== Enhanced Retrieval with HyDE & Multi-Query ====================

class EnhancedRetrieval:
    """
    Advanced retrieval strategies including:
    - HyDE (Hypothetical Document Embeddings)
    - Multi-Query Generation
    - Query Expansion
    """
    
    def __init__(self, litellm_client: LitellmClient, logger):
        self.litellm_client = litellm_client
        self.logger = logger
    
    async def generate_hypothetical_document(self, query: str, llm_params: dict, auth_token: str) -> str:
        """
        Generate a hypothetical document that would answer the query (HyDE).

        This improves retrieval by creating a document-like representation
        that better matches the embedding space of actual documents.

        Args:
            query: User's query
            llm_params: LLM configuration params from get_dynamic_llm_instance
            auth_token: Authentication token

        Returns:
            Hypothetical document text
        """
        hyde_prompt = f"""Given the following question, write a detailed, informative paragraph that would directly answer this question.
Write as if you are providing the answer in a well-structured document.

Question: {query}

Detailed Answer (write 2-3 paragraphs):"""

        try:
            messages = [{"role": "user", "content": hyde_prompt}]
            hypothetical_doc = await self.litellm_client.generate_response(llm_params, messages, auth_token)
            self.logger.info("Generated hypothetical document for HyDE retrieval")
            return hypothetical_doc

        except Exception as e:
            self.logger.error(f"Failed to generate hypothetical document: {e}")
            return query  # Fallback to original query
    
    async def generate_multi_queries(self, query: str, llm_params: dict, auth_token: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple variations of the query for better retrieval coverage.

        Args:
            query: Original query
            llm_params: LLM configuration params from get_dynamic_llm_instance
            auth_token: Authentication token
            num_queries: Number of query variations to generate

        Returns:
            List of query variations
        """
        multi_query_prompt = f"""Given the following question, generate {num_queries} different variations of the question that maintain the same intent but use different phrasing and keywords.

Original Question: {query}

Generate {num_queries} variations (one per line):"""

        try:
            messages = [{"role": "user", "content": multi_query_prompt}]
            response = await self.litellm_client.generate_response(llm_params, messages, auth_token)

            variations = response.strip().split('\n')
            # Clean up variations (remove numbering, extra whitespace)
            clean_variations = []
            for var in variations:
                cleaned = var.strip()
                # Remove common prefixes like "1. ", "- ", etc.
                if cleaned and len(cleaned) > 5:
                    for prefix in ['1. ', '2. ', '3. ', '4. ', '5. ', '- ', '* ']:
                        if cleaned.startswith(prefix):
                            cleaned = cleaned[len(prefix):]
                    clean_variations.append(cleaned)

            # Return up to num_queries variations
            result = clean_variations[:num_queries] if clean_variations else [query]
            self.logger.info(f"Generated {len(result)} query variations")
            return result

        except Exception as e:
            self.logger.error(f"Failed to generate multi-queries: {e}")
            return [query]  # Fallback to original query


# ==================== Milvus Vector Store ====================

class MilvusVectorStore:
    """
    Enhanced Milvus vector store with:
    - Session-based document management
    - HyDE retrieval
    - Multi-query search
    - Advanced LLM synthesis
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        database_name: Optional[str] = None,
    ):
        """
        Initialize the Milvus vector store with YAML configuration
        
        Args:
            config_path: Path to YAML config file (uses env var MILVUS_CONFIG_PATH if not provided)
            collection_name: Override collection name from config
            database_name: Override database name from config
        """
        # Set config path if provided
        if config_path:
            os.environ["MILVUS_CONFIG_PATH"] = config_path
        
        # Initialize logging
        self._setup_logging()
        
        # Load configuration
        self._load_configuration(collection_name, database_name)
        
        # Initialize Milvus connection
        self._connect_to_milvus()
        
        # Setup database
        self._setup_database()
        
        # Initialize Gemini client (used for embeddings)
        self.client = genai.Client()

        # Initialize LiteLLM client (used for text generation)
        self.litellm_client = LitellmClient()

        # Initialize text splitter
        self._init_text_splitter()

        # Initialize collection
        self._init_collection()

        # Initialize enhanced retrieval
        self.enhanced_retrieval = EnhancedRetrieval(
            self.litellm_client,
            self.logger
        )
    
    
    def _setup_logging(self):
        """Setup logging from configuration"""
        log_config = milvus_config.get_all('logging')
        
        log_level = log_config.get('level', 'INFO')
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Add file handler if enabled
        if log_config.get('file'):
            try:
                log_dir = os.path.dirname(log_config['file'])
                if log_dir:
                    os.makedirs(log_dir, exist_ok=True)
                file_handler = logging.FileHandler(log_config['file'])
                file_handler.setFormatter(logging.Formatter(log_format))
                self.logger.addHandler(file_handler)
            except OSError as e:
                self.logger.warning(f"Could not set up log file '{log_config['file']}': {e}. Logging to console only.")
    
    
    def _load_configuration(self, collection_name_override: Optional[str] = None, 
                           database_name_override: Optional[str] = None):
        """Load and validate configuration from YAML"""
        try:
            # Load Milvus settings
            milvus_cfg = milvus_config.get_all('milvus')
            self.milvus_host = milvus_cfg.get('host', 'localhost')
            self.milvus_port = str(milvus_cfg.get('port', '19530'))
            self.username = milvus_cfg.get('username')
            self.password = milvus_cfg.get('password')
            
            # Load collection settings
            collection_cfg = milvus_config.get_all('collection')
            self.collection_name = collection_name_override or collection_cfg.get('name', 'idp_collection')
            
            # Load database settings
            db_cfg = milvus_config.get_all('database')
            self.database_name = database_name_override or db_cfg.get('name', 'idp_db')
            
            # Load embedding settings
            embedding_cfg = milvus_config.get_all('embedding')
            self.embedding_model = embedding_cfg.get('model', 'gemini-embedding-001')
            self.embedding_dim = embedding_cfg.get('dimension', 768)
            
            # Load chunking settings
            chunking_cfg = milvus_config.get_all('chunking')
            self.chunk_size = chunking_cfg.get('chunk_size', 1000)
            self.chunk_overlap = chunking_cfg.get('chunk_overlap', 200)
            
            # Load search settings
            search_cfg = milvus_config.get_all('vector_search')
            self.metric_type = search_cfg.get('metric_type', 'IP')
            self.index_type = search_cfg.get('index_type', 'DISKANN')

            # DiskANN search-time params
            self.search_params = search_cfg.get('search_params', {})
            self.diskann_search_list = self.search_params.get('search_list', 100)

            # DiskANN index-time params
            self.index_params = search_cfg.get('index_params', {})
            self.diskann_max_degree = self.index_params.get('max_degree', 56)
            self.diskann_search_list_size = self.index_params.get('search_list_size', 100)
            
            # Load feature flags
            features_cfg = milvus_config.get_all('features')
            self.enable_logging = features_cfg.get('enable_logging', True)
            self.enable_error_fallback = features_cfg.get('enable_error_fallback', True)
            
            # Load retrieval settings
            milvus_config.get_all('retrieval')
            self.enable_hyde = True
            self.enable_multi_query = True
            self.num_multi_queries = 3
            
            self.logger.info("Configuration loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    
    def _connect_to_milvus(self):
        """Connect to Milvus server"""
        try:
            connections.connect(
                "default", 
                host=self.milvus_host, 
                port=self.milvus_port, 
                user=self.username, 
                password=self.password
            )
            self.logger.info(f"Connected to Milvus at {self.milvus_host}:{self.milvus_port}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _ensure_connected(self):
        """
        Re-establish the Milvus gRPC connection if it has dropped.

        On AWS Lambda the execution environment is frozen between invocations.
        The underlying TCP connection to Milvus times out during the freeze,
        but the Collection object still holds a stale reference.  Without this
        guard the next request fails with a gRPC transport error instead of
        transparently reconnecting.

        Strategy: attempt a cheap no-op call (has_collection) guarded by
        _MILVUS_CONNECT_PROBE_TIMEOUT.  If it raises or times out, disconnect,
        reconnect, re-setup the database, and re-attach a fresh Collection.
        """
        import concurrent.futures as _cf
        try:
            # Run the probe in a thread so we can enforce a hard wall-clock
            # timeout — gRPC calls are not cancellable from asyncio alone.
            with _cf.ThreadPoolExecutor(max_workers=1) as _probe_pool:
                _probe_pool.submit(
                    utility.has_collection, self.collection_name
                ).result(timeout=_MILVUS_CONNECT_PROBE_TIMEOUT)
        except Exception as conn_exc:
            self.logger.warning(
                "Milvus connection check failed (%s) — reconnecting...", conn_exc
            )
            try:
                connections.disconnect("default")
            except Exception:
                pass
            try:
                self._connect_to_milvus()
                self._setup_database()
                self.collection = Collection(self.collection_name)
                self.logger.info(
                    "Milvus reconnection successful, collection re-attached: %s",
                    self.collection_name,
                )
            except Exception as reconnect_exc:
                self.logger.error(
                    "Milvus reconnection failed: %s", reconnect_exc
                )
                raise
    
    
    def _setup_database(self):
        """Create and switch to the target database"""
        try:
            existing_databases = db.list_database()
            
            if self.database_name not in existing_databases:
                self.logger.info(f"Creating database: {self.database_name}")
                db.create_database(db_name=self.database_name)
            
            db.using_database(db_name=self.database_name)
            self.logger.info(f"Using database: {self.database_name}")
        except Exception as e:
            self.logger.error(f"Failed to setup database: {e}")
            raise
    
    
    def _init_text_splitter(self):
        """Initialize text splitter from configuration"""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        self.logger.info(f"Text splitter initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    
    def _init_collection(self):
        """Initialize or load existing collection"""
        try:
            # Build field schemas from configuration
            fields_config = milvus_config.get('field_schemas', [])
            
            if not fields_config:
                raise ValueError("No field_schemas found in configuration")
            
            self.logger.info(f"Building field schemas from config: {len(fields_config)} fields")
            field_schemas = FieldSchemaBuilder.build_schemas(fields_config)
            
            # Create collection schema
            collection_cfg = milvus_config.get_all('collection')
            schema = CollectionSchema(
                fields=field_schemas,
                description=collection_cfg.get('description', f"Vector store collection: {self.collection_name}")
            )
            
            # Create or load collection
            if not utility.has_collection(self.collection_name):
                self.logger.info(f"Creating collection: {self.collection_name}")
                self.collection = Collection(self.collection_name, schema)
                for field in field_schemas:
                    if field.dtype in [DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR]:
                        self._create_vector_index(field.name, field.dtype)
            else:
                self.logger.info(f"Loading existing collection: {self.collection_name}")
                self.collection = Collection(self.collection_name)
            self.logger.info(f"Collection initialized: {self.collection_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize collection: {e}")
            raise
    
    @contextmanager
    def manage_collection(self):
        """
        Context manager to handle collection lifecycle (load and release)
        at the request level.

        Lambda-safe design:
        - Calls _ensure_connected() first so a thawed Lambda with a dead gRPC
          connection transparently reconnects before any Milvus operation.
        - Tracks did_load so release() is only called by the context manager
          that actually performed the load (idempotent against nesting).
        - Skips release() on AWS Lambda to prevent cross-instance TaskStale.
        - Uses a separate try/finally for the load phase vs the body phase so
          exceptions raised inside the `with` block never bleed into the load
          exception handler — fixes `RuntimeError: generator didn't stop after
          throw()` which occurred when HTTPException from the body was caught
          by the load except-branch and caused a second yield.
        """
        # Reconnect if the gRPC channel dropped during a Lambda freeze
        self._ensure_connected()

        is_lambda = os.environ.get("AWS_LAMBDA_FUNCTION_NAME") is not None
        did_load = False

        # ── Phase 1: load the collection ──────────────────────────────────────
        # Isolated in its own try/except so its exception handler is exited
        # before yield — any exception raised inside the `with` body will
        # never be seen by this except clause.
        import concurrent.futures as _cf
        try:
            self.logger.info(f"Loading collection: {self.collection_name}")
            with _cf.ThreadPoolExecutor(max_workers=1) as _load_pool:
                _load_pool.submit(self.collection.load).result(
                    timeout=_MILVUS_LOAD_TIMEOUT
                )
            did_load = True
        except Exception as load_exc:
            # Collection may already be loaded by a concurrent instance,
            # or load timed out — proceed anyway, search may still work.
            self.logger.warning(
                "collection.load() raised or timed out (%s). Proceeding.",
                load_exc,
            )

        # ── Phase 2: run the body, then release ───────────────────────────────
        # Completely separate try/finally — exceptions from the body propagate
        # normally without touching the load logic above.
        try:
            yield
        finally:
            if did_load:
                if is_lambda:
                    self.logger.info(
                        "Lambda env detected — skipping collection.release() "
                        "to prevent cross-instance TaskStale."
                    )
                else:
                    try:
                        self.logger.info(
                            f"Releasing collection: {self.collection_name}"
                        )
                        self.collection.release()
                    except Exception as rel_exc:
                        self.logger.warning(
                            "collection.release() raised (non-fatal): %s", rel_exc
                        )
    
    
    def _create_vector_index(self, field_name: str, vector_type: DataType):
        """Create a vector index for *field_name* using the configured strategy.

        Supports DiskANN for float vectors and BIN_IVF_FLAT for binary vectors.
        DiskANN provides disk-resident approximate nearest-neighbour search,
        reducing in-memory footprint for large-scale deployments.
        """
        try:
            if vector_type == DataType.FLOAT_VECTOR:
                index_params = self._build_float_index_params()
            elif vector_type == DataType.BINARY_VECTOR:
                index_params = {
                    "metric_type": "JACCARD",
                    "index_type": "BIN_IVF_FLAT",
                    "params": {"nlist": 128},
                }
            else:
                return

            self.collection.create_index(field_name, index_params)
            self.logger.info(
                "Created index for field '%s' with params: %s",
                field_name,
                index_params,
            )
        except Exception as e:
            self.logger.error("Failed to create index for '%s': %s", field_name, e)

    def _build_float_index_params(self) -> dict:
        """Build index parameter dict for float-vector fields.

        Uses DiskANN as the default index type, with a generic fallback for
        any other index type that may be configured.
        """
        upper_type = self.index_type.upper()
        if upper_type == "DISKANN":
            return {
                "metric_type": self.metric_type,
                "index_type": "DISKANN",
                "params": {
                    "max_degree": self.diskann_max_degree,
                    "search_list_size": self.diskann_search_list_size,
                },
            }
        # Generic fallback for future index types (IVF_PQ, SCANN, etc.)
        return {
            "metric_type": self.metric_type,
            "index_type": self.index_type,
            "params": self.index_params,
        }
    
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embeddings using configured embedding model"""
        try:
            embedding_cfg = milvus_config.get_all('embedding')
            
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text,
                config=types.EmbedContentConfig(
                    task_type=embedding_cfg.get('task_type', 'SEMANTIC_SIMILARITY'),
                    output_dimensionality=embedding_cfg.get('output_dimensionality', 768)
                )
            )
            return result.embeddings[0].values
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Fallback to zero vector if error fallback is enabled
            if milvus_config.get('features.enable_error_fallback', True):
                return [0.0] * self.embedding_dim
            raise

    def batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings for *texts* by running batches concurrently.

        Delegates to ``async_batch_generate_embeddings`` via a safe
        sync→async bridge so existing sync call-sites (ingest_documents)
        continue to work without refactoring.

        Parameters
        ----------
        texts:
            Ordered list of text strings to embed.
        batch_size:
            Texts per Gemini API call.  Defaults to _EMBED_BATCH_SIZE (30).
        """
        if not texts:
            return []

        resolved_batch_size = batch_size or _EMBED_BATCH_SIZE

        # Run async embedding in a fresh event loop on the current thread.
        # This is safe because ingest_documents is always called from a
        # sync context (FastAPI background task / Lambda handler thread).
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already inside a running loop (e.g. called from async route).
                # Offload to a fresh thread with its own event loop.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(
                        asyncio.run,
                        self.async_batch_generate_embeddings(texts, resolved_batch_size),
                    )
                    return future.result()
            return loop.run_until_complete(
                self.async_batch_generate_embeddings(texts, resolved_batch_size)
            )
        except RuntimeError:
            return asyncio.run(
                self.async_batch_generate_embeddings(texts, resolved_batch_size)
            )

    async def async_batch_generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = _EMBED_BATCH_SIZE,
    ) -> List[List[float]]:
        """
        Concurrent async embedding.  All batches are dispatched simultaneously
        (bounded by _EMBED_CONCURRENCY semaphore) so total wall-clock time ≈
        single-batch latency instead of N × single-batch latency.

        Parameters
        ----------
        texts:
            Ordered list of text strings.
        batch_size:
            Texts per Gemini API call (default 30).

        Returns
        -------
        List[List[float]]
            Embeddings in the same order as *texts*.
        """
        if not texts:
            return []

        embedding_cfg = milvus_config.get_all("embedding")
        embed_config = types.EmbedContentConfig(
            task_type=embedding_cfg.get("task_type", "SEMANTIC_SIMILARITY"),
            output_dimensionality=embedding_cfg.get("output_dimensionality", 768),
        )

        # Build ordered list of (start_index, batch) pairs
        batches: List[tuple] = [
            (start, texts[start: start + batch_size])
            for start in range(0, len(texts), batch_size)
        ]

        self.logger.info(
            "Embedding %d texts in %d concurrent batches (batch_size=%d, concurrency=%d)",
            len(texts), len(batches), batch_size, _EMBED_CONCURRENCY,
        )

        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)
        loop = asyncio.get_event_loop()

        async def _embed_one_batch(start: int, batch: List[str]) -> tuple:
            """Embed a single batch; return (start_index, embeddings_list)."""
            async with sem:
                try:
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            _EMBED_EXECUTOR,
                            lambda b=batch, cfg=embed_config: self.client.models.embed_content(
                                model=self.embedding_model,
                                contents=b,
                                config=cfg,
                            ),
                        ),
                        timeout=_EMBED_BATCH_TIMEOUT,
                    )
                    embeddings = [emb.values for emb in result.embeddings]
                    self.logger.debug(
                        "Embedded batch starting at %d (%d items)", start, len(batch)
                    )
                    return start, embeddings
                except asyncio.TimeoutError:
                    self.logger.error(
                        "Embedding batch starting at %d timed out after %ds",
                        start, _EMBED_BATCH_TIMEOUT,
                    )
                    if milvus_config.get("features.enable_error_fallback", True):
                        return start, [[0.0] * self.embedding_dim] * len(batch)
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.logger.error(
                        "Embedding batch starting at %d failed: %s", start, exc
                    )
                    if milvus_config.get("features.enable_error_fallback", True):
                        return start, [[0.0] * self.embedding_dim] * len(batch)
                    raise

        # Dispatch all batches concurrently
        tasks = [_embed_one_batch(start, batch) for start, batch in batches]
        results = await asyncio.gather(*tasks)

        # Re-assemble in original order (gather preserves task order)
        all_embeddings: List[List[float]] = [None] * len(texts)  # type: ignore[list-item]
        for start, batch_embeddings in results:
            for offset, emb in enumerate(batch_embeddings):
                all_embeddings[start + offset] = emb

        self.logger.info(
            "Concurrent embedding complete: %d vectors generated", len(all_embeddings)
        )
        return all_embeddings
    
    
    def insert_data(self, data: List[List]):
        """
        Insert data into collection
        
        Args:
            data: List of lists, where each inner list corresponds to a field
        """
        # Reconnect if the gRPC channel dropped during a Lambda freeze
        self._ensure_connected()
        try:
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _ins_pool:
                _ins_pool.submit(
                    self.collection.insert, data
                ).result(timeout=_MILVUS_INSERT_TIMEOUT)
            self.logger.info(f"Inserted {len(data[0])} records")
        except Exception as e:
            self.logger.error(f"Error inserting data (timeout={_MILVUS_INSERT_TIMEOUT}s): {e}")
            raise
    
    def get_uploaded_filenames_for_session(self, session_id: str) -> set:
        """
        Get all filenames already uploaded in a session by querying metadata.
        Uses the existing metadata JSON field — no schema change needed.
        
        Returns:
            Set of filenames already present in this session
        """
        # Reconnect if the gRPC channel dropped during a Lambda freeze
        self._ensure_connected()
        try:
            filter_expr = f'session_id == "{session_id}"'
            results = self.query(
                filter_expr=filter_expr,
                output_fields=["metadata"],
                limit=milvus_config.get('ingestion.max_chunk_limit', 10000)
            )

            filenames = set()
            for result in results:
                metadata = result.get("metadata", {})
                if isinstance(metadata, dict) and "filename" in metadata:
                    filenames.add(metadata["filename"])

            self.logger.info(f"Found {len(filenames)} already-uploaded files in session {session_id}")
            return filenames

        except Exception as e:
            self.logger.error(f"Error fetching uploaded filenames for session {session_id}: {e}")
            return set()  # Safe fallback: treat as no files uploaded, allow processing
        
    def vector_search(
        self,
        query: str,
        vector_field: str = "embedding",
        filter_expr: Optional[str] = None,
        top_k: Optional[int] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        # Reconnect if the gRPC channel dropped during a Lambda freeze
        self._ensure_connected()

        if top_k is None:
            top_k = milvus_config.get('api.default_top_k', 5)
        
        query_embedding = self.generate_embedding(query)

        search_params = {
            "metric_type": self.metric_type,
            "params": {"search_list": self.diskann_search_list},
        }
        
        max_retries = 3

        # NOTE: collection.load() / release() are intentionally NOT called here.
        # Lifecycle is owned exclusively by manage_collection() at the router
        # level (query_documents / upload_documents).  Calling load()+release()
        # inside vector_search races with manage_collection() and produces
        # TaskStale errors on the subsequent HyDE / multi-query search calls.
        import concurrent.futures as _cf
        for attempt in range(max_retries):
            try:
                with _cf.ThreadPoolExecutor(max_workers=1) as _s_pool:
                    results = _s_pool.submit(
                        self.collection.search,
                        [query_embedding],   # data
                        vector_field,        # anns_field
                        search_params,       # param
                        top_k,               # limit
                        filter_expr,         # expr
                        None,                # partition_names
                        output_fields,       # output_fields
                    ).result(timeout=_MILVUS_SEARCH_TIMEOUT)
                return [
                    {
                        "score": hit.score,
                        "id": hit.id,
                        **{field: hit.entity.get(field) for field in (output_fields or [])}
                    }
                    for hit in results[0]
                ]
            
            except Exception as e:
                # Catch Milvus field type error -- collection likely empty
                if "Unsupported field type" in str(e):
                    raise HTTPException(
                        status_code=400,
                        detail="No documents found for this session. Please upload your files before querying."
                    )
                self.logger.error(
                    f"Error during vector search (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt == max_retries - 1:
                    raise
    
    def _run_direct_search(self, query, session_id, top_k, output_fields, seen_ids):
        results = self.search_by_session(query=query, session_id=session_id, top_k=top_k, output_fields=output_fields)
        new_results = []
        for result in results:
            if result['id'] not in seen_ids:
                result['retrieval_method'] = 'direct'
                new_results.append(result)
                seen_ids.add(result['id'])
        return new_results

    async def _run_hyde_search(self, query, session_id, top_k, output_fields, seen_ids, llm_params, auth_token):
        hypothetical_doc = await self.enhanced_retrieval.generate_hypothetical_document(query, llm_params, auth_token)
        results = self.search_by_session(query=hypothetical_doc, session_id=session_id, top_k=top_k, output_fields=output_fields)
        new_results = []
        for result in results:
            if result['id'] not in seen_ids:
                result['retrieval_method'] = 'hyde'
                new_results.append(result)
                seen_ids.add(result['id'])
        return new_results

    async def _run_multi_query_search(self, query, session_id, top_k, output_fields, seen_ids, llm_params, auth_token):
        variations = await self.enhanced_retrieval.generate_multi_queries(query, llm_params, auth_token, self.num_multi_queries)
        new_results = []
        for idx, query_var in enumerate(variations):
            results = self.search_by_session(query=query_var, session_id=session_id, top_k=top_k // 2, output_fields=output_fields)
            for result in results:
                if result['id'] not in seen_ids:
                    result['retrieval_method'] = f'multi_query_{idx+1}'
                    new_results.append(result)
                    seen_ids.add(result['id'])
        return new_results
    
    async def enhanced_search(self, query, session_id, llm_params, auth_token,
                          use_hyde=True, use_multi_query=True, top_k=None, output_fields=None):
        try:
            if top_k is None:
                top_k = milvus_config.get('api.default_top_k', 5)
            if output_fields is None:
                output_fields = ["text", "metadata", "session_id", "team_id", "created_at"]

            seen_ids = set()
            all_results = self._run_direct_search(query, session_id, top_k, output_fields, seen_ids)

            if use_hyde and self.enable_hyde:
                all_results += await self._run_hyde_search(query, session_id, top_k, output_fields, seen_ids, llm_params, auth_token)

            if use_multi_query and self.enable_multi_query:
                all_results += await self._run_multi_query_search(query, session_id, top_k, output_fields, seen_ids, llm_params, auth_token)

            all_results.sort(key=lambda x: x['score'], reverse=True)
            return all_results[:top_k * 2]
        except Exception as e:
            self.logger.error(f"Error during enhanced search: {e}")
            raise
    
    
    async def synthesize_answer_with_llm(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        llm_params: dict,
        auth_token: str,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Use LLM to synthesize final answer from retrieved chunks with enhanced formatting

        Args:
            query: Original user query
            retrieved_chunks: Retrieved document chunks
            llm_params: LLM configuration params from get_dynamic_llm_instance
            auth_token: Authentication token
            include_sources: Whether to include source citations

        Returns:
            Dictionary with synthesized answer and metadata
        """
        try:
            # Build context from retrieved chunks
            context_parts = []
            for idx, chunk in enumerate(retrieved_chunks, 1):
                text = chunk.get('text', '')
                metadata = chunk.get('metadata', {})
                source = metadata.get('source', metadata.get('filename', 'Unknown'))
                page = metadata.get('page', metadata.get('page_number', 'N/A'))
                
                # context_parts.append(f"[Source {idx}: {source}, Page {page}]\n{text}\n")
                context_parts.append(f"{text}\n")
            
            context = "\n---\n".join(context_parts)
            
            # Enhanced system prompt with detailed markdown formatting instructions
            system_prompt = """You are an expert document analyst and research assistant. Your role is to provide comprehensive, accurate, and well-formatted responses based on the provided context.

**CRITICAL FORMATTING REQUIREMENTS:**

1. **Use Proper Markdown Formatting:**
   - Use `#` for main headings, `##` for subheadings, `###` for sub-subheadings
   - Use **bold** for emphasis on key terms
   - Use *italics* for definitions or special terms
   - Use `code blocks` for technical terms, file names, or code
   - Use bullet points (-) or numbered lists (1., 2., 3.) for lists
   - Use > for important quotes or callouts
   - Use tables when presenting structured data

2. **Structure Your Response:**
   - Start with a brief overview or summary
   - Organize information into clear sections with headings
   - Use subheadings to break down complex topics
   - End with key takeaways or conclusions if applicable

3. **Content Guidelines:**
   - Answer directly and comprehensively
   - If information from multiple sources agrees, synthesize it
   - If the answer is not in the context, clearly state this
   - Provide context and explanations, not just facts
   - Use examples when they help clarify
   - If none of the retrieved context is relevant to the user's query, do NOT assume or fabricate any information. Instead, respond with a user-friendly message stating that the requested information is not available in the available data.

4. **Tables:**
   - When data is tabular in the source, preserve it in markdown table format
   - Use clear column headers
   - Align data appropriately

5. **Quality Standards:**
   - Be precise and factual
   - Use professional, clear language
   - Avoid redundancy
   - Ensure logical flow between sections
---

Now, answer the user's question using the provided context. {REASONING_SECTION_PROMPT}"""

            user_prompt = f"""**Context from Retrieved Documents:**

{context}

---

**User Question:** {query}

**Instructions:** Provide a comprehensive, well-formatted answer using the context above. Follow all markdown formatting guidelines. Include source citations where appropriate."""

            # Generate response using LiteLLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            answer = await self.litellm_client.generate_response(llm_params, messages, auth_token)
            
            # Build source references if requested
            sources = []
            if include_sources:
                seen_sources = set()
                for chunk in retrieved_chunks:
                    metadata = chunk.get('metadata', {})
                    source = metadata.get('source', metadata.get('filename', 'Unknown'))
                    page = metadata.get('page', metadata.get('page_number', 'N/A'))
                    source_key = f"{source}_{page}"
                    
                    if source_key not in seen_sources:
                        sources.append({
                            'source': source,
                            'page': page,
                            'score': chunk.get('score', 0.0)
                        })
                        seen_sources.add(source_key)
            
            return {
                'answer': answer,
                'sources': sources,
                'num_chunks_used': len(retrieved_chunks),
                'query': query
            }
            
        except Exception as e:
            self.logger.error(f"Error synthesizing answer: {e}")
            raise
    
    
    def query(
        self,
        filter_expr: str,
        output_fields: Optional[List[str]] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Query collection with filter expression
        
        Args:
            filter_expr: Filter expression
            output_fields: Fields to return
            limit: Maximum number of results
            
        Returns:
            List of matching records
        """
        try:
            if limit is None:
                limit = milvus_config.get('api.max_query_limit', 1000)

            # Pass all args as keyword arguments to avoid positional mapping
            # errors in pymilvus query() signature.
            # Bug fixed: previously limit was passed as positional arg 3 which
            # maps to partition_names (not limit), causing TypeError: Value
            # must be iterable.
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _q_pool:
                result = _q_pool.submit(
                    lambda: self.collection.query(
                        expr=filter_expr,
                        output_fields=output_fields or ["*"],
                        limit=limit,
                    )
                ).result(timeout=_MILVUS_QUERY_TIMEOUT)
            return result
        except Exception as e:
            self.logger.error(f"Error during query (timeout={_MILVUS_QUERY_TIMEOUT}s): {e}")
            raise
    
    
    def delete(self, filter_expr: str):
        """Delete records matching filter expression"""
        try:
            self.collection.delete(filter_expr)
            self.logger.info(f"Deleted records matching: {filter_expr}")
        except Exception as e:
            self.logger.error(f"Error deleting records: {e}")
            raise
    
    
    def drop_collection(self):
        """Drop the collection"""
        try:
            self.collection.drop()
            self.logger.info(f"Dropped collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error dropping collection: {e}")
            raise
    
    def _determine_file_type(self, filename: str) -> str:
        """
        Determine file type from filename extension
        
        Args:
            filename: Name of the file
            
        Returns:
            File type string (pdf, docx, pptx, xlsx, csv, etc.)
        """
        ext = filename.lower().split('.')[-1] if '.' in filename else 'unknown'
        
        # Normalize extensions
        if ext in ['doc', 'docx']:
            return 'docx'
        elif ext in ['xls', 'xlsx']:
            return 'xlsx'
        elif ext in ['ppt', 'pptx']:
            return 'pptx'
        elif ext == 'pdf':
            return 'pdf'
        elif ext == 'csv':
            return 'csv'
        else:
            return ext
    
    def _build_chunk_metadata(self, chunk: LangchainDocument, filename: str, file_type: str, attachment_id: Optional[str] = None, source_url: Optional[str] = None) -> Dict[str, Any]:
        """
        Build metadata dictionary from Langchain document
        
        Args:
            chunk: Langchain Document object
            filename: Name of the source file
            file_type: Type of file
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "filename": filename,
            "file_type": file_type,
            **chunk.metadata
        }

        if attachment_id:
            metadata["attachment_id"] = attachment_id

        if source_url:
            metadata["source_url"] = source_url
        
        return metadata
    
    def _process_chunk(
        self,
        chunk,
        chunk_idx,
        filename,
        file_type,
        session_id,
        generate_ids,
        timestamp,
        embedding: Optional[List[float]] = None,
        attachment_id: Optional[str] = None
    ):
        """
        Process a single chunk and return its data tuple.

        Parameters
        ----------
        embedding:
            Pre-computed embedding vector.  When provided it is used directly,
            avoiding a redundant API call.  When ``None`` (legacy / fallback)
            an embedding is generated on-the-fly via :meth:`generate_embedding`.
        """
        if not isinstance(chunk, LangchainDocument):
            self.logger.warning(f"Invalid chunk type at index {chunk_idx} in {filename}")
            return None
        chunk_id = (
            f"{session_id}_{uuid.uuid4().hex[:16]}_{chunk_idx}"
            if generate_ids
            else f"{session_id}_{filename}_{chunk_idx}"
        )
        MILVUS_MAX_TEXT_LENGTH = 65535
        text = chunk.page_content
        if len(text) > MILVUS_MAX_TEXT_LENGTH:
            self.logger.warning(
                f"Chunk {chunk_idx} in {filename} exceeds Milvus max text length "
                f"({len(text)} > {MILVUS_MAX_TEXT_LENGTH}), truncating."
            )
            text = text[:MILVUS_MAX_TEXT_LENGTH]
        emb = embedding if embedding is not None else self.generate_embedding(text)
        safe_attachment_id = attachment_id or ""
        return (
            chunk_id, session_id, text,
            self._build_chunk_metadata(chunk, filename, file_type),
            emb,
            timestamp,
            safe_attachment_id
        )

    def _flush_batch_if_needed(self, batch, batch_size, max_batch_bytes: int = 800 * 1024):
        """Insert and clear batch if chunk count or estimated byte size exceeds threshold.

        Threshold lowered to 800 KB (was 5 MB) to stay well under _insert_batch's
        1 MB sub-batch ceiling once embedding vector overhead is factored in.
        """
        all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids = batch
        if not all_ids:
            return batch
        estimated_bytes = sum(len(t.encode("utf-8", errors="replace")) for t in all_texts)
        if len(all_ids) >= batch_size or estimated_bytes >= max_batch_bytes:
            self._insert_batch(all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids)
            self.logger.info(f"Inserted batch of {len(all_ids)} chunks (~{estimated_bytes // 1024}KB)")
            return [], [], [], [], [], [], []
        return batch
    

    def ingest_documents(
        self,
        documents: Dict[str, List[LangchainDocument]],
        session_id: str,
        team_id: str,
        attachment_url_map: dict,
        document_source_map: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        generate_ids: Optional[bool] = None,
        # document_source_map: Optional[Dict[str, str]] = None,  # filename -> s3_url
    ) -> Dict[str, Any]:
        # Reconnect if the gRPC channel dropped during a Lambda freeze
        self._ensure_connected()
        try:
            if batch_size is None:
                batch_size = milvus_config.get('ingestion.batch_size', 100)
            if generate_ids is None:
                generate_ids = milvus_config.get('ingestion.generate_ids', True)

            total_docs = len(documents)
            total_chunks = 0
            failed_docs = []
            timestamp = datetime.now(timezone.utc).isoformat()

            # url_to_attachment_id = {url: aid for aid, url in (attachment_url_map or {}).items()}
            url_to_attachment_id = {url: aid for aid, url in (attachment_url_map or {}).items()}

            print("url_to_attachment_id", url_to_attachment_id)
            print("document_source_map", document_source_map)

            if document_source_map and len(document_source_map) != total_docs:
                raise ValueError(
                    f"document_source_map length ({len(document_source_map)}) must match "
                    f"number of documents ({total_docs})"
                )
        
            self.logger.info(f"Starting ingestion of {total_docs} documents for session: {session_id}, team: {team_id}")

            all_ids, all_session_ids, all_team_ids = [], [], []
            all_texts, all_metadata, all_embeddings, all_timestamps = [], [], [], []
            all_attachment_ids = []

            for idx, (filename, chunks) in enumerate(documents.items(), 1):
                try:
                    self.logger.info(f"Processing document {idx}/{total_docs}: {filename}")

                    # source_url = (document_source_map or {}).get(filename)
                    # attachment_id = url_to_attachment_id.get(source_url)

                    source_url = document_source_map[idx - 1] if document_source_map else None
                    attachment_id = url_to_attachment_id.get(source_url)

                    doc_chunks_count, all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids = self._process_single_document_ingestion(
                        filename, chunks, session_id, team_id, generate_ids, timestamp, batch_size,
                        (all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids), attachment_id=attachment_id, source_url=source_url
                    )
                    total_chunks += doc_chunks_count

                    # Flush remaining chunks for this file immediately to avoid accumulating
                    # too much data across files (prevents Kafka "Message size too large").
                    if all_ids:
                        self._insert_batch(all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids)
                        self.logger.info(f"Inserted {len(all_ids)} chunks for {filename}")
                        all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps = [], [], [], [], [], [], []

                    self.logger.info(f"Completed {filename}: {doc_chunks_count} chunks ingested")

                except Exception as e:
                    self.logger.error(f"Failed to process {filename}: {str(e)}")
                    failed_docs.append({"filename": filename, "error": str(e)})
                    # Reset batch so a failed file doesn't block subsequent files
                    all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps = [], [], [], [], [], [], []
                    all_attachment_ids = []

            # Flush removed: collection.flush() blocks for 30-120s waiting for
            # Milvus segment persistence and was the primary cause of upload hangs.
            # Milvus flushes automatically on segment seal / background schedule;
            # data inserted via insert_data() is immediately searchable without
            # an explicit flush call.
            stats = {
                "total_documents": total_docs,
                "successful_documents": total_docs - len(failed_docs),
                "failed_documents": len(failed_docs),
                "total_chunks_inserted": total_chunks,
                "failed_doc_details": failed_docs,
                "session_id": session_id,
                "team_id": team_id,
                "batch_size_used": batch_size
            }

            self.logger.info(f"Ingestion complete: {stats['successful_documents']}/{total_docs} docs, {total_chunks} chunks")
            return stats

        except Exception as e:
            self.logger.error(f"Critical error during document ingestion: {e}")
            raise    
    
    def _process_single_document_ingestion(
        self, filename, chunks, session_id, team_id, generate_ids, timestamp, batch_size, current_batch, attachment_id=None, source_url=None
    ):
        """Helper to process a single document's chunks and update the current batch."""
        all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids = current_batch
        
        if not chunks or not isinstance(chunks, list):
            self.logger.warning(f"No chunks or invalid chunks for {filename}")
            return 0, all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids,

        file_type = self._determine_file_type(filename)
        embedding_map = self._get_embedding_map(filename, chunks)
        
        doc_chunks_count = 0
        for chunk_idx, chunk in enumerate(chunks):
            pre_emb = embedding_map.get(chunk_idx)
            result = self._process_chunk(
                chunk, chunk_idx, filename, file_type,
                session_id, generate_ids, timestamp,
                embedding=pre_emb,
                attachment_id=attachment_id
            )
            if result is None:
                continue

            chunk_id, sid, text, metadata, embedding, ts, chunk_attachment_id = result

            all_ids.append(chunk_id)
            all_session_ids.append(sid)
            all_team_ids.append(team_id)
            all_texts.append(text)
            all_metadata.append(metadata)
            all_embeddings.append(embedding)
            all_timestamps.append(ts)
            all_attachment_ids.append(chunk_attachment_id)
            doc_chunks_count += 1

            batch = (all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids)
            all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids = self._flush_batch_if_needed(batch, batch_size)
            
        return doc_chunks_count, all_ids, all_session_ids, all_team_ids, all_texts, all_metadata, all_embeddings, all_timestamps, all_attachment_ids

    def _get_embedding_map(self, filename, chunks):
        """Generate embeddings for valid chunks in one batch call."""
        valid_chunks = [
            (ci, c)
            for ci, c in enumerate(chunks)
            if isinstance(c, LangchainDocument)
        ]
        if not valid_chunks:
            return {}
            
        MILVUS_MAX_TEXT_LENGTH = 65535
        texts_to_embed = [c.page_content[:MILVUS_MAX_TEXT_LENGTH] for _, c in valid_chunks]
        self.logger.info(
            "Batch-embedding %d chunks for document: %s",
            len(texts_to_embed),
            filename,
        )
        doc_embeddings = self.batch_generate_embeddings(texts_to_embed)
        return {
            ci: emb
            for (ci, _), emb in zip(valid_chunks, doc_embeddings)
        }

    def _insert_batch(
        self,
        ids: List[str],
        session_ids: List[str],
        team_ids: List[str],
        texts: List[str],
        metadata: List[Dict],
        embeddings: List[List[float]],
        timestamps: List[str],
        attachment_ids: List[Optional[str]]
    ):
        """
        Insert a batch of data into Milvus, splitting into sub-batches to stay
        under the Kafka broker message size limit.

        Reduced to 1 MB per sub-batch (was 4 MB) to account for embedding vector
        overhead (~3 KB per 768-dim float32 vector) on top of the text payload.
        A hard row cap of 20 per call prevents a single oversized row from
        slipping through the byte check.

        Args:
            ids: List of document IDs
            session_ids: List of session IDs
            team_ids: List of team IDs
            attachment_ids: List of chunks attachment IDs
            texts: List of text contents
            metadata: List of metadata dictionaries
            embeddings: List of embedding vectors
            timestamps: List of creation timestamps
            attachment_ids: List of attachment ids
        """
        attachment_ids = [a if isinstance(a, str) else "" for a in attachment_ids]
        MAX_INSERT_BYTES = 1 * 1024 * 1024  # 1 MB — safe margin below Kafka 4 MB default
        MAX_ROWS_PER_CALL = 20              # hard row cap regardless of byte size
        start = 0
        n = len(ids)
        while start < n:
            end = start
            current_bytes = 0
            while end < n and (end - start) < MAX_ROWS_PER_CALL:
                # Account for text bytes + embedding vector bytes (float32 = 4 bytes/dim)
                row_bytes = (
                    len(texts[end].encode("utf-8", errors="replace"))
                    + len(embeddings[end]) * 4
                )
                if current_bytes + row_bytes > MAX_INSERT_BYTES and end > start:
                    break
                current_bytes += row_bytes
                end += 1
            sub_data = [
                ids[start:end],
                session_ids[start:end],
                team_ids[start:end],
                attachment_ids[start:end],
                texts[start:end],
                metadata[start:end],
                embeddings[start:end],
                timestamps[start:end],
            ]
            self.insert_data(sub_data)
            self.logger.info(
                f"Inserted sub-batch {start}–{end-1} "
                f"({end - start} rows, ~{current_bytes // 1024}KB)"
            )
            start = end
    
    
    def search_by_session(
        self,
        query: str,
        session_id: str,
        vector_field: str = "embedding",
        top_k: Optional[int] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search documents for a specific session
        
        Args:
            query: Search query text
            session_id: Session identifier to filter by
            vector_field: Name of the vector field
            top_k: Number of results to return
            output_fields: Fields to return in results
            
        Returns:
            List of search results with scores and metadata
        """
        try:
            # Build filter expression
            filter_expr = f'session_id == "{session_id}"'
            
            if output_fields is None:
                output_fields = ["text", "metadata", "session_id", "team_id", "created_at"]
            
            return self.vector_search(
                query=query,
                vector_field=vector_field,
                filter_expr=filter_expr,
                top_k=top_k,
                output_fields=output_fields
            )
        except Exception as e:
            self.logger.error(f"Error searching by session: {e}")
            raise
    
    
    def get_session_documents(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get document statistics for a session
        
        Args:
            session_id: Session identifier
            limit: Maximum number of chunks to retrieve
            
        Returns:
            Dictionary with document statistics
        """
        try:
            if limit is None:
                limit = milvus_config.get('ingestion.max_chunk_limit', 10000)
            
            filter_expr = f'session_id == "{session_id}"'
            results = self.query(
                filter_expr=filter_expr,
                output_fields=["metadata", "team_id", "created_at"],
                limit=limit
            )
            
            filenames = set()
            file_types = set()
            for result in results:
                metadata = result.get("metadata", {})
                if "filename" in metadata:
                    filenames.add(metadata["filename"])
                if "file_type" in metadata:
                    file_types.add(metadata["file_type"])
            
            return {
                "session_id": session_id,
                "total_chunks": len(results),
                "unique_documents": len(filenames),
                "filenames": sorted(filenames),
                "file_types": sorted(file_types),
                "team_id": results[0].get("team_id") if results else None
            }
        except Exception as e:
            self.logger.error(f"Error getting session documents: {e}")
            raise
    
    
    def delete_session_documents(
        self,
        session_id: str,
        filename: Optional[str] = None
    ):
        """
        Delete documents for a session
        
        Args:
            session_id: Session identifier
            filename: If provided, delete only this document
        """
        try:
            filter_expr = f'session_id == "{session_id}"'
            if filename:
                filter_expr += f' && metadata["filename"] == "{filename}"'
            
            self.delete(filter_expr)
            self.logger.info(f"Deleted documents for session {session_id}: {filename or 'all'}")
        except Exception as e:
            self.logger.error(f"Error deleting session documents: {e}")
            raise
    
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = {
                "collection_name": self.collection_name,
                "database_name": self.database_name,
                "num_entities": self.collection.num_entities,
                "embedding_model": self.embedding_model,
                "embedding_dim": self.embedding_dim,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "hyde_enabled": self.enable_hyde,
                "multi_query_enabled": self.enable_multi_query
            }
            return stats
        except Exception as e:
            self.logger.error(f"Error getting collection stats: {e}")
            raise
