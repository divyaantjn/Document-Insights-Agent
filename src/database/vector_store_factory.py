"""
vector_store_factory.py

Returns the correct vector store instance based on the USE_MILVUS env flag.
The factory reads PG_* env vars and passes them to PgVectorStore via its
__init__. MILVUS_* credentials are read internally by MilvusVectorStore.

Usage:
    from src.database.vector_store_factory import get_vector_store
    milvus = get_vector_store()
"""
import os

from src.database.secrets_loader import load_pgvector_secrets

load_pgvector_secrets()

_USE_MILVUS = os.getenv("USE_MILVUS", "false").strip().lower() == "true"

_milvus_instance = None
_pgvector_instance = None


def get_vector_store():
    """Return a singleton vector store instance.

    Routes to MilvusVectorStore (reads MILVUS_*/YAML config internally) when
    USE_MILVUS=true, or to PgVectorStore (reads PG_* env vars) when USE_MILVUS=false.
    """
    global _milvus_instance, _pgvector_instance

    if _USE_MILVUS:
        if _milvus_instance is None:
            from src.database.milvus_db import MilvusVectorStore
            _milvus_instance = MilvusVectorStore()
        return _milvus_instance
    else:
        if _pgvector_instance is None:
            from src.database.pgvector_store import PgVectorStore
            _pgvector_instance = PgVectorStore()
        return _pgvector_instance
