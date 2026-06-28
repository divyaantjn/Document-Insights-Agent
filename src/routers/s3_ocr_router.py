"""
Unified Document Processing and RAG Router
Handles both document upload/ingestion and querying in a single endpoint
Designed for MCP/LangGraph integration
"""

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import logging
from datetime import datetime
import json
import os

logger = logging.getLogger(__name__)

from src.utils.kafka import create_event_logger
from src.utils.nia_embed_notifier import call_agent_embed_api
from src.utils.event_messages import EventMessages
from src.utils.otel_utils import set_message_id, set_user_context

from langchain_core.documents import Document as LangchainDocument
from src.extraction.s3_extraction import S3Extraction
from src.ocr.ocr import Ocr
from src.llm.litellm_client import LitellmClient
from src.database.vector_store_factory import get_vector_store as _get_vector_store
from src.database.semantic_cache import SemanticCache
from src.utils.opik_setup import track_llm_calls, update_current_trace
from src.utils.follow_up_generator import generate_follow_up_questions

# Initialize global instances
milvus = _get_vector_store()  # name kept to avoid cascading renames
router = APIRouter()
litellm_client = LitellmClient()

# Semantic cache — lazy-initialised on first query (needs milvus.generate_embedding)
_semantic_cache: SemanticCache = None


def get_semantic_cache() -> SemanticCache:
    """Return singleton SemanticCache, initialised on first call."""
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticCache(embed_fn=milvus.generate_embedding)
    return _semantic_cache


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    PPT = "ppt"
    IMAGE = "image"
    TEXT = "text"
    ZIP = "zip"
    AUTO = "auto"



class DocumentUploadRequest(BaseModel):
    user_metadata: str = Field(..., description="JSON string containing session_id and team_id")
    s3_urls: List[str] = Field(..., description="List of S3 URLs to process")

class DocumentQueryRequest(BaseModel):
    user_metadata: str = Field(..., description="JSON string containing session_id and team_id")
    query: str = Field(..., description="Search query text")

class DocumentUploadResponse(BaseModel):
    status: str
    message: str
    upload_results: Optional[Dict[str, Any]] = None

class DocumentQueryResponse(BaseModel):
    status: str
    message: str
    query_results: Optional[Dict[str, Any]] = None
    follow_up_questions: Optional[str] = None

class DocumentProcessingResult(BaseModel):
    """Result for a single processed document"""
    s3_url: str
    filename: str
    document_type: str
    status: str
    extracted_text: Optional[str] = None
    error: Optional[str] = None
    file_size: Optional[int] = None
    processing_time: Optional[float] = None


@router.post("/upload", response_model=DocumentUploadResponse, responses={400: {"description": "Bad Request"}})
@track_llm_calls(
        name="backend-upload-documents",
        tags=["upload", "document-processing"],
        metadata={"version": "1.0"},
        avoided_input_params=["request"]
    )
async def upload_documents(request: Request, payload: DocumentUploadRequest):
    event_logger = create_event_logger()
    try:
        user_metadata_dict = json.loads(payload.user_metadata) if payload.user_metadata else {}
        session_id = user_metadata_dict.get("session_id")
        team_id = user_metadata_dict.get("team_id")
        attachment_url_map = user_metadata_dict.get("attachment_url_map", {})
        auth_token = request.headers.get("Authorization", "")

        if not session_id or not team_id:
            detail = "session_id and team_id are required in user_metadata"
            raise HTTPException(status_code=400, detail=detail)

        _setup_request_tracing_and_context(request, user_metadata_dict)
        event_logger.log_event(EventMessages.UPLOAD_START, auth_token=auth_token)

        team_config = await litellm_client.get_dynamic_llm_instance(team_id=team_id)

        # NOTE: manage_collection() is NOT used here.
        # Milvus insert does not require the collection to be loaded into memory.
        # Loading during upload would cause unnecessary load/release cycles and
        # conflicts with the PDF streaming path's per-batch ingest calls.

        upload_results = await _process_document_upload(
            s3_urls=payload.s3_urls,
            document_type='auto',
            session_id=session_id,
            team_id=team_id,
            team_config=team_config,
            auth_token=auth_token,
            attachment_url_map=attachment_url_map
        )

        event_logger.log_event(EventMessages.UPLOAD_COMPLETE, auth_token=auth_token)

        if upload_results['skipped'] > 0 and upload_results['successful'] == 0:
            message = "Files are already uploaded, you can query them now."
        else:
            message = f"Uploaded {upload_results['successful']}/{upload_results['total_documents']} documents successfully"

        print("Upload API completed")

        return DocumentUploadResponse(
            status="success",
            message=message,
            upload_results=upload_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}", exc_info=True)
        event_logger.log_error(EventMessages.ERROR_UPLOAD_FAILED, auth_token=request.headers.get("Authorization", ""))
        return DocumentUploadResponse(
            status="failed",
            message=f"Failed to process upload: {str(e)}",
            upload_results=None
        )


@router.post("/query", response_model=DocumentQueryResponse, responses={400: {"description": "Bad Request"}})
@track_llm_calls(
        name="backend-qna",
        tags=["chat", "document-query"],
        metadata={"version": "1.0"},
        avoided_input_params=["request"]
    )
async def query_documents(request: Request, payload: DocumentQueryRequest):
    event_logger = create_event_logger()
    try:
        user_metadata_dict = json.loads(payload.user_metadata) if payload.user_metadata else {}
        session_id = user_metadata_dict.get("session_id")
        team_id = user_metadata_dict.get("team_id")
        auth_token = request.headers.get("Authorization", "")

        if not session_id or not team_id:
            detail = "session_id and team_id are required in user_metadata"
            raise HTTPException(status_code=400, detail=detail)

        _setup_request_tracing_and_context(request, user_metadata_dict)
        event_logger.log_event(EventMessages.QUERY_RECEIVED, auth_token=auth_token)

        team_config = await litellm_client.get_dynamic_llm_instance(team_id=team_id)

        # ── Semantic cache check (session-scoped namespace) ───────────────────
        sem_cache = get_semantic_cache()
        cached_payload = sem_cache.check(payload.query, namespace=session_id)
        if cached_payload:
            print("[SEMANTIC_CACHE] Returning cached response — skipping pipeline")
            event_logger.log_event(EventMessages.QUERY_COMPLETED, auth_token=auth_token)
            return DocumentQueryResponse(
                status="success",
                message=f"Retrieved {cached_payload.get('total_results', 0)} results for query",
                query_results=cached_payload,
            )
        # ──────────────────────────────────────────────────────────────────────

        with milvus.manage_collection():
            query_results = await _process_document_query(
                query=payload.query,
                session_id=session_id,
                top_k=10,
                llm_params=team_config,
                auth_token=auth_token
            )

        # ── Semantic cache store ───────────────────────────────────────────────
        sem_cache.store(payload.query, query_results, namespace=session_id)
        # ──────────────────────────────────────────────────────────────────────

        event_logger.log_event(EventMessages.QUERY_COMPLETED, auth_token=auth_token)

        return DocumentQueryResponse(
            status="success",
            message=f"Retrieved {query_results['total_results']} results for query",
            query_results=query_results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in document query: {str(e)}", exc_info=True)
        event_logger.log_error(EventMessages.ERROR_QUERY_FAILED, auth_token=request.headers.get("Authorization", ""))
        return DocumentQueryResponse(
            status="failed",
            message=f"Failed to process query: {str(e)}",
            query_results=None
        )



async def _process_document_query(
    query: str,
    session_id: str,
    top_k: int,
    llm_params: Dict[str, Any],
    auth_token: str
) -> Dict[str, Any]:
    """Process query with HyDE + multi-query + LLM synthesis"""
    event_logger = create_event_logger()
    try:
        logger.info(f"Enhanced query for session {session_id}: {query}")
        
        uploaded_files = milvus.get_uploaded_filenames_for_session(session_id)
        if not uploaded_files:
            raise HTTPException(
                status_code=400,
                detail="No documents found for this session. Please upload your files before querying."
            )
        
        output_fields = ["text", "metadata", "session_id", "team_id", "created_at"]
        
        event_logger.log_event(EventMessages.ANALYZING_DOCUMENT, auth_token=auth_token)
        search_results = await milvus.enhanced_search(
            query=query,
            session_id=session_id,
            llm_params=llm_params,
            auth_token=auth_token,
            use_hyde=True,
            use_multi_query=True,
            top_k=top_k,
            output_fields=output_fields
        )
        
        synthesis_result = await milvus.synthesize_answer_with_llm(
            query=query,
            retrieved_chunks=search_results,
            llm_params=llm_params,
            auth_token=auth_token,
            include_sources=True
        )
        
        event_logger.log_event(EventMessages.FINALIZING_RESULTS, auth_token=auth_token)

        llm_answer = synthesis_result['answer']

        agent_capabilities = [
            "Query documents and retrieve relevant information",
            "Search across uploaded documents using semantic search",
            "Summarize document content on a specific topic",
            "Compare information across multiple documents",
            "Extract specific data points or facts from documents",
            "Answer follow-up questions about document content",
        ]

        context = {
            "session_id": session_id,
            "documents_searched": search_results,
            "query": query,
        }

        follow_up_questions = await generate_follow_up_questions(
            user_query=query,
            response=llm_answer,
            agent_capabilities=agent_capabilities,
            context=context,
            llm_config=llm_params,
            num_questions=3
        )

        return {
            "query": query,
            "total_results": len(search_results),
            "llm_answer": llm_answer,
            "follow_up_questions": follow_up_questions
        }
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        event_logger.log_error(EventMessages.ERROR_QUERY_FAILED, auth_token=auth_token)
        raise

def _setup_request_tracing_and_context(request: Request, user_metadata_dict: Dict[str, Any]):
    """Helper to setup Opik tracing and OTEL context from request and metadata."""
    update_current_trace(
        user=user_metadata_dict.get('user_email'),
        message_id=user_metadata_dict.get('message_id'),
        team_id=user_metadata_dict.get('team_id'),
        organization_id=user_metadata_dict.get('organization_id'),
    )

    # ── OTEL: Set span attributes ──
    msg_id = user_metadata_dict.get("message_id")
    if msg_id:
        set_message_id(str(msg_id))
    
    uid = user_metadata_dict.get("user_id")
    if uid:
        set_user_context(user_id=str(uid), auth_mode="keycloak")

    # Extract user info from already-verified JWT payload in request state
    user_payload = getattr(request.state, "user", None)
    if user_payload:
        custom_data = user_payload.get("custom-data", {})
        email = custom_data.get("user_email") or user_payload.get("email")
        if email:
            set_user_context(user_email=email)

# ── Helpers for _process_document_upload ──────────────────────────────────────

def _build_langchain_docs(chunks: list) -> list:
    """Convert raw chunks to LangchainDocument objects"""
    langchain_docs = []
    for chunk in chunks:
        if isinstance(chunk, dict):
            metadata = {k: v for k, v in chunk.items() if k != "text"}
            langchain_docs.append(LangchainDocument(page_content=chunk.get("text", ""), metadata=metadata))
        elif isinstance(chunk, LangchainDocument):
            langchain_docs.append(chunk)
    return langchain_docs


def _build_preview_text(chunks: list) -> str:
    """Generate a short preview from first 3 chunks"""
    if not chunks:
        return ""
    preview = " ... ".join([c.get('text', '')[:200] for c in chunks[:3]])
    return preview[:500] + "..." if len(preview) > 500 else preview


def _ingest_to_milvus(documents_for_ingestion: dict, session_id: str, team_id: str, event_logger, auth_token: str, attachment_url_map: dict, document_source_map: List[str]) -> dict:
    """Ingest collected documents into Milvus"""
    if not documents_for_ingestion:
        return {"message": "No new documents to ingest"}
    try:
        event_logger.log_event(EventMessages.DOCUMENTS_INGESTED, auth_token=auth_token)
        return milvus.ingest_documents(
            documents=documents_for_ingestion,
            session_id=session_id,
            team_id=team_id,
            attachment_url_map=attachment_url_map,
            document_source_map=document_source_map
        )
    except Exception as e:
        logger.error(f"Error ingesting documents into Milvus: {str(e)}", exc_info=True)
        event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)
        return {"error": str(e)}

async def _process_single_file(
    s3_url: str,
    content: bytes,
    document_type,
    already_uploaded: set,
    ocr_processor,
    team_config: dict,
    event_logger,
    auth_token: str,
    session_id: str,
    team_id: str,
    attachment_url_map: dict,
) -> tuple:
    """
    Process one file. Returns (result, filename, langchain_docs or None).

    PDF streaming mode: chunks are ingested to Milvus per batch via callback.
    langchain_docs returned as [] (already ingested — skipped by _ingest_to_milvus).

    All other types: unchanged legacy behaviour, langchain_docs returned for
    bulk ingestion by _ingest_to_milvus.
    """
    from src.utils.s3_utility import extract_filename_from_s3_url
    from langchain_core.documents import Document as LangchainDocument

    doc_start_time = datetime.now()
    filename = extract_filename_from_s3_url(s3_url)

    if filename in already_uploaded:
        logger.info(f"Skipping already uploaded file: {filename}")
        return (
            DocumentProcessingResult(
                s3_url=s3_url, filename=filename, document_type="unknown",
                status="skipped", error="Document already exists in this session"
            ),
            filename, None
        )

    event_logger.log_event(EventMessages.ANALYZING_DOCUMENT, auth_token=auth_token)
    doc_type = _detect_document_type(filename) if document_type == DocumentType.AUTO else document_type.value
    event_logger.log_event(EventMessages.EXTRACTING_TEXT, auth_token=auth_token)

    # ── PDF: streaming ingestion path ─────────────────────────────────────────
    if doc_type == "pdf":
        return await _handle_pdf_processing(
            content, filename, s3_url, doc_type, doc_start_time,
            ocr_processor, team_config, auth_token, session_id, team_id, event_logger, attachment_url_map
        )

    # ── All other types: legacy path ──────────────────────────────────────────
    return await _handle_standard_processing(
        content, filename, s3_url, doc_type, doc_start_time,
        ocr_processor, team_config, auth_token, event_logger
    )

async def _handle_pdf_processing(
    content, filename, s3_url, doc_type, doc_start_time,
    ocr_processor, team_config, auth_token, session_id, team_id, event_logger,
    attachment_url_map
):
    """Handle PDF specific streaming ingestion."""

    event_logger.log_event(EventMessages.PROCESSING_OCR, auth_token=auth_token)
    streamed_chunk_count = 0

    async def _ingest_batch(chunks):
        nonlocal streamed_chunk_count
        if not chunks:
            return
        langchain_docs = [
            c if isinstance(c, LangchainDocument)
            else LangchainDocument(page_content=c.get("text", ""), metadata=c)
            for c in chunks
        ]
        print("langchain_docs", langchain_docs)
        import asyncio
        loop = asyncio.get_event_loop()
        def _sync_ingest():
            # NOTE: No manage_collection() here.
            # Collection lifecycle is owned by query_documents via manage_collection().
            # Wrapping ingest in a nested manage_collection() causes repeated
            # load/release cycles per PDF page batch (30-60s each) and produces
            # TaskStale on any concurrent search call.
            milvus.ingest_documents(
                documents={filename: langchain_docs},
                session_id=session_id,
                team_id=team_id,
                attachment_url_map=attachment_url_map,
                document_source_map=[s3_url]
            )
        await loop.run_in_executor(None, _sync_ingest)
        streamed_chunk_count += len(langchain_docs)

    extraction_result = await ocr_processor.extract_text_from_pdf_file(
        content=content,
        team_config=team_config,
        auth_token=auth_token,
        chunk_callback=_ingest_batch,
    )

    processing_time = (datetime.now() - doc_start_time).total_seconds()
    file_size = len(content) if isinstance(content, bytes) else None

    if "error" in extraction_result:
        event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)
        return (
            DocumentProcessingResult(
                s3_url=s3_url, filename=filename, document_type=doc_type,
                status="failed", error=extraction_result["error"],
                file_size=file_size, processing_time=processing_time
            ),
            filename, None
        )

    return (
        DocumentProcessingResult(
            s3_url=s3_url, filename=filename, document_type=doc_type,
            status="success",
            extracted_text=f"[Streamed {streamed_chunk_count} chunks to Milvus]",
            file_size=file_size, processing_time=processing_time
        ),
        filename, []
    )

async def _handle_standard_processing(
    content, filename, s3_url, doc_type, doc_start_time,
    ocr_processor, team_config, auth_token, event_logger
):
    """Handle non-PDF standard document ingestion."""
    extraction_result = await _process_document(
        content=content, doc_type=doc_type, filename=filename,
        ocr_processor=ocr_processor, team_config=team_config, auth_token=auth_token
    )

    processing_time = (datetime.now() - doc_start_time).total_seconds()
    file_size = len(content) if isinstance(content, bytes) else None

    if "error" in extraction_result:
        event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)
        return (
            DocumentProcessingResult(
                s3_url=s3_url, filename=filename, document_type=doc_type,
                status="failed", error=extraction_result["error"],
                file_size=file_size, processing_time=processing_time
            ),
            filename, None
        )

    chunks = extraction_result.get("chunks", [])
    langchain_docs = _build_langchain_docs(chunks) if chunks else None

    return (
        DocumentProcessingResult(
            s3_url=s3_url, filename=filename, document_type=doc_type,
            status="success", extracted_text=_build_preview_text(chunks),
            file_size=file_size, processing_time=processing_time
        ),
        filename, langchain_docs
    )

# ── Refactored _process_document_upload ───────────────────────────────────────

async def _process_document_upload(
    s3_urls: List[str],
    document_type: DocumentType,
    session_id: str,
    team_id: str,
    team_config: Dict[str, Any],
    auth_token: str,
    attachment_url_map: dict
) -> Dict[str, Any]:
    event_logger = create_event_logger()
    try:
        print("s3_urls 0:", s3_urls)
        s3_reader = S3Extraction()
        ocr_processor = Ocr(config_path="config/ocr_config.yaml")
        already_uploaded = milvus.get_uploaded_filenames_for_session(session_id)
        logger.info(f"Already uploaded files in session {session_id}: {already_uploaded}")

        file_contents = s3_reader.read_files(s3_urls)

        successful = failed = skipped = 0
        documents_for_ingestion = {}
        failed_details = []

        for s3_url, content in file_contents.items():
            try:
                MILVUS_DATABASE = os.getenv("MILVUS_DATABASE")
                MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION")
                # service = await call_agent_embed_api(
                #     file_url=s3_url,
                #     auth_token=auth_token,
                #     db_name=MILVUS_DATABASE,
                #     collection_name=MILVUS_COLLECTION,
                #     source="chat",
                # )
                # print("Archival Policy API Reponse: ", service)
                # logger.info(f"Archival Policy API Reponse {service}")

                result, filename, langchain_docs = await _process_single_file(
                    s3_url=s3_url,
                    content=content,
                    document_type=document_type,
                    already_uploaded=already_uploaded,
                    ocr_processor=ocr_processor,
                    team_config=team_config,
                    event_logger=event_logger,
                    auth_token=auth_token,
                    session_id=session_id,
                    team_id=team_id,
                    attachment_url_map=attachment_url_map,
                )
                if result.status == "skipped":
                    skipped += 1
                elif result.status == "failed":
                    failed += 1
                    failed_details.append({
                        "filename": result.filename,
                        "error": result.error
                    })
                else:
                    successful += 1
                    # langchain_docs == [] means PDF already streamed to Milvus
                    # langchain_docs == None means failed (caught above)
                    # langchain_docs == [...] means non-PDF, needs bulk ingest
                    if langchain_docs:
                        documents_for_ingestion[filename] = langchain_docs

                print("documents_for_ingestion", documents_for_ingestion)
                print("attachment_url_map", attachment_url_map)
            except Exception as e:
                failed += 1
                from src.utils.s3_utility import extract_filename_from_s3_url
                fname = extract_filename_from_s3_url(s3_url)
                error_msg = str(e)
                logger.error(f"Error processing {s3_url}: {error_msg}")
                failed_details.append({"filename": fname, "error": error_msg})
                event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)

        # Bulk ingest remaining non-PDF docs
        _ingest_to_milvus(documents_for_ingestion, session_id, team_id, event_logger, auth_token, attachment_url_map, s3_urls)

        return {
            "total_documents": len(s3_urls),
            "successful": successful,
            "failed": failed,
            "skipped": skipped,
            "failed_details": failed_details,
        }

    except Exception as e:
        logger.error(f"Error in document upload: {str(e)}", exc_info=True)
        event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)
        raise


# ── Helpers for _process_document ─────────────────────────────────────────────

async def _extract_zip_chunks(content: bytes, ocr_processor, team_config: dict, event_logger, auth_token: str) -> dict:
    """Extract and flatten chunks from a ZIP file"""
    event_logger.log_event(EventMessages.ZIP_PROCESSING_START, auth_token=auth_token)
    result = await ocr_processor.extract_text_from_zip(content, team_config, auth_token=auth_token)
    if isinstance(result, dict) and not result.get("error"):
        combined_chunks = []
        for file_name, file_chunks in result.items():
            event_logger.log_event(EventMessages.ZIP_FILES_ANALYZED, auth_token=auth_token)
            for chunk in file_chunks:
                chunk['zip_source'] = file_name
                combined_chunks.append(chunk)
        event_logger.log_event(EventMessages.ZIP_PROCESSING_COMPLETE, auth_token=auth_token)
        return {"chunks": combined_chunks}
    return result


async def _extract_image_chunks(content: bytes, ocr_processor, team_config: dict, event_logger, auth_token: str) -> dict:
    """Extract text from image and wrap in chunks format"""
    event_logger.log_event(EventMessages.PROCESSING_OCR, auth_token=auth_token)
    image_result = await ocr_processor.extract_text_from_image(content, team_config, auth_token=auth_token)
    if "text" in image_result:
        return {"chunks": [{"text": image_result["text"], "type": "image"}]}
    return image_result


# ── Refactored _process_document ──────────────────────────────────────────────

async def _process_document(
    content: bytes,
    doc_type: str,
    filename: str,
    ocr_processor,
    team_config: Dict[str, Any],
    auth_token: str
) -> Dict[str, Any]:
    event_logger = create_event_logger()
    try:
        logger.info(f"Processing {doc_type} document: {filename}")

        if doc_type == "pdf":
            event_logger.log_event(EventMessages.PROCESSING_OCR, auth_token=auth_token)
            return await ocr_processor.extract_text_from_pdf_file(content, team_config, auth_token=auth_token)

        if doc_type in ("docx", "doc"):
            event_logger.log_event(EventMessages.EXTRACTING_TEXT, auth_token=auth_token)
            return await ocr_processor.extract_text_from_docx(content, team_config, auth_token=auth_token)

        if doc_type in ("xlsx", "xls"):
            event_logger.log_event(EventMessages.EXTRACTING_TEXT, auth_token=auth_token)
            return await ocr_processor.extract_text_from_excel(content, filename)

        if doc_type in ("pptx", "ppt"):
            event_logger.log_event(EventMessages.EXTRACTING_TEXT, auth_token=auth_token)
            return await ocr_processor.extract_text_from_ppt(content, team_config, auth_token=auth_token)

        if doc_type == "zip":
            return await _extract_zip_chunks(content, ocr_processor, team_config, event_logger, auth_token)

        if doc_type in ("image", "png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"):
            return await _extract_image_chunks(content, ocr_processor, team_config, event_logger, auth_token)

        if doc_type in ("text", "txt", "md", "json", "xml", "html"):
            event_logger.log_event(EventMessages.EXTRACTING_TEXT, auth_token=auth_token)
            text = await ocr_processor.extract_text_from_text_file(content)
            return {"chunks": [{"text": text, "type": "text"}]}

        if doc_type == "csv":
            event_logger.log_event(EventMessages.EXTRACTING_TEXT, auth_token=auth_token)
            return ocr_processor.extract_text_from_csv(content, filename=filename)

        event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)
        return {"error": f"Unsupported document type: {doc_type}"}

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        event_logger.log_error(EventMessages.ERROR_PROCESSING_FAILED, auth_token=auth_token)
        return {"error": str(e)}

def _detect_document_type(filename: str) -> str:
    """
    Detect document type from filename extension or S3 URL
    """
    try:
        # Extract extension from filename
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        
        # Map extensions to document types
        type_mapping = {
            'pdf': 'pdf',
            'docx': 'docx',
            'doc': 'doc',
            'xlsx': 'xlsx',
            'xls': 'xls',
            'pptx': 'pptx',
            'ppt': 'ppt',
            'png': 'image',
            'jpg': 'image',
            'jpeg': 'image',
            'gif': 'image',
            'bmp': 'image',
            'tiff': 'image',
            'webp': 'image',
            'jfif': 'image',
            'txt': 'text',
            'md': 'text',
            'csv': 'text',
            'json': 'text',
            'xml': 'text',
            'html': 'text',
            'zip': 'zip'
        }
        
        detected_type = type_mapping.get(extension, 'unknown')
        logger.info(f"Detected document type: {detected_type} for file: {filename}")
        
        return detected_type
        
    except Exception as e:
        logger.warning(f"Could not detect document type: {str(e)}")
        return 'unknown'
