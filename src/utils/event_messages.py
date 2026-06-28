# src/utils/event_messages.py
"""
Constants file containing predefined messages for the custom Kafka event logger.
These messages provide user-friendly descriptions of what the IDP Backend is doing.
Simplified to fewer, more meaningful messages.
"""


class EventMessages:
    """
    Centralized collection of event messages for the Kafka event logger.
    Simplified to essential messages that provide clear value to users.
    """

    # CORE WORKFLOW EVENTS - What users care about
    TASK_RECEIVED = "Document processing request received"

    ANALYZING_DOCUMENT = "Analyzing document content"

    EXTRACTING_TEXT = "Extracting text from document"

    PROCESSING_OCR = "Running OCR on document"

    PROCESSING_NER = "Extracting entities from document"

    FINALIZING_RESULTS = "Finalizing processing results"

    TASK_COMPLETED = "Document processing completed"

    # DOCUMENT UPLOAD EVENTS
    UPLOAD_START = "Document upload started"

    UPLOAD_COMPLETE = "Document upload completed"

    DOCUMENTS_INGESTED = "Documents ingested into vector store"

    # QUERY EVENTS
    QUERY_RECEIVED = "Document query received"

    QUERY_COMPLETED = "Query results retrieved"

    # ZIP PROCESSING SPECIFIC EVENTS
    ZIP_PROCESSING_START = "ZIP file processing started"

    ZIP_FILES_ANALYZED = "ZIP files analyzed"

    ZIP_PROCESSING_COMPLETE = "ZIP processing completed"

    # ERROR EVENTS
    ERROR_INVALID_REQUEST = "Invalid request or content"

    ERROR_PROCESSING_FAILED = "Document processing failed"

    ERROR_OCR_FAILED = "OCR extraction failed"

    ERROR_NER_FAILED = "Entity extraction failed"

    ERROR_UPLOAD_FAILED = "Document upload failed"

    ERROR_QUERY_FAILED = "Document query failed"

    ERROR_SYSTEM_UNAVAILABLE = "System unavailable, try later"


class EventTypes:
    """
    Event type categories for classification
    """

    SYSTEM = "system"
    AGENT = "agent"
    TASK = "task"
    DOCUMENT = "document"
    OCR = "ocr"
    NER = "ner"
    UPLOAD = "upload"
    QUERY = "query"
    SUCCESS = "success"
    ERROR = "error"
    PROGRESS = "progress"


class EventPriority:
    """
    Event priority levels
    """

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"
