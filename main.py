"""
Main FastAPI Application
Handles email extraction and S3 document processing with OCR
"""

import os
import logging

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# TRACING INITIALIZATION - MUST happen before any other imports that use OTEL
# =============================================================================

# Initialize OpenTelemetry tracing FIRST!
from src.utils.tracing import setup_tracing, instrument_fastapi_app
setup_tracing()

# =============================================================================
# APPLICATION IMPORTS
# =============================================================================

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pathlib import Path
from src.middleware.auth_middleware import KeycloakAuthMiddleware
from src.middleware.license_middleware import LicenseMiddleware 
from src.utils.opik_setup import setup_opik_tracing, flush_traces, OPIKMiddleware

setup_opik_tracing(
    llm_only=False,
    enable_litellm=True,
    enable_genai=False
)

# Import routers
from src.routers import s3_ocr_router

# Import for exception handlers
from src.utils.error_capture import capture_http_error_details, capture_validation_error

# Import XRay tracing middleware
from middleware.xray_tracing_middleware import XRayTracingMiddleware

# Service name for tracing
SERVICE_NAME = os.getenv("OTEL_SERVICE_NAME", "yash-unified-idp-backend")

# =============================================================================
# FASTAPI APP CREATION
# =============================================================================
APP_NAME = "Email & Document Processing API"

app = FastAPI(
    title=APP_NAME,
    description="API for extracting emails from Gmail/Outlook and processing documents from S3 with OCR",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": APP_NAME
    }

app.add_middleware(OPIKMiddleware)
app.add_middleware(LicenseMiddleware)

# Authentication Middleware
app.add_middleware(KeycloakAuthMiddleware)

# =============================================================================
# EXCEPTION HANDLERS - Capture errors for tracing
# =============================================================================

@app.exception_handler(FastAPIHTTPException)
async def fastapi_http_exception_handler(
    request: Request,
    exc: FastAPIHTTPException,
):
    """Capture HTTP exceptions for tracing."""
    error_details = {
        "message": exc.detail,
        "status_code": exc.status_code
    }
    capture_http_error_details(exc.status_code, error_details)

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Capture validation errors for tracing."""
    capture_validation_error(exc.errors())

    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "details": exc.errors(),
        },
    )

# =============================================================================
# MIDDLEWARE - Order matters! Last added = First executed
# =============================================================================


# XRay Tracing middleware (added LAST = runs FIRST = creates segment)
app.add_middleware(XRayTracingMiddleware, service_name=SERVICE_NAME)

# Instrument FastAPI app for OpenTelemetry tracing
instrument_fastapi_app(app)

# =============================================================================
# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ROUTERS
# =============================================================================

app.include_router(
    s3_ocr_router.router,
    prefix="/api/v1/documents",
    tags=["Document Processing"]
)

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": APP_NAME,
        "version": "1.0.0",
        "endpoints": {
            "email_extraction": "/api/v1/email/extract",
            "s3_document_processing": "/api/v1/documents/process",
            "health_check": "/health"
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
