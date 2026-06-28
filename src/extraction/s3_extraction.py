
from src.utils.s3_utility import (
    get_s3_file,
    get_mime_type_from_s3_url,
    extract_filename_from_s3_url,
)
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import asyncio
from botocore.exceptions import ClientError
import mimetypes


# Configure logging
logger = logging.getLogger(__name__)


class S3Extraction:
    """
    A class to handle S3 file extraction, processing, and metadata management.
    Supports async operations, error handling, and file type validation.
    """

    def __init__(self, max_file_size: int = 50 * 1024 * 1024, timeout: int = 30):
        """
        Initialize S3Extraction instance.

        Args:
            max_file_size: Maximum file size in bytes (default: 50MB)
            timeout: Timeout for S3 operations in seconds (default: 30s)
        """
        self.urls = None
        self.file_content = None
        self.metadata = None
        self.max_file_size = max_file_size
        self.timeout = timeout
        self.supported_types = {
            "text": [".txt", ".md", ".csv", ".json"],
            "document": [".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx"],
            "media": [".jpg", ".jpeg", ".png", ".gif", ".mp4", ".mp3", ".wav"],
        }

    def read_files(self, s3_urls: List[str]) -> Dict[str, Any]:
        """
        Asynchronously read files from S3 URLs with error handling and validation.

        Args:
            s3_urls: List of S3 URLs to read

        Returns:
            Dictionary containing file contents keyed by filename

        Raises:
            ValueError: If s3_urls is empty or invalid
            Exception: If S3 operations fail
        """
        if not s3_urls:
            logger.error("S3 URLs list is empty")
            raise ValueError("s3_urls cannot be empty")

        if not isinstance(s3_urls, list):
            logger.error("s3_urls must be a list")
            raise TypeError("s3_urls must be a list of strings")

        try:
            file_content = {}
            
            for url in s3_urls:
                file_content[url] = get_s3_file(url)
                
            self.file_content = file_content
            self.urls = s3_urls
            return file_content

        except Exception as e:
            logger.error(f"[Error] in reading files: {str(e)}", exc_info=True)
            raise