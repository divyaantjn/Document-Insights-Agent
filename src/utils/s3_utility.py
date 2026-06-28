from typing import List
from fastapi import UploadFile, HTTPException
import logging
import os
import asyncio
import mimetypes
import boto3
from botocore.client import Config
import urllib.parse
from urllib.parse import urlparse
from dotenv import load_dotenv
from pathlib import Path


load_dotenv()

logger = logging.getLogger(__name__)

APPLICATION_STREAM = 'application/octet-stream'
DOMAIN = '.amazonaws.com'
HTTPS = 'https://'
S3_PREFIX = 's3://'
KEY_NOT_FOUND = "No file key found in URL"
# Create a consistent S3 client configuration
def get_s3_client():

    """Get properly configured S3 client with AWS4-HMAC-SHA256 signature"""
    return boto3.client(
        's3',
        region_name=os.getenv("AWS_REGION"),
        config=Config(
            signature_version='s3v4',
            region_name=os.getenv("AWS_REGION"),
            retries={'max_attempts': 3, 'mode': 'standard'},
            max_pool_connections=50
        )
    )

S3_BUCKET = os.getenv("S3_BUCKET_NAME")

async def upload_any_file_to_s3(file: UploadFile, folder_name: str) -> str:
    """
    Your generalized upload function which uploads the file to S3 under a 'temp/{user_id}/' prefix
    and returns a presigned S3 URL.
    """
    
    s3_client = get_s3_client()
    
    def sanitize_filename(filename: str) -> str:
        return "".join(
            c for c in filename if c.isalnum() or c in (' ', '.', '_', '-')
        ).rstrip()
    
    filename = sanitize_filename(file.filename)
    
    s3_key = f"{folder_name}/{filename}"
    
    content = await file.read()
    
    mime_type, _ = mimetypes.guess_type(filename)
    content_type = file.content_type or mime_type or APPLICATION_STREAM
    
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: s3_client.put_object(
                Bucket=S3_BUCKET,
                Key=s3_key,
                Body=content,
                ContentType=content_type
            )
        )
        
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": s3_key},
            ExpiresIn=3600
        )
        return presigned_url
        
    except Exception as e:
        logger.error(f"Failed to upload file to S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

def copy_s3_file_to_new_path(s3_url: str, new_folder: str, user_id: str) -> tuple[str, str]:
    """
    Copy an S3 file from a presigned URL or S3 path to another within the same bucket.
    """
    try:
        s3_client = get_s3_client()
        
        # Parse different URL formats
        if s3_url.startswith(HTTPS):
            parsed = urlparse(s3_url)
            
            # Handle both path-style and virtual-hosted-style URLs
            if parsed.netloc.endswith(DOMAIN):
                # Virtual-hosted-style: bucket.s3.region.amazonaws.com
                if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                    bucket_name = S3_BUCKET  # Use your defined bucket
                    source_key = urllib.parse.unquote(parsed.path.lstrip('/'))
                # Path-style: s3.region.amazonaws.com/bucket
                elif parsed.netloc.startswith('s3.') or parsed.netloc.startswith('s3-'):
                    path_parts = parsed.path.lstrip('/').split('/', 1)
                    bucket_name = path_parts[0] if path_parts else S3_BUCKET
                    source_key = urllib.parse.unquote(path_parts[1]) if len(path_parts) > 1 else ""
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_url}")
            else:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            
        elif s3_url.startswith(S3_PREFIX):
            path_parts = s3_url[5:].split('/', 1)
            bucket_name = str(path_parts[0])
            source_key = str(path_parts[1]) if len(path_parts) > 1 else ""
            
        else:
            raise ValueError(f"Unsupported URL format: {s3_url}")
        
        if not source_key:
            raise ValueError(KEY_NOT_FOUND)
        
        # Debug logging
        logger.info(f"Bucket: {bucket_name}")
        logger.info(f"Source key: {source_key}")
        
        # Extract filename
        filename = source_key.split('/')[-1]
        filename = urllib.parse.unquote(filename)
        
        # Create new key structure
        new_key = f"{new_folder}/{user_id}/{filename}"
        
        # Define copy source
        copy_source = {
            'Bucket': str(bucket_name), 
            'Key': str(source_key)
        }
        
        logger.info(f"Copying from {copy_source} to {new_key}")
        
        # Copy to new location
        s3_client.copy_object(
            CopySource=copy_source,
            Bucket=S3_BUCKET,  # Always copy to your defined bucket
            Key=str(new_key)
        )
        
        logger.info(f"Successfully copied {source_key} to {new_key}")
        
        # Generate presigned URL for new location
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": S3_BUCKET, "Key": str(new_key)},
            ExpiresIn=3600
        )
        
        return presigned_url, filename
        
    except Exception as e:
        logger.error(f"Failed to copy S3 file: {str(e)}")
        logger.error(f"URL: {s3_url}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"S3 copy operation failed: {str(e)}")
    
def get_s3_file(s3_url: str) -> bytes:
    """
    Get file bytes from S3 storage using various URL formats.
    
    Args:
        s3_url: S3 URL (presigned URL or s3:// format)
        user_id: User ID for logging/tracking purposes
    
    Returns:
        bytes: Binary content of the S3 object
    """
    try:
        s3_client = get_s3_client()
        
        # Parse different URL formats (same logic as copy function)
        if s3_url.startswith(HTTPS):
            parsed = urlparse(s3_url)
            
            # Handle both path-style and virtual-hosted-style URLs
            if parsed.netloc.endswith(DOMAIN):
                if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                    bucket_name = S3_BUCKET
                    source_key = urllib.parse.unquote(parsed.path.lstrip('/'))
                elif parsed.netloc.startswith('s3.') or parsed.netloc.startswith('s3-'):
                    path_parts = parsed.path.lstrip('/').split('/', 1)
                    bucket_name = path_parts[0] if path_parts else S3_BUCKET
                    source_key = urllib.parse.unquote(path_parts[1]) if len(path_parts) > 1 else ""
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_url}")
            else:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            
        elif s3_url.startswith(S3_PREFIX):
            path_parts = s3_url[5:].split('/', 1)
            bucket_name = str(path_parts[0])
            source_key = str(path_parts[1]) if len(path_parts) > 1 else ""
            
        else:
            raise ValueError(f"Unsupported URL format: {s3_url}")
        
        if not source_key:
            raise ValueError(KEY_NOT_FOUND)
        
        # Debug logging
        logger.info(f"Bucket: {bucket_name}")
        logger.info(f"Source key: {source_key}")
        
        # Get object from S3
        s3_response = s3_client.get_object(
            Bucket=str(bucket_name), 
            Key=str(source_key)
        )
        
        # Read binary data from StreamingBody
        binary_data = s3_response['Body'].read()
        
        logger.info(f"Successfully retrieved {len(binary_data)} bytes from {source_key}")
        
        return binary_data
        
    except Exception as e:
        logger.error(f"Failed to get S3 file: {str(e)}")
        logger.error(f"URL: {s3_url}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"S3 get operation failed: {str(e)}")

def delete_s3_file(s3_url: str):
    """Delete the file from S3 storage
    Args:
        s3_url: S3 URL
    """
    try:
        s3_client = get_s3_client()
        
        # Parse different URL formats (same logic as other functions)
        if s3_url.startswith(HTTPS):
            parsed = urlparse(s3_url)
            
            if parsed.netloc.endswith(DOMAIN):
                if '.s3.' in parsed.netloc or '.s3-' in parsed.netloc:
                    bucket_name = S3_BUCKET
                    source_key = urllib.parse.unquote(parsed.path.lstrip('/'))
                elif parsed.netloc.startswith('s3.') or parsed.netloc.startswith('s3-'):
                    path_parts = parsed.path.lstrip('/').split('/', 1)
                    bucket_name = path_parts[0] if path_parts else S3_BUCKET
                    source_key = urllib.parse.unquote(path_parts[1]) if len(path_parts) > 1 else ""
                else:
                    raise ValueError(f"Invalid S3 URL format: {s3_url}")
            else:
                raise ValueError(f"Invalid S3 URL format: {s3_url}")
            
        elif s3_url.startswith(S3_PREFIX):
            path_parts = s3_url[5:].split('/', 1)
            bucket_name = str(path_parts[0])
            source_key = str(path_parts[1]) if len(path_parts) > 1 else ""
            
        else:
            raise ValueError(f"Unsupported URL format: {s3_url}")
        
        if not source_key:
            raise ValueError(KEY_NOT_FOUND)
        
        # Debug logging
        logger.info(f"Deleting from bucket: {bucket_name}")
        logger.info(f"Source key: {source_key}")

        s3_client.delete_object(Bucket=bucket_name, Key=source_key)
        logger.info(f"Object '{source_key}' deleted from bucket '{bucket_name}'.")
    
    except Exception as e:
        logger.error(f"Failed to delete S3 file: {str(e)}")
        logger.error(f"URL: {s3_url}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise ValueError(f"S3 delete operation failed: {str(e)}")

# Additional helper function to test S3 connectivity
def test_s3_connection():
    """Test S3 connection and permissions"""
    try:
        get_s3_client()
        
        logger.info("S3 connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"S3 connection test failed: {str(e)}")
        return False
    
def get_mime_type_from_s3_url(s3_url: str) -> str:
    """Get MIME type from S3 URL using head_object"""
    try:
        s3_client = get_s3_client()
        
        # Parse S3 URL to extract bucket and key
        if s3_url.startswith(HTTPS):
            # Parse presigned URL or S3 endpoint URL
            parsed = urlparse(s3_url)
            
            # Extract bucket from hostname (for standard S3 URLs)
            host_parts = parsed.netloc.split('.')
            if len(host_parts) >= 2 and host_parts[1] == 's3':
                bucket_name = host_parts
            else:
                # Handle different S3 endpoint formats
                bucket_name = S3_BUCKET  # You may need to extract this differently
            
            object_key = parsed.path.lstrip('/')
            
        elif s3_url.startswith(S3_PREFIX):
            # Parse s3:// format
            path_parts = s3_url[5:].split('/', 1)
            bucket_name = path_parts
            object_key = path_parts[27] if len(path_parts) > 1 else ""
        
        # Get object metadata without downloading content
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        
        # Return ContentType from metadata
        return response.get('ContentType', APPLICATION_STREAM)
        
    except Exception as e:
        print(f"Error getting MIME type: {str(e)}")
        return APPLICATION_STREAM  # Default fallback

def extract_filename_from_s3_url(s3_url):
    parsed_url = urllib.parse.urlparse(s3_url)
    return Path(parsed_url.path).name

def get_content_type_from_extension(filename):
    extension = os.path.splitext(filename)[1].lower()
    content_type_map = {
        '.pdf': 'application/pdf',
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation'
    }
    return content_type_map.get(extension, APPLICATION_STREAM)