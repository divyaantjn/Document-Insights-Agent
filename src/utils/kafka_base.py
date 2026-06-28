# src/utils/kafka_base.py

import os
import json
import logging
import atexit
import threading
from typing import Optional, Any

# Kafka dependency handling
try:
    from kafka import KafkaProducer
    from kafka.errors import NoBrokersAvailable  # type: ignore[assignment]
    KAFKA_INSTALLED = True
except ImportError:
    KAFKA_INSTALLED = False
    KafkaProducer = None  # type: ignore[misc,assignment]
    class NoBrokersAvailable(Exception):
        pass

logger = logging.getLogger(__name__)

# Constants
CUSTOM_TOKEN_SEPARATOR = "$YashUnified2025$"


def extract_user_context(auth_token: Optional[str] = None) -> dict:
    """
    Extract encrypted_payload from auth token.
    
    Args:
        auth_token: Authorization header containing encrypted payload
        
    Returns:
        Dictionary with 'encrypted_payload' key
    """
    user_context = {"encrypted_payload": "NO_PAYLOAD"}
    if not auth_token:
        return user_context
    
    try:
        if CUSTOM_TOKEN_SEPARATOR in auth_token:
            _, encrypted_payload = auth_token.split(CUSTOM_TOKEN_SEPARATOR, 1)
            user_context["encrypted_payload"] = encrypted_payload
        else:
            jwt_part = auth_token
            if jwt_part.lower().startswith("bearer "):
                jwt_part = jwt_part[7:]
            user_context["encrypted_payload"] = f"mock-encrypted-payload-{jwt_part[-10:]}" if jwt_part else "NO_PAYLOAD"
    except Exception as e:
        logger.debug(f"Error extracting user context from token: {e}")
    
    return user_context


class KafkaManager:
    """Singleton Kafka Producer Manager."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self.producer = None
        self._producer_lock = threading.Lock()
        self._initialized = True
    
    def get_producer(self) -> Any:
        """Get or initialize the Kafka producer."""
        if not self.producer:
            with self._producer_lock:
                if not self.producer:
                    self._initialize_producer()
        return self.producer
    
    def _initialize_producer(self) -> bool:
        """Initialize the KafkaProducer singleton."""
        if not KAFKA_INSTALLED:
            logger.critical("kafka-python not installed. Kafka logging disabled.")
            return False
        
        bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        if not bootstrap_servers:
            logger.critical("KAFKA_BOOTSTRAP_SERVERS not set. Kafka logging disabled.")
            return False
        
        try:
            logger.info(f"Initializing Kafka producer and connecting to {bootstrap_servers}...")
            producer_config = {
                "bootstrap_servers": bootstrap_servers.split(","),
                "value_serializer": lambda v: json.dumps(v, default=str).encode("utf-8"),
                "key_serializer": lambda k: k.encode("utf-8") if k else None,
                "retries": 2,
                "request_timeout_ms": 10000,
                "acks": 1,
                "linger_ms": 10,
                "batch_size": 1024,
                "max_block_ms": 3000,
                "connections_max_idle_ms": 180000,
                "metadata_max_age_ms": 30000,
                "api_version_auto_timeout_ms": 2000
            }
            
            if os.getenv("KAFKA_USE_PLAINTEXT", "false").lower() == "true":
                producer_config["security_protocol"] = "PLAINTEXT"            
            if os.getenv("KAFKA_USE_SSL", "true").lower() == "true":
                producer_config["security_protocol"] = "SSL"
            
            if KafkaProducer is not None:
                self.producer = KafkaProducer(**producer_config)
                logger.info("Kafka Producer connected successfully.")
                return True
            else:
                logger.critical("KafkaProducer class not available")
                return False
        except (NoBrokersAvailable, Exception) as e:
            logger.critical(f"FATAL: Could not initialize Kafka producer. Error: {e}", exc_info=True)
            self.producer = None
            return False
    
    def close(self):
        """Close the Kafka producer."""
        if self.producer:
            try:
                logger.info("Closing Kafka producer...")
                self.producer.flush(timeout=5)
                self.producer.close(timeout=5)
                logger.info("Kafka producer closed.")
            except Exception as e:
                logger.debug(f"Error closing Kafka producer: {e}")
            finally:
                self.producer = None


# Register cleanup on exit
_kafka_manager = KafkaManager()
atexit.register(_kafka_manager.close)