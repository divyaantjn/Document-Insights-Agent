"""Failsafe Opik tracing for self-hosted deployment (Backend Server)."""
import os
import functools
import inspect
import json
import logging
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from enum import Enum
from contextlib import contextmanager
import traceback
import asyncio
from starlette.middleware.base import BaseHTTPMiddleware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state management
_opik_initialized = False
_opik_client = None
_tracing_enabled = True
_llm_only_mode = True  # Only track LLM calls, not all functions

class LLMProvider(Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    LITELLM = "litellm"
    LANGCHAIN_CHAT = "langchain_chat"
    UNKNOWN = "unknown"


class OpikTraceData:
    """Container for trace metadata."""
    def __init__(self, trace_id: str = None, span_id: str = None, 
                 provider: str = None, tokens: Dict = None):
        self.trace_id = trace_id
        self.span_id = span_id
        self.provider = provider
        self.tokens = tokens or {}
        self.created_at = datetime.now().isoformat()


def setup_opik_tracing(
    enable_tracing: bool = True,
    enable_litellm: bool = True,
    enable_genai: bool = True,
    llm_only: bool = True,
    project_name: Optional[str] = None,
    url: Optional[str] = None,
    workspace: str = "default"
) -> bool:
    """
    Initialize Opik tracing with multi-provider support.
    
    Args:
        enable_tracing: Enable/disable all tracing
        enable_litellm: Auto-integrate LiteLLM callbacks
        enable_genai: Auto-integrate Google GenAI
        llm_only: Only track LLM operations (recommended: True)
        project_name: Opik project name
        url: Opik server URL
        workspace: Opik workspace name
    
    Returns:
        bool: True if successfully initialized, False otherwise
    
    Example:
        >>> setup_opik_tracing(
        ...     project_name="email-agent",
        ...     url="http://localhost:5173",
        ...     llm_only=True
        ... )
        True
    """
    global _opik_initialized, _opik_client, _tracing_enabled, _llm_only_mode
    
    if _opik_initialized:
        print("Opik already initialized")
        return _opik_initialized
    
    _tracing_enabled = enable_tracing
    _llm_only_mode = llm_only
    
    if not _tracing_enabled:
        print("Tracing disabled by configuration")
        return True
    
    try:
        import opik
        
        # Configuration resolution
        project = project_name or os.getenv("OPIK_PROJECT_NAME") or \
                  os.getenv("AGENT_NAME_CONSTANT") or "DEFAULT_PROJECT"
        opik_url = url or os.getenv("OPIK_URL_OVERRIDE")
        opik_workspace = os.getenv("OPIK_WORKSPACE", "default")
        
        if not opik_url:
            print(
                "Opik URL not provided. Set OPIK_URL_OVERRIDE or pass url parameter. "
                "Tracing will be disabled."
            )
            return False
        
        # Set environment variables
        os.environ.setdefault("OPIK_PROJECT_NAME", project)
        os.environ["OPIK_URL_OVERRIDE"] = opik_url
        os.environ["OPIK_WORKSPACE"] = opik_workspace
        os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY", "")
        
        try:
            _opik_client = opik.Opik(host=opik_url, workspace=opik_workspace)
            _opik_initialized = True
            print(
                f"Opik initialized: project={project}, url={opik_url}, "
                f"workspace={opik_workspace}, llm_only_mode={llm_only}"
            )
        except ConnectionError as e:
            print(f"ERROR: Failed to connect to Opik server at {opik_url}: {e}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to initialize Opik client: {e}")
            return False
        
        # Setup provider integrations
        if enable_litellm:
            _setup_litellm_integration()
        
        if enable_genai:
            _setup_genai_integration()
        
        return True
        
    except ImportError:
        print("WARNING: Opik library not installed. Install with: pip install opik")
        return False
    except Exception as e:
        print(f"ERROR: Unexpected error during Opik setup: {e}\n{traceback.format_exc()}")
        return False


def _safe_update_span(opik_context, **kwargs):
    """Safely update the current Opik span if tracing is enabled."""
    if not _opik_initialized or not _tracing_enabled:
        return
    try:
        opik_context.update_current_span(**kwargs)
    except Exception:
        pass

def _prepare_litellm_input(args, kwargs):
    """Prepare standardized input for LiteLLM tracing."""
    model = kwargs.get('model', args[0] if args else 'unknown')
    messages = kwargs.get('messages', [])
    return {
        "model": model,
        "messages": [
            _serialize_data(m) if isinstance(m, dict)
            else {"content": str(m)}
            for m in (messages if isinstance(messages, list) else [])
        ]
    }

def _update_span_with_litellm_result(opik_context, result):
    """Extract and update span with LiteLLM result data."""
    try:
        content = ""
        if hasattr(result, 'choices') and result.choices:
            content = getattr(result.choices[0].message, 'content', '') or ''
        
        usage = getattr(result, 'usage', None)
        token_usage = {
            "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
            "completion_tokens": getattr(usage, 'completion_tokens', 0),
            "total_tokens": getattr(usage, 'total_tokens', 0),
        } if usage else {}

        opik_context.update_current_span(
            output={"content": content},
            metadata={"token_usage": token_usage}
        )
    except Exception:
        pass

def _perform_patched_litellm_call(func, input_data, *args, **kwargs):
    """Internal helper for Tier-2 litellm patching."""
    from opik import opik_context
    _safe_update_span(opik_context, input=input_data)
    result = func(*args, **kwargs)
    _update_span_with_litellm_result(opik_context, result)
    return result

async def _perform_patched_litellm_call_async(func, input_data, *args, **kwargs):
    """Internal helper for Tier-2 litellm patching (async)."""
    from opik import opik_context
    _safe_update_span(opik_context, input=input_data)
    result = await func(*args, **kwargs)
    _update_span_with_litellm_result(opik_context, result)
    return result

class _SafeFallbackCallback:
    """Fallback callback for LiteLLM when native integration is unavailable."""
    def _serialize_msgs(self, msgs):
        if isinstance(msgs, list):
            serialized = [
                _serialize_data(m) if isinstance(m, dict)
                else {"content": str(m)} for m in msgs
            ]
            return {"messages": serialized}
        return {"messages": str(msgs)}

    def log_pre_api_call(self, model, messages, _kwargs):
        try:
            from opik import opik_context
            _safe_update_span(
                opik_context,
                input=self._serialize_msgs(messages),
                tags=["litellm", model],
                metadata={"provider": "litellm", "model": model}
            )
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            from opik import opik_context
            content = ""
            if hasattr(response_obj, 'choices') and response_obj.choices:
                content = getattr(response_obj.choices[0].message, 'content', '') or ''
            
            usage = getattr(response_obj, 'usage', None)
            duration = int((end_time - start_time).total_seconds() * 1000)
            
            _safe_update_span(
                opik_context,
                output={"content": content[:2000]},
                metadata={
                    "model": kwargs.get("model", "unknown"),
                    "duration_ms": duration,
                    "token_usage": {
                        "prompt_tokens": getattr(usage, 'prompt_tokens', 0),
                        "completion_tokens": getattr(usage, 'completion_tokens', 0),
                        "total_tokens": getattr(usage, 'total_tokens', 0),
                    } if usage else {}
                }
            )
        except Exception:
            pass

    def log_failure_event(self, kwargs, response_obj, _start_time, _end_time):
        try:
            from opik import opik_context
            error_msg = str(kwargs.get("exception", response_obj))
            _safe_update_span(
                opik_context,
                output={"error": error_msg[:1000]},
                metadata={"model": kwargs.get("model", "unknown"), "status": "error"}
            )
        except Exception:
            pass

def _get_safe_fallback_callback(custom_logger_class):
    """Factory for SafeFallbackCallback to avoid class nesting complexity."""
    class SafeFallbackCallback(_SafeFallbackCallback, custom_logger_class):
        pass
    return SafeFallbackCallback

class LiteLLMIntegration:
    """Encapsulates LiteLLM tracing integration logic."""
    def __init__(self, litellm, opik):
        self.litellm = litellm
        self.opik = opik

    def setup(self):
        """Try tiers in order and return True if any succeed."""
        # Tier 1: Native OpikLogger
        if self._setup_tier1():
            return True

        # Tier 2: Monkey-patching
        if self._setup_tier2():
            return True

        # Tier 3: Callback fallback
        return self._setup_tier3()

    def _setup_tier1(self) -> bool:
        """Native opik LiteLLM integration."""
        try:
            from opik.integrations.litellm import OpikLogger
            if hasattr(self.litellm, 'callbacks') and isinstance(self.litellm.callbacks, list):
                self.litellm.callbacks = [
                    cb for cb in self.litellm.callbacks 
                    if not (hasattr(cb, '__class__') and cb.__class__.__name__ == 'OpikLogger')
                ]
            else:
                self.litellm.callbacks = []
            self.litellm.callbacks.append(OpikLogger())
            print("LiteLLM Tier-1 (native OpikLogger) enabled")
            return True
        except Exception as err:
            print(f"ERROR: Tier-1 failed ({err}), trying Tier 2")
            return False

    def _setup_tier2(self) -> bool:
        """Monkey-patch litellm.completion / acompletion."""
        try:
            _orig_completion = getattr(self.litellm, 'completion', None)
            _orig_acompletion = getattr(self.litellm, 'acompletion', None)

            if _orig_completion and not getattr(_orig_completion, '_opik_patched', False):
                self._patch_sync(_orig_completion)

            if _orig_acompletion and not getattr(_orig_acompletion, '_opik_patched', False):
                self._patch_async(_orig_acompletion)

            print("LiteLLM Tier-2 (monkey-patch @opik.track) enabled — nested spans supported")
            return True
        except Exception as err:
            print(f"ERROR: Tier-2 monkey-patch failed ({err}), trying Tier 3")
            return False

    def _patch_sync(self, _orig):
        @functools.wraps(_orig)
        def _patched(*args, **kwargs):
            safe_input = _prepare_litellm_input(args, kwargs)
            model = safe_input["model"]

            @self.opik.track(
                name=f"litellm.{model}",
                tags=["litellm", "llm_call"],
                metadata={"provider": "litellm", "model": model},
                capture_input=False
            )
            def _do(input_data=None):
                return _perform_patched_litellm_call(_orig, input_data, *args, **kwargs)

            return _do(input_data=safe_input)
        _patched._opik_patched = True
        self.litellm.completion = _patched

    def _patch_async(self, _orig):
        @functools.wraps(_orig)
        async def _patched(*args, **kwargs):
            safe_input = _prepare_litellm_input(args, kwargs)
            model = safe_input["model"]

            @self.opik.track(
                name=f"litellm.{model}",
                tags=["litellm", "llm_call"],
                metadata={"provider": "litellm", "model": model},
                capture_input=False
            )
            async def _do_async(input_data=None):
                return await _perform_patched_litellm_call_async(_orig, input_data, *args, **kwargs)

            return await _do_async(input_data=safe_input)
        _patched._opik_patched = True
        self.litellm.acompletion = _patched

    def _setup_tier3(self) -> bool:
        """Custom callback (last resort)."""
        try:
            from litellm.integrations.custom_logger import CustomLogger
            callback_class = _get_safe_fallback_callback(CustomLogger)

            if not hasattr(self.litellm, 'callbacks') or not isinstance(self.litellm.callbacks, list):
                self.litellm.callbacks = []
            
            self.litellm.callbacks = [
                cb for cb in self.litellm.callbacks
                if not (hasattr(cb, '__class__') and 'Opik' in cb.__class__.__name__)
            ]
            self.litellm.callbacks.append(callback_class())
            print("LiteLLM Tier-3 (callback fallback) enabled — parent span annotation only")
            return True
        except Exception as e:
            print(f"ERROR: LiteLLM tier-3 fallback setup failed: {e}")
            return False

def _setup_litellm_integration():
    """Setup LiteLLM -> Opik integration (3-tier strategy)."""
    try:
        import litellm
        import opik
        integration = LiteLLMIntegration(litellm, opik)
        integration.setup()
    except ImportError:
        print("ERROR: LiteLLM or Opik not installed, skipping integration")
    except Exception as e:
        print(f"ERROR: LiteLLM integration setup failed: {e}")

def _setup_genai_integration():
    """Setup Google GenAI integration."""
    print("GenAI integration ready (manual wrapping required)")

# ============================================================================
# LLM Client Wrapper - Zero-modification integration
# ============================================================================

class LLMClient:
    """
    Universal LLM client wrapper that detects and wraps LLM calls.
    
    Supports: Gemini, LiteLLM, LangChain ChatModels, OpenAI, Anthropic
    
    Features:
        - Automatic provider detection
        - Method interception for LLM operations only
        - Transparent fallback on errors
        - Token usage tracking
    
    Usage:
        >>> from google import genai
        >>> client = genai.Client()
        >>> wrapped_client = LLMClient.wrap(client, provider="gemini")
        >>> response = wrapped_client.models.generate_content(...)
    """
    
    # Method names that indicate LLM operations
    LLM_METHOD_PATTERNS = {
        'generate', 'complete', 'chat', 'create', 'embed',
        'invoke', 'agenerate', 'astream', 'predict'
    }
    
    @staticmethod
    def wrap(client: Any, provider: Union[str, LLMProvider] = None, 
             track_all: bool = False) -> Any:
        """
        Wrap an LLM client for transparent tracing.
        
        Args:
            client: The LLM client to wrap (genai.Client, ChatOpenAI, etc.)
            provider: Provider name or LLMProvider enum
            track_all: Track all methods (default: False = LLM methods only)
        
        Returns:
            Wrapped client (same interface as original)
        
        Raises:
            No exceptions - returns original client on wrapper failure
        
        Example:
            >>> # Gemini
            >>> from google import genai
            >>> client = LLMClient.wrap(genai.Client(), provider="gemini")
            >>> 
            >>> # LangChain
            >>> from langchain.chat_models import ChatOpenAI
            >>> llm = LLMClient.wrap(ChatOpenAI(), provider="langchain_chat")
            >>> 
            >>> # LiteLLM
            >>> import litellm
            >>> llm = LLMClient.wrap(litellm, provider="litellm")
        """
        if not _opik_initialized or not _tracing_enabled:
            return client
        
        try:
            # Resolve provider
            if provider:
                if isinstance(provider, str):
                    provider = LLMProvider[provider.upper()] \
                        if provider.upper() in LLMProvider.__members__ \
                        else LLMProvider.UNKNOWN
                elif not isinstance(provider, LLMProvider):
                    provider = LLMProvider.UNKNOWN
            else:
                provider = LLMClient._detect_provider(client)
            
            # Provider-specific wrapping
            if provider == LLMProvider.GEMINI:
                return _wrap_gemini(client)
            elif provider == LLMProvider.LANGCHAIN_CHAT:
                return _wrap_langchain(client)
            elif provider == LLMProvider.LITELLM:
                return _wrap_litellm_client(client)
            else:
                # Generic wrapper for unknown providers
                return _wrap_generic_client(client, track_all)
        
        except Exception as e:
            print(f"ERROR: Failed to wrap LLM client: {e}")
            return client
    
    @staticmethod
    def _detect_provider(client: Any) -> LLMProvider:
        """Detect LLM provider from client object."""
        try:
            client_module = client.__class__.__module__
            
            if 'genai' in client_module or 'generative_ai' in client_module:
                return LLMProvider.GEMINI
            elif 'langchain' in client_module:
                return LLMProvider.LANGCHAIN_CHAT
            elif 'litellm' in client_module:
                return LLMProvider.LITELLM
        except Exception:
            pass
        
        return LLMProvider.UNKNOWN


def _wrap_gemini(client):
    """Wrap Google Gemini client with Opik tracking."""
    try:
        from opik.integrations.genai import track_genai
        tracked_client = track_genai(client)
        logger.debug("Gemini client wrapped successfully")
        return tracked_client
    except Exception as e:
        print(f"ERROR: Gemini wrapping failed, returning original: {e}")
        return client


def _wrap_langchain(client):
    """Wrap LangChain ChatModel for tracking."""
    try:
        from opik.integrations.langchain import track_langchain
        tracked_client = track_langchain(client)
        logger.debug("LangChain client wrapped successfully")
        return tracked_client
    except Exception as e:
        print(f"ERROR: LangChain wrapping failed, returning original: {e}")
        return client


def _wrap_litellm_client(client):
    """Wrap LiteLLM client for tracking."""
    return client


def _wrap_generic_client(client, track_all=False):
    """Generic wrapper for unknown LLM clients."""
    try:
        # Create method interceptor
        def create_tracked_method(original_method, method_name):
            @functools.wraps(original_method)
            async def async_wrapper(*args, **kwargs):
                return await original_method(*args, **kwargs)
            
            @functools.wraps(original_method)
            def sync_wrapper(*args, **kwargs):
                return original_method(*args, **kwargs)
            
            # Decide which wrapper to use
            if inspect.iscoroutinefunction(original_method):
                wrapper = async_wrapper
            else:
                wrapper = sync_wrapper
            
            # Wrap for tracing
            is_llm_method = any(
                pattern in method_name.lower() 
                for pattern in LLMClient.LLM_METHOD_PATTERNS
            )
            
            if track_all or is_llm_method:
                return _create_llm_tracker(method_name)(wrapper)
            return wrapper
        
        # Intercept methods
        class WrappedClient:
            def __init__(self, original_client):
                self._client = original_client
            
            def __getattr__(self, name):
                attr = getattr(self._client, name)
                if callable(attr):
                    return create_tracked_method(attr, name)
                return attr
        
        logger.debug("Generic client wrapped successfully")
        return WrappedClient(client)
    
    except Exception as e:
        print(f"ERROR: Generic wrapping failed: {e}")
        return client


# ============================================================================
# Decorators for LLM function tracking
# ============================================================================

def track_llm_calls(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    avoided_input_params: Optional[List[str]] = None,
    capture_input: bool = True,
    capture_output: bool = True
):
    """
    Decorator to track LLM function calls (and nested operations).
    
    Only tracks when Opik is initialized. Gracefully degrades if tracing fails.
    
    Args:
        name: Operation name (default: function name)
        tags: List of tags for categorization
        metadata: Additional metadata dict
        capture_input: Capture function arguments
        capture_output: Capture function result
    
    Returns:
        Decorated function with LLM tracking
    
    Example:
        >>> @track_llm_calls(name="generate_email", tags=["email", "draft"])
        ... def generate_email(prompt: str):
        ...     return llm.complete(prompt)
        >>> 
        >>> email = generate_email("Write professional email")
    """
    def decorator(func):
        operation_name = name or func.__name__
        
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _execute_tracked_operation_async(
                    func, args, kwargs, operation_name, tags, metadata,
                    avoided_input_params,
                    capture_input, capture_output
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return _execute_tracked_operation_sync(
                    func, args, kwargs, operation_name, tags, metadata,
                    avoided_input_params,
                    capture_input, capture_output
                )
            return sync_wrapper
    
    return decorator


def _create_llm_tracker(operation_name: str):
    """Create a tracker decorator for a specific operation."""
    def decorator(func):
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await _execute_tracked_operation_async(
                    func, args, kwargs, operation_name, None, None, None,
                    capture_input=True, capture_output=True
                )
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return _execute_tracked_operation_sync(
                    func, args, kwargs, operation_name, None, None, None,
                    capture_input=True, capture_output=True
                )
            return sync_wrapper
    
    return decorator

def _prepare_span_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Prepare common span metadata."""
    span_meta = dict(metadata or {})
    span_meta.update({
        "timestamp": datetime.now().isoformat(),
        "llm_only_mode": _llm_only_mode
    })
    return span_meta

def _get_serialized_inputs(capture_input, avoided_input_params, args, kwargs):
    """Process and serialize inputs for tracing."""
    if not capture_input:
        return None
    
    kwargs_cloned = kwargs.copy()
    if avoided_input_params:
        for param in avoided_input_params:
            if param in kwargs_cloned:
                del kwargs_cloned[param]

    return {
        "args": args,
        "kwargs": kwargs_cloned
    }

def _execute_tracked_operation_sync(
    func, args, kwargs, operation_name, tags, metadata, avoided_input_params,
    capture_input, capture_output
):
    """Execute a function with LLM operation tracking (sync version)."""

    if not _opik_initialized or not _tracing_enabled:
        return func(*args, **kwargs)

    try:
        import opik
        from opik import opik_context

        trace_metadata = _prepare_span_metadata(metadata)
        serialized_input = _get_serialized_inputs(capture_input, avoided_input_params, args, kwargs)
        @opik.track(name=operation_name, tags=tags or [], metadata=trace_metadata, capture_input=False, capture_output=False)
        def _tracked(input_data=None):
            # Annotate the span input from inside the active span context
            if capture_input and input_data is not None:
                _safe_update_span(opik_context, input=input_data)

            result = func(*args, **kwargs)

            # Annotate output from inside the active span context
            if capture_output:
                _safe_update_span(
                    opik_context,
                    output=_serialize_data(result, stringify=False)
                )

            return result

        return _tracked(input_data=serialized_input)

    except Exception as e:
        print(f"ERROR: Error in {operation_name}: {e}")
        raise


async def _execute_tracked_operation_async(
    func, args, kwargs, operation_name, tags, metadata, avoided_input_params,
    capture_input, capture_output
):
    """Execute a function with LLM operation tracking (async version)."""

    if not _opik_initialized or not _tracing_enabled:
        return await func(*args, **kwargs)

    try:
        import opik
        from opik import opik_context

        trace_metadata = _prepare_span_metadata(metadata)
        serialized_input = _get_serialized_inputs(capture_input, avoided_input_params, args, kwargs)

        @opik.track(name=operation_name, tags=tags or [], metadata=trace_metadata, capture_input=False, capture_output=False)
        async def _tracked(input_data=None):
            # Annotate the span input from inside the active span context
            if capture_input and input_data is not None:
                try:
                    opik_context.update_current_span(input=input_data)
                except Exception:
                    pass

            result = await func(*args, **kwargs)

            # Annotate output from inside the active span context
            if capture_output:
                try:
                    opik_context.update_current_span(
                        output=_serialize_data(result, stringify=False)
                    )
                except Exception:
                    pass

            return result

        return await _tracked(input_data=serialized_input)

    except Exception as e:
        print(f"ERROR: Error in {operation_name}: {e}")
        raise


# ============================================================================
# Span management for LLM operations
# ============================================================================

def start_llm_span(
    name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    input_data: Optional[Dict] = None,
    depth: int = 0
) -> Optional[str]:
    """
    Start a new LLM operation span.

    Note: In the current Opik SDK, standalone span creation outside of a tracked
    function is not supported via opik_context. This function is a no-op stub
    that returns a synthetic span ID; use @track_llm_calls or llm_operation
    for reliable span tracking.

    Args:
        name: Operation name
        tags: Tags for categorization
        metadata: Additional metadata
        input_data: Captured input data
        depth: Nesting depth

    Returns:
        Span ID if successful, None otherwise

    Example:
        >>> span_id = start_llm_span("email_generation", tags=["email"])
        >>> result = llm.generate(...)
        >>> end_span(span_id, output=result)
    """
    if not _opik_initialized or not _tracing_enabled:
        return None

    try:
        from opik import opik_context

        span_metadata = dict(metadata or {})
        span_metadata.update({
            "depth": depth,
            "timestamp": datetime.now().isoformat(),
            "llm_only_mode": _llm_only_mode,
            "span_name": name,
        })

        if input_data:
            serialized_input = _serialize_data(input_data, stringify=True)
            if isinstance(serialized_input, list):
                serialized_input = {"data": serialized_input}
            span_metadata["input"] = serialized_input

        # If we are currently inside a tracked function we can annotate the span.
        # Otherwise this is a no-op — the caller should prefer @track_llm_calls.
        try:
            opik_context.update_current_span(metadata=span_metadata, tags=tags or [])
        except Exception:
            pass  # Not inside a tracked scope; silently ignore

        # Return a synthetic ID so callers that use (span_id, end_span) still work.
        synthetic_id = f"{name}:{datetime.now().isoformat()}"
        logger.debug(f"Started LLM span: {name} (depth={depth})")
        return synthetic_id

    except Exception as e:
        print(f"ERROR: Failed to start LLM span: {e}")
        return None


def end_span(
    span_id: str,
    output: Optional[Any] = None,
    error: Optional[str] = None,
    error_traceback: Optional[str] = None,
    status: str = "success",
    tokens: Optional[Dict[str, int]] = None
) -> bool:
    """
    End an LLM operation span.

    Note: In the current Opik SDK, standalone span lifecycle management
    (start/end) is not exposed via opik_context. This function updates
    the *current* span's metadata if one is active; otherwise it is a
    no-op. Prefer @track_llm_calls or llm_operation for reliable span tracking.

    Args:
        span_id: Span ID from start_llm_span()
        output: Operation output/result
        error: Error message if failed
        error_traceback: Full error traceback
        status: "success", "error", or "partial"
        tokens: Token usage {"input": N, "output": M}

    Returns:
        True if successful, False otherwise

    Example:
        >>> span_id = start_llm_span("completion")
        >>> try:
        ...     result = llm.complete(prompt)
        ...     end_span(span_id, output=result, tokens={"input": 50, "output": 100})
        ... except Exception as e:
        ...     end_span(span_id, error=str(e), error_traceback=traceback.format_exc())
    """
    if not _opik_initialized or not _tracing_enabled or not span_id:
        return False

    try:
        from opik import opik_context

        end_metadata: Dict[str, Any] = {
            "status": status,
            "end_time": datetime.now().isoformat()
        }

        if output is not None:
            serialized = _serialize_data(output, stringify=True)
            if isinstance(serialized, list):
                serialized = {"result": serialized}
            end_metadata["output"] = serialized

        if error:
            end_metadata["error"] = error
        if error_traceback:
            end_metadata["error_traceback"] = error_traceback
        if tokens:
            end_metadata["tokens"] = tokens

        # Annotate current span if we happen to be inside a tracked scope.
        try:
            serialized_output = _serialize_data(output, stringify=True) if output is not None else None
            if isinstance(serialized_output, list):
                serialized_output = {"result": serialized_output}
            opik_context.update_current_span(
                output=serialized_output,
                metadata=end_metadata
            )
        except Exception:
            pass  # Not inside a tracked scope; silently ignore

        logger.debug(f"Ended LLM span: {span_id} (status={status})")
        return True

    except Exception as e:
        print(f"ERROR: Failed to end LLM span: {e}")
        return False

@contextmanager
def llm_operation(
    name: str,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Context manager for tracking LLM operations.

    Usage:
        >>> with llm_operation("summarization", tags=["summary"]) as ctx:
        ...     result = llm.generate_summary(text)
        ...     ctx.output = result

    Yields:
        Span context object
    """
    class _SpanContext:
        """Holds output/error set by the caller inside the with-block."""
        def __init__(self):
            self.span_id: Optional[str] = None
            self.output = None
            self.error: Optional[str] = None
            self.tokens: Optional[Dict[str, int]] = None

    ctx = _SpanContext()

    if not _opik_initialized or not _tracing_enabled:
        yield ctx
        return

    try:
        import opik
        from opik import opik_context

        span_metadata = dict(metadata or {})
        span_metadata.update({"timestamp": datetime.now().isoformat()})

        # Start the synthetic span id so callers can reference it
        ctx.span_id = f"{name}:{datetime.now().isoformat()}"

        yield ctx

        # After the body, create the tracked span with the captured output
        final_meta = dict(span_metadata)
        if ctx.output is not None:
            serialized = _serialize_data(ctx.output, stringify=True)
            if isinstance(serialized, list):
                serialized = {"result": serialized}
            final_meta["output"] = serialized
        if ctx.tokens:
            final_meta["tokens"] = ctx.tokens
        final_meta["status"] = "success"

        @opik.track(name=name, tags=tags or [], metadata=final_meta)
        def _record_success():
            """Placeholder function used by Opik decorator to record success."""
            pass

        _record_success()

    except Exception as e:
        # Record the failure span then re-raise
        try:
            import opik
            err_meta = dict(metadata or {})
            err_meta.update({
                "status": "error",
                "error": str(e),
                "error_traceback": traceback.format_exc(),
            })

            @opik.track(name=name, tags=tags or [], metadata=err_meta)
            def _record_error():
                """Placeholder function used by Opik decorator to record error."""
                pass

            _record_error()
        except Exception:
            pass
        raise


# ============================================================================
# Data utilities
# ============================================================================
def _serialize_data(data: Any, stringify: bool = False) -> Any:
    def _process(obj):
        if isinstance(obj, (tuple, list)):
            # Wrap in a dict immediately so Opik's attachment extractor
            # never receives a bare list (it calls .items() on top-level data)
            return {"result": [_process(item) for item in obj]}
        elif isinstance(obj, dict):
            return {str(k): _process(v) for k, v in obj.items()}
        elif isinstance(obj, str):
            return obj
        elif isinstance(obj, (int, float, bool, type(None))):
            return obj
        elif hasattr(obj, '__dict__'):
            try:
                return {str(k): _process(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            except Exception:
                return str(obj)
        else:
            return str(obj)

    try:
        processed = _process(data)
        if stringify and not isinstance(processed, (str, int, float, bool, type(None))):
            return json.dumps(processed)
        return processed
    except Exception:
        return str(data)
# ============================================================================
# Feedback and trace management
# ============================================================================

def log_trace_feedback(
    trace_id: str,
    score: float,
    comment: Optional[str] = None
) -> bool:
    """
    Log feedback for a trace (1.0 = good, 0.0 = bad).
    
    Args:
        trace_id: Trace ID from get_current_trace_id()
        score: Score between 0.0 and 1.0
        comment: Optional feedback comment
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> trace_id = get_current_trace_id()
        >>> if trace_id:
        ...     log_trace_feedback(trace_id, score=0.95, comment="Good response")
    """
    if not _opik_initialized or not _opik_client or not trace_id:
        return False
    
    try:
        _opik_client.log_feedback(
            trace_id=trace_id,
            score=score,
            comment=comment
        )
        logger.debug(f"Logged feedback for trace {trace_id}: score={score}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to log feedback: {e}")
        return False


def get_current_trace_id() -> Optional[str]:
    """
    Get the current trace ID if available.
    
    Returns:
        Current trace ID or None
    
    Example:
        >>> trace_id = get_current_trace_id()
        >>> if trace_id:
        ...     print(f"Tracing as: {trace_id}")
    """
    if not _opik_initialized:
        return None
    
    try:
        from opik import opik_context
        return opik_context.get_current_trace_id()
    except Exception as e:
        print(f"ERROR: Failed to get trace ID: {e}")
        return None


def flush_traces(timeout: int = 30) -> bool:
    """
    Flush pending traces to Opik server.
    
    Args:
        timeout: Flush timeout in seconds
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> # At application shutdown
        >>> flush_traces(timeout=10)
    """
    if not _opik_initialized or not _opik_client:
        return False
    
    try:
        _opik_client.flush()
        print("Traces flushed successfully")
        return True
    except Exception as e:
        print(f"ERROR: Failed to flush traces: {e}")
        return False


def update_current_span(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Update the current span with additional data.
    
    Args:
        name: New span name
        tags: Tags to add/update
        metadata: Metadata to merge
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> update_current_span(
        ...     tags=["email", "generated"],
        ...     metadata={"model": "gemini-pro"}
        ... )
    """
    if not _opik_initialized:
        return False
    
    try:
        from opik import opik_context
        
        span_data = opik_context.get_current_span_data()
        if span_data:
            if name:
                span_data.name = name
            if tags:
                span_data.tags = tags
            if metadata:
                span_data.metadata.update(metadata)
        
        logger.debug("Updated current span")
        return True
    except Exception as e:
        print(f"ERROR: Failed to update span: {e}")
        return False


def update_current_trace(
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
    user: Optional[str] = None,
    team_id: Optional[str] = None,
    organization_id: Optional[str] = None,
    message_id: Optional[str] = None,       
    session_id: Optional[str] = None
) -> bool:
    """
    Update the current trace with additional context.
    
    Args:
        metadata: Metadata to add
        tags: Tags to add
        user_id: User ID for this trace
        session_id: Session ID for this trace
    
    Returns:
        True if successful, False otherwise
    
    Example:
        >>> update_current_trace(
        ...     user_id="user@example.com",
        ...     session_id="sess-123",
        ...     tags=["email-draft"]
        ... )
    """
    if not _opik_initialized:
        return False
    
    try:
        from opik import opik_context
        
        # Build the metadata update
        combined_meta: Dict[str, Any] = dict(metadata or {})
        if user: combined_meta['user'] = user
        if team_id: combined_meta['team_id'] = team_id
        if organization_id: combined_meta['organization_id'] = organization_id
        if message_id: combined_meta['message_id'] = message_id
        if session_id: combined_meta['session_id'] = session_id

        opik_context.update_current_trace(
            metadata=combined_meta if combined_meta else None,
            tags=tags,
        )
        
        print(f"Updated current trace: message_id={message_id}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to update trace: {e}")
        return False


def is_tracing_enabled() -> bool:
    """Check if tracing is currently enabled."""
    return _opik_initialized and _tracing_enabled


def get_tracing_status() -> Dict[str, Any]:
    """Get detailed tracing status."""
    return {
        "initialized": _opik_initialized,
        "enabled": _tracing_enabled,
        "llm_only_mode": _llm_only_mode,
        "client": _opik_client is not None
    }

def get_distributed_headers():
    """Get distributed trace headers for HTTP requests."""
    if not _opik_initialized:
        return {}

    from opik import opik_context
    return opik_context.get_distributed_trace_headers()

def distributed_headers(headers: Dict[str, str]):
    """Get distributed trace headers for HTTP requests."""
    if not _opik_initialized:
        from contextlib import nullcontext
        return nullcontext()

    from opik.decorator.context_manager import distributed_headers
    
    return distributed_headers(headers=headers)

def setup_other_server_span(user_metadata: dict):
    if not _opik_initialized:
        return user_metadata
    
    headers = get_distributed_headers()
    user_metadata.update(headers)
    return user_metadata

class OPIKMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        headers = {}
 
        try:
            content_type = request.headers.get("content-type", "")
           
            if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
                # Form data - need to read body and re-inject it
                body = await request.body()
               
                # Re-inject body so downstream handlers can still read it
                async def receive():
                    # The 'no-op' await: tells Sonar/Python this is truly async
                    await asyncio.sleep(0) 
                    return {"type": "http.request", "body": body, "more_body": False}
                request._receive = receive
                
                form = await request.form()
                user_metadata_str = form.get('user_metadata', '{}') or '{}'
            else:
                body = await request.body()
                user_metadata_str = json.loads(body).get('user_metadata', '{}') if body else '{}'
 
            try:
                user_metadata = json.loads(user_metadata_str) if isinstance(user_metadata_str, str) else user_metadata_str
            except (json.JSONDecodeError, TypeError):
                user_metadata = {}
 
            headers = {
                "opik_trace_id": user_metadata.get('opik_trace_id'),
                "opik_parent_span_id": user_metadata.get('opik_parent_span_id'),
            }
        except Exception as e:
            print("ERROR: OPIK MIDDLEWARE: Failed to extract user metadata: " + str(e))
 
        with distributed_headers(headers=headers):
            if inspect.iscoroutinefunction(call_next):        
                response = await call_next(request)
            else:
                response = call_next(request)
            
        return response

class OPIKMiddlewareA2A(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Extract grouping ID from header or generate a new one
        # Use a real correlation ID if available (e.g., from request headers)

        if (
            request.url.path.startswith("/.well-known") or 
            request.url.path.startswith("/health") or 
            request.url.path.startswith("/openapi.json") or
            request.url.path.startswith("/favicon.ico")
        ):
            if inspect.iscoroutinefunction(call_next):        
                response = await call_next(request)
            else:
                response = call_next(request)

            return response

        headers = {}

        try:
            body = await request.body()

            json_body = json.loads(body)

            try:
                user_metadata = json_body.get('params', {}).get('metadata', {}).get('user_metadata', {})
            except (AttributeError, TypeError):
                user_metadata = {}

            headers = {
                "opik_trace_id": user_metadata.get('opik_trace_id'),
                "opik_parent_span_id": user_metadata.get('opik_parent_span_id'),
            }
        except Exception as e:
            print("ERROR: OPIK MIDDLEWARE: Failed to extract user metadata from request body: " + str(e))
        
        with distributed_headers(headers=headers):
            if inspect.iscoroutinefunction(call_next):        
                response = await call_next(request)
            else:
                response = call_next(request)

            return response
            
        return response

__all__ = [
    # Setup
    "setup_opik_tracing",
    
    # Client wrapping
    "LLMClient",
    "LLMProvider",
    
    # Decorators
    "track_llm_calls",
    "llm_operation",
    
    # Span management
    "start_llm_span",
    "end_span",
    
    # Trace management
    "log_trace_feedback",
    "get_current_trace_id",
    "flush_traces",
    "update_current_span",
    "update_current_trace",
    "get_distributed_headers",
    "distributed_headers",
    "OPIKMiddleware",
    "OPIKMiddlewareA2A",
    "setup_other_server_span",
    
    # Status
    "is_tracing_enabled",
    "get_tracing_status",
]
