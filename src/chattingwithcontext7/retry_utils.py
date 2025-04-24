"""
Retry utilities for handling transient API failures.

This module provides decorators and utilities for implementing retry logic
for API calls to handle transient failures gracefully.
"""

import asyncio
import functools
import logging
import random
import time
from typing import Any, Callable, Optional, Type, TypeVar, Union, List, Dict

from src.chattingwithcontext7.config import RetryConfig

# Type variable for return type
T = TypeVar('T')


class RetryableError(Exception):
    """Base class for errors that should trigger a retry."""
    pass


def is_retryable_exception(exception: Exception) -> bool:
    """Determine if an exception should trigger a retry.
    
    Args:
        exception: The exception to check
        
    Returns:
        True if the exception should trigger a retry, False otherwise
    """
    # Retry on network errors, timeouts, and rate limits
    retryable_error_types = [
        'ConnectionError', 
        'Timeout', 
        'TimeoutError',
        'ReadTimeout',
        'ConnectTimeout',
        'ConnectionTimeout',
        'ServiceUnavailable',
        'TooManyRequests',
        'RetryableError',
        'RateLimitError'
    ]
    
    # Check if the exception type name matches any retryable error type
    exception_type = type(exception).__name__
    if any(error_type in exception_type for error_type in retryable_error_types):
        return True
    
    # Check if it's a RetryableError or subclass
    if isinstance(exception, RetryableError):
        return True
    
    # Check error message for common retryable patterns
    error_message = str(exception).lower()
    retryable_patterns = [
        'timeout', 
        'timed out',
        'connection', 
        'network',
        'rate limit', 
        'too many requests',
        'server error',
        'internal server error',
        'service unavailable',
        'try again',
        'temporary',
        'retry',
        '429',  # HTTP 429 Too Many Requests
        '503',  # HTTP 503 Service Unavailable
        '504'   # HTTP 504 Gateway Timeout
    ]
    
    if any(pattern in error_message for pattern in retryable_patterns):
        return True
    
    return False


def calculate_delay(attempt: int, config: RetryConfig) -> float:
    """Calculate the delay before the next retry attempt.
    
    Args:
        attempt: The current attempt number (0-based)
        config: Retry configuration
        
    Returns:
        Delay in seconds before the next retry
    """
    # Exponential backoff with jitter
    delay = min(
        config.max_delay,
        config.initial_delay * (config.backoff_factor ** attempt)
    )
    
    # Add jitter (±25%)
    jitter = delay * 0.25
    delay = delay + random.uniform(-jitter, jitter)
    
    return max(0.1, delay)  # Ensure minimum delay of 0.1 seconds


async def retry_async(
    func: Callable[..., Any],
    *args: Any,
    retry_config: RetryConfig,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any
) -> Any:
    """Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        *args: Positional arguments to pass to the function
        retry_config: Retry configuration
        logger: Logger for logging retry attempts
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception raised by the function
    """
    max_retries = retry_config.max_retries
    
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check if we should retry
            if attempt < max_retries and is_retryable_exception(e):
                delay = calculate_delay(attempt, retry_config)
                
                if logger:
                    logger.warning(
                        f"Retry {attempt+1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {str(e)}"
                    )
                
                await asyncio.sleep(delay)
            else:
                # Either we've exhausted retries or it's not a retryable error
                if logger and attempt == max_retries:
                    logger.error(
                        f"Failed after {max_retries} retries for {func.__name__}: {str(e)}"
                    )
                raise


def retry_sync(
    func: Callable[..., Any],
    *args: Any,
    retry_config: RetryConfig,
    logger: Optional[logging.Logger] = None,
    **kwargs: Any
) -> Any:
    """Retry a synchronous function with exponential backoff.
    
    Args:
        func: Synchronous function to retry
        *args: Positional arguments to pass to the function
        retry_config: Retry configuration
        logger: Logger for logging retry attempts
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: The last exception raised by the function
    """
    max_retries = retry_config.max_retries
    
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Check if we should retry
            if attempt < max_retries and is_retryable_exception(e):
                delay = calculate_delay(attempt, retry_config)
                
                if logger:
                    logger.warning(
                        f"Retry {attempt+1}/{max_retries} for {func.__name__} "
                        f"after {delay:.2f}s due to: {str(e)}"
                    )
                
                time.sleep(delay)
            else:
                # Either we've exhausted retries or it's not a retryable error
                if logger and attempt == max_retries:
                    logger.error(
                        f"Failed after {max_retries} retries for {func.__name__}: {str(e)}"
                    )
                raise


def with_retry_async(retry_config: RetryConfig, logger: Optional[logging.Logger] = None):
    """Decorator for retrying async functions.
    
    Args:
        retry_config: Retry configuration
        logger: Logger for logging retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await retry_async(func, *args, retry_config=retry_config, logger=logger, **kwargs)
        return wrapper
    return decorator


def with_retry_sync(retry_config: RetryConfig, logger: Optional[logging.Logger] = None):
    """Decorator for retrying synchronous functions.
    
    Args:
        retry_config: Retry configuration
        logger: Logger for logging retry attempts
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return retry_sync(func, *args, retry_config=retry_config, logger=logger, **kwargs)
        return wrapper
    return decorator