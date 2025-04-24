"""
Configuration module for the EBS Wiki Chatbot.

This module provides a centralized configuration system that allows users to customize
parameters like chunk size, overlap, concurrency limits, and retry settings without
modifying the code.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, field_validator


class RetryConfig(BaseModel):
    """Configuration for API retry mechanism."""
    max_retries: int = Field(3, description="Maximum number of retry attempts")
    initial_delay: float = Field(1.0, description="Initial delay in seconds before retrying")
    max_delay: float = Field(60.0, description="Maximum delay in seconds before retrying")
    backoff_factor: float = Field(2.0, description="Backoff factor for exponential delay")
    
    @field_validator('max_retries', 'initial_delay', 'max_delay', 'backoff_factor')
    def validate_positive(cls, v, field):
        """Validate that values are positive."""
        if v <= 0:
            raise ValueError(f"{field.name} must be positive")
        return v


class ProcessingConfig(BaseModel):
    """Configuration for PDF processing."""
    chunk_size: int = Field(1000, description="Maximum size of text chunks")
    chunk_overlap: int = Field(100, description="Number of characters to overlap between chunks")
    concurrency_limit: int = Field(5, description="Maximum number of concurrent PDF processing tasks")
    
    @field_validator('chunk_size', 'chunk_overlap', 'concurrency_limit')
    def validate_positive(cls, v, field):
        """Validate that values are positive."""
        if v <= 0:
            raise ValueError(f"{field.name} must be positive")
        return v
    
    @field_validator('concurrency_limit')
    def validate_concurrency_limit(cls, v):
        """Validate that concurrency limit is reasonable."""
        if v > 20:
            raise ValueError("Concurrency limit should not exceed 20 to avoid API rate limits")
        return v


class APIConfig(BaseModel):
    """Configuration for API settings."""
    gemini_api_key: Optional[str] = Field(None, description="Google API key for Gemini")
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key for embeddings and LLM")
    gemini_model: str = Field("gemini-2.0-flash", description="Gemini model to use")
    openai_embedding_model: str = Field("text-embedding-3-small", description="OpenAI embedding model to use")
    openai_llm_model: str = Field("gpt-4o-mini", description="OpenAI LLM model to use")


class AppConfig(BaseModel):
    """Main application configuration."""
    input_dir: str = Field("./entersoft-docs", description="Directory containing PDF files to process")
    output_dir: str = Field("./output", description="Directory to save processed output")
    database_dir: str = Field("./lightrag_data", description="Directory to store the lightrag database")
    log_level: str = Field("INFO", description="Logging level")
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig, description="PDF processing configuration")
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")


def load_config(config_path: str = None) -> AppConfig:
    """Load configuration from a file or create default configuration.
    
    Args:
        config_path: Path to the configuration file (optional)
        
    Returns:
        Application configuration
    """
    # Default configuration
    config = AppConfig()
    
    # Try to load from file if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                config = AppConfig(**config_data)
        except Exception as e:
            logging.warning(f"Error loading configuration from {config_path}: {str(e)}")
            logging.warning("Using default configuration")
    
    # Override with environment variables if set
    if os.getenv("GEMINI_API_KEY"):
        config.api.gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if os.getenv("OPENAI_API_KEY"):
        config.api.openai_api_key = os.getenv("OPENAI_API_KEY")
    
    return config


def save_config(config: AppConfig, config_path: str) -> None:
    """Save configuration to a file.
    
    Args:
        config: Application configuration
        config_path: Path to save the configuration file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
    
    # Save to file
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.dict(), f, ensure_ascii=False, indent=2)


def create_default_config_file(config_path: str = "./output/lightrag_config.json") -> str:
    """Create a default configuration file.
    
    Args:
        config_path: Path to save the configuration file
        
    Returns:
        Path to the created configuration file
    """
    config = AppConfig()
    save_config(config, config_path)
    return config_path