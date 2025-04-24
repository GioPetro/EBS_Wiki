"""
Utility functions for lightrag database operations.

This module provides common utilities for building and querying
a lightrag database from Entersoft Docs.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from lightrag import LightRAG
from lightrag.llm.openai import openai_embed, gpt_4o_mini_complete
from lightrag.kg.shared_storage import initialize_pipeline_status

# Set OpenAI API configuration
import os
os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"


async def initialize_lightrag(
    database_dir: str,
    logger: Optional[logging.Logger] = None
) -> LightRAG:
    """Initialize LightRAG with OpenAI embeddings and GPT-4o-mini.
    
    Args:
        database_dir: Directory to store the lightrag database
        logger: Logger instance for logging
        
    Returns:
        Initialized LightRAG instance
    """
    if logger:
        logger.info(f"Initializing LightRAG in {database_dir}")
    
    # Create directory if it doesn't exist
    os.makedirs(database_dir, exist_ok=True)
    
    # Create LightRAG instance
    lightrag = LightRAG(
        working_dir=database_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    
    # Initialize storages
    if logger:
        logger.info("Initializing LightRAG storages")
    await lightrag.initialize_storages()
    await initialize_pipeline_status()
    
    if logger:
        logger.info(f"LightRAG initialized successfully in {database_dir}")
    return lightrag


def find_pdf_files(
    input_dir: str,
    recursive: bool = True,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Find all PDF files in the input directory.
    
    Args:
        input_dir: Directory to search for PDF files
        recursive: Whether to search recursively in subdirectories
        logger: Logger instance for logging
        
    Returns:
        List of Path objects for PDF files
    """
    if logger:
        logger.info(f"Finding PDF files in {input_dir} (recursive={recursive})")
    
    # Find PDF files
    if recursive:
        pdf_files = list(Path(input_dir).glob("**/*.pdf"))
    else:
        pdf_files = list(Path(input_dir).glob("*.pdf"))
    
    if logger:
        logger.info(f"Found {len(pdf_files)} PDF files")
    
    return pdf_files


def save_database_info(
    info: Dict[str, Any],
    output_dir: str,
    filename: str = "database_info.json",
    logger: Optional[logging.Logger] = None
) -> str:
    """Save database information to a JSON file.
    
    Args:
        info: Dictionary with database information
        output_dir: Directory to save the file
        filename: Name of the file
        logger: Logger instance for logging
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create file path
    file_path = os.path.join(output_dir, filename)
    
    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    
    if logger:
        logger.info(f"Saved database information to {file_path}")
    
    return file_path


def load_database_info(
    file_path: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """Load database information from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        logger: Logger instance for logging
        
    Returns:
        Dictionary with database information
    """
    if not os.path.exists(file_path):
        if logger:
            logger.warning(f"Database info file not found: {file_path}")
        return {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        if logger:
            logger.info(f"Loaded database information from {file_path}")
        
        return info
    except Exception as e:
        if logger:
            logger.error(f"Error loading database information: {str(e)}")
        return {}


def format_time_elapsed(seconds: float) -> str:
    """Format time elapsed in a human-readable format.
    
    Args:
        seconds: Time elapsed in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"


def create_progress_tracker(
    total: int,
    description: str = "Processing",
    logger: Optional[logging.Logger] = None
) -> callable:
    """Create a progress tracker function.
    
    Args:
        total: Total number of items to process
        description: Description of the process
        logger: Logger instance for logging
        
    Returns:
        Function to update progress
    """
    start_time = datetime.now()
    
    def update_progress(current: int, additional_info: str = ""):
        """Update progress.
        
        Args:
            current: Current number of items processed
            additional_info: Additional information to log
        """
        if current <= 0 or total <= 0:
            return
        
        # Calculate progress percentage
        percentage = (current / total) * 100
        
        # Calculate time elapsed and estimated time remaining
        elapsed = (datetime.now() - start_time).total_seconds()
        if current > 0 and elapsed > 0:
            items_per_second = current / elapsed
            estimated_total = elapsed * (total / current)
            remaining = estimated_total - elapsed
        else:
            items_per_second = 0
            remaining = 0
        
        # Format message
        message = f"{description}: {current}/{total} ({percentage:.1f}%) - "
        message += f"Elapsed: {format_time_elapsed(elapsed)}, "
        message += f"Remaining: {format_time_elapsed(remaining)}, "
        message += f"Speed: {items_per_second:.2f} items/sec"
        
        if additional_info:
            message += f" - {additional_info}"
        
        # Log progress
        if logger:
            logger.info(message)
        else:
            print(message)
    
    return update_progress


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging.
    
    Args:
        log_level: Logging level
        log_file: Path to log file (optional)
        
    Returns:
        Logger instance
    """
    # Convert log level string to logging level
    level = getattr(logging, log_level.upper())
    
    # Create logger
    logger = logging.getLogger("lightrag_database")
    logger.setLevel(level)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Add file handler if log file is provided
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
    
    return logger