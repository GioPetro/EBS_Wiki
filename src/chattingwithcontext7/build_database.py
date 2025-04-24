#!/usr/bin/env python
"""
Build a persistent lightrag database from Entersoft Docs.

This script processes all PDF documents in the specified directory (including subfolders),
extracts text and images, and builds a persistent lightrag knowledge base that can be
queried later.
"""

import os
import sys
import logging
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import dotenv

from lightrag import LightRAG

from src.chattingwithcontext7.pdf_processor import PDFProcessor
from src.chattingwithcontext7.models import ProcessedDocument
from src.chattingwithcontext7.config import AppConfig, load_config, save_config, create_default_config_file
from src.chattingwithcontext7.retry_utils import with_retry_async, retry_async
from src.chattingwithcontext7.utils import (
    initialize_lightrag,
    find_pdf_files,
    save_database_info,
    create_progress_tracker,
    setup_logging
)

# Load environment variables from .env file
dotenv.load_dotenv()


class LightRAGDatabaseBuilder:
    """Build a persistent lightrag database from PDF documents."""

    def __init__(
        self,
        config: AppConfig,
        log_level: int = logging.INFO,
    ):
        """Initialize the database builder.
        
        Args:
            config: Application configuration
            log_level: Logging level
        """
        # Set up configuration
        self.config = config
        
        # Set up directories
        self.input_dir = self.config.input_dir
        self.output_dir = self.config.output_dir
        self.database_dir = self.config.database_dir
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.database_dir, exist_ok=True)
        
        # Set up logging
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"build_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = setup_logging(self.config.log_level, log_file)
        
        # Set API keys
        self.gemini_api_key = self.config.api.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required for image analysis. Set GEMINI_API_KEY in .env")
        
        self.openai_api_key = self.config.api.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings. Set OPENAI_API_KEY in .env")
        
        # Set OpenAI API key in environment
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        
        # Initialize PDF processor with configuration
        self.pdf_processor = PDFProcessor(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            gemini_api_key=self.gemini_api_key,
            log_level=log_level,
            config={
                "chunk_size": self.config.processing.chunk_size,
                "chunk_overlap": self.config.processing.chunk_overlap,
                "max_retries": self.config.retry.max_retries,
                "initial_delay": self.config.retry.initial_delay,
                "max_delay": self.config.retry.max_delay,
                "backoff_factor": self.config.retry.backoff_factor,
                "gemini_model": self.config.api.gemini_model
            }
        )
        
        # Initialize LightRAG instance
        self.lightrag = None
        
        # Statistics
        self.stats = {
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_documents": 0,
            "total_chunks": 0,
            "total_pages": 0,
            "total_images": 0,
            "processing_time_seconds": 0,
            "database_path": self.database_dir,
            "lightrag_docs_path": None,
            "concurrency_limit": self.config.processing.concurrency_limit
        }

    # Removed redundant initialize_lightrag method as it just calls the imported function

    async def process_pdfs(self) -> Tuple[List[ProcessedDocument], str]:
        """Process all PDF files in the input directory and its subdirectories concurrently.
        
        Returns:
            Tuple containing:
                - List of processed documents
                - Path to the saved LightRAG documents
        """
        self.logger.info(f"Processing PDF files in {self.input_dir} (including subdirectories)")
        
        # Find all PDF files recursively
        pdf_files = find_pdf_files(self.input_dir, recursive=True, logger=self.logger)
        
        # Process PDF files concurrently with a limit on concurrency
        all_processed_documents = []
        all_lightrag_docs = []
        
        start_time = datetime.now()
        
        # Create a progress tracker for processing PDFs
        progress = create_progress_tracker(len(pdf_files), "Processing PDF files", self.logger)
        
        # Process PDFs in batches to control concurrency
        concurrency_limit = self.config.processing.concurrency_limit
        self.logger.info(f"Using concurrency limit of {concurrency_limit} for PDF processing")
        
        # Track completed PDFs for progress reporting
        completed_count = 0
        
        # Process PDFs in batches
        for i in range(0, len(pdf_files), concurrency_limit):
            batch = pdf_files[i:i + concurrency_limit]
            
            # Create tasks for concurrent processing
            tasks = []
            for pdf_path in batch:
                self.logger.debug(f"Creating task for processing: {pdf_path.name}")
                task = self.process_single_pdf(str(pdf_path))
                tasks.append(task)
            
            # Process batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result, pdf_path in zip(batch_results, batch):
                completed_count += 1
                progress(completed_count, f"Processed: {pdf_path.name}")
                
                if isinstance(result, Exception):
                    self.logger.error(f"Error processing {pdf_path}: {str(result)}")
                    continue
                
                if result:
                    document, lightrag_docs = result
                    all_processed_documents.append(document)
                    all_lightrag_docs.extend(lightrag_docs)
                    
                    # Update statistics
                    self.stats["total_documents"] += 1
                    self.stats["total_pages"] += document.total_pages
                    self.stats["total_images"] += document.metadata.get("total_images", 0)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        self.stats["processing_time_seconds"] = processing_time
        
        # Save all LightRAG documents to a single file
        lightrag_path = os.path.join(self.output_dir, "combined_lightrag.json")
        with open(lightrag_path, 'w', encoding='utf-8') as f:
            json.dump(all_lightrag_docs, f, ensure_ascii=False, indent=2)
        
        self.stats["total_chunks"] = len(all_lightrag_docs)
        self.stats["lightrag_docs_path"] = lightrag_path
        
        self.logger.info(f"Processed {len(all_processed_documents)} documents with {len(all_lightrag_docs)} chunks")
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"LightRAG documents saved to {lightrag_path}")
        
        return all_processed_documents, lightrag_path
    
    async def process_single_pdf(self, pdf_path: str) -> Optional[Tuple[ProcessedDocument, List[Dict[str, Any]]]]:
        """Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple containing the processed document and its LightRAG documents, or None if processing failed
        """
        try:
            # Process the PDF
            document = await self.pdf_processor.process_pdf(pdf_path)
            
            if document:
                # Transform to LightRAG format
                lightrag_docs = self.pdf_processor.transform_for_lightrag(document)
                return document, lightrag_docs
            
            return None
        except FileNotFoundError as e:
            self.logger.error(f"File not found: {pdf_path}: {str(e)}")
            raise
        except PermissionError as e:
            self.logger.error(f"Permission denied when accessing {pdf_path}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
            raise

    async def load_documents_into_lightrag(self, lightrag_path: str) -> None:
        """Load documents into LightRAG with retry mechanism.
        
        Args:
            lightrag_path: Path to the LightRAG documents JSON file
        """
        self.logger.info(f"Loading documents into LightRAG from {lightrag_path}")
        
        # Load LightRAG documents
        with open(lightrag_path, 'r', encoding='utf-8') as f:
            lightrag_docs = json.load(f)
        
        # Create a progress tracker for loading documents
        progress = create_progress_tracker(len(lightrag_docs), "Loading documents into LightRAG", self.logger)
        
        # Insert documents into LightRAG with retry
        for i, doc in enumerate(lightrag_docs):
            try:
                # Update progress
                if (i + 1) % 100 == 0 or i == 0 or i == len(lightrag_docs) - 1:
                    progress(i + 1)
                
                # Insert document with retry
                await self.insert_document_with_retry(doc["text"], doc["metadata"])
            except Exception as e:
                self.logger.error(f"Error inserting document {i+1} after retries: {str(e)}")
        
        self.logger.info(f"Loaded {len(lightrag_docs)} chunks into LightRAG")
    
    async def insert_document_with_retry(self, text: str, metadata: Dict[str, Any]) -> None:
        """Insert a document into LightRAG with retry mechanism.
        
        Args:
            text: Document text
            metadata: Document metadata
        """
        # Create a unique ID for the document based on metadata
        doc_id = metadata.get("document_id", "") + "_" + metadata.get("segment_id", "")
        
        # Use retry mechanism for API calls
        await retry_async(
            self.lightrag.ainsert,
            text,
            ids=[doc_id],  # Pass metadata as IDs instead of a separate parameter
            retry_config=self.config.retry,
            logger=self.logger
        )

    async def build_database(self) -> Dict[str, Any]:
        """Build the lightrag database.
        
        Returns:
            Dictionary with database information
        """
        self.logger.info("Starting database build process")
        
        # Initialize LightRAG
        self.lightrag = await initialize_lightrag(self.database_dir, self.logger)
        
        # Process PDFs
        processed_documents, lightrag_path = await self.process_pdfs()
        
        # Load documents into LightRAG
        await self.load_documents_into_lightrag(lightrag_path)
        
        # Save database information
        self.stats["end_time"] = datetime.now().isoformat()
        
        # Save database information to a file
        db_info_path = save_database_info(self.stats, self.output_dir, "database_info.json", self.logger)
        
        self.logger.info(f"Database build complete. Information saved to {db_info_path}")
        self.logger.info(f"Database directory: {self.database_dir}")
        self.logger.info(f"Total documents: {self.stats['total_documents']}")
        self.logger.info(f"Total chunks: {self.stats['total_chunks']}")
        
        return self.stats


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Build a persistent lightrag database from Entersoft Docs"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "entersoft-docs"),
        help="Directory containing PDF files to process (including subdirectories)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save processed output"
    )
    parser.add_argument(
        "--database-dir",
        type=str,
        default="./lightrag_data",
        help="Directory to store the lightrag database"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=os.getenv("GEMINI_API_KEY"),
        help="Google API key for Gemini (for image analysis)"
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=os.getenv("OPENAI_API_KEY"),
        help="OpenAI API key for embeddings and LLM"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="./output/lightrag_config.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--concurrency-limit",
        type=int,
        default=5,
        help="Maximum number of concurrent PDF processing tasks"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum size of text chunks"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=100,
        help="Number of characters to overlap between chunks"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for API calls"
    )
    parser.add_argument(
        "--create-default-config",
        action="store_true",
        help="Create a default configuration file and exit"
    )
    
    args = parser.parse_args()
    
    # Create default configuration file if requested
    if args.create_default_config:
        config_path = create_default_config_file(args.config_file)
        print(f"Default configuration file created at: {config_path}")
        return
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Override configuration with command line arguments
    if args.input_dir:
        config.input_dir = args.input_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.database_dir:
        config.database_dir = args.database_dir
    if args.gemini_api_key:
        config.api.gemini_api_key = args.gemini_api_key
    if args.openai_api_key:
        config.api.openai_api_key = args.openai_api_key
    if args.log_level:
        config.log_level = args.log_level
    if args.concurrency_limit:
        config.processing.concurrency_limit = args.concurrency_limit
    if args.chunk_size:
        config.processing.chunk_size = args.chunk_size
    if args.chunk_overlap:
        config.processing.chunk_overlap = args.chunk_overlap
    if args.max_retries:
        config.retry.max_retries = args.max_retries
    
    # Save updated configuration
    save_config(config, args.config_file)
    
    # Build the database
    builder = LightRAGDatabaseBuilder(
        config=config,
        log_level=getattr(logging, config.log_level)
    )
    
    try:
        stats = await builder.build_database()
        
        # Print summary
        print("\nDatabase build complete!")
        print(f"Total documents processed: {stats['total_documents']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Total pages processed: {stats['total_pages']}")
        print(f"Total images processed: {stats['total_images']}")
        print(f"Processing time: {stats['processing_time_seconds']:.2f} seconds")
        print(f"Concurrency limit: {stats['concurrency_limit']}")
        print(f"\nDatabase directory: {stats['database_path']}")
        print(f"LightRAG documents: {stats['lightrag_docs_path']}")
        print(f"Configuration file: {args.config_file}")
        print("\nTo query the database, run:")
        print(f"python -m src.chattingwithcontext7.query_database --database-dir {config.database_dir} --query \"Your question here\"")
    
    except ValueError as e:
        logging.error(f"Configuration error: {str(e)}")
        sys.exit(1)
    except FileNotFoundError as e:
        logging.error(f"File not found: {str(e)}")
        sys.exit(1)
    except PermissionError as e:
        logging.error(f"Permission denied: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error building database: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())