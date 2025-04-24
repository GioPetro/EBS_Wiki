"""Entry point script for processing PDFs with images using LightRAG and pydantic_ai."""

import os
import argparse
import asyncio
import logging
from typing import List, Optional
import json
import dotenv
from datetime import datetime

from src.chattingwithcontext7.pdf_processor import PDFProcessor
from src.chattingwithcontext7.agent import initialize_rag, run_rag_agent
from src.chattingwithcontext7.models import ProcessedDocument

# Load environment variables from .env file
dotenv.load_dotenv()


async def process_pdfs(
    input_dir: str,
    output_dir: str,
    gemini_api_key: str,
    log_level: int = logging.INFO
) -> tuple[List[ProcessedDocument], str]:
    """Process PDFs with images.
    
    Args:
        input_dir: Directory containing PDF files to process
        output_dir: Directory to save processed output
        gemini_api_key: Google API key for Gemini
        log_level: Logging level
        
    Returns:
        Tuple containing:
            - List of processed documents
            - Path to the saved LightRAG documents
    """
    # Initialize PDF processor
    processor = PDFProcessor(
        input_dir=input_dir,
        output_dir=output_dir,
        gemini_api_key=gemini_api_key,
        log_level=log_level
    )
    
    # Process PDFs
    processed_documents, lightrag_path = await processor.process_directory()
    
    return processed_documents, lightrag_path


async def load_processed_documents(lightrag_path: str) -> List[ProcessedDocument]:
    """Load processed documents from a LightRAG file.
    
    Args:
        lightrag_path: Path to the LightRAG documents JSON file
        
    Returns:
        List of processed documents
    """
    # Load LightRAG documents
    with open(lightrag_path, 'r', encoding='utf-8') as f:
        lightrag_docs = json.load(f)
    
    # Extract unique document IDs
    document_ids = set()
    for doc in lightrag_docs:
        if "metadata" in doc and "document_id" in doc["metadata"]:
            document_ids.add(doc["metadata"]["document_id"])
    
    # Create simplified document objects
    documents = []
    for doc_id in document_ids:
        # Find a representative document to get metadata
        for rag_doc in lightrag_docs:
            metadata = rag_doc.get("metadata", {})
            if metadata.get("document_id") == doc_id:
                # Create a simplified document
                document = ProcessedDocument(
                    document_id=doc_id,
                    filename=metadata.get("filename", "unknown"),
                    title=metadata.get("title", "Unknown Document"),
                    total_pages=metadata.get("page", 1),  # Use the highest page number as total
                    segments=[]  # We don't need segments for the agent
                )
                documents.append(document)
                break
    
    return documents


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process PDFs with images using LightRAG and pydantic_ai")
    parser.add_argument("--input-dir", type=str, default="C:\\Users\\georg\\Desktop\\AEGIS\\Projects\\EnterSoftData",
                        help="Directory containing PDF files to process")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="Directory to save processed output")
    parser.add_argument("--rag-dir", type=str, default="./lightrag_data",
                        help="Working directory for LightRAG")
    parser.add_argument("--gemini-api-key", type=str, default=os.getenv("GEMINI_API_KEY"),
                        help="Google API key for Gemini")
    parser.add_argument("--openai-api-key", type=str, default=os.getenv("OPENAI_API_KEY"),
                        help="OpenAI API key for LightRAG")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--query", type=str,
                        help="Query to run against the processed documents")
    parser.add_argument("--skip-processing", action="store_true",
                        help="Skip PDF processing and use existing LightRAG documents")
    parser.add_argument("--lightrag-path", type=str,
                        help="Path to existing LightRAG documents JSON file")
    parser.add_argument("--config", type=str,
                        help="Path to saved LightRAG configuration file")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Check for required API keys
    if not args.gemini_api_key and not args.skip_processing:
        logger.error("Gemini API key is required for processing. Set it via --gemini-api-key or GEMINI_API_KEY env var.")
        return
    
    if not args.openai_api_key:
        logger.error("OpenAI API key is required for LightRAG. Set it via --openai-api-key or OPENAI_API_KEY env var.")
        return
    
    # Set OpenAI API key and base URL
    os.environ["OPENAI_API_KEY"] = args.openai_api_key
    # Set OPENAI_API_BASE to the default OpenAI API endpoint if not already set
    if "OPENAI_API_BASE" not in os.environ:
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
    
    # Check if a configuration file was provided
    if args.config:
        logger.info(f"Loading configuration from {args.config}")
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Override arguments with values from config
            if not args.lightrag_path and "lightrag_path" in config:
                args.lightrag_path = config["lightrag_path"]
                logger.info(f"Using LightRAG documents path from config: {args.lightrag_path}")
            
            if not args.rag_dir and "rag_dir" in config:
                args.rag_dir = config["rag_dir"]
                logger.info(f"Using LightRAG working directory from config: {args.rag_dir}")
            
            # Print some information from the config
            if "processed_at" in config:
                logger.info(f"Documents were processed at: {config['processed_at']}")
            if "total_documents" in config:
                logger.info(f"Total documents in the database: {config['total_documents']}")
            if "total_chunks" in config:
                logger.info(f"Total chunks in the database: {config['total_chunks']}")
            
            # Force skip processing if config is provided
            if not args.skip_processing:
                logger.info("Config file provided, setting --skip-processing")
                args.skip_processing = True
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return
    
    # Process PDFs or use existing LightRAG documents
    if not args.skip_processing:
        logger.info(f"Processing PDFs from {args.input_dir}")
        processed_documents, lightrag_path = await process_pdfs(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            gemini_api_key=args.gemini_api_key,
            log_level=log_level
        )
        logger.info(f"Processed {len(processed_documents)} documents")
        logger.info(f"LightRAG documents saved to {lightrag_path}")
    else:
        if not args.lightrag_path:
            logger.error("--lightrag-path is required when using --skip-processing")
            return
        lightrag_path = args.lightrag_path
        logger.info(f"Using existing LightRAG documents from {lightrag_path}")
        processed_documents = await load_processed_documents(lightrag_path)
        logger.info(f"Loaded {len(processed_documents)} documents")
    
    # Initialize LightRAG
    logger.info(f"Initializing LightRAG in {args.rag_dir}")
    logger.info(f"This will create a persistent database that can be reused in future runs")
    lightrag = await initialize_rag(args.rag_dir)
    logger.info(f"LightRAG initialized successfully with the following storages:")
    logger.info(f"  - Vector DB for entities: {args.rag_dir}/vdb_entities.json")
    logger.info(f"  - Vector DB for relationships: {args.rag_dir}/vdb_relationships.json")
    logger.info(f"  - Vector DB for chunks: {args.rag_dir}/vdb_chunks.json")
    logger.info(f"  - Knowledge graph: {args.rag_dir}/graph_chunk_entity_relation.graphml")
    
    # Load documents into LightRAG
    logger.info("Loading documents into LightRAG")
    with open(lightrag_path, 'r', encoding='utf-8') as f:
        lightrag_docs = json.load(f)
    
    # Insert documents asynchronously
    async def insert_documents():
        for doc in lightrag_docs:
            # The LightRAG API expects a string for content, not a dictionary
            # Let's check the API documentation and use the correct method
            try:
                # Try with metadata as a keyword argument
                await lightrag.ainsert(doc["text"], metadata=doc["metadata"])
            except TypeError:
                # If that fails, try with just the text
                logger.warning("Failed to insert with metadata, trying without metadata")
                await lightrag.ainsert(doc["text"])
    
    await insert_documents()
    
    logger.info(f"Loaded {len(lightrag_docs)} chunks into LightRAG")
    
    # Save the LightRAG database path to a file for easy reuse
    config_path = os.path.join(args.output_dir, "lightrag_config.json")
    config = {
        "lightrag_path": lightrag_path,
        "rag_dir": args.rag_dir,
        "processed_at": datetime.now().isoformat(),
        "total_documents": len(processed_documents),
        "total_chunks": len(lightrag_docs)
    }
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved LightRAG configuration to {config_path}")
    
    # Run query if provided
    if args.query:
        logger.info(f"Running query: {args.query}")
        response = await run_rag_agent(args.query, lightrag, processed_documents)
        print("\nResponse:")
        print(response)
    else:
        logger.info("No query provided. Use --query to ask a question about the documents.")
        print("\nPDF processing complete. Documents loaded into LightRAG.")
        print(f"Total documents: {len(processed_documents)}")
        print(f"Total chunks: {len(lightrag_docs)}")
        print(f"LightRAG working directory: {args.rag_dir}")
        print(f"LightRAG documents path: {lightrag_path}")
        print(f"LightRAG configuration saved to: {config_path}")
        print("\nTo query the documents, run:")
        print(f"python -m src.chattingwithcontext7.main --skip-processing --lightrag-path {lightrag_path} --query \"Your question here\"")


if __name__ == "__main__":
    asyncio.run(main())