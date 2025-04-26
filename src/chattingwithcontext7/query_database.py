#!/usr/bin/env python
"""
Query a persistent lightrag database built from Entersoft Docs.

This script loads a pre-built lightrag database and allows querying it
with natural language questions. Results include citations to trace back
to source documents.
"""

import os
import sys
import logging
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import dotenv

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.gemini import GeminiModel
from dataclasses import dataclass

from lightrag import LightRAG, QueryParam

from src.chattingwithcontext7.models import ProcessedDocument
from src.chattingwithcontext7.utils import (
    initialize_lightrag,
    load_database_info,
    setup_logging
)

# Load environment variables from .env file
dotenv.load_dotenv()


@dataclass
class RAGQueryDeps:
    """Dependencies for the RAG query agent."""
    lightrag: LightRAG
    database_info: Dict[str, Any]


class LightRAGDatabaseQuerier:
    """Query a persistent lightrag database."""

    def __init__(
        self,
        database_dir: str,
        database_info_path: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        log_level: int = logging.INFO,
    ):
        """Initialize the database querier.
        
        Args:
            database_dir: Directory containing the lightrag database
            database_info_path: Path to the database info JSON file (optional)
            gemini_api_key: Google API key for Gemini (for the agent)
            openai_api_key: OpenAI API key for embeddings and LLM
            log_level: Logging level
        """
        # Set up directories
        self.database_dir = database_dir
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Set API keys
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("Gemini API key is required for the agent. Set GEMINI_API_KEY in .env")
        
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required for embeddings. Set OPENAI_API_KEY in .env")
        
        # Set OpenAI API key and base URL in environment
        os.environ["OPENAI_API_KEY"] = self.openai_api_key
        os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
        
        # Initialize LightRAG instance
        self.lightrag = None
        
        # Load database info if provided
        self.database_info = {"database_dir": database_dir}
        if database_info_path:
            loaded_info = load_database_info(database_info_path, self.logger)
            if loaded_info:
                self.database_info = loaded_info
        
        # Initialize the agent
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        """Create a pydantic_ai agent for querying the database.
        
        Returns:
            Configured Agent instance
        """
        # Create the Gemini model with the correct provider initialization
        from pydantic_ai.providers.google_gla import GoogleGLAProvider
        gemini_model = GeminiModel(
            'gemini-2.0-flash',
            provider=GoogleGLAProvider(api_key=self.gemini_api_key)
        )
        
        # Create the agent
        agent = Agent(
            gemini_model,
            deps_type=RAGQueryDeps,
            system_prompt="""
            You are an Entersoft ERP documentation assistant that answers questions using the knowledge base.
            
            When answering questions:
            1. Always cite your sources with complete metadata including:
               - Document filename
               - Page number
               - Document version (when available)
            2. Format each citation as: (Source: [filename], Page: [page], Version: [version])
            3. If information comes from an image, mention that it's from an image and describe what the image shows
            4. If the knowledge base doesn't contain the answer, clearly state that and provide your best general knowledge response
            5. When multiple documents provide relevant information, synthesize the information and cite all sources
            6. Be concise, accurate, and helpful
            7. Format your responses in a clear, readable way with appropriate headings and bullet points when needed
            8. ALWAYS include all available metadata for each piece of information in your response
            
            The knowledge base contains Entersoft ERP documentation that has been processed and stored in a LightRAG database.
            Use the retrieve tool to search for relevant information before answering.
            """
        )
        
        # Add tools to the agent
        @agent.tool
        async def retrieve(context: RunContext[RAGQueryDeps], search_query: str) -> str:
            """Retrieve relevant information from the knowledge base.
            
            Args:
                context: The run context containing dependencies
                search_query: The search query to find relevant information
                
            Returns:
                Formatted context information from the retrieved documents
            """
            try:
                # Query LightRAG
                results = await context.deps.lightrag.aquery(
                    search_query, param=QueryParam(mode="mix", top_k=10)
                )
                
                # Format results
                if not results:
                    return "No relevant information found in the knowledge base."
                
                # Check if results is a string (API might have changed)
                if isinstance(results, str):
                    return f"Retrieved content:\n{results}"
                
                # Document-level version cache - stores version info by document ID
                document_versions = {}
                
                # First pass: Extract version information from early pages (1-3)
                # This implements the knowledge that version info is in first 3 pages
                for result in results:
                    if not isinstance(result, dict):
                        continue
                        
                    metadata = result.get("metadata", {})
                    if not isinstance(metadata, dict):
                        continue
                        
                    # Get document identifiers
                    document_id = metadata.get("document_id", "unknown")
                    filename = metadata.get("filename", "Unknown Filename")
                    page = metadata.get("page", "unknown")
                    
                    # Only process pages 1-3 for version detection
                    try:
                        page_num = int(page) if str(page).isdigit() else 999
                    except:
                        page_num = 999
                        
                    if page_num <= 3:
                        # Extract version from early page content
                        text = result.get('text', '')
                        
                        # Look for version patterns in text
                        version_patterns = [
                            r'version\s*(\d+(?:\.\d+)?)',
                            r'v(\d+(?:\.\d+)?)\s',
                            r'CRM\s*®?\s*(\d+(?:\.\d+)?)',
                            r'ERP\s*®?\s*(\d+(?:\.\d+)?)',
                        ]
                        
                        extracted_version = None
                        for pattern in version_patterns:
                            match = re.search(pattern, text, re.IGNORECASE)
                            if match:
                                extracted_version = match.group(1)
                                print(f"Found version {extracted_version} on page {page} of {filename}")
                                break
                                
                        # If not found in text, try filename
                        if not extracted_version and filename != "Unknown Filename":
                            version_match = re.search(r'v(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
                            if version_match:
                                extracted_version = version_match.group(1)
                                print(f"Extracted version {extracted_version} from filename: {filename}")
                                
                        # Store the version for this document
                        if extracted_version and document_id != "unknown":
                            document_versions[document_id] = extracted_version
                            print(f"Stored version {extracted_version} for document {document_id}")
                
                # Always log ALL results in detail
                print("\n==== RAW QUERY RESULTS ====")
                try:
                    if isinstance(results, list) and len(results) > 0:
                        print(f"Total results: {len(results)}")
                        for i, result in enumerate(results):  # Log all results
                            if isinstance(result, dict):
                                text = result.get('text', 'No text')[:100] + "..." if len(result.get('text', '')) > 100 else result.get('text', 'No text')
                                metadata = result.get('metadata', {})
                                print(f"\nResult {i+1}:")
                                print(f"Text snippet: {text}")
                                print(f"COMPLETE METADATA: {json.dumps(metadata, indent=2)}")
                                
                                # Special debug for version detection
                                version_info = metadata.get('version', 'Not found')
                                print(f"Version info: {version_info}")
                                
                                # Check if filename contains version info
                                filename = metadata.get('filename', '')
                                if 'v' in filename.lower():
                                    print(f"Filename with possible version: {filename}")
                                    version_match = re.search(r'v(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
                                    if version_match:
                                        extracted_version = version_match.group(1)
                                        print(f"Version from filename: {extracted_version}")
                                
                                # Check if text contains possible version information
                                if 'v' in text.lower():
                                    version_match = re.search(r'v(\d+(?:\.\d+)?)', text, re.IGNORECASE)
                                    if version_match:
                                        extracted_version = version_match.group(1)
                                        print(f"Version from text: {extracted_version}")
                            else:
                                print(f"Result {i+1} is not a dictionary: {type(result)}")
                    else:
                        print(f"Results is not a list or is empty: {type(results)}")
                except Exception as e:
                    print(f"Error printing raw results: {str(e)}")
                print("============================\n")
                
                # Format the results with citations
                formatted_results = []
                for i, result in enumerate(results):
                    try:
                        # Ensure result is a dictionary
                        if not isinstance(result, dict):
                            formatted_results.append(f"Result {i+1}: Invalid result format - {type(result)}")
                            continue
                            
                        # Extract all available metadata
                        metadata = result.get("metadata", {})
                        if not isinstance(metadata, dict):
                            metadata = {}
                            
                        document_id = metadata.get("document_id", "unknown")
                        title = metadata.get("title", "Unknown Document")
                        filename = metadata.get("filename", "Unknown Filename")
                        page = metadata.get("page", "unknown")
                        content_type = metadata.get("content_type", "TEXT")
                        version = metadata.get("version", "unknown")
                        segment_id = metadata.get("segment_id", "unknown")
                        
                        # First, check document_versions cache for this document_id
                        if document_id != "unknown" and document_id in document_versions:
                            version = document_versions[document_id]
                            print(f"Using cached version {version} for document {document_id}, page {page}")
                        
                        # If version still unknown, try to extract from filename
                        elif version == "unknown" and filename != "Unknown Filename":
                            version_match = re.search(r'v(\d+(?:\.\d+)?)', filename, re.IGNORECASE)
                            if version_match:
                                version = version_match.group(1)
                                print(f"Extracted version '{version}' from filename: {filename}")
                        
                        # Format the citation with comprehensive metadata
                        citation = f"Source: {filename}, Page: {page}"
                        
                        # Add version if available
                        if version != "unknown":
                            citation += f", Version: {version}"
                        
                        # Add content type info
                        if content_type == "IMAGE":
                            citation += " (Image)"
    
                        # Add document title if different from filename
                        if title != filename and title != "Unknown Document":
                            citation += f"\nTitle: {title}"
    
                        # Check for knowledge graph source indicators
                        if "[KG]" in result.get('text', ''):
                            citation = f"Source: Knowledge Graph Entry"
                            # Try to extract the document source from KG reference
                            kg_match = re.search(r'\[KG\]\s+(.+?)(,|\)|\s+|$)', result.get('text', ''))
                            if kg_match:
                                citation += f" - {kg_match.group(1)}"
                        
                        # Format the result with detailed metadata
                        text = result.get('text', 'No text available')
                        formatted_result = f"Result {i+1}:\n{citation}\n\n{text}\n"
                        
                        # Include additional metadata details at the end of each result
                        metadata_details = []
                        if segment_id != "unknown":
                            metadata_details.append(f"Segment ID: {segment_id}")
                        
                        # Add any other relevant metadata that might be available
                        for key, value in metadata.items():
                            if key not in ["document_id", "title", "filename", "page", "content_type", "version", "segment_id"] and value:
                                if isinstance(value, str) and len(value) < 100:  # Only include reasonably sized string values
                                    metadata_details.append(f"{key.replace('_', ' ').title()}: {value}")
                        
                        if metadata_details:
                            formatted_result += "Additional Metadata:\n- " + "\n- ".join(metadata_details) + "\n"
                        
                        formatted_results.append(formatted_result)
                    except Exception as e:
                        formatted_results.append(f"Error formatting result {i+1}: {str(e)}")
                
                return "\n".join(formatted_results)
            
            except Exception as e:
                return f"Error retrieving information: {str(e)}"
        
        @agent.tool
        async def database_info(context: RunContext[RAGQueryDeps]) -> str:
            """Get information about the database.
            
            Args:
                context: The run context containing dependencies
                
            Returns:
                Information about the database
            """
            info = context.deps.database_info
            
            # Format the information
            formatted_info = ["Database Information:"]
            
            # Add basic information
            formatted_info.append(f"Database directory: {info.get('database_path', 'Unknown')}")
            formatted_info.append(f"Total documents: {info.get('total_documents', 'Unknown')}")
            formatted_info.append(f"Total chunks: {info.get('total_chunks', 'Unknown')}")
            formatted_info.append(f"Total pages: {info.get('total_pages', 'Unknown')}")
            formatted_info.append(f"Total images: {info.get('total_images', 'Unknown')}")
            
            # Add build information if available
            if "start_time" in info:
                formatted_info.append(f"Build start time: {info['start_time']}")
            if "end_time" in info:
                formatted_info.append(f"Build end time: {info['end_time']}")
            if "processing_time_seconds" in info:
                formatted_info.append(f"Processing time: {info['processing_time_seconds']:.2f} seconds")
            
            return "\n".join(formatted_info)
        
        return agent

    async def initialize_lightrag(self) -> LightRAG:
        """Initialize LightRAG with OpenAI embeddings and GPT-4o-mini.
        
        Returns:
            Initialized LightRAG instance
        """
        return await initialize_lightrag(self.database_dir, self.logger)

    async def query(self, query_text: str) -> str:
        """Query the database with a natural language question.
        
        Args:
            query_text: The question to ask
            
        Returns:
            The answer with citations
        """
        self.logger.info(f"Querying database with: {query_text}")
        
        # Initialize LightRAG if not already initialized
        if self.lightrag is None:
            self.lightrag = await self.initialize_lightrag()
        
        # Create dependencies
        deps = RAGQueryDeps(
            lightrag=self.lightrag,
            database_info=self.database_info
        )
        
        # Run the agent
        self.logger.info("Running query through the agent")
        result = await self.agent.run(query_text, deps=deps)
        
        # Extract the response
        if hasattr(result, 'data'):
            response = result.output
        elif hasattr(result, 'output'):
            response = result.output
        elif hasattr(result, 'response'):
            response = result.response
        elif hasattr(result, 'content'):
            response = result.content
        else:
            response = str(result)
        
        self.logger.info("Query completed successfully")
        return response

    async def interactive_mode(self):
        """Run an interactive query session.
        
        This allows the user to ask multiple questions in a row.
        """
        self.logger.info("Starting interactive query session")
        
        # Initialize LightRAG if not already initialized
        if self.lightrag is None:
            self.lightrag = await self.initialize_lightrag()
        
        print("\nEntersoft ERP Documentation Assistant")
        print("====================================")
        print("Type your questions and press Enter. Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            # Get user input
            query_text = input("\nYour question: ")
            
            # Check if user wants to exit
            if query_text.lower() in ['exit', 'quit', 'q', 'bye']:
                print("\nThank you for using the Entersoft ERP Documentation Assistant. Goodbye!")
                break
            
            # Skip empty queries
            if not query_text.strip():
                continue
            
            # Process the query
            try:
                print("\nSearching knowledge base...")
                response = await self.query(query_text)
                print("\nAnswer:")
                print(response)
            except Exception as e:
                print(f"\nError: {str(e)}")


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Query a persistent lightrag database built from Entersoft Docs"
    )
    parser.add_argument(
        "--database-dir", 
        type=str, 
        default="./lightrag_data",
        help="Directory containing the lightrag database"
    )
    parser.add_argument(
        "--database-info", 
        type=str, 
        default="./output/database_info.json",
        help="Path to the database info JSON file"
    )
    parser.add_argument(
        "--query", 
        type=str,
        help="Question to ask (if not provided, interactive mode will be started)"
    )
    parser.add_argument(
        "--gemini-api-key", 
        type=str, 
        default=os.getenv("GEMINI_API_KEY"),
        help="Google API key for Gemini (for the agent)"
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
        "--interactive",
        action="store_true",
        help="Start in interactive mode (ignores --query)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logs_dir = os.path.join(os.path.dirname(args.database_dir), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f"query_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(args.log_level, log_file)
    
    # Create the querier - force DEBUG logging for testing
    querier = LightRAGDatabaseQuerier(
        database_dir=args.database_dir,
        database_info_path=args.database_info,
        gemini_api_key=args.gemini_api_key,
        openai_api_key=args.openai_api_key,
        log_level=logging.DEBUG
    )
    
    try:
        # Check if interactive mode or single query
        if args.interactive or not args.query:
            await querier.interactive_mode()
        else:
            # Run a single query
            response = await querier.query(args.query)
            print("\nAnswer:")
            print(response)
    
    except Exception as e:
        logging.error(f"Error querying database: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
