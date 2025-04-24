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
            1. Always cite your sources with document titles and page numbers
            2. If information comes from an image, mention that it's from an image and describe what the image shows
            3. If the knowledge base doesn't contain the answer, clearly state that and provide your best general knowledge response
            4. When multiple documents provide relevant information, synthesize the information and cite all sources
            5. Be concise, accurate, and helpful
            6. Format your responses in a clear, readable way with appropriate headings and bullet points when needed
            
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
                    search_query, param=QueryParam(mode="mix", top_k=7)
                )
                
                # Format results
                if not results:
                    return "No relevant information found in the knowledge base."
                
                # Check if results is a string (API might have changed)
                if isinstance(results, str):
                    return f"Retrieved content:\n{results}"
                
                # Format the results with citations
                formatted_results = []
                for i, result in enumerate(results):
                    # Extract metadata
                    metadata = result.get("metadata", {})
                    document_id = metadata.get("document_id", "unknown")
                    title = metadata.get("title", "Unknown Document")
                    page = metadata.get("page", "unknown")
                    content_type = metadata.get("content_type", "TEXT")
                    
                    # Format the citation
                    citation = f"Source: {title}, Page: {page}"
                    if content_type == "IMAGE":
                        citation += " (Image)"
                    
                    # Format the result
                    formatted_result = f"Result {i+1}:\n{citation}\n{result['text']}\n"
                    formatted_results.append(formatted_result)
                
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
            response = result.data
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
    
    # Create the querier
    querier = LightRAGDatabaseQuerier(
        database_dir=args.database_dir,
        database_info_path=args.database_info,
        gemini_api_key=args.gemini_api_key,
        openai_api_key=args.openai_api_key,
        log_level=getattr(logging, args.log_level)
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