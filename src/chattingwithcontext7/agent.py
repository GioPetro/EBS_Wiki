"""Agent implementation using pydantic_ai for PDF processing with LightRAG."""

import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import asyncio
import dotenv

from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.gemini import GeminiModel
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

from src.chattingwithcontext7.models import ProcessedDocument

# Load environment variables from .env file
dotenv.load_dotenv()


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG
    documents: List[ProcessedDocument]


async def initialize_rag(working_dir: str) -> LightRAG:
    """Initialize LightRAG.
    
    Args:
        working_dir: Working directory for LightRAG
        
    Returns:
        Initialized LightRAG instance
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Create working directory if it doesn't exist
    os.makedirs(working_dir, exist_ok=True)
    logger.info(f"Created or verified working directory: {working_dir}")
    
    # Initialize LightRAG
    logger.info("Initializing LightRAG with OpenAI embedding and GPT-4o-mini LLM")
    rag = LightRAG(
        working_dir=working_dir,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete
    )
    
    # Initialize storages
    logger.info("Initializing LightRAG storages")
    await rag.initialize_storages()
    logger.info("Initializing pipeline status")
    await initialize_pipeline_status()
    
    # Save the configuration
    logger.info(f"LightRAG initialized successfully in {working_dir}")
    logger.info("This database can be reused in future runs by using --skip-processing and --lightrag-path")
    
    return rag


# Create the Gemini model using pydantic_ai
# Use the API key from the environment variable
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API key is required. Set GEMINI_API_KEY in .env")

gemini_model = GeminiModel(
    'gemini-2.0-flash',
    api_key=gemini_api_key
)

# Create the Pydantic AI agent with Gemini model
pdf_agent = Agent(
    gemini_model,
    deps_type=RAGDeps,
    system_prompt="""
    You are a helpful assistant that answers questions about PDF documents with images.
    Use the retrieve tool to get relevant information from the processed documents before answering.
    
    When answering:
    1. Cite the specific document and page number where you found the information
    2. If the information comes from an image, mention that it's from an image and describe what the image shows
    3. If the documents don't contain the answer, clearly state that the information isn't available
       in the current documents and provide your best general knowledge response
    4. When multiple documents provide relevant information, synthesize the information and cite all sources
    
    When asked about what information is contained in the PDFs, automatically use the retrieve tool
    to get the content of all documents and provide a summary of the information they contain.
    
    Be concise, accurate, and helpful.
    """
)


@pdf_agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    """Retrieve relevant documents from LightRAG based on a search query.
    
    Args:
        context: The run context containing dependencies
        search_query: The search query to find relevant documents
        
    Returns:
        Formatted context information from the retrieved documents
    """
    try:
        # Query LightRAG
        results = await context.deps.lightrag.aquery(
            search_query, param=QueryParam(mode="mix", top_k=5)
        )
        
        # Format results
        if not results:
            return "No relevant information found in the documents."
        
        # Check if results is a string (API might have changed)
        if isinstance(results, str):
            return f"Retrieved content:\n{results}"
        
        # If results is a list of dictionaries (original API)
        formatted_results = []
        for i, result in enumerate(results):
            # Extract metadata
            metadata = result.get("metadata", {})
            document_id = metadata.get("document_id", "unknown")
            page = metadata.get("page", "unknown")
            content_type = metadata.get("content_type", "TEXT")
            
            # Find document title
            document_title = "Unknown Document"
            for doc in context.deps.documents:
                if doc.document_id == document_id:
                    document_title = doc.title
                    break
            
            # Format the result
            source_info = f"Document: {document_title}, Page: {page}"
            if content_type == "IMAGE":
                source_info += " (Image)"
            
            formatted_result = f"Result {i+1}:\n{source_info}\n{result['text']}\n"
            formatted_results.append(formatted_result)
    except Exception as e:
        return f"Error retrieving information: {str(e)}\n\nFallback: Using document list instead.\n\n{await list_documents(context)}"
    
    return "\n".join(formatted_results)


@pdf_agent.tool
async def list_documents(context: RunContext[RAGDeps]) -> str:
    """List all available documents.
    
    Args:
        context: The run context containing dependencies
        
    Returns:
        Formatted list of available documents
    """
    if not context.deps.documents:
        return "No documents available."
    
    formatted_list = ["Available Documents:"]
    for i, doc in enumerate(context.deps.documents):
        formatted_list.append(f"{i+1}. {doc.title} ({doc.total_pages} pages, {len(doc.segments)} segments)")
    
    return "\n".join(formatted_list)


@pdf_agent.tool
async def document_details(context: RunContext[RAGDeps], document_index: int) -> str:
    """Get detailed information about a specific document.
    
    Args:
        context: The run context containing dependencies
        document_index: The index of the document (1-based)
        
    Returns:
        Detailed information about the document
    """
    if not context.deps.documents:
        return "No documents available."
    
    if document_index < 1 or document_index > len(context.deps.documents):
        return f"Invalid document index. Please provide a number between 1 and {len(context.deps.documents)}."
    
    doc = context.deps.documents[document_index - 1]
    
    # Count segment types
    segment_counts = {}
    for segment in doc.segments:
        content_type = segment.content_type
        if content_type not in segment_counts:
            segment_counts[content_type] = 0
        segment_counts[content_type] += 1
    
    # Format segment counts
    segment_info = []
    for content_type, count in segment_counts.items():
        segment_info.append(f"{content_type}: {count}")
    
    # Format document details
    details = [
        f"Document: {doc.title}",
        f"Filename: {doc.filename}",
        f"ID: {doc.document_id}",
        f"Total Pages: {doc.total_pages}",
        f"Total Segments: {len(doc.segments)}",
        f"Segment Types: {', '.join(segment_info)}",
        f"Processed At: {doc.processed_at.isoformat()}"
    ]
    
    return "\n".join(details)


async def run_rag_agent(query: str, lightrag: LightRAG, documents: List[ProcessedDocument]) -> str:
    """Run the RAG agent to answer a question about the documents.
    
    Args:
        query: The question to answer
        lightrag: Initialized LightRAG instance
        documents: List of processed documents
        
    Returns:
        The agent's response
    """
    # Create dependencies
    deps = RAGDeps(lightrag=lightrag, documents=documents)
    
    # Run the agent
    result = await pdf_agent.run(query, deps=deps)
    
    # Extract the data attribute which contains the actual response
    if hasattr(result, 'data'):
        return result.data
    elif hasattr(result, 'output'):
        return result.output
    elif hasattr(result, 'response'):
        return result.response
    elif hasattr(result, 'content'):
        return result.content
    else:
        # If we can't find a suitable attribute, convert the result to a string
        return str(result)