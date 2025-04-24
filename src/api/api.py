"""API interface for the Entersoft ERP chatbot."""

import os
import sys
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.config import load_config_from_env, load_config_from_file, ChatbotConfig
from src.api.chatbot import create_chatbot, EntersoftChatbot


# Pydantic models for API requests and responses
class QueryRequest(BaseModel):
    """Model for query request."""
    query: str = Field(..., description="User query")
    session_id: Optional[str] = Field(None, description="Session ID (created if not provided)")


class QueryResponse(BaseModel):
    """Model for query response."""
    response: str = Field(..., description="Generated response")
    session_id: str = Field(..., description="Session ID")
    query_id: Optional[str] = Field(None, description="Query ID for feedback")
    is_enhanced: bool = Field(False, description="Whether the response was enhanced with self-learning")
    is_insufficient: bool = Field(False, description="Whether the response was detected as insufficient")
    error: Optional[str] = Field(None, description="Error message if any")


class FeedbackRequest(BaseModel):
    """Model for feedback request."""
    query_id: str = Field(..., description="Query ID")
    feedback_type: str = Field(..., description="Type of feedback")
    user_feedback: str = Field(..., description="Feedback provided by the user")
    corrected_answer: Optional[str] = Field(None, description="Corrected or additional answer")


class FeedbackResponse(BaseModel):
    """Model for feedback response."""
    feedback_id: Optional[str] = Field(None, description="Feedback ID")
    accepted: bool = Field(..., description="Whether the feedback was accepted")
    reason: Optional[str] = Field(None, description="Reason for rejection if rejected")


class DocumentRequest(BaseModel):
    """Model for document processing request."""
    file_path: str = Field(..., description="Path to the document file")


class DocumentResponse(BaseModel):
    """Model for document processing response."""
    success: bool = Field(..., description="Whether the processing was successful")
    document_id: Optional[str] = Field(None, description="Document ID")
    chunks_ingested: Optional[int] = Field(None, description="Number of chunks ingested")
    title: Optional[str] = Field(None, description="Document title")
    total_pages: Optional[int] = Field(None, description="Total number of pages")
    total_segments: Optional[int] = Field(None, description="Total number of segments")
    error: Optional[str] = Field(None, description="Error message if any")


class SessionInfo(BaseModel):
    """Model for session information."""
    session_id: str = Field(..., description="Session ID")
    created_at: str = Field(..., description="Creation timestamp")
    last_activity: str = Field(..., description="Last activity timestamp")
    message_count: int = Field(..., description="Number of messages in the session")


class PerformanceMetrics(BaseModel):
    """Model for performance metrics."""
    total_queries: int = Field(..., description="Total number of queries processed")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    average_response_time: float = Field(..., description="Average response time in seconds")
    insufficient_answers: int = Field(..., description="Number of insufficient answers")
    enhanced_answers: int = Field(..., description="Number of enhanced answers")
    queries_by_hour: Dict[str, int] = Field(..., description="Queries by hour")


# Global chatbot instance
chatbot: Optional[EntersoftChatbot] = None


async def get_chatbot() -> EntersoftChatbot:
    """Get the chatbot instance.
    
    Returns:
        Chatbot instance
    """
    global chatbot
    
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return chatbot


# Create FastAPI app
app = FastAPI(
    title="Entersoft ERP Chatbot API",
    description="API for interacting with the Entersoft ERP chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.on_event("startup")
async def startup_event():
    """Initialize chatbot on startup."""
    global chatbot
    
    try:
        # Load configuration
        config_path = os.environ.get("CHATBOT_CONFIG")
        
        if config_path and os.path.exists(config_path):
            chatbot = await create_chatbot(config_path)
        else:
            chatbot = await create_chatbot()
        
        logging.info("Chatbot initialized successfully")
    except Exception as e:
        logging.error(f"Error initializing chatbot: {str(e)}", exc_info=True)
        # We'll let the app start anyway, but endpoints will return 503 until chatbot is initialized


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown chatbot on app shutdown."""
    global chatbot
    
    if chatbot:
        await chatbot.shutdown()
        logging.info("Chatbot shut down")


@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint.
    
    Returns:
        Basic information about the API
    """
    return {
        "name": "Entersoft ERP Chatbot API",
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Process a user query.
    
    Args:
        request: Query request
        chatbot: Chatbot instance
        
    Returns:
        Query response
    """
    try:
        result = await chatbot.process_query(request.query, request.session_id)
        return result
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        return QueryResponse(
            response="I'm sorry, I encountered an error processing your query. Please try again.",
            session_id=request.session_id or "error",
            error=str(e)
        )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest, chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Provide feedback for a response.
    
    Args:
        request: Feedback request
        chatbot: Chatbot instance
        
    Returns:
        Feedback response
    """
    try:
        result = await chatbot.provide_feedback(
            request.query_id,
            request.feedback_type,
            request.user_feedback,
            request.corrected_answer
        )
        return result
    except Exception as e:
        logging.error(f"Error providing feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-document", response_model=DocumentResponse)
async def process_document(
    request: DocumentRequest, 
    background_tasks: BackgroundTasks,
    chatbot: EntersoftChatbot = Depends(get_chatbot)
):
    """Process a document and ingest it into the RAG component.
    
    Args:
        request: Document processing request
        background_tasks: Background tasks
        chatbot: Chatbot instance
        
    Returns:
        Document processing response
    """
    # Check if file exists
    if not os.path.exists(request.file_path):
        raise HTTPException(status_code=404, detail=f"File not found: {request.file_path}")
    
    try:
        # Process document (this can be slow, so we'll do it in the background)
        background_tasks.add_task(chatbot.process_document, request.file_path)
        
        return DocumentResponse(
            success=True,
            document_id="processing",  # Placeholder
            error=None,
            chunks_ingested=0,  # Will be updated when processing completes
            title="Processing...",  # Placeholder
            total_pages=0,  # Will be updated when processing completes
            total_segments=0  # Will be updated when processing completes
        )
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}", exc_info=True)
        return DocumentResponse(
            success=False,
            error=str(e)
        )


@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions(chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Get all sessions.
    
    Args:
        chatbot: Chatbot instance
        
    Returns:
        List of session information
    """
    sessions = chatbot.get_all_sessions()
    
    return [
        SessionInfo(
            session_id=session["session_id"],
            created_at=session["created_at"],
            last_activity=session["last_activity"],
            message_count=len(session["messages"])
        )
        for session in sessions
    ]


@app.get("/session/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str, chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Get a session by ID.
    
    Args:
        session_id: Session ID
        chatbot: Chatbot instance
        
    Returns:
        Session information
    """
    session = chatbot.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    return session.to_dict()


@app.get("/metrics", response_model=PerformanceMetrics)
async def get_metrics(chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Get performance metrics.
    
    Args:
        chatbot: Chatbot instance
        
    Returns:
        Performance metrics
    """
    return chatbot.get_performance_metrics()


@app.get("/insufficient-responses", response_model=List[Dict[str, Any]])
async def get_insufficient_responses(chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Get all insufficient responses.
    
    Args:
        chatbot: Chatbot instance
        
    Returns:
        List of insufficient responses
    """
    return await chatbot.get_insufficient_responses()


@app.post("/save-state", response_model=Dict[str, Any])
async def save_state(file_path: Optional[str] = None, chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Save chatbot state to a file.
    
    Args:
        file_path: Path to save the state (optional)
        chatbot: Chatbot instance
        
    Returns:
        Result of the operation
    """
    try:
        saved_path = await chatbot.save_state(file_path)
        
        return {
            "success": True,
            "file_path": saved_path
        }
    except Exception as e:
        logging.error(f"Error saving state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-state", response_model=Dict[str, Any])
async def load_state(file_path: Optional[str] = None, chatbot: EntersoftChatbot = Depends(get_chatbot)):
    """Load chatbot state from a file.
    
    Args:
        file_path: Path to load the state from (optional)
        chatbot: Chatbot instance
        
    Returns:
        Result of the operation
    """
    try:
        success = await chatbot.load_state(file_path)
        
        return {
            "success": success
        }
    except Exception as e:
        logging.error(f"Error loading state: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler.
    
    Args:
        request: Request
        exc: Exception
        
    Returns:
        JSON response with error details
    """
    logging.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


def start_api(host: str = "127.0.0.1", port: int = 8000, workers: int = 4, timeout: int = 60):
    """Start the API server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        workers: Number of worker processes
        timeout: Timeout in seconds
    """
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        timeout_keep_alive=timeout
    )


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    config = load_config_from_env()
    
    # Start API
    start_api(
        host=config.ui.api_host,
        port=config.ui.api_port,
        workers=config.ui.api_workers,
        timeout=config.ui.api_timeout
    )