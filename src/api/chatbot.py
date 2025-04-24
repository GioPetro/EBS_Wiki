"""Main chatbot application for the Entersoft ERP chatbot."""

import os
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json

from src.core.config import ChatbotConfig, load_config_from_env, load_config_from_file
from src.core.models import ProcessedDocument, UserFeedback, FeedbackType
from src.rag.rag_component import RAGComponent, create_rag_component
from src.rag.query_processor import QueryProcessor, create_query_processor
from src.rag.self_learning_component import (
    SelfLearningComponent, create_self_learning_component,
    enhance_rag_with_self_learning, QueryResponse
)
from src.processors.pdf_processor import PDFProcessor


class ChatMessage:
    """Class representing a chat message."""
    
    def __init__(self, role: str, content: str, message_id: Optional[str] = None):
        """Initialize a chat message.
        
        Args:
            role: Role of the message sender (user or assistant)
            content: Content of the message
            message_id: Unique identifier for the message (generated if not provided)
        """
        self.role = role
        self.content = content
        self.message_id = message_id or str(uuid.uuid4())
        self.timestamp = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary.
        
        Returns:
            Dictionary representation of the message
        """
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary.
        
        Args:
            data: Dictionary representation of the message
            
        Returns:
            ChatMessage instance
        """
        return cls(
            role=data["role"],
            content=data["content"],
            message_id=data["message_id"]
        )


class ChatSession:
    """Class representing a chat session."""
    
    def __init__(self, session_id: Optional[str] = None, max_history: int = 10):
        """Initialize a chat session.
        
        Args:
            session_id: Unique identifier for the session (generated if not provided)
            max_history: Maximum number of messages to keep in history
        """
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[ChatMessage] = []
        self.created_at = datetime.now()
        self.last_activity = self.created_at
        self.metadata: Dict[str, Any] = {}
        self.max_history = max_history
        
    def add_message(self, message: ChatMessage) -> None:
        """Add a message to the session.
        
        Args:
            message: Message to add
        """
        self.messages.append(message)
        self.last_activity = datetime.now()
        
        # Trim history if needed
        if len(self.messages) > self.max_history * 2:  # Keep pairs of messages
            self.messages = self.messages[-self.max_history * 2:]
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get message history.
        
        Returns:
            List of message dictionaries
        """
        return [msg.to_dict() for msg in self.messages]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary.
        
        Returns:
            Dictionary representation of the session
        """
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatSession':
        """Create session from dictionary.
        
        Args:
            data: Dictionary representation of the session
            
        Returns:
            ChatSession instance
        """
        session = cls(session_id=data["session_id"])
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.metadata = data["metadata"]
        
        for msg_data in data["messages"]:
            session.messages.append(ChatMessage.from_dict(msg_data))
        
        return session


class EntersoftChatbot:
    """Main chatbot application for the Entersoft ERP chatbot."""
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the chatbot application.
        
        Args:
            config: Configuration for the chatbot
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Initialize components
        self.rag_component: Optional[RAGComponent] = None
        self.query_processor: Optional[QueryProcessor] = None
        self.self_learning_component: Optional[SelfLearningComponent] = None
        self.pdf_processor: Optional[PDFProcessor] = None
        
        # Session management
        self.sessions: Dict[str, ChatSession] = {}
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Any] = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0,
            "total_response_time": 0.0,
            "queries_by_hour": {},
            "insufficient_answers": 0,
            "enhanced_answers": 0
        }
        
        self.logger.info(f"Initialized Entersoft Chatbot v{config.version}")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger("entersoft_chatbot")
        logger.setLevel(getattr(logging, self.config.logging.level))
        
        # Clear existing handlers
        logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(self.config.logging.format)
        
        # Add console handler if enabled
        if self.config.logging.console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # Add file handler if specified
        if self.config.logging.file:
            # Create directory if it doesn't exist
            log_dir = os.path.dirname(self.config.logging.file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                
            file_handler = logging.FileHandler(self.config.logging.file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    async def initialize(self) -> None:
        """Initialize all components of the chatbot."""
        self.logger.info("Initializing Entersoft Chatbot components")
        
        try:
            # Initialize RAG component
            self.logger.info("Initializing RAG component")
            self.rag_component = await create_rag_component(self.config.rag, self.logger)
            
            # Initialize query processor
            self.logger.info("Initializing query processor")
            self.query_processor = create_query_processor(logger=self.logger)
            
            # Initialize self-learning component if enabled
            if self.config.enable_self_learning and self.config.self_learning:
                self.logger.info("Initializing self-learning component")
                self.self_learning_component = await create_self_learning_component(
                    self.config.self_learning, self.logger
                )
            
            # Initialize PDF processor
            self.logger.info("Initializing PDF processor")
            self.pdf_processor = PDFProcessor(
                input_dir=self.config.data_processing.input_dir,
                output_dir=self.config.data_processing.output_dir,
                gemini_api_key=self.config.api.gemini_api_key,
                log_level=getattr(logging, self.config.logging.level)
            )
            
            self.logger.info("All components initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}", exc_info=True)
            raise
    
    def is_initialized(self) -> bool:
        """Check if all required components are initialized.
        
        Returns:
            True if all required components are initialized, False otherwise
        """
        return (
            self.rag_component is not None and
            self.query_processor is not None and
            self.pdf_processor is not None
        )
    
    async def ensure_initialized(self) -> None:
        """Ensure all components are initialized."""
        if not self.is_initialized():
            await self.initialize()
    
    async def process_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user query.
        
        Args:
            query: User query
            session_id: Session ID (created if not provided)
            
        Returns:
            Dictionary containing:
                - response: Generated response
                - session_id: Session ID
                - query_id: Query ID for feedback (if self-learning is enabled)
                - is_enhanced: Whether the response was enhanced with self-learning
                - is_insufficient: Whether the response was detected as insufficient
        """
        await self.ensure_initialized()
        
        # Get or create session
        session = self._get_or_create_session(session_id)
        
        # Add user message to session
        user_message = ChatMessage(role="user", content=query)
        session.add_message(user_message)
        
        # Update performance metrics
        self.performance_metrics["total_queries"] += 1
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        self.performance_metrics["queries_by_hour"][hour] = self.performance_metrics["queries_by_hour"].get(hour, 0) + 1
        
        start_time = datetime.now()
        
        try:
            # Process query
            self.logger.info(f"Processing query: '{query}'")
            
            # Enhance query using query processor
            enhanced_queries = self.query_processor.enhance_query(query)
            self.logger.debug(f"Enhanced queries: {enhanced_queries}")
            
            # Query RAG component
            response, query_result = await self.rag_component.query_and_generate(
                enhanced_queries[0],  # Use first enhanced query
                self.config.system_prompt
            )
            
            # Assemble context
            context = query_result.context
            
            result = {
                "response": response,
                "session_id": session.session_id,
                "query_id": None,
                "is_enhanced": False,
                "is_insufficient": False
            }
            
            # Apply self-learning if enabled
            if self.config.enable_self_learning and self.self_learning_component:
                # Analyze response
                query_response = await self.self_learning_component.analyze_response(
                    query, response, context
                )
                
                result["query_id"] = query_response.query_id
                result["is_insufficient"] = query_response.is_insufficient
                
                if query_response.is_insufficient:
                    self.performance_metrics["insufficient_answers"] += 1
                    
                    # Enhance response if insufficient
                    enhanced_response, is_enhanced = await enhance_rag_with_self_learning(
                        query, response, context, self.self_learning_component
                    )
                    
                    if is_enhanced:
                        result["response"] = enhanced_response
                        result["is_enhanced"] = True
                        self.performance_metrics["enhanced_answers"] += 1
            
            # Add assistant message to session
            assistant_message = ChatMessage(role="assistant", content=result["response"])
            session.add_message(assistant_message)
            
            # Update performance metrics
            self.performance_metrics["successful_queries"] += 1
            
            return result
        except Exception as e:
            self.logger.error(f"Error processing query: {str(e)}", exc_info=True)
            self.performance_metrics["failed_queries"] += 1
            
            # Add error message to session
            error_message = ChatMessage(
                role="assistant", 
                content="I'm sorry, I encountered an error processing your query. Please try again."
            )
            session.add_message(error_message)
            
            return {
                "response": "I'm sorry, I encountered an error processing your query. Please try again.",
                "session_id": session.session_id,
                "query_id": None,
                "is_enhanced": False,
                "is_insufficient": False,
                "error": str(e)
            }
        finally:
            # Update response time metrics
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self.performance_metrics["total_response_time"] += response_time
            self.performance_metrics["average_response_time"] = (
                self.performance_metrics["total_response_time"] / 
                self.performance_metrics["total_queries"]
            )
    
    async def provide_feedback(self, query_id: str, feedback_type: str, 
                              user_feedback: str, corrected_answer: Optional[str] = None) -> Dict[str, Any]:
        """Provide feedback for a response.
        
        Args:
            query_id: Query ID
            feedback_type: Type of feedback (correction, addition, clarification, confirmation, rejection)
            user_feedback: Feedback provided by the user
            corrected_answer: Corrected or additional answer (optional)
            
        Returns:
            Dictionary containing:
                - feedback_id: Feedback ID
                - accepted: Whether the feedback was accepted
                - reason: Reason for rejection (if rejected)
        """
        await self.ensure_initialized()
        
        if not self.self_learning_component:
            raise RuntimeError("Self-learning component not initialized")
        
        # Convert feedback type string to enum
        try:
            feedback_type_enum = FeedbackType(feedback_type.lower())
        except ValueError:
            return {
                "feedback_id": None,
                "accepted": False,
                "reason": f"Invalid feedback type: {feedback_type}"
            }
        
        # Create feedback
        feedback = await self.self_learning_component.create_feedback(
            query_id, feedback_type_enum, user_feedback, corrected_answer
        )
        
        # Process feedback
        accepted, reason = await self.self_learning_component.process_user_feedback(feedback)
        
        return {
            "feedback_id": feedback.feedback_id,
            "accepted": accepted,
            "reason": reason
        }
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """Process a document and ingest it into the RAG component.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing processing results
        """
        await self.ensure_initialized()
        
        try:
            self.logger.info(f"Processing document: {file_path}")
            
            # Process document
            processed_docs, lightrag_path = self.pdf_processor.process_files(
                [file_path],
                chunk_size=self.config.data_processing.chunk_size,
                overlap=self.config.data_processing.overlap
            )
            
            if not processed_docs:
                return {
                    "success": False,
                    "error": "No documents were processed",
                    "document_id": None
                }
            
            # Ingest into RAG component
            doc_count = await self.rag_component.ingest_from_file(lightrag_path)
            
            return {
                "success": True,
                "document_id": processed_docs[0].document_id,
                "chunks_ingested": doc_count,
                "title": processed_docs[0].title,
                "total_pages": processed_docs[0].total_pages,
                "total_segments": len(processed_docs[0].segments)
            }
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "document_id": None
            }
    
    def _get_or_create_session(self, session_id: Optional[str] = None) -> ChatSession:
        """Get an existing session or create a new one.
        
        Args:
            session_id: Session ID (created if not provided or not found)
            
        Returns:
            Chat session
        """
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]
        
        # Create new session
        session = ChatSession(
            session_id=session_id,
            max_history=self.config.max_history_length
        )
        self.sessions[session.session_id] = session
        
        return session
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get a session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Chat session or None if not found
        """
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions.
        
        Returns:
            List of session dictionaries
        """
        return [session.to_dict() for session in self.sessions.values()]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return self.performance_metrics
    
    async def get_insufficient_responses(self) -> List[Dict[str, Any]]:
        """Get all insufficient responses.
        
        Returns:
            List of insufficient responses
        """
        await self.ensure_initialized()
        
        if not self.self_learning_component:
            return []
        
        insufficient_responses = self.self_learning_component.get_insufficient_responses()
        
        return [
            {
                "query_id": resp.query_id,
                "query": resp.query,
                "response": resp.response,
                "timestamp": resp.timestamp.isoformat(),
                "completeness": resp.quality_metrics.completeness,
                "relevance": resp.quality_metrics.relevance,
                "accuracy": resp.quality_metrics.accuracy,
                "clarity": resp.quality_metrics.clarity,
                "overall_score": resp.quality_metrics.overall_score()
            }
            for resp in insufficient_responses
        ]
    
    async def save_state(self, file_path: Optional[str] = None) -> str:
        """Save chatbot state to a file.
        
        Args:
            file_path: Path to save the state (default: working_dir/chatbot_state.json)
            
        Returns:
            Path to the saved state file
        """
        if file_path is None:
            file_path = os.path.join(self.config.working_dir, "chatbot_state.json")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare state
        state = {
            "version": self.config.version,
            "timestamp": datetime.now().isoformat(),
            "sessions": [session.to_dict() for session in self.sessions.values()],
            "performance_metrics": self.performance_metrics
        }
        
        # Save state
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Chatbot state saved to {file_path}")
        return file_path
    
    async def load_state(self, file_path: Optional[str] = None) -> bool:
        """Load chatbot state from a file.
        
        Args:
            file_path: Path to load the state from (default: working_dir/chatbot_state.json)
            
        Returns:
            True if state was loaded successfully, False otherwise
        """
        if file_path is None:
            file_path = os.path.join(self.config.working_dir, "chatbot_state.json")
        
        if not os.path.exists(file_path):
            self.logger.warning(f"State file not found: {file_path}")
            return False
        
        try:
            # Load state
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Load sessions
            self.sessions = {}
            for session_data in state["sessions"]:
                session = ChatSession.from_dict(session_data)
                self.sessions[session.session_id] = session
            
            # Load performance metrics
            self.performance_metrics = state["performance_metrics"]
            
            self.logger.info(f"Chatbot state loaded from {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading state: {str(e)}", exc_info=True)
            return False
    
    async def shutdown(self) -> None:
        """Shutdown the chatbot and save state."""
        self.logger.info("Shutting down Entersoft Chatbot")
        
        # Save state
        await self.save_state()
        
        # Close any resources
        # (Add any cleanup code here)
        
        self.logger.info("Shutdown complete")


async def create_chatbot(config: Optional[Union[ChatbotConfig, str, Dict[str, Any]]] = None) -> EntersoftChatbot:
    """Create and initialize an Entersoft ERP chatbot.
    
    Args:
        config: Configuration for the chatbot (ChatbotConfig, path to JSON file, or dict)
        
    Returns:
        Initialized chatbot
    """
    # Determine configuration
    if config is None:
        # Load from environment
        chatbot_config = load_config_from_env()
    elif isinstance(config, str):
        # Load from file
        chatbot_config = load_config_from_file(config)
    elif isinstance(config, dict):
        # Create from dict
        chatbot_config = ChatbotConfig(**config)
    else:
        # Use provided config
        chatbot_config = config
    
    # Create chatbot
    chatbot = EntersoftChatbot(chatbot_config)
    
    # Initialize
    await chatbot.initialize()
    
    return chatbot