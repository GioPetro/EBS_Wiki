"""Main entry point for the Entersoft ERP chatbot application."""

import os
import sys
import argparse
import asyncio
import logging
import signal
from typing import Optional, Dict, Any
import json

from dotenv import load_dotenv

from src.core.config import load_config_from_env, load_config_from_file, ChatbotConfig
from src.api.chatbot import create_chatbot, EntersoftChatbot
from src.cli.cli import ChatbotCLI
from src.core.monitoring import PerformanceMonitor, get_enhanced_logger
from src.core.error_handler import create_error_handler, set_global_exception_hook


class EntersoftChatbotApp:
    """Main application class for the Entersoft ERP chatbot."""
    
    def __init__(self):
        """Initialize the application."""
        self.config: Optional[ChatbotConfig] = None
        self.chatbot: Optional[EntersoftChatbot] = None
        self.cli: Optional[ChatbotCLI] = None
        self.logger: Optional[logging.Logger] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.error_handler = None
        self.running = True
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)
    
    def _handle_interrupt(self, sig, frame):
        """Handle interrupt signal."""
        print("\nShutting down...")
        self.running = False
        
        # Stop event loop if running
        if asyncio.get_event_loop().is_running():
            asyncio.get_event_loop().stop()
    
    async def initialize(self, config_path: Optional[str] = None) -> None:
        """Initialize the application.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Load environment variables
        load_dotenv()
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = load_config_from_file(config_path)
        else:
            self.config = load_config_from_env()
        
        # Set up logging
        log_file = self.config.logging.file
        log_level = getattr(logging, self.config.logging.level)
        
        self.logger = get_enhanced_logger(
            "entersoft_chatbot",
            level=log_level,
            file_path=log_file,
            console=self.config.logging.console
        )
        
        # Set up performance monitoring
        metrics_file = os.path.join(self.config.working_dir, "metrics.jsonl")
        self.performance_monitor = PerformanceMonitor(
            log_interval=3600,  # Log metrics every hour
            log_file=metrics_file
        )
        
        # Set performance monitor for logger
        if isinstance(self.logger, logging.Logger):
            self.logger.set_performance_monitor(self.performance_monitor)
        
        # Set up error handling
        self.error_handler = create_error_handler(self.logger, self.performance_monitor)
        set_global_exception_hook(self.error_handler)
        
        # Initialize chatbot
        self.logger.info(f"Initializing Entersoft Chatbot v{self.config.version}")
        self.chatbot = await create_chatbot(self.config)
        
        # Initialize CLI
        self.cli = ChatbotCLI()
        self.cli.chatbot = self.chatbot
        
        self.logger.info("Application initialized successfully")
    
    async def run_cli(self) -> None:
        """Run the CLI interface."""
        if not self.cli or not self.chatbot:
            raise RuntimeError("Application not initialized")
        
        await self.cli.start_interactive_session()
    
    async def run_api(self) -> None:
        """Run the API interface."""
        if not self.chatbot or not self.config:
            raise RuntimeError("Application not initialized")
        
        # Import here to avoid circular imports
        from src.api.api import start_api
        
        self.logger.info(f"Starting API server on {self.config.ui.api_host}:{self.config.ui.api_port}")
        
        # Set environment variable for API to find config
        os.environ["CHATBOT_CONFIG"] = os.path.join(self.config.working_dir, "config.json")
        
        # Save config for API to use
        config_path = os.path.join(self.config.working_dir, "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(self.config.json(indent=2))
        
        # Start API
        start_api(
            host=self.config.ui.api_host,
            port=self.config.ui.api_port,
            workers=self.config.ui.api_workers,
            timeout=self.config.ui.api_timeout
        )
    
    async def process_single_query(self, query: str) -> None:
        """Process a single query and exit.
        
        Args:
            query: Query string
        """
        if not self.chatbot:
            raise RuntimeError("Application not initialized")
        
        result = await self.chatbot.process_query(query)
        print(result["response"])
    
    async def process_document(self, file_path: str) -> None:
        """Process a document and exit.
        
        Args:
            file_path: Path to document file
        """
        if not self.chatbot:
            raise RuntimeError("Application not initialized")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        print(f"Processing document: {file_path}")
        
        result = await self.chatbot.process_document(file_path)
        
        if result["success"]:
            print(f"Document processed successfully:")
            print(f"  Document ID: {result['document_id']}")
            print(f"  Title: {result['title']}")
            print(f"  Pages: {result['total_pages']}")
            print(f"  Segments: {result['total_segments']}")
            print(f"  Chunks ingested: {result['chunks_ingested']}")
        else:
            print(f"Error processing document: {result['error']}")
    
    async def shutdown(self) -> None:
        """Shutdown the application."""
        if self.logger:
            self.logger.info("Shutting down application")
        
        # Shutdown chatbot
        if self.chatbot:
            await self.chatbot.shutdown()
        
        # Log final metrics
        if self.performance_monitor:
            self.performance_monitor.log_metrics()
        
        if self.logger:
            self.logger.info("Application shutdown complete")


async def main():
    """Main entry point."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Entersoft ERP Chatbot"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--query", 
        type=str, 
        help="Process a single query and exit"
    )
    parser.add_argument(
        "--process", 
        type=str, 
        help="Process a document and exit"
    )
    parser.add_argument(
        "--api", 
        action="store_true",
        help="Start API server"
    )
    parser.add_argument(
        "--cli", 
        action="store_true",
        help="Start CLI interface"
    )
    
    args = parser.parse_args()
    
    # Create application
    app = EntersoftChatbotApp()
    
    try:
        # Initialize application
        await app.initialize(args.config)
        
        # Determine what to run
        if args.query:
            # Process single query
            await app.process_single_query(args.query)
        elif args.process:
            # Process document
            await app.process_document(args.process)
        elif args.api:
            # Run API
            await app.run_api()
        elif args.cli:
            # Run CLI
            await app.run_cli()
        else:
            # Default to CLI
            await app.run_cli()
    except Exception as e:
        print(f"Error: {str(e)}")
        if hasattr(app, 'logger') and app.logger:
            app.logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return 1
    finally:
        # Shutdown
        await app.shutdown()
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)