"""PDF processor for extracting text and images from PDFs using Gemini 2.0 Flash."""

import os
import logging
import base64
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import uuid
import time
import asyncio
from datetime import datetime
import dotenv

import fitz  # PyMuPDF
from PIL import Image
import io

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.models.gemini import GeminiModel

from src.chattingwithcontext7.models import (
    ProcessedDocument,
    DocumentSegment,
    ImageAnalysisResult,
    ProcessingStats
)

# Load environment variables from .env file
dotenv.load_dotenv()


class PDFProcessor:
    """Process PDFs with images using multiple AI models (Gemini, OpenAI, Claude)."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        gemini_api_key: str = None,
        openai_api_key: str = None,
        claude_api_key: str = None,
        image_reader_model: str = "gemini",
        image_handling: str = "semi-structured",
        log_level: int = logging.INFO,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize PDF processor.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_dir: Directory to save processed output
            gemini_api_key: Google API key for Gemini
            openai_api_key: OpenAI API key
            claude_api_key: Anthropic API key for Claude
            image_reader_model: Model to use for image analysis ("gemini", "openai", or "claude")
            log_level: Logging level
            config: Configuration dictionary with processing parameters
        """
        # Set up directories
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.lightrag_dir = os.path.join(output_dir, "lightrag")
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.lightrag_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
            # Add file handler
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, f"pdf_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Set up configuration
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.image_reader_model = image_reader_model.lower()
        self.image_handling = image_handling.lower()
        
        # Validate image reader model
        if self.image_reader_model not in ["gemini", "openai", "claude"]:
            raise ValueError("Image reader model must be one of: 'gemini', 'openai', 'claude'")
            
        # Validate image handling mode
        if self.image_handling not in ["plain-text", "semi-structured"]:
            raise ValueError("Image handling must be one of: 'plain-text', 'semi-structured'")
        
        # Set up API keys
        self.gemini_api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.claude_api_key = claude_api_key or os.getenv("CLAUDE_API_KEY")
            
        # Initialize AI models based on selected image reader
        self.agent = None
        
        # Setup models based on selected image reader
        if self.image_reader_model == "gemini":
            if not self.gemini_api_key:
                raise ValueError("Gemini API key is required for Gemini image analysis. Provide it as an argument or set GEMINI_API_KEY in .env")
            
            # Set the API key in the environment
            os.environ["GEMINI_API_KEY"] = self.gemini_api_key
            
            # Get Gemini model name from config or use default
            gemini_model = self.config.get("gemini_model", "gemini-2.0-flash")
                
            # Create the model with the correct provider initialization
            from pydantic_ai.providers.google_gla import GoogleGLAProvider
            from pydantic_ai.models.gemini import GeminiModel
            model = GeminiModel(gemini_model, provider=GoogleGLAProvider(api_key=self.gemini_api_key))
            
            # Create the agent
            self.agent = Agent(model)
            
        if self.image_reader_model == "openai" or self.openai_api_key:
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required for OpenAI image analysis. Provide it as an argument or set OPENAI_API_KEY in .env")
                
            # Set the API key in the environment
            os.environ["OPENAI_API_KEY"] = self.openai_api_key
            
            # Initialize OpenAI model if it's the selected reader
            if self.image_reader_model == "openai":
                from pydantic_ai.models.openai import OpenAIModel
                from pydantic_ai.providers.openai import OpenAIProvider
                
                # Use GPT-4o for image analysis
                openai_model = "gpt-4.1-nano"
                model = OpenAIModel(openai_model, provider=OpenAIProvider(api_key=self.openai_api_key))
                
                # Create the agent
                self.agent = Agent(model)
                
        if self.image_reader_model == "claude":
            if not self.claude_api_key:
                raise ValueError("Claude API key is required for Claude image analysis. Provide it as an argument or set CLAUDE_API_KEY in .env")
                
            # Set the API key in the environment
            os.environ["ANTHROPIC_API_KEY"] = self.claude_api_key
            
            # Initialize Claude model
            from pydantic_ai.models.anthropic import AnthropicModel
            from pydantic_ai.providers.anthropic import AnthropicProvider
            
            # Use Claude 3 Sonnet for image analysis
            claude_model = "claude-3.5-sonnet"
            model = AnthropicModel(claude_model, provider=AnthropicProvider(api_key=self.claude_api_key))
            
            # Create the agent
            self.agent = Agent(model)
        
        # Ensure an agent was initialized
        if not self.agent:
            raise ValueError(f"Failed to initialize an agent for the selected image reader model: {self.image_reader_model}")
        
        # Initialize statistics
        self.stats = ProcessingStats()
        
        # Set up retry configuration
        from src.chattingwithcontext7.config import RetryConfig
        self.retry_config = RetryConfig(
            max_retries=self.config.get("max_retries", 3),
            initial_delay=self.config.get("initial_delay", 1.0),
            max_delay=self.config.get("max_delay", 60.0),
            backoff_factor=self.config.get("backoff_factor", 2.0)
        )
    
    async def process_directory(self, file_pattern: str = "*.pdf") -> Tuple[List[ProcessedDocument], str]:
        """Process all PDF files in the input directory.
        
        Args:
            file_pattern: File pattern to match PDF files
            
        Returns:
            Tuple containing:
                - List of processed documents
                - Path to the saved LightRAG documents
        """
        self.logger.info(f"Processing PDF files in {self.input_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Images directory: {self.images_dir}")
        self.logger.info(f"LightRAG directory: {self.lightrag_dir}")
        
        # Find all PDF files
        pdf_files = list(Path(self.input_dir).glob(file_pattern))
        self.logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Log the list of PDF files
        for i, pdf_path in enumerate(pdf_files):
            self.logger.info(f"  {i+1}. {pdf_path}")
        
        # Process each PDF
        processed_documents = []
        start_time = time.time()
        
        for i, pdf_path in enumerate(pdf_files):
            self.logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_path}")
            document = await self.process_pdf(str(pdf_path))
            
            if document:
                processed_documents.append(document)
                
                # Save individual document to LightRAG format
                lightrag_docs = self.transform_for_lightrag(document)
                doc_filename = Path(pdf_path).stem
                self.save_lightrag_documents(
                    lightrag_docs,
                    self.lightrag_dir,
                    f"{doc_filename}_lightrag"
                )
        
        # Update processing time
        self.stats.processing_time = time.time() - start_time
        
        # Create combined LightRAG documents
        all_lightrag_docs = self.create_lightrag_batch(processed_documents)
        
        # Save combined LightRAG documents
        lightrag_path = self.save_lightrag_documents(
            all_lightrag_docs,
            self.lightrag_dir,
            "combined_lightrag"
        )
        
        # Save processing stats
        self._save_processing_stats()
        
        return processed_documents, lightrag_path
    
    async def process_pdf(self, pdf_path: str) -> Optional[ProcessedDocument]:
        """Process a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Processed document or None if processing failed
        """
        self.logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Generate document ID
            document_id = str(uuid.uuid4())
            
            # Create document-specific image directory
            doc_images_dir = os.path.join(self.images_dir, document_id)
            os.makedirs(doc_images_dir, exist_ok=True)
            
            # Extract text and images
            segments, metadata = await self.extract_text_and_images(pdf_path, doc_images_dir)
            
            # Create processed document
            document = ProcessedDocument(
                document_id=document_id,
                filename=Path(pdf_path).name,
                title=Path(pdf_path).stem,
                total_pages=metadata["total_pages"],
                segments=segments,
                metadata=metadata
            )
            
            # Update statistics
            self._update_stats_for_document(document)
            
            return document
            
        except Exception as e:
            self.logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            self.stats.failed_documents += 1
            return None
    
    def detect_version(self, text: str) -> Optional[str]:
        """Detect version information from text using simple regex patterns.
        
        This is a minimal implementation that looks for common version patterns like
        'v4', 'version 4', etc. in the extracted text.
        
        Args:
            text: The text content to analyze
            
        Returns:
            Detected version string or None if no version is found
        """
        # Simple patterns to match version information
        patterns = [
            r'v(\d+(\.\d+)*)',                # Matches v4, v4.0, v4.1.2
            r'version\s+(\d+(\.\d+)*)',       # Matches version 4, version 4.0
            r'ver\.\s*(\d+(\.\d+)*)',         # Matches ver. 4, ver.4.0
            r'release\s+(\d+(\.\d+)*)'        # Matches release 4, release 4.0
        ]
        
        # Convert text to lowercase for case-insensitive matching
        lower_text = text.lower()
        
        for pattern in patterns:
            matches = re.findall(pattern, lower_text, re.IGNORECASE)
            if matches:
                # Return the first match (first captured group)
                return matches[0][0]
                
        # Check filename in case version isn't in the text
        if "pdf_path" in self.__dict__:
            filename = Path(self.pdf_path).stem
            for pattern in patterns:
                matches = re.findall(pattern, filename, re.IGNORECASE)
                if matches:
                    return matches[0][0]
        
        return None
        
    async def extract_text_and_images(self, pdf_path: str, images_dir: str) -> Tuple[List[DocumentSegment], Dict[str, Any]]:
        """Extract text and images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            images_dir: Directory to save extracted images
            
        Returns:
            Tuple containing:
                - List of document segments
                - Metadata dictionary
        """
        segments = []
        metadata = {"total_pages": 0, "total_images": 0}
        
        # Store pdf_path for potential version detection from filename
        self.pdf_path = pdf_path
        
        # Open the PDF using context manager for proper resource management
        with fitz.open(pdf_path) as doc:
            metadata["total_pages"] = len(doc)
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    # Detect version from text
                    detected_version = self.detect_version(text)
                    
                    # Prepare segment metadata
                    segment_metadata = {
                        "source": "text_extraction"
                    }
                    
                    # Add version information if detected
                    if detected_version:
                        segment_metadata["version"] = detected_version
                    
                    text_segment = DocumentSegment(
                        segment_id=f"{Path(pdf_path).stem}_p{page_num+1}_text",
                        page_num=page_num + 1,
                        content_type="TEXT",
                        content=text,
                        position={"page": page_num + 1},
                        metadata=segment_metadata
                    )
                    segments.append(text_segment)
                
                # Extract images
                image_list = page.get_images(full=True)
                
                # Prepare tasks for concurrent image processing
                image_tasks = []
                image_info = []
                
                for img_index, img_info in enumerate(image_list):
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Save the image using context manager
                    image_filename = f"p{page_num+1}_img{img_index+1}.png"
                    image_path = os.path.join(images_dir, image_filename)
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    # Skip processing the first image on each page (likely logos/headers)
                    # This matches patterns like p1_img1.png, p2_img1.png, p3_img1.png, etc.
                    if img_index == 0:  # This is the first image on the page
                        self.logger.info(f"Skipping analysis of likely logo/header image: {image_filename}")
                        continue
                    
                    # Convert to base64 for Gemini
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Create task for image analysis with retry using the selected model
                    task = self._analyze_image_with_ai(base64_image)
                    image_tasks.append(task)
                    image_info.append({
                        "page_num": page_num,
                        "img_index": img_index,
                        "image_path": image_path
                    })
                
                # Process all images concurrently if there are any
                if image_tasks:
                    analysis_results = await asyncio.gather(*image_tasks, return_exceptions=True)
                    
                    # Process the results
                    for i, (result, info) in enumerate(zip(analysis_results, image_info)):
                        page_num = info["page_num"]
                        img_index = info["img_index"]
                        image_path = info["image_path"]
                        
                        # Handle exceptions
                        if isinstance(result, Exception):
                            self.logger.warning(f"Error analyzing image: {str(result)}. Using fallback.")
                            analysis_result = ImageAnalysisResult(
                                description=f"Image from page {page_num+1}",
                                detected_elements=[],
                                confidence_score=0.0,
                                tags=["analysis_failed", "fallback_used"]
                            )
                        else:
                            analysis_result = result
                    
                        # Create image segment
                        image_segment = DocumentSegment(
                            segment_id=f"{Path(pdf_path).stem}_p{page_num+1}_img{img_index+1}",
                            page_num=page_num + 1,
                            content_type="IMAGE",
                            content=f"[Image: {analysis_result.description}]",
                            position={"page": page_num + 1},
                            metadata={
                                "source": "image_extraction",
                                "image_path": image_path,
                                "tags": analysis_result.tags,
                                "detected_elements": analysis_result.detected_elements
                            },
                            image_analysis=analysis_result
                        )
                        segments.append(image_segment)
                        metadata["total_images"] += 1
        
        return segments, metadata
    
    async def _analyze_image_with_ai(self, base64_image: str) -> ImageAnalysisResult:
        """Analyze an image using the selected AI model.
        
        Args:
            base64_image: Base64-encoded image
            
        Returns:
            Image analysis result
        """
        # Log the start of image analysis with model information
        image_size_kb = len(base64_image) / 1024
        self.logger.info(f"Starting image analysis with {self.image_reader_model} model (image size: {image_size_kb:.2f} KB)")
        
        start_time = time.time()
        
        # Dispatch to the appropriate model-specific analysis method
        try:
            if self.image_reader_model == "gemini":
                result = await self._analyze_image_with_gemini(base64_image)
            elif self.image_reader_model == "openai":
                result = await self._analyze_image_with_openai(base64_image)
            elif self.image_reader_model == "claude":
                result = await self._analyze_image_with_claude(base64_image)
            else:
                # Default to Gemini if the model is not recognized
                self.logger.warning(f"Unrecognized image reader model: {self.image_reader_model}. Falling back to Gemini.")
                result = await self._analyze_image_with_gemini(base64_image)
            
            # Log the completion of image analysis with timing information
            elapsed_time = time.time() - start_time
            self.logger.info(f"Completed image analysis with {self.image_reader_model} model in {elapsed_time:.2f} seconds")
            
            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Image analysis with {self.image_reader_model} failed after {elapsed_time:.2f} seconds: {str(e)}")
            raise
            
    async def _analyze_image_with_gemini(self, base64_image: str) -> ImageAnalysisResult:
        """Analyze an image using Gemini 2 Flash with retry mechanism.
        
        Args:
            base64_image: Base64-encoded image
            
        Returns:
            Image analysis result
        """
        # Import retry utilities
        from src.chattingwithcontext7.retry_utils import retry_async
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_image)
        
        # Select the appropriate prompt based on image handling mode
        if self.image_handling == "semi-structured":
            # Enhanced prompt for extracting detailed UI information in structured format
            prompt = """
            You are a specialized analyzer for Enterprise Resource Planning (ERP) software screenshots. Analyze this image thoroughly, focusing on extracting detailed information about the UI elements and their relationships.
            
            Provide the following information:
            1. A comprehensive description of what the image shows, including its context within an ERP system
            2. Detailed identification of all UI elements present, with specific attention to:
               - Buttons (standard, action, toggle)
               - Tables (data grids, with columns and content)
               - Form fields (input boxes, dropdowns, checkboxes)
               - Information panels or sections
               - Navigation elements (menus, breadcrumbs)
               - Arrows or connectors showing relationships
               - Icons and their apparent functions
               - Modal windows or popups
            
            3. For each UI element identified, determine:
               - Its type and visual appearance
               - Its approximate location in the image
               - Any text or data it contains
               - Its relationship to other elements (what it connects to, is contained within, etc.)
            
            4. Analyze the overall UI layout and how elements are arranged spatially
            5. Determine if this represents a specific workflow or process in the ERP system
            6. Extract all visible text in the image, especially labels, headers, and data values
            
            Format your response as structured JSON with the following schema:
            
            {
                "description": "Comprehensive description of what the ERP screenshot shows",
                "detected_elements": ["button", "table", "form", "panel", "arrow", "icon", "etc"],
                "ui_elements": [
                    {
                        "type": "button/table/panel/arrow/etc",
                        "description": "Detailed description of this element",
                        "location": "top-left/center/bottom-right/etc",
                        "content": "Text or data contained in this element",
                        "relationship": "points to/contains/is part of/etc",
                        "related_elements": ["names or brief descriptions of related elements"]
                    }
                ],
                "ui_layout": "Description of overall UI structure and organization",
                "workflow_context": "Description of the business process or workflow shown",
                "image_type": "ERP screenshot/diagram/chart/etc",
                "extracted_text": "All text visible in the image organized by context",
                "tags": ["relevant", "tags", "for", "this", "image"],
                "confidence_score": 0.95
            }
            
            Be comprehensive in your analysis, capture as much detail as possible about complex UI elements like tables, and clearly explain the relationships between elements (especially what arrows are pointing to or connecting). Format the output in clean, structured markdown within each field.
            
            Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
            """
        else:  # plain-text mode
            # Simpler prompt for extracting all information in plain text format
            prompt = """
            You are a specialized analyzer for images in technical documentation. Analyze this image thoroughly and extract all information you can see in it.
            
            Provide a comprehensive description of the image that includes:
            1. What the image shows (screenshot, diagram, chart, photo, etc.)
            2. All visible text in the image
            3. Any important visual elements and their significance
            4. The overall context and purpose of the image
            5. Any technical information that would be relevant to a user of this documentation
            
            Format your response as structured JSON with the following schema:
            
            {
                "description": "Comprehensive plain-text description of everything visible in the image",
                "extracted_text": "All text visible in the image",
                "image_type": "Screenshot/diagram/chart/photo/etc",
                "tags": ["relevant", "tags", "for", "this", "image"],
                "confidence_score": 0.95
            }
            
            Be thorough and detailed in your description. Include all information that would be useful to someone who cannot see the image.
            Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
            """
        
        # Create binary content for the image
        binary_content = BinaryContent(image_bytes, media_type="image/png")
        
        # Define the API call function to retry
        async def call_gemini_api():
            self.logger.info(f"Calling Gemini API with model: {self.config.get('gemini_model', 'gemini-2.0-flash')}")
            api_start_time = time.time()
            
            try:
                response = await self.agent.run(
                    [prompt, binary_content],
                    model_settings={"temperature": 0.2, "max_tokens": 10000},
                    usage_limits=UsageLimits(response_tokens_limit=11000)
                )
                
                api_elapsed_time = time.time() - api_start_time
                self.logger.info(f"Gemini API call completed in {api_elapsed_time:.2f} seconds")
                
                return response
            except Exception as e:
                api_elapsed_time = time.time() - api_start_time
                self.logger.error(f"Gemini API call failed after {api_elapsed_time:.2f} seconds: {str(e)}")
                raise
        
        # Generate content with Gemini using pydantic_ai agent with retry
        try:
            response = await retry_async(
                call_gemini_api,
                retry_config=self.retry_config,
                logger=self.logger
            )
            
            # Extract the response text
            if hasattr(response, 'data'):
                response_text = response.data
            elif hasattr(response, 'output'):
                response_text = response.output
            elif hasattr(response, 'response'):
                response_text = response.response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception as e:
            self.logger.error(f"Error calling Gemini API after retries: {str(e)}", exc_info=True)
            return ImageAnalysisResult(
                description="Failed to analyze image due to API error",
                detected_elements=[],
                confidence_score=0.0,
                tags=["analysis_failed", "api_error"]
            )
        
        # Parse the response
        try:
            # Extract JSON from response
            
            # Handle potential formatting issues in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
            
            result_dict = json.loads(json_str)
            
            # Process UI elements if available
            ui_elements = []
            if "ui_elements" in result_dict and isinstance(result_dict["ui_elements"], list):
                for element_data in result_dict["ui_elements"]:
                    if isinstance(element_data, dict):
                        try:
                            # Create UIElement from the data
                            from src.chattingwithcontext7.models import UIElement
                            ui_element = UIElement(
                                type=element_data.get("type", "unknown"),
                                description=element_data.get("description", "No description"),
                                location=element_data.get("location", ""),
                                content=element_data.get("content", ""),
                                relationship=element_data.get("relationship"),
                                related_elements=element_data.get("related_elements", [])
                            )
                            ui_elements.append(ui_element)
                        except Exception as e:
                            self.logger.warning(f"Error creating UIElement: {str(e)}")
            
            # Create enhanced ImageAnalysisResult
            analysis_result = ImageAnalysisResult(
                description=result_dict.get("description", "No description available"),
                detected_elements=result_dict.get("detected_elements", []),
                ui_elements=ui_elements,
                ui_layout=result_dict.get("ui_layout", ""),
                workflow_context=result_dict.get("workflow_context", ""),
                confidence_score=result_dict.get("confidence_score", 0.5),
                tags=result_dict.get("tags", []),
                extracted_text=result_dict.get("extracted_text", "")
            )
            
            # Add additional information to tags if available
            if "image_type" in result_dict:
                analysis_result.tags.append(result_dict["image_type"])
            
            # Add workflow context as a tag if available
            if "workflow_context" in result_dict and result_dict["workflow_context"]:
                context_tag = f"workflow:{result_dict['workflow_context'].split()[0]}"  # First word as tag
                if context_tag not in analysis_result.tags:
                    analysis_result.tags.append(context_tag)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}", exc_info=True)
            
            # Return a default result
            return ImageAnalysisResult(
                description="Failed to analyze image",
                detected_elements=[],
                confidence_score=0.0,
                tags=["analysis_failed"]
            )
    
    async def _analyze_image_with_openai(self, base64_image: str) -> ImageAnalysisResult:
        """Analyze an image using OpenAI with retry mechanism.
        
        Args:
            base64_image: Base64-encoded image
            
        Returns:
            Image analysis result
        """
        # Import retry utilities
        from src.chattingwithcontext7.retry_utils import retry_async
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_image)
        
        # Select the appropriate prompt based on image handling mode
        if self.image_handling == "semi-structured":
            # Enhanced prompt for extracting detailed UI information in structured format
            prompt = """
            You are a specialized analyzer for Enterprise Resource Planning (ERP) software screenshots. Analyze this image thoroughly, focusing on extracting detailed information about the UI elements and their relationships.
            
            Provide the following information:
            1. A comprehensive description of what the image shows, including its context within an ERP system
            2. Detailed identification of all UI elements present, with specific attention to:
               - Buttons (standard, action, toggle)
               - Tables (data grids, with columns and content)
               - Form fields (input boxes, dropdowns, checkboxes)
               - Information panels or sections
               - Navigation elements (menus, breadcrumbs)
               - Arrows or connectors showing relationships
               - Icons and their apparent functions
               - Modal windows or popups
            
            3. For each UI element identified, determine:
               - Its type and visual appearance
               - Its approximate location in the image
               - Any text or data it contains
               - Its relationship to other elements (what it connects to, is contained within, etc.)
            
            4. Analyze the overall UI layout and how elements are arranged spatially
            5. Determine if this represents a specific workflow or process in the ERP system
            6. Extract all visible text in the image, especially labels, headers, and data values
            
            Format your response as structured JSON with the following schema:
            
            {
                "description": "Comprehensive description of what the ERP screenshot shows",
                "detected_elements": ["button", "table", "form", "panel", "arrow", "icon", "etc"],
                "ui_elements": [
                    {
                        "type": "button/table/panel/arrow/etc",
                        "description": "Detailed description of this element",
                        "location": "top-left/center/bottom-right/etc",
                        "content": "Text or data contained in this element",
                        "relationship": "points to/contains/is part of/etc",
                        "related_elements": ["names or brief descriptions of related elements"]
                    }
                ],
                "ui_layout": "Description of overall UI structure and organization",
                "workflow_context": "Description of the business process or workflow shown",
                "image_type": "ERP screenshot/diagram/chart/etc",
                "extracted_text": "All text visible in the image organized by context",
                "tags": ["relevant", "tags", "for", "this", "image"],
                "confidence_score": 0.95
            }
            
            Be comprehensive in your analysis, capture as much detail as possible about complex UI elements like tables, and clearly explain the relationships between elements (especially what arrows are pointing to or connecting). Format the output in clean, structured markdown within each field.
            
            Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
            """
        else:  # plain-text mode
            # Simpler prompt for extracting all information in plain text format
            prompt = """
            You are a specialized analyzer for images in technical documentation. Analyze this image thoroughly and extract all information you can see in it.
            
            Provide a comprehensive description of the image that includes:
            1. What the image shows (screenshot, diagram, chart, photo, etc.)
            2. All visible text in the image
            3. Any important visual elements and their significance
            4. The overall context and purpose of the image
            5. Any technical information that would be relevant to a user of this documentation
            
            Format your response as structured JSON with the following schema:
            
            {
                "description": "Comprehensive plain-text description of everything visible in the image",
                "extracted_text": "All text visible in the image",
                "image_type": "Screenshot/diagram/chart/photo/etc",
                "tags": ["relevant", "tags", "for", "this", "image"],
                "confidence_score": 0.95
            }
            
            Be thorough and detailed in your description. Include all information that would be useful to someone who cannot see the image.
            Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
            """
        
        # Create binary content for the image
        binary_content = BinaryContent(image_bytes, media_type="image/png")
        
        # Define the API call function to retry
        async def call_openai_api():
            self.logger.info(f"Calling OpenAI API with model: gpt-4.1-nano")
            api_start_time = time.time()
            
            try:
                response = await self.agent.run(
                    [prompt, binary_content],
                    model_settings={"temperature": 0.2, "max_tokens": 10000},
                    usage_limits=UsageLimits(response_tokens_limit=11000)
                )
                
                api_elapsed_time = time.time() - api_start_time
                self.logger.info(f"OpenAI API call completed in {api_elapsed_time:.2f} seconds")
                
                return response
            except Exception as e:
                api_elapsed_time = time.time() - api_start_time
                self.logger.error(f"OpenAI API call failed after {api_elapsed_time:.2f} seconds: {str(e)}")
                raise
        
        # Generate content with OpenAI using pydantic_ai agent with retry
        try:
            response = await retry_async(
                call_openai_api,
                retry_config=self.retry_config,
                logger=self.logger
            )
            
            # Extract the response text
            if hasattr(response, 'data'):
                response_text = response.data
            elif hasattr(response, 'output'):
                response_text = response.output
            elif hasattr(response, 'response'):
                response_text = response.response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception as e:
            self.logger.error(f"Error calling OpenAI API after retries: {str(e)}", exc_info=True)
            return ImageAnalysisResult(
                description="Failed to analyze image due to API error",
                detected_elements=[],
                confidence_score=0.0,
                tags=["analysis_failed", "api_error"]
            )
        
        # Parse the response
        try:
            # Extract JSON from response
            
            # Handle potential formatting issues in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
            
            # Try to fix common JSON syntax errors before parsing
            try:
                result_dict = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                self.logger.warning(f"Initial JSON parsing failed: {str(json_err)}. Attempting repair...")
                
                # 1. Try to fix missing commas in the most common locations
                if "Expecting ',' delimiter" in str(json_err):
                    # Get approximate error location
                    err_parts = str(json_err).split("char ")
                    if len(err_parts) > 1:
                        try:
                            err_pos = int(err_parts[1])
                            # Add comma at the error position and try again
                            fixed_json = json_str[:err_pos] + "," + json_str[err_pos:]
                            try:
                                result_dict = json.loads(fixed_json)
                                self.logger.info("Successfully fixed JSON by adding missing comma")
                            except json.JSONDecodeError:
                                # If that doesn't work, try a more general approach
                                raise
                        except (ValueError, IndexError):
                            # Fall back to general repair if we can't parse the position
                            raise
                
                # 2. If specific fixes fail, try a more general clean-up approach
                if 'result_dict' not in locals():
                    self.logger.warning("Specific JSON repair failed, trying general cleanup...")
                    # Replace common syntax errors
                    fixed_json = json_str
                    # Fix trailing commas in arrays and objects
                    fixed_json = fixed_json.replace(",\n}", "\n}")
                    fixed_json = fixed_json.replace(",\n]", "\n]")
                    # Fix missing commas after closing quotes in key-value pairs
                    fixed_json = fixed_json.replace('"\n', '",\n')
                    
                    try:
                        result_dict = json.loads(fixed_json)
                        self.logger.info("Successfully fixed JSON with general cleanup")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON repair failed: {str(e)}")
                        # Create a minimal valid result
                        result_dict = {
                            "description": "Image analysis result unavailable due to JSON parsing error",
                            "detected_elements": [],
                            "tags": ["analysis_failed", "json_error"],
                            "confidence_score": 0.0
                        }
            
            # Process UI elements if available
            ui_elements = []
            if "ui_elements" in result_dict and isinstance(result_dict["ui_elements"], list):
                for element_data in result_dict["ui_elements"]:
                    if isinstance(element_data, dict):
                        try:
                            # Create UIElement from the data
                            from src.chattingwithcontext7.models import UIElement
                            ui_element = UIElement(
                                type=element_data.get("type", "unknown"),
                                description=element_data.get("description", "No description"),
                                location=element_data.get("location", ""),
                                content=element_data.get("content", ""),
                                relationship=element_data.get("relationship"),
                                related_elements=element_data.get("related_elements", [])
                            )
                            ui_elements.append(ui_element)
                        except Exception as e:
                            self.logger.warning(f"Error creating UIElement: {str(e)}")
            
            # Create enhanced ImageAnalysisResult
            analysis_result = ImageAnalysisResult(
                description=result_dict.get("description", "No description available"),
                detected_elements=result_dict.get("detected_elements", []),
                ui_elements=ui_elements,
                ui_layout=result_dict.get("ui_layout", ""),
                workflow_context=result_dict.get("workflow_context", ""),
                confidence_score=result_dict.get("confidence_score", 0.5),
                tags=result_dict.get("tags", []),
                extracted_text=result_dict.get("extracted_text", "")
            )
            
            # Add additional information to tags if available
            if "image_type" in result_dict:
                analysis_result.tags.append(result_dict["image_type"])
            
            # Add workflow context as a tag if available
            if "workflow_context" in result_dict and result_dict["workflow_context"]:
                context_tag = f"workflow:{result_dict['workflow_context'].split()[0]}"  # First word as tag
                if context_tag not in analysis_result.tags:
                    analysis_result.tags.append(context_tag)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error parsing OpenAI response: {str(e)}", exc_info=True)
            
            # Log the problematic JSON for debugging
            truncated_response = response_text[:500] + "..." if len(response_text) > 500 else response_text
            self.logger.error(f"Problematic JSON content (truncated): {truncated_response}")
            
            # Return a more informative default result
            return ImageAnalysisResult(
                description="Failed to analyze image due to JSON parsing error",
                detected_elements=[],
                confidence_score=0.0,
                tags=["analysis_failed", "json_error", "openai_response_error"]
            )
    
    async def _analyze_image_with_claude(self, base64_image: str) -> ImageAnalysisResult:
        """Analyze an image using Claude with retry mechanism.
        
        Args:
            base64_image: Base64-encoded image
            
        Returns:
            Image analysis result
        """
        # Import retry utilities
        from src.chattingwithcontext7.retry_utils import retry_async
        
        # Decode base64 image
        image_bytes = base64.b64decode(base64_image)
        
        # Select the appropriate prompt based on image handling mode
        if self.image_handling == "semi-structured":
            # Enhanced prompt for extracting detailed UI information in structured format
            prompt = """
            You are a specialized analyzer for Enterprise Resource Planning (ERP) software screenshots. Analyze this image thoroughly, focusing on extracting detailed information about the UI elements and their relationships.
            
            Provide the following information:
            1. A comprehensive description of what the image shows, including its context within an ERP system
            2. Detailed identification of all UI elements present, with specific attention to:
               - Buttons (standard, action, toggle)
               - Tables (data grids, with columns and content)
               - Form fields (input boxes, dropdowns, checkboxes)
               - Information panels or sections
               - Navigation elements (menus, breadcrumbs)
               - Arrows or connectors showing relationships
               - Icons and their apparent functions
               - Modal windows or popups
            
            3. For each UI element identified, determine:
               - Its type and visual appearance
               - Its approximate location in the image
               - Any text or data it contains
               - Its relationship to other elements (what it connects to, is contained within, etc.)
            
            4. Analyze the overall UI layout and how elements are arranged spatially
            5. Determine if this represents a specific workflow or process in the ERP system
            6. Extract all visible text in the image, especially labels, headers, and data values
            
            Format your response as structured JSON with the following schema:
            
            {
                "description": "Comprehensive description of what the ERP screenshot shows",
                "detected_elements": ["button", "table", "form", "panel", "arrow", "icon", "etc"],
                "ui_elements": [
                    {
                        "type": "button/table/panel/arrow/etc",
                        "description": "Detailed description of this element",
                        "location": "top-left/center/bottom-right/etc",
                        "content": "Text or data contained in this element",
                        "relationship": "points to/contains/is part of/etc",
                        "related_elements": ["names or brief descriptions of related elements"]
                    }
                ],
                "ui_layout": "Description of overall UI structure and organization",
                "workflow_context": "Description of the business process or workflow shown",
                "image_type": "ERP screenshot/diagram/chart/etc",
                "extracted_text": "All text visible in the image organized by context",
                "tags": ["relevant", "tags", "for", "this", "image"],
                "confidence_score": 0.95
            }
            
            Be comprehensive in your analysis, capture as much detail as possible about complex UI elements like tables, and clearly explain the relationships between elements (especially what arrows are pointing to or connecting). Format the output in clean, structured markdown within each field.
            
            Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
            """
        else:  # plain-text mode
            # Simpler prompt for extracting all information in plain text format
            prompt = """
            You are a specialized analyzer for images in technical documentation. Analyze this image thoroughly and extract all information you can see in it.
            
            Provide a comprehensive description of the image that includes:
            1. What the image shows (screenshot, diagram, chart, photo, etc.)
            2. All visible text in the image
            3. Any important visual elements and their significance
            4. The overall context and purpose of the image
            5. Any technical information that would be relevant to a user of this documentation
            
            Format your response as structured JSON with the following schema:
            
            {
                "description": "Comprehensive plain-text description of everything visible in the image",
                "extracted_text": "All text visible in the image",
                "image_type": "Screenshot/diagram/chart/photo/etc",
                "tags": ["relevant", "tags", "for", "this", "image"],
                "confidence_score": 0.95
            }
            
            Be thorough and detailed in your description. Include all information that would be useful to someone who cannot see the image.
            Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
            """
        
        # Create binary content for the image
        binary_content = BinaryContent(image_bytes, media_type="image/png")
        
        # Define the API call function to retry
        async def call_claude_api():
            self.logger.info(f"Calling Claude API with model: claude-3.5-sonnet")
            api_start_time = time.time()
            
            try:
                response = await self.agent.run(
                    [prompt, binary_content],
                    model_settings={"temperature": 0.2, "max_tokens": 10000},
                    usage_limits=UsageLimits(response_tokens_limit=11000)
                )
                
                api_elapsed_time = time.time() - api_start_time
                self.logger.info(f"Claude API call completed in {api_elapsed_time:.2f} seconds")
                
                return response
            except Exception as e:
                api_elapsed_time = time.time() - api_start_time
                self.logger.error(f"Claude API call failed after {api_elapsed_time:.2f} seconds: {str(e)}")
                raise
        
        # Generate content with Claude using pydantic_ai agent with retry
        try:
            response = await retry_async(
                call_claude_api,
                retry_config=self.retry_config,
                logger=self.logger
            )
            
            # Extract the response text
            if hasattr(response, 'data'):
                response_text = response.data
            elif hasattr(response, 'output'):
                response_text = response.output
            elif hasattr(response, 'response'):
                response_text = response.response
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
        except Exception as e:
            self.logger.error(f"Error calling Claude API after retries: {str(e)}", exc_info=True)
            return ImageAnalysisResult(
                description="Failed to analyze image due to API error",
                detected_elements=[],
                confidence_score=0.0,
                tags=["analysis_failed", "api_error"]
            )
        
        # Parse the response
        try:
            # Extract JSON from response
            
            # Handle potential formatting issues in the response
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].strip()
            else:
                json_str = response_text.strip()
            
            # Try to fix common JSON syntax errors before parsing
            try:
                result_dict = json.loads(json_str)
            except json.JSONDecodeError as json_err:
                self.logger.warning(f"Initial JSON parsing failed: {str(json_err)}. Attempting repair...")
                
                # 1. Try to fix missing commas in the most common locations
                if "Expecting ',' delimiter" in str(json_err):
                    # Get approximate error location
                    err_parts = str(json_err).split("char ")
                    if len(err_parts) > 1:
                        try:
                            err_pos = int(err_parts[1])
                            # Add comma at the error position and try again
                            fixed_json = json_str[:err_pos] + "," + json_str[err_pos:]
                            try:
                                result_dict = json.loads(fixed_json)
                                self.logger.info("Successfully fixed JSON by adding missing comma")
                            except json.JSONDecodeError:
                                # If that doesn't work, try a more general approach
                                raise
                        except (ValueError, IndexError):
                            # Fall back to general repair if we can't parse the position
                            raise
                
                # 2. If specific fixes fail, try a more general clean-up approach
                if 'result_dict' not in locals():
                    self.logger.warning("Specific JSON repair failed, trying general cleanup...")
                    # Replace common syntax errors
                    fixed_json = json_str
                    # Fix trailing commas in arrays and objects
                    fixed_json = fixed_json.replace(",\n}", "\n}")
                    fixed_json = fixed_json.replace(",\n]", "\n]")
                    # Fix missing commas after closing quotes in key-value pairs
                    fixed_json = fixed_json.replace('"\n', '",\n')
                    
                    try:
                        result_dict = json.loads(fixed_json)
                        self.logger.info("Successfully fixed JSON with general cleanup")
                    except json.JSONDecodeError as e:
                        self.logger.error(f"JSON repair failed: {str(e)}")
                        # Create a minimal valid result
                        result_dict = {
                            "description": "Image analysis result unavailable due to JSON parsing error",
                            "detected_elements": [],
                            "tags": ["analysis_failed", "json_error"],
                            "confidence_score": 0.0
                        }
            
            # Process UI elements if available
            ui_elements = []
            if "ui_elements" in result_dict and isinstance(result_dict["ui_elements"], list):
                for element_data in result_dict["ui_elements"]:
                    if isinstance(element_data, dict):
                        try:
                            # Create UIElement from the data
                            from src.chattingwithcontext7.models import UIElement
                            ui_element = UIElement(
                                type=element_data.get("type", "unknown"),
                                description=element_data.get("description", "No description"),
                                location=element_data.get("location", ""),
                                content=element_data.get("content", ""),
                                relationship=element_data.get("relationship"),
                                related_elements=element_data.get("related_elements", [])
                            )
                            ui_elements.append(ui_element)
                        except Exception as e:
                            self.logger.warning(f"Error creating UIElement: {str(e)}")
            
            # Create enhanced ImageAnalysisResult
            analysis_result = ImageAnalysisResult(
                description=result_dict.get("description", "No description available"),
                detected_elements=result_dict.get("detected_elements", []),
                ui_elements=ui_elements,
                ui_layout=result_dict.get("ui_layout", ""),
                workflow_context=result_dict.get("workflow_context", ""),
                confidence_score=result_dict.get("confidence_score", 0.5),
                tags=result_dict.get("tags", []),
                extracted_text=result_dict.get("extracted_text", "")
            )
            
            # Add additional information to tags if available
            if "image_type" in result_dict:
                analysis_result.tags.append(result_dict["image_type"])
            
            # Add workflow context as a tag if available
            if "workflow_context" in result_dict and result_dict["workflow_context"]:
                context_tag = f"workflow:{result_dict['workflow_context'].split()[0]}"  # First word as tag
                if context_tag not in analysis_result.tags:
                    analysis_result.tags.append(context_tag)
            
            return analysis_result
            
        except Exception as e:
            self.logger.error(f"Error parsing Claude response: {str(e)}", exc_info=True)
            
            # Log the problematic JSON for debugging
            truncated_response = response_text[:500] + "..." if len(response_text) > 500 else response_text
            self.logger.error(f"Problematic JSON content (truncated): {truncated_response}")
            
            # Return a more informative default result
            return ImageAnalysisResult(
                description="Failed to analyze image due to JSON parsing error",
                detected_elements=[],
                confidence_score=0.0,
                tags=["analysis_failed", "json_error", "claude_response_error"]
            )
    
    def transform_for_lightrag(self, document: ProcessedDocument, chunk_size: int = None, overlap: int = None) -> List[Dict[str, Any]]:
        """Transform a processed document into LightRAG format.
        
        Args:
            document: Processed document
            chunk_size: Maximum size of text chunks (defaults to self.chunk_size)
            overlap: Number of characters to overlap between chunks (defaults to self.chunk_overlap)
            
        Returns:
            List of LightRAG documents
        """
        # Use provided values or fall back to instance defaults
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        lightrag_documents = []
        
        # Process text segments
        text_segments = [s for s in document.segments if s.content_type == "TEXT"]
        for segment in text_segments:
            text = segment.content
            
            # Split text into chunks
            if len(text) > chunk_size:
                chunks = []
                start = 0
                while start < len(text):
                    end = min(start + chunk_size, len(text))
                    if end < len(text) and end - start < chunk_size:
                        # Find the last space to avoid cutting words
                        last_space = text.rfind(' ', start, end)
                        if last_space > start:
                            end = last_space
                    chunks.append(text[start:end])
                    start = end - overlap if end - overlap > start else end
            else:
                chunks = [text]
            
            # Create LightRAG documents for each chunk
            for i, chunk in enumerate(chunks):
                lightrag_doc = {
                    "id": f"{document.document_id}_p{segment.page_num}_c{i}",
                    "text": chunk,
                    "metadata": {
                        "document_id": document.document_id,
                        "filename": document.filename,
                        "title": document.title,
                        "page": segment.page_num,
                        "chunk": i,
                        "content_type": "TEXT",
                        "segment_id": segment.segment_id
                    }
                }
                
                # Add version information to metadata if available
                if "version" in segment.metadata:
                    lightrag_doc["metadata"]["version"] = segment.metadata["version"]
                
                lightrag_documents.append(lightrag_doc)
        
        # Process image segments with enhanced UI detail extraction
        image_segments = [s for s in document.segments if s.content_type == "IMAGE"]
        for segment in image_segments:
            if segment.image_analysis:
                # Create structured markdown representation of the image
                image_text = f"# ERP Screenshot Analysis\n\n## Description\n{segment.image_analysis.description}\n\n"
                
                # Add UI layout if available
                if hasattr(segment.image_analysis, 'ui_layout') and segment.image_analysis.ui_layout:
                    image_text += f"## UI Layout\n{segment.image_analysis.ui_layout}\n\n"
                
                # Add workflow context if available
                if hasattr(segment.image_analysis, 'workflow_context') and segment.image_analysis.workflow_context:
                    image_text += f"## Workflow Context\n{segment.image_analysis.workflow_context}\n\n"
                
                # Add detected elements list
                if segment.image_analysis.detected_elements:
                    image_text += "## Detected UI Elements\n"
                    elements_str = ", ".join(segment.image_analysis.detected_elements)
                    image_text += f"{elements_str}\n\n"
                
                # Add detailed UI elements if available
                if hasattr(segment.image_analysis, 'ui_elements') and segment.image_analysis.ui_elements:
                    image_text += "## Detailed UI Element Analysis\n\n"
                    for i, element in enumerate(segment.image_analysis.ui_elements):
                        image_text += f"### {i+1}. {element.type.upper()}\n"
                        image_text += f"- **Description**: {element.description}\n"
                        if element.location:
                            image_text += f"- **Location**: {element.location}\n"
                        if element.content:
                            image_text += f"- **Content**: {element.content}\n"
                        if element.relationship:
                            image_text += f"- **Relationship**: {element.relationship}\n"
                        if element.related_elements:
                            related = ", ".join(element.related_elements)
                            image_text += f"- **Related Elements**: {related}\n"
                        image_text += "\n"
                
                # Build enhanced metadata dictionary with UI details
                enhanced_metadata = {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "title": document.title,
                    "page": segment.page_num,
                    "content_type": "IMAGE",
                    "segment_id": segment.segment_id,
                    "tags": segment.image_analysis.tags,
                    "confidence_score": segment.image_analysis.confidence_score,
                    "ui_elements_count": len(segment.image_analysis.ui_elements) if hasattr(segment.image_analysis, 'ui_elements') else 0
                }
                
                # Add version information to metadata if available
                if "version" in segment.metadata:
                    enhanced_metadata["version"] = segment.metadata["version"]
                
                # Add UI layout and workflow context to metadata if available
                if hasattr(segment.image_analysis, 'ui_layout') and segment.image_analysis.ui_layout:
                    enhanced_metadata["ui_layout_summary"] = segment.image_analysis.ui_layout[:100] + "..." if len(segment.image_analysis.ui_layout) > 100 else segment.image_analysis.ui_layout
                
                if hasattr(segment.image_analysis, 'workflow_context') and segment.image_analysis.workflow_context:
                    enhanced_metadata["workflow_context"] = segment.image_analysis.workflow_context
                
                # Add element types for improved searchability
                if hasattr(segment.image_analysis, 'ui_elements') and segment.image_analysis.ui_elements:
                    element_types = list(set(element.type for element in segment.image_analysis.ui_elements))
                    enhanced_metadata["ui_element_types"] = element_types
                
                # Create LightRAG document with enhanced content and metadata
                lightrag_doc = {
                    "id": f"{document.document_id}_img{segment.segment_id}",
                    "text": image_text,
                    "metadata": enhanced_metadata
                }
                lightrag_documents.append(lightrag_doc)
        
        return lightrag_documents
    
    def save_lightrag_documents(self, lightrag_documents: List[Dict[str, Any]], output_dir: str, filename: str) -> str:
        """Save LightRAG documents to a file.
        
        Args:
            lightrag_documents: List of LightRAG documents
            output_dir: Directory to save the file
            filename: Base filename (without extension)
            
        Returns:
            Path to the saved file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output path
        output_path = os.path.join(output_dir, f"{filename}.json")
        
        # Save to file using context manager
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lightrag_documents, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved {len(lightrag_documents)} LightRAG documents to {output_path}")
        return output_path
    
    def create_lightrag_batch(self, documents: List[ProcessedDocument], chunk_size: int = 1000, overlap: int = 100) -> List[Dict[str, Any]]:
        """Create a batch of LightRAG documents from multiple processed documents.
        
        Args:
            documents: List of processed documents
            chunk_size: Maximum size of text chunks
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of LightRAG documents
        """
        all_lightrag_documents = []
        
        for document in documents:
            lightrag_docs = self.transform_for_lightrag(document, chunk_size, overlap)
            all_lightrag_documents.extend(lightrag_docs)
        
        return all_lightrag_documents
    
    def _update_stats_for_document(self, document: ProcessedDocument) -> None:
        """Update processing statistics for a document.
        
        Args:
            document: Processed document
        """
        self.stats.total_documents += 1
        self.stats.successful_documents += 1
        self.stats.total_pages += document.total_pages
        self.stats.total_segments += len(document.segments)
        
        # Update segment counts by type
        for segment in document.segments:
            content_type = segment.content_type
            if content_type not in self.stats.segments_by_type:
                self.stats.segments_by_type[content_type] = 0
            self.stats.segments_by_type[content_type] += 1
    
    def _save_processing_stats(self) -> None:
        """Save processing statistics to a file."""
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        stats_path = os.path.join(self.output_dir, "processing_stats.json")
        
        # Save to file using context manager
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats.dict(), f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Saved processing statistics to {stats_path}")