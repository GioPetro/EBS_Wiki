"""PDF processor for extracting text and images from PDFs using Gemini 2.0 Flash."""

import os
import logging
import base64
import json
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
    """Process PDFs with images using Gemini 2.0 Flash."""
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        gemini_api_key: str = None,
        log_level: int = logging.INFO,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize PDF processor.
        
        Args:
            input_dir: Directory containing PDF files to process
            output_dir: Directory to save processed output
            gemini_api_key: Google API key for Gemini
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
        
        # Set up the Gemini model and agent using pydantic_ai
        # Use the provided API key or fall back to the one from .env
        api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key is required. Provide it as an argument or set GEMINI_API_KEY in .env")
            
        # Set the API key in the environment
        os.environ["GEMINI_API_KEY"] = api_key
        
        # Get Gemini model name from config or use default
        gemini_model = self.config.get("gemini_model", "gemini-2.0-flash")
            
        # Create the model with the correct provider initialization
        from pydantic_ai.providers.google_gla import GoogleGLAProvider
        model = GeminiModel(gemini_model, provider=GoogleGLAProvider(api_key=api_key))
        
        # Create the agent
        self.agent = Agent(model)
        
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
        
        # Open the PDF using context manager for proper resource management
        with fitz.open(pdf_path) as doc:
            metadata["total_pages"] = len(doc)
            
            # Process each page
            for page_num, page in enumerate(doc):
                # Extract text
                text = page.get_text()
                if text.strip():
                    text_segment = DocumentSegment(
                        segment_id=f"{Path(pdf_path).stem}_p{page_num+1}_text",
                        page_num=page_num + 1,
                        content_type="TEXT",
                        content=text,
                        position={"page": page_num + 1},
                        metadata={"source": "text_extraction"}
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
                    
                    # Convert to base64 for Gemini
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    # Create task for image analysis with retry
                    task = self._analyze_image_with_gemini(base64_image)
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
        
        # Prompt for Gemini
        prompt = """
        Analyze this image from a PDF document.
        
        Provide the following information:
        1. A detailed description of what the image shows
        2. Identify any UI elements present (buttons, forms, tables, charts, etc.)
        3. Determine if this is a screenshot, diagram, chart, or other type of image
        4. Extract any visible text in the image
        5. Identify the purpose or context of this image
        
        Format your response as JSON with the following structure:
        {
            "description": "Detailed description of the image",
            "detected_elements": ["list", "of", "UI elements", "detected"],
            "image_type": "screenshot/diagram/chart/etc",
            "extracted_text": "Any text visible in the image",
            "purpose": "The likely purpose of this image",
            "tags": ["relevant", "tags", "for", "this", "image"],
            "confidence_score": 0.95
        }
        
        Ensure the confidence_score is between 0 and 1, reflecting your confidence in this analysis.
        """
        
        # Create binary content for the image
        binary_content = BinaryContent(image_bytes, media_type="image/png")
        
        # Define the API call function to retry
        async def call_gemini_api():
            return await self.agent.run(
                [prompt, binary_content],
                model_settings={"temperature": 0.0, "max_tokens": 4000},
                usage_limits=UsageLimits(response_tokens_limit=4000)
            )
        
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
            
            # Create ImageAnalysisResult
            analysis_result = ImageAnalysisResult(
                description=result_dict.get("description", "No description available"),
                detected_elements=result_dict.get("detected_elements", []),
                confidence_score=result_dict.get("confidence_score", 0.5),
                tags=result_dict.get("tags", [])
            )
            
            # Add additional information to tags if available
            if "image_type" in result_dict:
                analysis_result.tags.append(result_dict["image_type"])
            
            if "purpose" in result_dict:
                analysis_result.tags.append(result_dict["purpose"])
            
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
                lightrag_documents.append(lightrag_doc)
        
        # Process image segments
        image_segments = [s for s in document.segments if s.content_type == "IMAGE"]
        for segment in image_segments:
            if segment.image_analysis:
                # Create text representation of the image
                image_text = f"[Image: {segment.image_analysis.description}]"
                
                # Add detected elements if available
                if segment.image_analysis.detected_elements:
                    elements_str = ", ".join(segment.image_analysis.detected_elements)
                    image_text += f"\nDetected elements: {elements_str}"
                
                # Create LightRAG document
                lightrag_doc = {
                    "id": f"{document.document_id}_img{segment.segment_id}",
                    "text": image_text,
                    "metadata": {
                        "document_id": document.document_id,
                        "filename": document.filename,
                        "title": document.title,
                        "page": segment.page_num,
                        "content_type": "IMAGE",
                        "segment_id": segment.segment_id,
                        "tags": segment.image_analysis.tags,
                        "confidence_score": segment.image_analysis.confidence_score
                    }
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