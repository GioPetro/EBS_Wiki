"""Pydantic models for PDF processing with images."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class UIElement(BaseModel):
    """Detailed information about a UI element detected in an image."""
    
    type: str = Field(..., description="Type of UI element (button, table, panel, arrow, etc.)")
    description: str = Field(..., description="Description of the UI element")
    location: str = Field(default="", description="Relative location in the image (top-left, center, etc.)")
    content: Any = Field(default="", description="Text or data content of the element (can be string or list)")
    relationship: Optional[str] = Field(None, description="Relationship to other elements (points to, contains, etc.)")
    related_elements: List[Any] = Field(default_factory=list, description="Names/IDs of related elements")
    
    def model_post_init(self, __context):
        """Process fields after initialization to handle flexible inputs."""
        # Convert list content to string if needed
        if isinstance(self.content, list):
            self.content = ", ".join(str(item) for item in self.content if item)
        elif self.content is None:
            self.content = ""
        
        # Convert related_elements items to strings if needed
        if self.related_elements:
            self.related_elements = [str(item) if item is not None else "" for item in self.related_elements]
        
class ImageAnalysisResult(BaseModel):
    """Result of image analysis from various AI models (Gemini, OpenAI, Claude)."""
    
    description: str = Field(..., description="Detailed description of the image")
    detected_elements: List[str] = Field(default_factory=list, description="Simple list of UI elements detected in the image")
    ui_elements: List[UIElement] = Field(default_factory=list, description="Detailed information about UI elements")
    ui_layout: str = Field(default="", description="Description of overall UI layout and structure")
    workflow_context: str = Field(default="", description="Context about the workflow or process shown")
    confidence_score: float = Field(default=0.5, description="Confidence score of the analysis (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the image")
    extracted_text: str = Field(default="", description="All text extracted from the image")
    
    def model_post_init(self, __context):
        """Process fields after initialization to handle flexible inputs."""
        # Convert detected_elements to list of strings if needed
        if self.detected_elements and not all(isinstance(e, str) for e in self.detected_elements):
            self.detected_elements = [str(e) for e in self.detected_elements if e is not None]
        
        # Convert tags to list of strings if needed
        if self.tags and not all(isinstance(t, str) for t in self.tags):
            self.tags = [str(t) for t in self.tags if t is not None]


class DocumentSegment(BaseModel):
    """
    Segment of a document (text or image).

    For image segments, 'image_path' stores the file path to the extracted image (if applicable).
    'content' contains the text content or image description (e.g., from OCR or captioning).
    'metadata' should include any additional context needed to trace the answer back to its origin,
    such as the source document name, extraction method, or other relevant details.
    """

    segment_id: str = Field(..., description="Unique identifier for the segment")
    page_num: int = Field(..., description="Page number where the segment appears")
    content_type: str = Field(..., description="Type of content (TEXT, IMAGE, TABLE, etc.)")
    content: str = Field(..., description="Text content or image description")
    image_path: Optional[str] = Field(None, description="File path to the extracted image, if applicable")
    position: Dict[str, Any] = Field(default_factory=dict, description="Position information on the page")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (e.g., source document name, extraction context)")
    image_analysis: Optional[ImageAnalysisResult] = Field(None, description="Image analysis result if applicable")


class ProcessedDocument(BaseModel):
    """Processed document with text and image segments."""
    
    document_id: str = Field(..., description="Unique identifier for the document")
    filename: str = Field(..., description="Original filename")
    title: str = Field(..., description="Document title")
    total_pages: int = Field(..., description="Total number of pages")
    segments: List[DocumentSegment] = Field(default_factory=list, description="Document segments")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    processed_at: datetime = Field(default_factory=datetime.now, description="Processing timestamp")


class ProcessingStats(BaseModel):
    """Statistics for PDF processing."""
    
    total_documents: int = Field(0, description="Total number of documents processed")
    successful_documents: int = Field(0, description="Number of successfully processed documents")
    failed_documents: int = Field(0, description="Number of documents that failed processing")
    total_pages: int = Field(0, description="Total number of pages processed")
    total_segments: int = Field(0, description="Total number of segments extracted")
    segments_by_type: Dict[str, int] = Field(default_factory=dict, description="Segments count by type")
    processing_time: float = Field(0.0, description="Total processing time in seconds")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="List of processing errors")