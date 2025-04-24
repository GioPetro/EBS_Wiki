"""Pydantic models for PDF processing with images."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class ImageAnalysisResult(BaseModel):
    """Result of image analysis using Gemini 2.0 Flash."""
    
    description: str = Field(..., description="Detailed description of the image")
    detected_elements: List[str] = Field(default_factory=list, description="UI elements detected in the image")
    confidence_score: float = Field(..., description="Confidence score of the analysis (0-1)")
    tags: List[str] = Field(default_factory=list, description="Tags associated with the image")


class DocumentSegment(BaseModel):
    """Segment of a document (text or image)."""
    
    segment_id: str = Field(..., description="Unique identifier for the segment")
    page_num: int = Field(..., description="Page number where the segment appears")
    content_type: str = Field(..., description="Type of content (TEXT, IMAGE, TABLE, etc.)")
    content: str = Field(..., description="Text content or image description")
    position: Dict[str, Any] = Field(default_factory=dict, description="Position information on the page")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
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