# Entersoft ERP Chatbot Data Processing Pipeline

A data processing pipeline for the Entersoft ERP chatbot that processes PDF documents, extracts text and images, analyzes images using Gemini 2 Flash Lite, and structures the data for ingestion by LightRAG.

## Features

- PDF document processing with text and image extraction
- Image analysis using Google's Gemini 2 Flash Lite model
- Identification of UI elements, diagrams, tables, and other content in images
- Data transformation for LightRAG ingestion
- Comprehensive error handling and logging
- Modular architecture with clean separation of concerns

## Architecture

The pipeline consists of the following components:

1. **PDF Parser**: Extracts text and images from PDF documents
2. **Image Processor**: Analyzes images using Gemini 2 Flash Lite
3. **Data Transformer**: Prepares extracted data for LightRAG
4. **Main Processor**: Orchestrates the entire pipeline

## Requirements

- Python 3.8+
- Google API key for Gemini 2 Flash Lite
- OpenAI API key for LightRAG (if using the LightRAG integration)

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Run the pipeline with default settings:

```bash
python main.py
```

### Command Line Arguments

- `--input-dir`: Directory containing PDF files to process (default: C:/Users/georg/Desktop/AEGIS/Projects/EnterSoftData)
- `--output-dir`: Directory to save processed output (default: ./output)
- `--gemini-api-key`: Google API key for Gemini (can also be set via GEMINI_API_KEY env var)
- `--openai-api-key`: OpenAI API key for LightRAG (can also be set via OPENAI_API_KEY env var)
- `--chunk-size`: Maximum size of text chunks for LightRAG (default: 1000)
- `--overlap`: Number of characters to overlap between chunks (default: 100)
- `--log-level`: Logging level (default: INFO)
- `--file-pattern`: File pattern to match PDF files (default: *.pdf)
- `--insert-lightrag`: Insert processed documents into LightRAG
- `--lightrag-dir`: Working directory for LightRAG (default: ./entersoft-docs)

Example:

```bash
python main.py --input-dir /path/to/pdfs --output-dir ./processed --log-level DEBUG --insert-lightrag
```

## Output Structure

The pipeline generates the following output:

- `output/images/`: Extracted images from PDFs
- `output/lightrag/`: Processed data in LightRAG format
- `output/logs/`: Processing logs
- `output/processing_stats.json`: Statistics about the processing run

## Data Models

The pipeline uses Pydantic models to represent data:

- `ProcessedDocument`: Represents a processed PDF document
- `DocumentSegment`: Represents a segment of a document (text or image)
- `ImageAnalysisResult`: Contains analysis results for an image
- `ProcessingStats`: Contains statistics about the processing run
- `ProcessingError`: Represents an error encountered during processing

## Integration with LightRAG

The pipeline can optionally insert processed documents into LightRAG for use with the chatbot. Use the `--insert-lightrag` flag to enable this feature.

## Error Handling

The pipeline includes comprehensive error handling and logging. Errors are logged and collected in the processing statistics, allowing for easy troubleshooting.

## License

MIT