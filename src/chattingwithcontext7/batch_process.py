#!/usr/bin/env python
"""
Batch process PDFs with pauses to avoid rate limits.
This script processes PDFs in batches with pauses between batches to avoid hitting API rate limits.
"""

import os
import sys
import time
import logging
import argparse
import asyncio
import subprocess
from pathlib import Path
from datetime import datetime

def setup_logging():
    """Set up logging."""
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"batch_process_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Set up logging
    logger = logging.getLogger("batch_processor")
    logger.setLevel(logging.INFO)
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    
    return logger

def find_pdf_files(input_dir):
    """Find all PDF files in the input directory."""
    return list(Path(input_dir).glob("**/*.pdf"))

def process_batch(pdf_files, batch_index, batch_size, output_dir, logger):
    """Process a batch of PDF files."""
    start_idx = batch_index * batch_size
    end_idx = min(start_idx + batch_size, len(pdf_files))
    batch_files = pdf_files[start_idx:end_idx]
    
    # Create batch directory
    batch_dir = os.path.join(output_dir, f"batch_{batch_index}")
    os.makedirs(batch_dir, exist_ok=True)
    
    # Create a temporary directory for this batch
    temp_input_dir = os.path.join(batch_dir, "input")
    os.makedirs(temp_input_dir, exist_ok=True)
    
    # Create symbolic links to the PDF files for this batch
    for pdf_file in batch_files:
        # Create a symbolic link in the temp directory
        link_path = os.path.join(temp_input_dir, pdf_file.name)
        if os.path.exists(link_path):
            os.remove(link_path)
        
        # On Windows, we need to copy the file instead of creating a symlink
        if os.name == 'nt':
            import shutil
            shutil.copy2(pdf_file, link_path)
        else:
            os.symlink(pdf_file, link_path)
    
    # Process the batch
    logger.info(f"Processing batch {batch_index+1} ({start_idx+1}-{end_idx} of {len(pdf_files)})")
    logger.info(f"Batch files: {[f.name for f in batch_files]}")
    
    # Run the build_database.py script for this batch
    cmd = [
        sys.executable,
        "-m",
        "src.chattingwithcontext7.build_database",
        "--input-dir", temp_input_dir,
        "--output-dir", batch_dir,
        "--database-dir", os.path.join(batch_dir, "lightrag_data"),
        "--concurrency-limit", "1"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output to the logger
        for line in process.stdout:
            logger.info(line.strip())
        
        # Wait for the process to complete
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Batch {batch_index+1} completed successfully")
        else:
            logger.error(f"Batch {batch_index+1} failed with return code {process.returncode}")
        
    except Exception as e:
        logger.error(f"Error processing batch {batch_index+1}: {str(e)}")
        return False
    
    return True

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Process PDFs in batches with pauses to avoid rate limits"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./entersoft-docs",
        help="Directory containing PDF files to process"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output/batches",
        help="Directory to save processed output"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of PDFs to process in each batch"
    )
    parser.add_argument(
        "--pause-minutes",
        type=int,
        default=10,
        help="Number of minutes to pause between batches"
    )
    parser.add_argument(
        "--start-batch",
        type=int,
        default=0,
        help="Batch index to start from (0-based)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Find all PDF files
    pdf_files = find_pdf_files(args.input_dir)
    logger.info(f"Found {len(pdf_files)} PDF files in {args.input_dir}")
    
    # Calculate number of batches
    num_batches = (len(pdf_files) + args.batch_size - 1) // args.batch_size
    logger.info(f"Processing in {num_batches} batches of {args.batch_size} files")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process batches
    for batch_index in range(args.start_batch, num_batches):
        logger.info(f"Starting batch {batch_index+1}/{num_batches}")
        
        # Process the batch
        success = process_batch(pdf_files, batch_index, args.batch_size, args.output_dir, logger)
        
        # Check if this is the last batch
        if batch_index == num_batches - 1:
            logger.info("All batches completed")
            break
        
        # Pause between batches
        if success:
            pause_seconds = args.pause_minutes * 60
            logger.info(f"Pausing for {args.pause_minutes} minutes before next batch")
            
            # Display a countdown
            for remaining in range(pause_seconds, 0, -60):
                minutes_left = remaining // 60
                logger.info(f"Resuming in {minutes_left} minutes...")
                time.sleep(60)
        else:
            logger.error(f"Batch {batch_index+1} failed, stopping processing")
            break
    
    logger.info("Batch processing completed")

if __name__ == "__main__":
    main()