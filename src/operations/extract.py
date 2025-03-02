import os
import logging
import concurrent.futures
import time
from typing import Dict, List, Optional, Union, Set
from pathlib import Path
import json
from llama_parse import LlamaParse
from os import environ
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import extract_msg
import re

logger = logging.getLogger('markdown-extractor')

LLAMA_API_KEY = environ.get('LLAMA_API_KEY')

class MarkdownExtractor:
    """
    Class for extracting content from various file types (PDF, XLSX, DOCX, MSG)
    and converting it to Markdown format using LlamaParser.
    """
    
    def __init__(self, output_dir: str = "./markdown_output", max_workers: int = 4):
        """
        Initialize the MarkdownExtractor.
        
        Args:
            output_dir: Directory to save the extracted Markdown files
            max_workers: Maximum number of worker threads for parallel processing

        Note: max_workers currently irrelevant due to llama-parse being synchronous. See README.md on further improvements part.
        """
        self.output_dir = output_dir
        self.api_key = LLAMA_API_KEY
        self.max_workers = max_workers
        
        # Define supported file extensions
        self.supported_extensions = {'.pdf', '.xlsx', '.xls', '.docx', '.doc', '.msg'}
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        logging.info(f"MarkdownExtractor initialized with output directory: {output_dir}")
        
        # Initialize the parser
        self.parser = LlamaParse(result_type="markdown", api_key=LLAMA_API_KEY)
    
    def extract_document(self, file_path: str, max_retries: int = 3) -> str:
        """
        Extract text from a document file and convert it to Markdown format.
        
        Args:
            file_path: Path to the document file
            max_retries: Maximum number of retries for API calls
            
        Returns:
            Markdown string of the document content
        """
        if not os.path.exists(file_path):
            logging.error(f"File not found: {file_path}")
            return ""
        
        file_ext = os.path.splitext(file_path.lower())[1]
        if file_ext not in self.supported_extensions:
            logging.error(f"Unsupported file type: {file_ext}")
            return f"# Unsupported File Type: {os.path.basename(file_path)}\n\nThe file type {file_ext} is not supported."
        
        # Special handling for MSG files because llama parse cannot handle them
        if file_ext.lower() == '.msg':
            return self._extract_from_msg(file_path)
        
        # Normal processing for other file types with LlamaParse
        # Normalize path for logging because windows paths use backslashes
        normalized_path = file_path.replace('\\', '/')
        logging.info(f"Parsing {normalized_path} with LlamaParse")
        
        for attempt in range(max_retries):
            try:
                # Parse the document with llama-parse
                logging.info(f"Processing {os.path.basename(file_path)} - attempt {attempt+1}")
                
                # Informing user about processing time.
                start_time = time.time()
                results = self.parser.load_data(file_path)
                processing_time = time.time() - start_time
                logging.info(f"Success! Processed in {processing_time:.2f} seconds")
                
                # Check if results are empty
                if not results:
                    logging.warning(f"No content extracted from {os.path.basename(file_path)}")
                    return f"# No Content Extracted: {os.path.basename(file_path)}\n\nLlamaParse did not return any content for this file."
                
                # For multi-page documents, combine all the content
                markdown_parts = []
                metadata = { 'page_count' : len(results), 'source' : file_path.replace('\\', '/'), 'file_type' : file_ext.replace('.', '') }
                
                for idx, doc in enumerate(results):
                    if hasattr(doc, 'text'):
                        markdown_parts.append(doc.text)
                    
                    # Collect metadata from the first document or any document that has it
                    if hasattr(doc, 'metadata'):
                        metadata.update(doc.metadata)
                
                # Combine all text parts
                markdown_content = "\n\n".join(markdown_parts)
                
                # Add metadata as YAML frontmatter if available almost always available since it's added by me.
                if metadata:
                    metadata_yaml = "---\n"
                    for key, value in metadata.items():
                        if value:
                            metadata_yaml += f"{key}: \"{value}\"\n"
                    metadata_yaml += "---\n\n"
                    
                    markdown_content = metadata_yaml + markdown_content
                
                return markdown_content
                
            except Exception as e:
                logging.error(f"Error on attempt {attempt+1}: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # Progressive backoff
                    logging.warning(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Failed after {max_retries} attempts: {str(e)}"
                    logging.error(error_msg)
                    return f"# Error Processing {os.path.basename(file_path)}\n\n{error_msg}"
    
    def _extract_from_msg(self, file_path: str) -> str:
        """
        Extract content from an MSG (Outlook email) file and convert to Markdown.
        
        Args:
            file_path: Path to the MSG file
            
        Returns:
            Markdown string representation of the email
        """
        try:
            logging.info(f"Processing MSG file {os.path.basename(file_path)}")
            
            # Parse the MSG file
            msg = extract_msg.Message(file_path)
            
            # Build metadata dictionary
            metadata = {
                "subject": msg.subject,
                "sender": msg.sender,
                "to": msg.to,
                "cc": msg.cc if hasattr(msg, 'cc') else None,
                "date": msg.date.strftime("%Y-%m-%d %H:%M:%S") if msg.date else None,
                "source": file_path.replace('\\', '/'),
                "file_type": "msg"
            }
            
            # Start building markdown
            markdown_parts = []
            
            # Add metadata as YAML frontmatter
            frontmatter = "---\n"
            for key, value in metadata.items():
                if value:
                    frontmatter += f"{key}: \"{value}\"\n"
            frontmatter += "---\n\n"
            
            # Add email header
            markdown_parts.append(f"# {msg.subject}\n")
            markdown_parts.append(f"**From:** {msg.sender}")
            markdown_parts.append(f"**To:** {msg.to}")
            if hasattr(msg, 'cc') and msg.cc:
                markdown_parts.append(f"**CC:** {msg.cc}")
            if msg.date:
                markdown_parts.append(f"**Date:** {msg.date.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Add email body
            markdown_parts.append("\n## Email Body\n")
            
            # Get email body content
            body_text = ""
            # Try to get HTML body first for better formatting
            if msg.htmlBody:
                from bs4 import BeautifulSoup
                try:
                    soup = BeautifulSoup(msg.htmlBody, 'html.parser')
                    body_text = soup.get_text(separator='\n')
                except Exception as e:
                    # Fallback to plain text if HTML parsing fails
                    body_text = msg.body
            else:
                # Use plain text body
                body_text = msg.body
                
            # Clean up the body text for less tokens in the future
            if body_text:
                # Replace consecutive blank lines with a maximum of one blank line
                cleaned_body = re.sub(r'\n\s*\n\s*\n+', '\n\n', body_text)
                
                # Remove excessive spaces
                cleaned_body = re.sub(r' {2,}', ' ', cleaned_body)
                
                # Ensure paragraphs have consistent spacing
                cleaned_body = re.sub(r'\n{3,}', '\n\n', cleaned_body)
                
                # Remove leading/trailing whitespace
                cleaned_body = cleaned_body.strip()
                
                markdown_parts.append(cleaned_body)
            else:
                markdown_parts.append("*No email body content*")
            
            # Add attachments section if there are any
            # This part mostly irrelevant for the project, but it is more complete with the following code.
            if msg.attachments:
                markdown_parts.append("\n\n## Attachments\n")
                for i, attachment in enumerate(msg.attachments):
                    markdown_parts.append(f"{i+1}. {attachment.longFilename}")
                    
                    # Save attachments to a subdirectory
                    attachments_dir = os.path.join(self.output_dir, "attachments")
                    if not os.path.exists(attachments_dir):
                        os.makedirs(attachments_dir)
                    
                    try:
                        msg_basename = os.path.splitext(os.path.basename(file_path))[0]
                        attachment_path = os.path.join(
                            attachments_dir, 
                            f"{msg_basename}_{attachment.longFilename}"
                        )
                        with open(attachment_path, 'wb') as f:
                            f.write(attachment.data)
                        markdown_parts.append(f"   - Saved to: `{attachment_path}`")
                    except Exception as e:
                        markdown_parts.append(f"   - Error saving attachment: {str(e)}")
            
            # Combine all parts
            markdown_content = frontmatter + "\n".join(markdown_parts)
            
            # Clean up msg object
            msg.close()
            
            return markdown_content
            
        except Exception as e:
            error_msg = f"Error extracting content from MSG file: {str(e)}"
            logging.error(error_msg)
            return f"# Error Processing {os.path.basename(file_path)}\n\n{error_msg}"
    
    def extract_all_documents(self, doc_dir: str = "./documents", 
                             file_extensions: Set[str] = None, 
                             skip_existing: bool = False) -> Dict[str, str]:
        """
        Extract content from all documents in a directory.
        
        Args:
            doc_dir: Directory containing document files
            file_extensions: Set of file extensions to process (default: all supported)
            skip_existing: Skip files that already have markdown output
            
        Returns:
            Dictionary mapping filenames to markdown content
        """
        # Default to all supported extensions if no extensions specified
        if file_extensions is None:
            file_extensions = self.supported_extensions
        
        # Convert to lowercase set for case-insensitive matching
        file_extensions = {ext.lower() for ext in file_extensions}
        
        # Get all matching files in directory and subdirectories
        files_to_process = []
        for root, _, files in os.walk(doc_dir):
            for file in files:
                file_ext = os.path.splitext(file.lower())[1] # Extract the extension
                if file_ext in file_extensions:
                    full_path = os.path.join(root, file)
                    output_path = self._get_output_path(full_path)
                    
                    # Check if file already has a corresponding markdown file
                    if skip_existing and os.path.exists(output_path):
                        logger.info(f"Skipping {file} - markdown already exists")
                        continue
                        
                    files_to_process.append(full_path)
        
        logging.info(f"Found {len(files_to_process)} files to process")

        if not files_to_process:
            logger.warning(f"No matching files found in {doc_dir}")
            return {}
        
        results = {}
        
        # Process files SEQUENTIALLY with progress reporting, 
        # again it is pointed as sequentially because this part can be improved with parallel processing.
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("({task.completed}/{task.total})"),
        ) as progress:
            task = progress.add_task("Converting documents to Markdown...", total=len(files_to_process))
            
            # Process files one by one (sequentially)
            for file_path in files_to_process:
                file_name = os.path.basename(file_path)
                try:
                    # Process single file to save into md file
                    markdown_content = self._process_file(file_path)
                    results[file_name] = markdown_content
                    progress.update(task, advance=1)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_name}: {str(e)}")
                    results[file_name] = f"# Error Processing {file_name}\n\nAn error occurred: {str(e)}"
                    progress.update(task, advance=1)
        
        logger.info(f"Processed {len(results)} files from {doc_dir}")
        return results
    
    def _process_file(self, file_path: str) -> str:
        """Process a single file and save the output"""

        print("File path process file: ", file_path)
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path.lower())[1]
        # Use normalized path for logging
        normalized_path = file_path.replace('\\', '/')
        logger.info(f"Processing {file_name} ({file_ext}) from {normalized_path}...")
        
        # Extract content as string to write to markdown file
        markdown_content = self.extract_document(file_path)
        
        # Save to file in the MARKDOWN Folder
        output_path = self._get_output_path(file_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        logger.info(f"Extracted and saved markdown for {file_name}")
        return markdown_content
            
    
    def _get_output_path(self, input_path: str) -> str:
        """Generate output path for markdown file based on input path"""
        file_name = os.path.basename(input_path)
        output_filename = os.path.splitext(file_name)[0] + '.md'
        return os.path.join(self.output_dir, output_filename)