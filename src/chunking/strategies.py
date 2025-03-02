from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies"""
    
    @abstractmethod
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split a document into chunks based on the strategy
        
        Args:
            content: The document content to be chunked
            metadata: The metadata associated with the document
            
        Returns:
            A list of Document objects representing the chunks
        """
        pass

class NoChunkingStrategy(ChunkingStrategy):
    """Strategy that doesn't chunk the document at all"""
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Keep the document as a single chunk"""
        return [Document(page_content=content, metadata=metadata)]

class HeaderBasedChunkingStrategy(ChunkingStrategy):
    """Strategy that chunks the document based on Markdown headers"""
    
    def __init__(self, headers_to_split_on=None):
        """
        Initialize with header levels to split on
        
        Args:
            headers_to_split_on: List of (header_marker, header_name) tuples
        """
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "heading1"),
            ("##", "heading2"),
            ("###", "heading3"),
            ("####", "heading4"),
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on
        )
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """
        Split document based on markdown headers
        
        This can be improved with the tree structure like header storage with smaller chunks
        """
        try:
            header_splits = self.md_splitter.split_text(content)
            
            # Enhance each split with the file's metadata
            documents = []
            for split in header_splits:
                # Make a copy of the metadata to avoid shared references
                split_metadata = metadata.copy()
                
                # Add heading info to metadata
                for key, value in split.metadata.items():
                    if key.startswith('heading'):
                        split_metadata[key] = value
                
                # Create a Document with enhanced metadata
                documents.append(Document(
                    page_content=split.page_content,
                    metadata=split_metadata
                ))
            
            return documents
        except Exception as e:
            # If header splitting fails, return the whole document
            print(f"Header splitting failed: {str(e)}, returning document as single chunk")
            return [Document(page_content=content, metadata=metadata)]

class SizeBasedChunkingStrategy(ChunkingStrategy):
    """Strategy that chunks the document based on size/character count"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=100):
        """
        Initialize with chunk size parameters
        
        Args:
            chunk_size: The target size of each chunk
            chunk_overlap: The amount of overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len,
        )
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Split document based on size"""
        chunks = self.text_splitter.split_text(content)
        
        # Create Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            # Make a copy of metadata and add chunk info
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_count"] = len(chunks)
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents

class AdaptiveChunkingStrategy(ChunkingStrategy):
    """Strategy that adapts based on document properties (tries headers first, then size)"""
    
    def __init__(self):
        """Initialize with both header and size-based strategies"""
        self.header_strategy = HeaderBasedChunkingStrategy()
        self.size_strategy = SizeBasedChunkingStrategy()
    
    def chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[Document]:
        """Tries header-based chunking first, falls back to size-based"""
        try:
            # Try header-based chunking first
            header_chunks = self.header_strategy.chunk_document(content, metadata)
            

            print(len(content))
            # If header chunking resulted in only one chunk and the content is large,
            # try size-based chunking instead
            if len(header_chunks) <= 1 and len(content) > 2000:
                # Get page count if available to adjust chunk size
                page_count = 1
                if "page_count" in metadata:
                    try:
                        page_count = int(metadata.get("page_count", "1").strip('"'))
                    except (ValueError, TypeError):
                        page_count = 1
                
                print("AdaptiveChunking switched to SizeBasedChunkingStrategy")
                # Create a custom size-based strategy with page-aware chunk size
                total_length = len(content)
                chunk_size = max(1000, total_length // (page_count * 2))
                custom_size_strategy = SizeBasedChunkingStrategy(
                    chunk_size=chunk_size, 
                    chunk_overlap=chunk_size//10
                )
                
                return custom_size_strategy.chunk_document(content, metadata)
            else:
                return header_chunks
                
        except Exception as e:
            # Fall back to size-based if header-based fails
            print(f"Adaptive chunking fallback: {str(e)}")
            return self.size_strategy.chunk_document(content, metadata)


def get_strategy(strategy_name: str = "adaptive") -> ChunkingStrategy:
    """
    Factory function to get the appropriate chunking strategy.
    Allows developers to easily switch between strategies using CLI Argument.
    
    Args:
        strategy_name: Name of the strategy to use
        
    Returns:
        A chunking strategy instance
    """
    strategies = {
        "none": NoChunkingStrategy(),
        "header": HeaderBasedChunkingStrategy(),
        "size": SizeBasedChunkingStrategy(),
        "adaptive": AdaptiveChunkingStrategy(),
    }
    
    return strategies.get(strategy_name.lower(), AdaptiveChunkingStrategy())
