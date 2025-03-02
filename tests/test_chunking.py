import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.chunking.strategies import (
    NoChunkingStrategy,
    HeaderBasedChunkingStrategy,
    SizeBasedChunkingStrategy,
    AdaptiveChunkingStrategy,
    get_strategy
)

class TestChunkingStrategies(unittest.TestCase):
    def setUp(self):
        self.sample_content = """# Header 1
This is content under header 1.

## Header 2
This is content under header 2.

### Header 3
This is content under header 3.

# Another Header 1
This is more content.
"""
        self.metadata = {"source": "./documents/test.pdf", "filename": "test.md", "page_count": "3"}
    
    def test_no_chunking_strategy(self):
        strategy = NoChunkingStrategy()
        chunks = strategy.chunk_document(self.sample_content, self.metadata)
        
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].page_content, self.sample_content)
        self.assertEqual(chunks[0].metadata["source"], "./documents/test.pdf")
    
    def test_header_based_chunking_strategy(self):
        strategy = HeaderBasedChunkingStrategy()
        chunks = strategy.chunk_document(self.sample_content, self.metadata)
        
        # Should split into at least 3 chunks (2 H1 headers and 1 H2 header)
        # Header division, and naming convention is not perfect, so there is an improvement possibility
        # print("Chunks:", chunks)
        self.assertGreaterEqual(len(chunks), 3)
        
        # Check metadata is preserved
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "./documents/test.pdf")
            self.assertEqual(chunk.metadata["filename"], "test.md")
    
    def test_size_based_chunking_strategy(self):
        strategy = SizeBasedChunkingStrategy(chunk_size=100, chunk_overlap=10)
        chunks = strategy.chunk_document(self.sample_content, self.metadata)
        
        # Content should be split into multiple chunks
        self.assertGreater(len(chunks), 1)
        
        # Check that no chunk exceeds the specified chunk size by too much
        for chunk in chunks:
            self.assertLessEqual(len(chunk.page_content), 150)  # Allow some flexibility
    
    def test_adaptive_chunking_strategy(self):
        """
        Since the adaptive chunking strategy is based on two strategies combined (header and size),
        this test has to cover up for both of them.
        """
        strategy = AdaptiveChunkingStrategy()
        chunks = strategy.chunk_document(self.sample_content, self.metadata)
        
        # Should have at least one chunk
        self.assertGreater(len(chunks), 0)
        
        # Test with a document that doesn't have headers
        no_header_content = "This is a document with no headers. It just has plain text content. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. End of the testing adaptive chunking stratgey."

        no_header_chunks = strategy.chunk_document(no_header_content, self.metadata)
        
        # Should still create at least one chunk
        self.assertGreater(len(no_header_chunks), 0)
    
    def test_get_strategy_function(self):
        # Test that get_strategy returns the right strategy
        self.assertIsInstance(get_strategy("none"), NoChunkingStrategy)
        self.assertIsInstance(get_strategy("header"), HeaderBasedChunkingStrategy)
        self.assertIsInstance(get_strategy("size"), SizeBasedChunkingStrategy)
        self.assertIsInstance(get_strategy("adaptive"), AdaptiveChunkingStrategy)
        
        # Test default (should return AdaptiveChunkingStrategy)
        self.assertIsInstance(get_strategy("invalid_name"), AdaptiveChunkingStrategy)

if __name__ == "__main__":
    unittest.main()
