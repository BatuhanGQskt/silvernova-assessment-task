import unittest
import sys
import os
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.operations.ask import LLMAsker, estimate_token_count

class TestLLMAsker(unittest.TestCase):
    def test_token_estimation(self):
        # Test token estimation function
        text = "This is a test string with exactly 12 words and should be around 12 tokens."
        estimate = estimate_token_count(text)
        
        # The estimation should be roughly len(text) / 4
        self.assertAlmostEqual(estimate, len(text) // 4)
    
    @patch('src.operations.ask.execute_prompt') # Patch the execute_prompt functionality to avoid actual API calls, and it will be mocked
    @patch('src.operations.ask.EmbedService')
    @patch('src.operations.ask.SearchEngine')
    def test_construct_prompt_intelligent(self, mock_search_engine, mock_embed_service, mock_execute_prompt):
        # Configure mocks
        mock_embed_instance = MagicMock()
        mock_embed_instance.embed.return_value = [0.1] * 1024  # Create a 1024-dimensional vector to represent return value of embeddings
        mock_embed_service.return_value = mock_embed_instance # Return value of the object as a mocked EmbedService class
        
        mock_search_instance = MagicMock()
        mock_search_instance.search.return_value = [
            {
                "content": "This is document 1",
                "metadata": {"source": "./documents/doc1.pdf", "filename": "doc1.md"},
                "similarity": 0.9
            },
            {
                "content": "This is document 2",
                "metadata": {"source": "./documents/doc2.docx", "filename": "doc2.md"},
                "similarity": 0.8
            }
        ]
        mock_search_engine.return_value = mock_search_instance # Return value of the object as a mocked SearchEngine class
        
        mock_execute_prompt.return_value = {"response": "This is a test response"} # Always returns this response for testing purposes
        
        # Create LLMAsker instance with small max_tokens for testing
        asker = LLMAsker(max_tokens=1000, construction_method="intelligent")
        
        # Test ask method
        response = asker.ask("What is the capital of France?")
        
        # Verify that the correct methods were called
        mock_embed_instance.embed.assert_called_once_with("What is the capital of France?")
        mock_search_instance.search.assert_called_once()
        mock_execute_prompt.assert_called_once()
        
        expected = "This is a test response\n\nSources:\n1. ./documents/doc1.pdf\n2. ./documents/doc2.docx\n"
        # Verify response
        self.assertEqual(response, expected)
    
    @patch('src.operations.ask.execute_prompt')
    @patch('src.operations.ask.EmbedService')
    @patch('src.operations.ask.SearchEngine')
    def test_construct_prompt_representative(self, mock_search_engine, mock_embed_service, mock_execute_prompt):
        # Configure mocks
        mock_embed_instance = MagicMock()
        mock_embed_instance.embed.return_value = [0.1, 0.2, 0.3]
        mock_embed_service.return_value = mock_embed_instance
        
        # Mock search results with two documents from same source
        mock_search_instance = MagicMock()
        mock_search_instance.search.return_value = [
            {
                "content": "This is document 1, part 1",
                "metadata": {"source": "./documents/doc1.pdf", "filename": "doc1.md"},
                "similarity": 0.9
            },
            {
                "content": "This is document 1, part 2",
                "metadata": {"source": "./documents/doc1.pdf", "filename": "doc1.md"},
                "similarity": 0.8
            },
            {
                "content": "This is document 2",
                "metadata": {"source": "./documents/doc2.pdf", "filename": "doc2.md"},
                "similarity": 0.7
            }
        ]
        mock_search_engine.return_value = mock_search_instance
        
        mock_execute_prompt.return_value = {"response": "This is a representative response"}
        
        # Create LLMAsker instance with representative construction method
        asker = LLMAsker(max_tokens=1000, construction_method="representative")
        
        # Test ask method
        response = asker.ask("What is the capital of France?")
        
        # Verify that the correct methods were called
        mock_embed_instance.embed.assert_called()  # Called multiple times
        mock_search_instance.search.assert_called_once()
        mock_execute_prompt.assert_called_once()
        

        # If id or source should have been created more accurately depending on the chunk/page number, this test would be more reasonable.
        expected = "This is a representative response\n\nSources:\n1. ./documents/doc1.pdf\n2. ./documents/doc2.pdf\n"
        # Verify response
        self.assertEqual(response, expected)
    
    def test_format_answer_with_sources(self):
        # Test formatting answer with sources
        asker = LLMAsker()
        
        answer = "Paris is the capital of France."
        context = [
            {
                "content": "Paris is the capital of France.",
                "metadata": {"source": "./documents/france.pdf", "filename": "france.md", "heading1": "France"},
                "similarity": 0.9
            },
            {
                "content": "Berlin is the capital of Germany.",
                "metadata": {"source": "./documents/germany.docx", "filename": "germany.md"},
                "similarity": 0.5,
                "truncated": True
            }
        ]
        
        formatted_answer = asker._format_answer(answer, context)
        
        # Formatted answer should contain the answer and sources
        self.assertIn("Paris is the capital of France.", formatted_answer)
        self.assertIn("./documents/france.pdf", formatted_answer)
        self.assertIn("./documents/germany.docx", formatted_answer)
        self.assertIn("France", formatted_answer)  # heading included
        self.assertIn("[truncated]", formatted_answer)  # truncation noted

    def test_format_answer_without_sources(self):
        # Test formatting answer without sources
        asker = LLMAsker(include_sources=False)
        
        answer = "Paris is the capital of France."
        context = [
            {
                "content": "Paris is the capital of France.",
                "metadata": {"source": "france.md", "filename": "france.md"},
                "similarity": 0.9
            }
        ]
        
        formatted_answer = asker._format_answer(answer, context)
        
        # Formatted answer should only contain the answer
        self.assertEqual(formatted_answer, answer)

if __name__ == "__main__":
    unittest.main()
