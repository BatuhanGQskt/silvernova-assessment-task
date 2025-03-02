import unittest
import sys
import os
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.__init__ import App, LLM_DB
from src.operations.embed import EmbedService
from src.operations.extract import MarkdownExtractor
from src.chunking import NoChunkingStrategy

class TestIntegration(unittest.TestCase):
    def setUp(self):
        # Create temporary directories for testing
        # Basically, same structure of the project, but in the temporary directory
        self.temp_dir = tempfile.mkdtemp(suffix="silvernova")
        print("Temporary directory created:", self.temp_dir)

        self.docs_dir = os.path.join(self.temp_dir, "documents")
        self.md_dir = os.path.join(self.temp_dir, "markdown_output")
        self.db_dir = os.path.join(self.temp_dir, "llm_db")
        
        # Create directories
        os.makedirs(self.docs_dir, exist_ok=True)
        os.makedirs(self.md_dir, exist_ok=True)
        os.makedirs(self.db_dir, exist_ok=True)
        
        # Create a test document that can be writable to the temporary file
        with open(os.path.join(self.docs_dir, "test.docx"), "w") as f:
            f.write("This is a test document for integration testing.")
    
    def tearDown(self):
        # Remove temporary directories
        shutil.rmtree(self.temp_dir)
    
    def test_workflow_extract_to_db(self):
        """Test the workflow from document extraction to database storage"""
        # Use MarkdownExtractor to convert documents to markdown
        extractor = MarkdownExtractor(output_dir=self.md_dir)
        results = extractor.extract_all_documents(doc_dir=self.docs_dir) # This extracts files and saves them into the markdown_output directory
        
        # Verify extraction
        self.assertGreaterEqual(len(results), 1) # test.docx file should be extracted as md file
        md_files = os.listdir(self.md_dir)
        self.assertGreaterEqual(len(md_files), 1)
        
        # Create a markdown file for chunking test
        md_content = """# Test Document

This is a test document for integration testing.

## Section 1

This is section 1 content.

## Section 2

This is section 2 content.
"""
        with open(os.path.join(self.md_dir, "test_md.md"), "w") as f:
            f.write(md_content)
        

        # Create LLM_DB instance
        db_path = os.path.join(self.db_dir, "test.db")
        embed_service = EmbedService("document")
        llm_db = LLM_DB(db_path, embed_service)
        
        # Create test document with NoChunkingStrategy
        strategy = NoChunkingStrategy()
        with open(os.path.join(self.md_dir, "test_md.md"), "r") as f:
            content = f.read()
        
        metadata = {
            "source": "./documents/test.docx",
            "filename": "test_md.md"
        }
        
        # Create chunks
        chunks = strategy.chunk_document(content, metadata)
        
        # Add chunks to database
        llm_db.add_chunks(chunks)
        
        # Verify database has entries
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertGreaterEqual(count, 1)

    @unittest.skipUnless(os.getenv("API_KEY"), "This test requires a real LLM API and is meant for manual execution")
    def test_search_and_ask_workflow(self):
        """Test the search and ask workflow (requires real API access)"""
        # This is a placeholder for a manual test
        # You would need to run this with actual API credentials
        
        app = App()
        
        # First index some files
        app.get_markdown()
        app.load_files(chunking_strategy="none", skip_duplicates=True)
        
        # Test search
        app.search("test document", similarity_method="cosine", top_k=3)
        
        # Test ask question
        app.ask_question("Was ist die Betrag gesamt EUR für die Haufe Gruppe?", similarity="cosine", top_k=3, construction="intelligent")

if __name__ == "__main__":
    unittest.main()
