import unittest
import sys
import os
import sqlite3
import pickle
import numpy as np
from pathlib import Path

# Add parent directory to path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent))

from src.operations.search import SearchEngine

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        # Create a test database
        self.db_path = "./tests/test_db.db"
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Create a test database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create the embeddings table
        cursor.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB NOT NULL
            )''')
        
        # Add some test embeddings
        # Data of the test embeddings totally irrelevant to the actual case. 
        # Yet, it is necessary to have some data to test the search engine.
        embeddings = [
            {
                "id": "1",
                "content": "This is a document about cats",
                "metadata": "{'source': './documents/cats.pdf', 'filename': 'cats.md'}",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            },
            {
                "id": "2",
                "content": "This is a document about dogs",
                "metadata": "{'source': './documents/dogs.msg', 'filename': 'dogs.md'}",
                "embedding": [0.5, 0.4, 0.3, 0.2, 0.1]
            },
            {
                "id": "3",
                "content": "This is a document about space",
                "metadata": "{'source': './documents/space.docx', 'filename': 'space.md'}",
                "embedding": [0.9, 0.1, 0.1, 0.1, 0.1]
            }
        ]
        
        for item in embeddings:
            cursor.execute(
                "INSERT INTO embeddings (id, content, metadata, embedding) VALUES (?, ?, ?, ?)",
                (item["id"], item["content"], item["metadata"], pickle.dumps(item["embedding"]))
            )
        
        conn.commit()
        conn.close()
    
    def tearDown(self):
        # Remove the test database
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
    
    def test_search_cosine_similarity(self):
        # Test cosine similarity search
        engine = SearchEngine(db_path=self.db_path, similarity_method="cosine", top_k=2)
        
        # Query embedding exactly same to cats
        # Best case scenario to find a match
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = engine.search(query_embedding)
        
        # Should return 2 results due to top_k = 2
        self.assertEqual(len(results), 2)
        
        # First result should be about cats (most similar)
        self.assertIn("cats", results[0]["content"])
        
        # Check similarity score (should be 1.0 for exact match)
        self.assertAlmostEqual(results[0]["similarity"], 1.0, places=5)
    
    def test_search_dot_product(self):
        # Test dot product similarity search
        engine = SearchEngine(db_path=self.db_path, similarity_method="dot_product", top_k=2)
        
        # Query embedding similar to cats
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = engine.search(query_embedding)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # First result should be about cats (highest dot product)
        self.assertIn("cats", results[0]["content"])
    
    def test_search_euclidean(self):
        # Test euclidean distance search
        engine = SearchEngine(db_path=self.db_path, similarity_method="euclidean", top_k=2)
        
        # Query embedding similar to cats
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = engine.search(query_embedding)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # First result should be about cats (lowest distance)
        self.assertIn("cats", results[0]["content"])
        
        # For euclidean, lowest score is best (distance=0 for exact match)
        self.assertEqual(results[0]["similarity"], 0.0)
    
    def test_invalid_similarity_method(self):
        # Test invalid similarity method (should default to cosine)
        engine = SearchEngine(db_path=self.db_path, similarity_method="invalid", top_k=2)
        
        # Query embedding similar to cats
        query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        results = engine.search(query_embedding)
        
        # Should return 2 results
        self.assertEqual(len(results), 2)
        
        # First result should be about cats (most similar by cosine)
        self.assertIn("cats", results[0]["content"])

if __name__ == "__main__":
    unittest.main()
