from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import sqlite3
import pickle
import logging

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

class SearchEngine:
    """
    Search engine for finding similar documents based on vector embeddings.
    Supports multiple similarity measures.
    """
    
    def __init__(self, db_path: str, similarity_method: str = "cosine", top_k: int = 5):
        """
        Initialize the search engine.
        
        Args:
            db_path: Path to the SQLite database with embeddings
            similarity_method: Similarity method to use (cosine, dot_product, or euclidean)
            top_k: Number of top results to return
        """
        self.db_path = db_path
        self.similarity_method = similarity_method.lower()
        self.top_k = top_k
        
        # Validate similarity method
        valid_methods = ["cosine", "dot_product", "euclidean"]
        if self.similarity_method not in valid_methods:
            logging.warning(f"Invalid similarity method '{similarity_method}'. Using 'cosine' instead.")
            self.similarity_method = "cosine"
            
        logging.info(f"Search engine initialized with {self.similarity_method} similarity method")
    
    def search(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Search for similar documents using the specified similarity method.
        
        Args:
            query_embedding: Vector embedding of the query
            
        Returns:
            List of dictionaries with content, metadata, and similarity score
        """
        query_embedding_array = np.array(query_embedding)
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Fetch all embeddings
        cursor.execute("SELECT id, content, metadata, embedding FROM embeddings")
        results = cursor.fetchall()
        
        if not results:
            logging.warning("No documents found in the database")
            return []
        
        # Calculate similarity for each embedding
        similarity_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} file searched"),
        ) as progress:
          task = progress.add_task("Searching files...", total=len(results))

          for id_val, content, metadata_str, embedding_blob in results:
              # Deserialize the embedding
              embedding = pickle.loads(embedding_blob)
              embedding_array = np.array(embedding)
              
              # Calculate similarity based on the selected method
              similarity = self._calculate_similarity(query_embedding_array, embedding_array)
              
              similarity_results.append({
                  "id": id_val,
                  "content": content,
                  "metadata": eval(metadata_str),  # Convert string representation back to dict
                  "similarity": similarity
              })

              progress.update(task, advance=1)
        
        # Sort by similarity (higher is better for cosine and dot product, 
        # lower is better for euclidean due to euclidean using distance measure)
        reverse_sort = self.similarity_method != "euclidean"
        similarity_results.sort(key=lambda x: x["similarity"], reverse=reverse_sort)
        
        # Take top_k results
        top_results = similarity_results[:self.top_k]
        
        conn.close()
        return top_results
    
    def _calculate_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings using the specified method.
        
        Args:
            query_embedding: The query embedding as a numpy array
            doc_embedding: The document embedding as a numpy array
            
        Returns:
            Similarity score (higher is better for cosine and dot product, lower is better for euclidean)
        """
        if self.similarity_method == "cosine":
            return self._cosine_similarity(query_embedding, doc_embedding)
        elif self.similarity_method == "dot_product":
            return self._dot_product(query_embedding, doc_embedding)
        elif self.similarity_method == "euclidean":
            return self._euclidean_distance(query_embedding, doc_embedding)
        else:
            # Default to cosine if something goes wrong
            return self._cosine_similarity(query_embedding, doc_embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate dot product between two vectors."""
        return np.dot(a, b)
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate euclidean distance between two vectors."""
        return np.linalg.norm(a - b)