from src.api import execute_prompt
from src.operations.search import SearchEngine
from src.operations.embed import EmbedService
import time
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
import logging
import numpy as np

console = Console()

# Approximate token count - rough estimate is ~4 chars per token
def estimate_token_count(text: str) -> int:
    return len(text) // 4

class LLMAsker:
    """
    Provides question answering capabilities using the LLM API with search augmentation.
    """

    def __init__(self, 
                 db_path: str = "./llm_db/markdown_embeddings.db",
                 similarity_method: str = "cosine",
                 top_k: int = 3,
                 include_sources: bool = True,
                 max_tokens: int = 100000,  # Safe limit below 128k
                 construction_method: str = "intelligent"): 
        """
        Initialize the LLM Asker with search configuration.
        
        Args:
            db_path: Path to the SQLite database with document embeddings
            similarity_method: Method to use for similarity search (cosine, dot_product, euclidean)
            top_k: Number of relevant documents to include in context
            include_sources: Whether to include source information in responses
            max_tokens: Maximum tokens to allow in prompt (default: 100k to be safe)
            construction_method: Method to use for prompt construction ("intelligent" or "representative")
        """
        self.db_path = db_path
        self.similarity_method = similarity_method
        self.top_k = top_k
        self.include_sources = include_sources
        self.max_tokens = max_tokens
        self.construction_method = construction_method
        self.embed_service = EmbedService('query')
        self.search_engine = SearchEngine(
            db_path=db_path, 
            similarity_method=similarity_method,
            top_k=top_k
        )

    def ask(self, question: str) -> str:
        """
        Ask a question and get a response augmented with relevant document context.
        
        Args:
            question: The question to ask
            
        Returns:
            The LLM response
        """
        # Display thinking animation
        with console.status("[bold green]Searching for relevant information...[/bold green]"):
            # Get relevant context through search
            relevant_context = self._get_relevant_context(question)
            time.sleep(0.5)  # Small pause for UX
        
        # Construct prompt with context using the selected method
        if self.construction_method == "representative":
            prompt, used_sources = self._construct_prompt_with_representation(question, relevant_context)
        else:  # Default to "intelligent"
            prompt, used_sources = self._construct_prompt(question, relevant_context)
            
        # Show that we're generating a response
        with console.status("[bold green]Generating answer based on found information...[/bold green]"):
            try:
                #print("Prompt is: ", prompt)
                #print("Len prompt is: ", len(prompt))
                #print("Token promt is: ", estimate_token_count(prompt))
                #print("Max token is: ", self.max_tokens)

                response = execute_prompt(prompt)
                answer = response.get('response', "No response generated.")
                
                # Process answer to include source attribution if requested
                final_answer = self._format_answer(answer, used_sources)
                return final_answer
                
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] Failed to get response from LLM API: {str(e)}")
                return f"Sorry, I encountered an error while generating your answer: {str(e)}"
    
    def _get_relevant_context(self, question: str) -> List[Dict[str, Any]]:
        """Get relevant document chunks for the question"""
        # Generate embedding for the query
        query_embedding = self.embed_service.embed(question)
        
        if not query_embedding:
            logging.warning("Failed to generate embedding for query")
            return []
        
        # Search for relevant documents
        results = self.search_engine.search(query_embedding)
        return results
    
    def _construct_prompt(self, question: str, context: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """
        Construct a prompt with the question and relevant context,
        ensuring it stays within token limits.
        
        When token limits are exceeded, intelligently select the most relevant
        documents based on their similarity scores.
        
        Returns:
            tuple: (prompt_text, used_sources) where used_sources is a list of contexts actually used
        """
        if not context:
            return question, []
            
        # Create the prompt base structure
        base_prompt = "Please answer the following question based on the provided context:\n\n"
        question_part = f"Question: {question}\n\n"
        instruction_part = "Answer the question concisely and accurately based on the provided context. Also, provide the context directly in the prompt if exist with proper Markdown structure. If the context doesn't contain relevant information, please state that clearly."
        
        # Calculate tokens for fixed parts
        fixed_tokens = estimate_token_count(base_prompt + question_part + instruction_part)
        
        # Calculate max available tokens for context
        available_tokens = self.max_tokens - fixed_tokens - 100  # 100 token buffer
        
        # Initialize context string and token counter
        context_str = "Context:\n"
        current_tokens = estimate_token_count(context_str)
        
        # Initialize list to track which sources are actually used
        used_sources = []
        
        # Calculate all document token counts and total
        doc_tokens = []
        for doc in context:
            doc_context = f"Document X (from ./documents/{doc['metadata'].get('source', 'Unknown')}):\n{doc['content']}\n\n"
            doc_tokens.append(estimate_token_count(doc_context))
        
        total_context_tokens = sum(doc_tokens)
        
        # Check if we need to intelligently select documents due to token limit
        if current_tokens + total_context_tokens > available_tokens:
            logging.warning(f"Context exceeds token limit of available {available_tokens} tokens. Using intelligent document selection.")
            
            # Since context is already ordered by similarity from search engine,
            # create a new prioritized list based on similarity, focusing on the most relevant
            prioritized_context = context[:max(1, self.top_k//2)]
            
            # Log the documents we're keeping
            logging.warning(f"Selected {len(prioritized_context)} most relevant documents out of {len(context)}")
            
            # Replace our context with the prioritized subset
            context = prioritized_context
        
        # Add contexts while respecting token limit. 
        # This is applied due to potential exceeding of token limit after concatenating all contexts.
        added_contexts = 0
        for i, doc in enumerate(context):
            # Extract document information
            source = doc["metadata"].get("source", "Unknown")
            content = doc['content']
            similarity = doc.get("similarity", 0)
            
            # Create this document's context entry with similarity score
            doc_context = f"Document {i+1} (from ./documents/{source}, similarity: {similarity:.4f}):\n{content}\n\n"
            doc_tokens = estimate_token_count(doc_context)
            
            # Check if adding this context would exceed the limit
            if current_tokens + doc_tokens > available_tokens:
                # If no contexts added yet, include a truncated version
                if added_contexts == 0:
                    # Take whatever we can fit
                    chars_to_keep = (available_tokens - current_tokens) * 4
                    truncated_content = content[:chars_to_keep] + "... [truncated due to length]"
                    doc_context = f"Document {i+1} (from ./documents/{source}, similarity: {similarity:.4f}):\n{truncated_content}\n\n"

                    # This can still fail due to approximation of token size. Therefore, I used warning message to inform the user.
                    context_str += doc_context
                    console.print(f"[yellow]Warning:[/yellow] Document truncated to fit token limit")
                    # Add to used sources with truncation indicator
                    doc["truncated"] = True
                    used_sources.append(doc)
                break
            
            # Add this context
            context_str += doc_context
            current_tokens += doc_tokens
            added_contexts += 1
            # Add to used sources
            used_sources.append(doc)
        
        # Combine all parts
        prompt = base_prompt + context_str + question_part + instruction_part
        
        # Log the final token count estimation
        token_estimate = estimate_token_count(prompt)
        console.print(f"Constructed prompt with ~{token_estimate} tokens (limit: {self.max_tokens})")
        
        return prompt, used_sources
    
    def _construct_prompt_with_representation(self, question: str, context: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
        """
        Construct a prompt by selecting the most representative chunk from each document.
        
        This approach:
        1. Groups chunks by their source document
        2. Re-embeds the question
        3. Finds the most similar chunk from each document
        4. Uses those representative chunks in the prompt
        
        Returns:
            tuple: (prompt_text, used_sources) where used_sources is a list of contexts actually used
        """
        if not context:
            return question, []
        
        # Create the prompt base structure
        base_prompt = "Please answer the following question based on the provided context:\n\n"
        question_part = f"Question: {question}\n\n"
        instruction_part = "Answer the question concisely and accurately based on the provided context. Also, provide the context directly in the prompt if exist with proper Markdown structure. If the context doesn't contain relevant information, please state that clearly."
        
        # Calculate tokens for fixed parts
        fixed_tokens = estimate_token_count(base_prompt + question_part + instruction_part)
        
        # Calculate max available tokens for context
        available_tokens = self.max_tokens - fixed_tokens - 100  # 100 token buffer
        
        # Group chunks by source document
        document_groups = {}
        for doc in context:
            source = doc["metadata"].get("source", "Unknown")
            if source not in document_groups:
                document_groups[source] = []
            document_groups[source].append(doc)
        
        console.print(f"Found content from [bold]{len(document_groups)}[/bold] different documents")
        
        # Re-embed the question for direct comparison
        question_embedding = self.embed_service.embed(question)
        
        # Get the most representative chunk from each document
        representative_chunks = []
        for source, chunks in document_groups.items():
            # Skip if no embedding available
            if not question_embedding:
                representative_chunks.extend(chunks[:1])  # Just take first chunk
                continue
            
            # Find most similar chunk in this document
            most_similar_chunk = None
            highest_similarity = -1
            
            for chunk in chunks:
                # Get the chunk's embedding
                if "embedding" in chunk:
                    chunk_embedding = chunk["embedding"]
                else:
                    # If embedding not stored with the chunk, recalculate it
                    chunk_embedding = self.embed_service.embed(chunk["content"])
                
                if chunk_embedding:
                    # Calculate similarity with question
                    similarity = self._calculate_similarity(
                        np.array(question_embedding), 
                        np.array(chunk_embedding)
                    )
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        most_similar_chunk = chunk
            
            # Add the most similar chunk, or the first chunk if no similarity could be calculated
            if most_similar_chunk:
                representative_chunks.append(most_similar_chunk)
            elif chunks:
                representative_chunks.append(chunks[0])
        
        # Sort by similarity if available
        representative_chunks.sort(
            key=lambda x: x.get("similarity", 0), 
            reverse=self.similarity_method != "euclidean"
        )
        
        # Now use the same token-aware approach as _construct_prompt
        context_str = "Context:\n"
        current_tokens = estimate_token_count(context_str)
        used_sources = []
        
        for i, doc in enumerate(representative_chunks):
            source = doc["metadata"].get("source", "Unknown")
            content = doc["content"]
            similarity = doc.get("similarity", 0)
            
            # Create context entry
            doc_context = f"Document {i+1} (from ./documents/{source}, similarity: {similarity:.4f}):\n{content}\n\n"
            doc_tokens = estimate_token_count(doc_context)
            
            # Check if adding this would exceed the limit
            if current_tokens + doc_tokens > available_tokens:
                if i == 0:  # If we can't even add one document
                    chars_to_keep = (available_tokens - current_tokens) * 4
                    truncated_content = content[:chars_to_keep] + "... [truncated due to length]"
                    doc_context = f"Document {i+1} (from ./documents/{source}, similarity: {similarity:.4f}):\n{truncated_content}\n\n"
                    context_str += doc_context
                    console.print(f"[yellow]Warning:[/yellow] Document truncated to fit token limit")
                    doc["truncated"] = True
                    used_sources.append(doc)
                break
            
            # Add this context
            context_str += doc_context
            current_tokens += doc_tokens
            used_sources.append(doc)
        
        # Combine all parts
        prompt = base_prompt + context_str + question_part + instruction_part
        
        # Log the final token count estimation
        token_estimate = estimate_token_count(prompt)
        console.print(f"Constructed prompt with [bold]{len(used_sources)}[/bold] document representations, ~{token_estimate} tokens (limit: {self.max_tokens})")
        
        return prompt, used_sources
    
    # Following code might be look like a duplicate but it is not. 
    # It is because of _construct_prompt_with_representation 
    # because of its requirement of calculating similarity between question and the documents.
    def _calculate_similarity(self, query_embedding: np.ndarray, doc_embedding: np.ndarray) -> float:
        """
        Calculate similarity between two embeddings using the configured method.
        
        Args:
            query_embedding: The query embedding as a numpy array
            doc_embedding: The document embedding as a numpy array
            
        Returns:
            Similarity score
        """
        if self.similarity_method == "cosine":
            return self._cosine_similarity(query_embedding, doc_embedding)
        elif self.similarity_method == "dot_product":
            return self._dot_product(query_embedding, doc_embedding)
        elif self.similarity_method == "euclidean":
            return self._euclidean_distance(query_embedding, doc_embedding)
        else:
            # Default to cosine
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
        
    def _format_answer(self, answer: str, context: List[Dict[str, Any]]) -> str:
        """Format the answer with source attribution if requested"""
        if not self.include_sources or not context:
            return answer
            
        # Add source information
        formatted_answer = answer + "\n\n"
        formatted_answer += "Sources:\n"
        
        for i, doc in enumerate(context):
            source = doc["metadata"].get("source", "Unknown")
            heading = ""
            
            # Try to get heading information
            for key, value in doc["metadata"].items():
                if key.startswith("heading") and value:
                    heading = f" - {value}"
                    break
            
            # Note whether the source was truncated        
            truncated_note = " [truncated]" if doc.get("truncated", False) else ""
            formatted_answer += f"{i+1}. {source}{heading}{truncated_note}\n"
            
        return formatted_answer
