import logging
from src.operations.ask import LLMAsker
from src.operations.search import SearchEngine
from src.operations.embed import EmbedService
from src.operations.extract import MarkdownExtractor
import argparse
from typing import List
import sqlite3
import pickle
import os
import glob

from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from langchain_core.documents import Document
import uuid

# Import our new chunking strategies
from src.chunking import get_strategy

# Used only for search and ask-question modes to display results in a nice format
from rich.console import Console
console = Console()

class App:
    """ The main class of the application. """

    def __init__(self):
        pass

    def run(self):
        parser = argparse.ArgumentParser(description='Ask questions about the files of a case.')

        parser.add_argument('--mode', 
                            choices=['index-files', 'ask-question', 'search', 'get-markdown', 'load-files'], 
                            default='ask-question', 
                            help='The mode of the application.')
        parser.add_argument('--chunking', 
                            choices=['none', 'header', 'size', 'adaptive'], 
                            default='adaptive', 
                            help='Chunking strategy for document indexing')
        parser.add_argument('--overwrite', 
                            action='store_true',
                            help='Overwrite existing chunks instead of skipping them')
        parser.add_argument('--similarity',
                            choices=['cosine', 'dot_product', 'euclidean'],
                            default='cosine',
                            help='Similarity method for search')
        parser.add_argument('--top-k',
                            type=int,
                            default=5,
                            help='Number of top results to show in search')
        parser.add_argument('--construction',
                            choices=['intelligent', 'representative'],
                            default='intelligent',
                            help='Prompt construction method')
        parser.add_argument('question', nargs='?', type=str, 
                            help='The question to ask about the files of a case.')

        args = parser.parse_args()

        if args.mode == 'index-files':
            self.load_files(chunking_strategy=args.chunking, skip_duplicates=not args.overwrite)
        elif args.mode == 'ask-question':
            question = args.question
            if not question or question.isspace():
                parser.error('The question argument is required in "ask-question" mode.')
            self.ask_question(question, similarity=args.similarity, top_k=args.top_k, construction=args.construction)
        elif args.mode == 'search':
            query = args.question
            if not query or query.isspace():
                parser.error('The query argument is required in "search" mode.')
            self.search(query, similarity_method=args.similarity, top_k=args.top_k)
        elif args.mode == 'get-markdown':
            self.get_markdown()

    def load_files(self, chunking_strategy: str = 'adaptive', skip_duplicates: bool = True):
        """
        Load and index markdown files from the output directory using a specified chunking strategy to the SQLITE3 Database.
        
        Args:
            chunking_strategy: Strategy to use for document chunking ('none', 'header', 'size', 'adaptive')
            skip_duplicates: Whether to skip documents that already exist in the database
        """
        logging.info(f"Loading markdown files with '{chunking_strategy}' chunking strategy...")
        logging.info(f"Skip duplicates: {'enabled' if skip_duplicates else 'disabled'}")

        # Get the appropriate chunking strategy
        strategy = get_strategy(chunking_strategy)
        logging.info(f"Using {strategy.__class__.__name__}")

        # Find markdown files in the output directory even if nested
        markdown_files = glob.glob('./markdown_output/**/*.md', recursive=True)
        logging.info(f"Found {len(markdown_files)} markdown files")
        
        # Process files with the selected chunking strategy
        all_documents = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            task = progress.add_task("Processing markdown files...", total=len(markdown_files))
            
            for file_path in markdown_files:
                try:
                    # Read the markdown file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create metadata from file info from md files
                    metadata = {
                        "source": file_path, # Store the source file path e.g '/markdown_output\\file.md'
                        "filename": os.path.basename(file_path) # Store the filename e.g 'file.md'
                    }
                    
                    # Extract YAML frontmatter if present
                    # This is required due to additional metadata in front of the content
                    content_without_frontmatter = content
                    if content.startswith('---'):
                        end_idx = content.find('---', 3)
                        if end_idx != -1:
                            frontmatter = content[3:end_idx].strip()
                            # Parse frontmatter into metadata
                            for line in frontmatter.split('\n'):
                                if ':' in line:
                                    key, value = line.split(':', 1)
                                    cleaned_value = value.strip().strip('"')
                                    metadata[key.strip()] = cleaned_value # Overwrite the metadata['source'] with the correct extension
                                    
                            # Remove frontmatter from content for processing
                            content_without_frontmatter = content[end_idx+3:].strip()
                    
                    # Apply the chunking strategy to get document chunks
                    document_chunks = strategy.chunk_document(content_without_frontmatter, metadata)
                    all_documents.extend(document_chunks)
                
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                
                progress.update(task, advance=1)
        
        # Show chunking results
        logging.info(f"Processed into {len(all_documents)} document chunks")

        # Embed each chunk using EmbedService
        embed_service = EmbedService('document')
        DB_FILE = "./llm_db/markdown_embeddings.db"
        llm_db = LLM_DB(DB_FILE, embed_service)
        llm_db.add_chunks(all_documents, skip_duplicates=skip_duplicates)
        # llm_db.fetch_first_embedding()
        # llm_db.fetch_data_by_metadata_source("NDA_filled.docx")
        
        logging.info("✅ Successfully indexed markdown files")

    def search(self, query, similarity_method="cosine", top_k=5):
        """
        Search for documents similar to the query
        
        Args:
            query: Text query to search for
            similarity_method: Method to use for similarity calculation
            top_k: Number of top results to return
        """
        
        console.print(f"Searching for: [bold cyan]{query}[/bold cyan]")
        console.print(f"Using [bold]{similarity_method}[/bold] similarity method, showing top {top_k} results")
        
        # Generate embedding for the query
        embed_service = EmbedService('query')
        query_embedding = embed_service.embed(query)
        
        if not query_embedding:
            logging.error("Failed to generate embedding for query")
            return
        
        # Initialize search engine and perform search
        DB_FILE = "./llm_db/markdown_embeddings.db"
        search_engine = SearchEngine(db_path=DB_FILE, similarity_method=similarity_method, top_k=top_k)
        results = search_engine.search(query_embedding)
        
        if not results:
            logging.warning("No matching results found.")
            return
        
        # Display results
        console.print(f"\n[bold green]Found {len(results)} matching results:[/bold green]")
        for i, result in enumerate(results):
            similarity_score = result["similarity"]
            content_preview = result["content"][:200] + "..." if len(result["content"]) > 200 else result["content"]
            
            # Prepare a clean format for metadata
            source = result["metadata"].get("source", "Unknown source")
            filename = result["metadata"].get("filename", os.path.basename(source) if source else "Unknown file")
            
            console.print(f"\n[bold]{i+1}. {filename}[/bold] [cyan](Score: {similarity_score:.4f})[/cyan]")
            
            # Show heading info if available
            heading_info = []
            for key, value in result["metadata"].items():
                if key.startswith("chunk_headers"):
                    heading_info.append(f"{value}")
            
            if heading_info:
                console.print("[italic]" + " > ".join(heading_info) + "[/italic]")
                
            console.print(content_preview)
            console.print("-" * 80)

    def get_markdown(self):
        print("Converting documents to Markdown...")
        extractor = MarkdownExtractor(output_dir="./markdown_output", max_workers=1)  # Reduced workers for stability
        
        # Extracting all documents from the directory
        results = extractor.extract_all_documents(doc_dir="./documents")
        
        # Group results by file type for reporting
        # Counting up the extensions of the saved files
        file_types = {}
        for filename in results.keys():
            ext = os.path.splitext(filename.lower())[1]
            if ext not in file_types:
                file_types[ext] = 0
            file_types[ext] += 1
        
        logging.info(f"✅ Successfully converted {len(results)} files to Markdown:")
        
        # Reporting saved file types with the number of files
        for ext, count in file_types.items():
            print(f"  - {ext} files: {count}")
        logging.info(f"Markdown files saved to: ./markdown_output")

    def ask_question(self, question, similarity, top_k, construction):
        logging.info(f'Asking question: {question}')
        operator = LLMAsker(
            similarity_method=similarity, 
            top_k=top_k, 
            max_tokens=10000,
            construction_method=construction
        )
        response = operator.ask(question)
        print(response)


class LLM_DB:
    """
    Class to handle SQLite database operations for storing and retrieving document embeddings.
    Storing strategy is to store the document content, metadata, and embedding as a BLOB.
    """
    def __init__(self, DB_FILE : str, embed_service : EmbedService):
        """
        Initialize the database connection and create the table if it doesn't exist.
        """
        self.DB_FILE = DB_FILE
        self._embed_service = embed_service
        self.create_table(DB_FILE)

    def create_table(self, DB_PATH : str):
        """Ensure the folder exists and create the SQLite database."""
        db_folder = os.path.dirname(DB_PATH)  # Extract folder from path

        if db_folder and not os.path.exists(db_folder):  # Ensure folder exists if not create a folder
            os.makedirs(db_folder)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        # Only metadata can be NULL, but it is also not expected to be NULL most of the time.
        c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                metadata TEXT,
                embedding BLOB NOT NULL
            )''')
        conn.commit()
        conn.close()

    def add_chunks(self, chunks: List[Document], skip_duplicates: bool = True):
        """
        Stores document chunks and their embeddings into SQLite.
        
        Args:
            chunks: List of Document objects to store
            skip_duplicates: If True, skip chunks that already exist in the database
                            If False, overwrite existing chunks
        """
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} chunks added"),
        ) as progress:

            task = progress.add_task("Indexing files...", total=len(chunks))

            for chunk in chunks:
                content = chunk.page_content.strip()
                metadata = str(chunk.metadata)
                embedding = self._embed_service.embed(content)

                if not embedding:  # Validate that embeddings are generated
                    logging.warning(f"⚠️ Embedding failed for chunk: {content[:50]}...")
                    progress.update(task, advance=1)
                    continue

                embedding_blob = pickle.dumps(embedding)

                # Check if the chunk already exists
                cursor.execute("SELECT id FROM embeddings WHERE content = ?", (content,))
                existing_id = cursor.fetchone()
                
                if existing_id:
                    # Handle existing chunks based on skip_duplicates parameter
                    if skip_duplicates:
                        logging.info(f"🔹 Skipping duplicate chunk: {content[:50]}...")
                    else:
                        # Update the existing entry with ALL fields
                        logging.info(f"🔄 Updating existing chunk: {content[:50]}...")
                        cursor.execute("UPDATE embeddings SET content = ?, metadata = ?, embedding = ? WHERE id = ?",
                                    (content, metadata, embedding_blob, existing_id[0]))
                else:
                    # Insert new chunk
                    cursor.execute("INSERT INTO embeddings (id, content, metadata, embedding) VALUES (?, ?, ?, ?)",
                                (str(uuid.uuid4()), content, metadata, embedding_blob))
                
                progress.update(task, advance=1)

        conn.commit()
        conn.close()
        logging.info("✅ Files successfully indexed into SQLite.")

    def fetch_first_embedding(self):
        """
        Usage: Debugging method to verify that embeddings are stored correctly.
        Fetch the first embedding from the database and print the first 10 elements.
        """
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()

        # Retrieve the first embedding from the database
        cursor.execute("SELECT embedding FROM embeddings LIMIT 1")
        row = cursor.fetchone()

        if row:
            embedding_blob = row[0]  # Extract BLOB data
            embedding = pickle.loads(embedding_blob)  # Deserialize BLOB to a list
            print("First 10 elements of the embedding:", embedding[:10])  # Print first 10 elements
        else:
            print("No embeddings found in the database.")

        conn.close()


    def fetch_data_by_metadata_source(self, source: str):
        """
        Usage: Debugging method to verify that metadata is stored correctly.
        Fetch data from the database based on metadata containing the source string.
        """
        conn = sqlite3.connect(self.DB_FILE)
        cursor = conn.cursor()

        # Retrieve data based on metadata
        cursor.execute("SELECT content FROM embeddings WHERE metadata=?", (source,))
        rows = cursor.fetchall()

        if rows:
            print(f"Found {len(rows)} entries with metadata containing {source}")
            for row in rows:
                print(row[0][:50] + "...")
        else:
            print(f"No entries found with metadata containing {source}")
            embeddings = cursor.execute("SELECT content, metadata FROM embeddings").fetchall()
            print(embeddings)

        conn.close()