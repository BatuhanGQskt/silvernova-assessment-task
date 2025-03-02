import sqlite3
import os

def verify_indexed_docs():
    """Verify that documents have been properly indexed in the database."""
    DB_FILE = "./llm_db/markdown_embeddings.db"
    
    if not os.path.exists(DB_FILE):
        print(f"❌ Database file {DB_FILE} not found. Run load_files() first.")
        return
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        total_count = cursor.fetchone()[0]
        
        # Get sample of content
        cursor.execute("SELECT content, metadata FROM embeddings LIMIT 5")
        samples = cursor.fetchall()
        
        print(f"✅ Found {total_count} embedded chunks in the database.")
        print("\n--- Sample content ---")
        
        for i, (content, metadata) in enumerate(samples):
            print(f"\nSample {i+1}:")
            print(f"Metadata: {metadata}")
            print(f"Content snippet: {content[:150]}...")
        
    except Exception as e:
        print(f"❌ Error accessing database: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    verify_indexed_docs()
