from typing import List, Literal
from src.api import embed_texts
import logging

logger = logging.getLogger('embed')

class EmbedService:

  def __init__(self, embed_type : Literal['document', 'query']):
    self.embed_type = embed_type

  def embed(self, text: str) -> List[float]:
    embeddings = embed_texts([text], self.embed_type)
    
    return embeddings['embeddings'][0]  # Ensure to return embedding vector only